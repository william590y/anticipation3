#!/usr/bin/env python3
"""Model soups (Wortsman et al., ICML 2022) over packed-format ASAP checkpoints.

A "soup" is the elementwise AVERAGE OF THE WEIGHTS of several models that were
fine-tuned from one common initialization.  Two variants are built here:

  * UNIFORM soup  -- mean of every candidate.  Reported as a reference point;
    it is the variant that can be dragged below the best single model by one
    bad ingredient.
  * GREEDY soup   -- candidates sorted best-first by their own validation
    score, then each is folded in only if the resulting soup does NOT get
    worse on validation.  This is the variant the paper shows reliably matches
    or beats the best individual model, so it is the one written to --out_dir.

The validation score is a TEACHER-FORCED cross-entropy on a FIXED subset of
--val_file, evaluated with train.py's own `TokenizedDataset` (is_training=False,
so no augmentation) and `forward_batch`.  The subset is drawn once with
--subset_seed, the batches are materialized once, and every soup is scored on
those identical batches, so all comparisons here are exactly paired.  The loss
is token-weighted (sum of NLL over non-ignored label positions / token count),
NOT a mean of per-batch means, so batch composition cannot shift it.

Numbers printed here are computed in fp32 without autocast; train.py validates
under bf16 autocast, so the absolute values are not directly comparable to a
`val/loss` from a training log.  Only the paired differences within one run of
this script are meaningful -- which is all the greedy procedure needs.

Usage
-----
    # explicit candidate list
    python soup.py --checkpoints run_soup_seed1/checkpoint-3000 ... \
        --out_dir ./run_soup_greedy

    # let the script pick each run's best checkpoint from its own train.log
    python soup.py --run_dirs run_soup_seed1 ... run_soup_seed10 \
        --out_dir ./run_soup_greedy

    # just print the per-run best-checkpoint picks and exit
    python soup.py --run_dirs run_soup_seed* --print_best

    # numerical self-test (souping N copies of one checkpoint must be a no-op)
    python soup.py --selftest
"""

import argparse
import os
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INIT_CHECKPOINT = "run_paper_split_v2/checkpoint-2500"


# --------------------------------------------------------------------------
# checkpoint I/O
# --------------------------------------------------------------------------

def load_state_dict(checkpoint_dir):
    """Read one HF checkpoint's weights as a CPU fp32 state dict."""
    checkpoint_dir = Path(checkpoint_dir)
    safetensors_path = checkpoint_dir / "model.safetensors"
    bin_path = checkpoint_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_path), device="cpu")
    elif bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"No model.safetensors or pytorch_model.bin in {checkpoint_dir}"
        )

    return {k: v.to(torch.float32) if v.is_floating_point() else v for k, v in state.items()}


def accumulate_into(accumulator, state):
    """accumulator += state, elementwise. Returns the accumulator."""
    if accumulator is None:
        return {k: v.clone() for k, v in state.items()}
    if set(accumulator.keys()) != set(state.keys()):
        missing = set(accumulator.keys()) ^ set(state.keys())
        raise RuntimeError(f"Checkpoints have mismatched parameter sets: {sorted(missing)[:8]}")
    for k, v in state.items():
        if accumulator[k].shape != v.shape:
            raise RuntimeError(
                f"Shape mismatch for {k}: {tuple(accumulator[k].shape)} vs {tuple(v.shape)}"
            )
        accumulator[k].add_(v)
    return accumulator


def scale_state(accumulator, denominator):
    """Return accumulator / denominator as a fresh state dict."""
    inv = 1.0 / float(denominator)
    return {k: (v * inv) for k, v in accumulator.items()}


def save_soup(state, out_dir, config_source):
    """Write `state` as a plain HF checkpoint dir (config.json + safetensors).

    `evaluate_muster_asap.load_model` (== `evaluate_muster.load_model`) takes the
    `config.json` branch, i.e. plain `AutoModelForCausalLM.from_pretrained`.
    """
    from transformers import AutoModelForCausalLM

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(str(config_source), local_files_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when materializing soup: {unexpected}")
    allowed_missing = {"lm_head.weight"}  # tied to transformer.wte.weight in GPT-2
    if set(missing) - allowed_missing:
        raise RuntimeError(f"Missing keys when materializing soup: {missing}")
    model.tie_weights()
    model.save_pretrained(str(out_dir), safe_serialization=True)
    return out_dir


# --------------------------------------------------------------------------
# per-run best-checkpoint selection (the "tiny helper")
# --------------------------------------------------------------------------

_VAL_EVENT_RE = re.compile(
    r"Running validation at (?P<kind>(?:checkpoint |final )?)step (?P<step>\d+)\.\.\."
    r"|Validation Loss:\s*(?P<loss>[0-9]+(?:\.[0-9]+)?)"
)


def parse_validation_losses(log_path):
    """Pair each 'Running validation at ... step N' with the 'Validation Loss:' that follows.

    Returns a list of (kind, step, loss) in file order.  `kind` is '' for a plain
    eval_steps validation, 'checkpoint' for one that precedes a checkpoint save,
    'final' for the end-of-run validation.
    """
    text = Path(log_path).read_text(errors="replace")
    events, pending = [], None
    for m in _VAL_EVENT_RE.finditer(text):
        if m.group("loss") is not None:
            if pending is not None:
                events.append((pending[0], pending[1], float(m.group("loss"))))
                pending = None
        else:
            pending = (m.group("kind").strip(), int(m.group("step")))
    return events


def pick_best_checkpoint(run_dir, log_path=None):
    """Best (lowest val/loss) SAVED checkpoint of one run.

    Only validations that immediately precede a checkpoint save correspond to a
    directory on disk, so plain eval_steps validations are ignored.  Returns
    (checkpoint_dir, val_loss, note).  Falls back to the highest-numbered
    checkpoint with a loud note if the log is unusable -- the last checkpoint is
    NOT assumed to be the best, it is only the last resort.
    """
    run_dir = Path(run_dir)
    log_path = Path(log_path) if log_path else run_dir / "train.log"

    on_disk = {}
    for d in run_dir.glob("checkpoint-*"):
        if (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists():
            try:
                on_disk[int(d.name.split("-")[1])] = d
            except (IndexError, ValueError):
                continue
    if not on_disk:
        raise FileNotFoundError(f"No usable checkpoint-* dirs in {run_dir}")

    if log_path.exists():
        candidates = [
            (loss, step)
            for kind, step, loss in parse_validation_losses(log_path)
            if kind == "checkpoint" and step in on_disk
        ]
        if candidates:
            best_loss, best_step = min(candidates)
            note = f"best of {len(candidates)} logged checkpoint validations"
            return on_disk[best_step], best_loss, note

    last = max(on_disk)
    note = (
        f"WARNING: no usable validation losses in {log_path} -- "
        f"FELL BACK to the last checkpoint ({last}), which is not necessarily the best"
    )
    return on_disk[last], float("nan"), note


# --------------------------------------------------------------------------
# fixed-subset teacher-forced validation
# --------------------------------------------------------------------------

class FixedSubsetScorer:
    """Teacher-forced val loss on one fixed, pre-materialized batch list."""

    def __init__(self, val_file, num_windows, batch_size, device, subset_seed, config_source):
        import train as base_train
        from transformers import AutoModelForCausalLM

        self.base_train = base_train

        t0 = time.time()
        dataset = base_train.TokenizedDataset(val_file, is_training=False)
        if len(dataset) == 0:
            raise ValueError(f"Validation file is empty: {val_file}")
        n = min(num_windows, len(dataset))
        indices = sorted(random.Random(subset_seed).sample(range(len(dataset)), n))

        def collate_fn(batch):
            return {
                key: torch.stack([item[key] for item in batch])
                for key in ("input_ids", "attention_mask", "labels",
                            "score_token_mask", "score_mask")
            }

        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        # Materialize once: every soup is then scored on byte-identical inputs.
        self.batches = list(loader)
        self.num_windows = n
        print(
            f"Fixed validation subset: {n} windows of {len(dataset)} from {val_file} "
            f"(subset_seed={subset_seed}, {len(self.batches)} batches, "
            f"{time.time() - t0:.1f}s)"
        )

        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(config_source), local_files_only=True
        )
        self.model.config.use_cache = False
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, state):
        """Token-weighted mean teacher-forced CE of `state` on the fixed subset."""
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys while scoring: {unexpected}")
        if set(missing) - {"lm_head.weight"}:
            raise RuntimeError(f"Missing keys while scoring: {missing}")
        self.model.tie_weights()
        self.model.eval()

        total_nll = 0.0
        total_tokens = 0
        for batch in self.batches:
            moved = {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            outputs = self.base_train.forward_batch(self.model, moved)
            logits = outputs.logits[:, :-1, :]
            labels = moved["labels"][:, 1:]
            # Row-by-row so the fp32 log_softmax never materializes a
            # (batch * 1019, 55028) intermediate.
            for row in range(logits.shape[0]):
                target = labels[row]
                nll = F.cross_entropy(
                    logits[row].float(), target, ignore_index=-100, reduction="sum"
                )
                total_nll += float(nll)
                total_tokens += int((target != -100).sum())
        if total_tokens == 0:
            raise RuntimeError("Validation subset produced zero supervised tokens")
        return total_nll / total_tokens


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def run_selftest(checkpoint, workdir):
    print("=" * 78)
    print("SELF-TEST: souping copies of one checkpoint must be a numerical no-op")
    print("=" * 78)

    # (a) tiny synthetic check of the averaging arithmetic itself
    a = {"w": torch.tensor([0.0, 2.0]), "b": torch.tensor([[1.0]])}
    b = {"w": torch.tensor([4.0, 8.0]), "b": torch.tensor([[3.0]])}
    mixed = scale_state(accumulate_into(accumulate_into(None, a), b), 2)
    assert torch.equal(mixed["w"], torch.tensor([2.0, 5.0])), mixed["w"]
    assert torch.equal(mixed["b"], torch.tensor([[2.0]])), mixed["b"]
    print("  [1/4] synthetic 2-model average is exact               OK")

    # (b) real checkpoint, 2 copies -> bit-identical
    original = load_state_dict(checkpoint)
    accum = None
    for _ in range(2):
        accum = accumulate_into(accum, load_state_dict(checkpoint))
    soup2 = scale_state(accum, 2)
    bad = [k for k in original if not torch.equal(soup2[k], original[k])]
    assert not bad, f"2-copy soup differs from input for {len(bad)} tensors, e.g. {bad[:5]}"
    print(f"  [2/4] 2-copy soup == input for all {len(original)} tensors   OK")

    # (c) 3 copies.  The denominator is not a power of two, so this one is only
    # exact up to fp32 rounding -- the bar is "within a couple of ULPs of the
    # largest weight", not bit-equality.
    accum = None
    for _ in range(3):
        accum = accumulate_into(accum, load_state_dict(checkpoint))
    soup3 = scale_state(accum, 3)
    del accum
    worst = 0.0
    scale = 0.0
    for k, v in original.items():
        if not v.is_floating_point():
            continue
        worst = max(worst, float((soup3[k] - v).abs().max()))
        scale = max(scale, float(v.abs().max()))
    del soup3
    tolerance = 4 * float(torch.finfo(torch.float32).eps) * max(scale, 1.0)
    assert worst <= tolerance, (
        f"3-copy soup max abs deviation {worst:g} exceeds the fp32 rounding "
        f"floor {tolerance:g} (max |w| = {scale:g})"
    )
    print(
        f"  [3/4] 3-copy soup max|soup - input| = {worst:.3g} <= {tolerance:.3g}  OK"
    )

    # (d) round-trip through the on-disk format the eval scripts load
    tmp = Path(tempfile.mkdtemp(prefix=".soup_selftest_", dir=str(workdir)))
    try:
        save_soup(soup2, tmp, config_source=checkpoint)
        from evaluate_muster import load_model

        model, device = load_model(str(tmp))
        reloaded = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
        checked = 0
        for k, v in original.items():
            if k not in reloaded:
                raise AssertionError(f"{k} missing after save/load round-trip")
            if not torch.equal(reloaded[k], v):
                raise AssertionError(f"{k} changed in the save/load round-trip")
            checked += 1
        del model
        print(
            f"  [4/4] evaluate_muster_asap.load_model round-trip on {device}: "
            f"{checked} tensors identical  OK"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("SELF-TEST PASSED")
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def resolve_device(requested):
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="explicit list of HF checkpoint dirs to soup")
    parser.add_argument("--run_dirs", nargs="+", default=None,
                        help="run dirs; each contributes its own BEST checkpoint, "
                             "chosen from the run's logged validation losses")
    parser.add_argument("--out_dir", type=str, default="./run_soup_greedy",
                        help="where the GREEDY soup is written as an HF checkpoint dir")
    parser.add_argument("--uniform_out_dir", type=str, default=None,
                        help="optional dir for the uniform soup (default: <out_dir>_uniform)")
    parser.add_argument("--no_save_uniform", action="store_true",
                        help="do not write the uniform soup to disk (it is still scored)")
    parser.add_argument("--val_file", type=str, default="data/val_paper.txt")
    parser.add_argument("--val_windows", type=int, default=384,
                        help="size of the FIXED validation subset")
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--subset_seed", type=int, default=20260830,
                        help="seed for the fixed validation subset (keep it constant "
                             "across runs to keep results comparable)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--results_file", type=str, default="results/model_soup.txt")
    parser.add_argument("--print_best", action="store_true",
                        help="only resolve --run_dirs to their best checkpoints and exit")
    parser.add_argument("--selftest", action="store_true",
                        help="soup copies of one checkpoint and assert the soup is "
                             "numerically identical to the input")
    parser.add_argument("--selftest_checkpoint", type=str, default=DEFAULT_INIT_CHECKPOINT)
    args = parser.parse_args()

    if args.selftest:
        return run_selftest(args.selftest_checkpoint, REPO_ROOT)

    # ---- resolve the candidate list -------------------------------------
    picks = []  # (label, path, logged_val_loss_or_nan, note)
    if args.run_dirs:
        for run_dir in args.run_dirs:
            path, loss, note = pick_best_checkpoint(run_dir)
            picks.append((Path(run_dir).name, path, loss, note))
    if args.checkpoints:
        for ckpt in args.checkpoints:
            picks.append((Path(ckpt).parent.name + "/" + Path(ckpt).name, Path(ckpt),
                          float("nan"), "explicit"))
    if not picks:
        parser.error("give --checkpoints and/or --run_dirs (or --selftest)")

    print("Candidates:")
    for label, path, loss, note in picks:
        logged = f"{loss:.4f}" if loss == loss else "n/a"
        print(f"  {label:24s} -> {str(path):48s} logged val/loss {logged}  [{note}]")
    if args.print_best:
        return 0

    for _, path, _, _ in picks:
        if not (Path(path) / "config.json").exists():
            raise FileNotFoundError(f"{path} has no config.json (not an HF checkpoint dir)")

    device = resolve_device(args.device)
    print(f"\nScoring device: {device}")
    config_source = picks[0][1]
    scorer = FixedSubsetScorer(
        val_file=args.val_file,
        num_windows=args.val_windows,
        batch_size=args.val_batch_size,
        device=device,
        subset_seed=args.subset_seed,
        config_source=config_source,
    )

    # ---- pass 1: solo scores + uniform accumulator ------------------------
    print("\nPass 1: solo validation loss of each candidate (and uniform sum)")
    uniform_accum = None
    solo = []
    for label, path, logged, _ in picks:
        t0 = time.time()
        state = load_state_dict(path)
        loss = scorer.score(state)
        uniform_accum = accumulate_into(uniform_accum, state)
        del state
        solo.append({"label": label, "path": str(path), "logged": logged, "solo": loss})
        print(f"  {label:24s} solo val loss {loss:.6f}   ({time.time() - t0:.1f}s)")

    uniform_state = scale_state(uniform_accum, len(picks))
    del uniform_accum
    uniform_loss = scorer.score(uniform_state)
    print(f"\nUNIFORM soup of all {len(picks)}: val loss {uniform_loss:.6f}")

    # Write (and free) the uniform soup now: holding a second full state dict
    # through the greedy loop is 1.4 GB of RSS for nothing.
    uniform_dir = None
    if not args.no_save_uniform:
        uniform_dir = Path(args.uniform_out_dir or (str(args.out_dir).rstrip("/") + "_uniform"))
        save_soup(uniform_state, uniform_dir, config_source=config_source)
        print(f"Wrote UNIFORM soup ({len(picks)} ingredients) to {uniform_dir}")
    del uniform_state

    # ---- greedy soup ------------------------------------------------------
    order = sorted(solo, key=lambda r: r["solo"])
    print("\nPass 2: greedy soup (candidates in ascending solo val loss)")
    greedy_accum = load_state_dict(order[0]["path"])
    greedy_loss = order[0]["solo"]
    accepted = [order[0]["label"]]
    decisions = [{
        "label": order[0]["label"],
        "solo": order[0]["solo"],
        "trial": order[0]["solo"],
        "accepted": True,
        "soup_size": 1,
    }]
    print(f"  seed the soup with {order[0]['label']} (val loss {greedy_loss:.6f})")

    for record in order[1:]:
        candidate = load_state_dict(record["path"])
        inv = 1.0 / (len(accepted) + 1)
        trial = {k: (greedy_accum[k] + candidate[k]) * inv for k in greedy_accum}
        trial_loss = scorer.score(trial)
        del trial
        take = trial_loss <= greedy_loss
        if take:
            greedy_accum = accumulate_into(greedy_accum, candidate)
            accepted.append(record["label"])
            greedy_loss = trial_loss
        del candidate
        decisions.append({
            "label": record["label"],
            "solo": record["solo"],
            "trial": trial_loss,
            "accepted": take,
            "soup_size": len(accepted),
        })
        verdict = "ACCEPT" if take else "reject"
        print(
            f"  + {record['label']:24s} solo {record['solo']:.6f}  "
            f"soup-with-it {trial_loss:.6f}  -> {verdict}  (soup size {len(accepted)})"
        )

    greedy_state = scale_state(greedy_accum, len(accepted))
    del greedy_accum

    best_solo = order[0]
    print(f"\nBest single model : {best_solo['label']}  {best_solo['solo']:.6f}")
    print(f"Uniform soup      : {len(picks)} models        {uniform_loss:.6f} "
          f"({uniform_loss - best_solo['solo']:+.6f} vs best single)")
    print(f"Greedy soup       : {len(accepted)} models        {greedy_loss:.6f} "
          f"({greedy_loss - best_solo['solo']:+.6f} vs best single)")

    # ---- save -------------------------------------------------------------
    out_dir = Path(args.out_dir)
    save_soup(greedy_state, out_dir, config_source=config_source)
    print(f"\nWrote GREEDY soup ({len(accepted)} ingredients) to {out_dir}")

    # ---- report -----------------------------------------------------------
    lines = []
    lines.append("MODEL SOUP (Wortsman et al., ICML 2022) -- greedy + uniform")
    lines.append(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"val_file : {args.val_file}")
    lines.append(
        f"scoring  : teacher-forced token-weighted CE on a FIXED subset of "
        f"{scorer.num_windows} windows (subset_seed={args.subset_seed}), fp32, "
        f"device={device}"
    )
    lines.append(
        "note     : train.py validates under bf16 autocast on a resampled subset, so "
        "these\n           absolute losses are not comparable to a training log's "
        "val/loss; the\n           paired differences below are."
    )
    label_width = max([len("candidate")] + [len(r["label"]) for r in solo]) + 2
    path_width = max([len("checkpoint")] + [len(r["path"]) for r in solo]) + 2
    lines.append("")
    lines.append("CANDIDATES")
    lines.append(
        f"  {'run':<{label_width}}{'checkpoint':<{path_width}}{'logged':>9}{'solo val loss':>16}"
    )
    for record in solo:
        logged = f"{record['logged']:.4f}" if record["logged"] == record["logged"] else "n/a"
        lines.append(
            f"  {record['label']:<{label_width}}{record['path']:<{path_width}}"
            f"{logged:>9}{record['solo']:>16.6f}"
        )
    lines.append("")
    lines.append("GREEDY INCLUSION (candidates in ascending solo val loss)")
    lines.append(
        f"  {'candidate':<{label_width}}{'solo':>12}{'soup with it':>15}"
        f"{'decision':>11}{'soup size':>11}"
    )
    for step in decisions:
        verdict = "seed" if step is decisions[0] else ("ACCEPT" if step["accepted"] else "reject")
        lines.append(
            f"  {step['label']:<{label_width}}{step['solo']:>12.6f}{step['trial']:>15.6f}"
            f"{verdict:>11}{step['soup_size']:>11d}"
        )
    lines.append("")
    lines.append("RESULT")
    summary_width = label_width + 12
    lines.append(
        f"  best single model : {best_solo['label']:<{summary_width}}{best_solo['solo']:.6f}"
    )
    lines.append(
        f"  uniform soup      : {str(len(picks)) + ' models':<{summary_width}}{uniform_loss:.6f}  "
        f"({uniform_loss - best_solo['solo']:+.6f})"
    )
    lines.append(
        f"  greedy soup       : {str(len(accepted)) + ' models':<{summary_width}}{greedy_loss:.6f}  "
        f"({greedy_loss - best_solo['solo']:+.6f})"
    )
    lines.append(f"  greedy ingredients: {', '.join(accepted)}")
    lines.append(f"  greedy soup dir   : {out_dir}")
    if uniform_dir is not None:
        lines.append(f"  uniform soup dir  : {uniform_dir}")
    report = "\n".join(lines) + "\n"

    results_path = Path(args.results_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(report)
    print(f"\nWrote {results_path}")
    print()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
