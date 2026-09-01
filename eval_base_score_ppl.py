#!/usr/bin/env python
"""GT-score vs generated-score perplexity under the UNTUNED base model.

Question: under the raw pretrained anticipatory music transformer
(``stanford-crfm/music-medium-800k``, no fine-tuning), which has higher
perplexity — the original (ground-truth) scores, or the scores the base model
itself generates from the performance?

Protocol, per packed window:

1. **Generate.** Greedy-decode a score with the base model in the packed
   format, conditioned on the ground-truth performance controls — the repo's
   standard rollout (``inference.batched_autoregressive_generate_score`` with
   ``ground_truth_score_tokens_to_feed=0``, slot-constrained logits), exactly
   as ``train.py``'s autoregressive validation pass runs it.
2. **Measure score-only.** Flatten each side's score triplets into a pure
   score-token sequence (no interleaved performance controls), prepend the
   ``AUTOREGRESS`` mode flag (the base model's pretraining prepends this exact
   token to control-free sequences — see ``anticipation/tokenize.py``), and
   compute teacher-forced NLL of every score token under the base model.
   This is the primary metric: perplexity of the score *as a score*, not of
   the interleaved packed sequence.
3. **Measure in-context (secondary).** Also record the NLL of the same score
   tokens inside their packed sequence (GT packed line / generated context),
   i.e. conditioned on the interleaved performance. Reported separately.

Every NLL is computed in two variants: **unconstrained** (softmax over the
full vocab) and **constrained** (softmax renormalized over the slot-legal
token range via ``constrain_score_token_logits`` — the distribution the
decoding actually samples from). GT and generated sides always get identical
treatment, so either variant supports the comparison.

The model forward runs in bf16 on CUDA; NLLs are computed from float32-cast
logits. Both sides of every pair share the same token count (the generator
fills exactly the GT slot layout), so per-window mean NLLs are comparable.

Sharding for SLURM arrays: ``--shard-index i --num-shards N`` stripes the
(optionally ``--stride``-subsampled) line indices i::N. Each shard writes a
JSON of per-window results; ``--merge`` aggregates shards and prints the
answer.

Usage:
  # one shard on one GPU
  python eval_base_score_ppl.py --token-file data/val_paper.txt \
      --shard-index 0 --num-shards 8 \
      --output base_score_ppl_results/val_shard00.json

  # after all shards finish
  python eval_base_score_ppl.py --merge 'base_score_ppl_results/val_shard*.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from anticipation.packed_sequence import (
    ALTERNATING_START,
    iter_score_slot_positions,
)
from anticipation.score_constraints import constrain_score_token_logits
from anticipation.vocab import AUTOREGRESS, REST, VOCAB_SIZE
from inference import batched_autoregressive_generate_score

BASE_MODEL = "stanford-crfm/music-medium-800k"

# metric keys: (score-only | packed) x (constrained | unconstrained)
METRICS = ("so_c", "so_u", "pk_c", "pk_u")
ARMS = ("gt", "gen")
SLOT_NAMES = ("onset", "duration", "pitch")


def load_base_model(device: torch.device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM

    print(f"Loading untuned base model {BASE_MODEL} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    if model.config.vocab_size != VOCAB_SIZE:
        # The released checkpoint already has vocab 55028; resizing would add
        # untrained rows and invalidate the measurement, so refuse loudly.
        raise SystemExit(
            f"Base model vocab {model.config.vocab_size} != VOCAB_SIZE {VOCAB_SIZE}; "
            "an untrained resize would make base-model perplexities meaningless."
        )
    if device.type == "cuda":
        # The base model's WEIGHTS are cast, not merely autocast -- so a caller
        # that only disables autocast still gets a bf16 base model. That is how
        # nbest_data/fp32_*.pt came to be fp32 on the FT channel and bf16 on
        # the `logp_base` channel despite the name. Pass dtype=torch.float32
        # for a genuinely full-precision base.
        model = model.to(device=device, dtype=dtype)
    else:
        model = model.to(device)
    model.eval()
    return model


def read_shard_windows(path: str, stride: int, shard_index: int, num_shards: int,
                       max_windows: int | None, truncate_slots: int | None):
    """Return (line_indices, list of 1-D long tensors) for this shard.

    Lines are subsampled to every ``stride``-th, then striped ``i::num_shards``
    over the subsample so shards interleave evenly across the file.
    """
    line_indices, windows = [], []
    selected = 0
    with open(path) as f:
        for line_idx, line in enumerate(f):
            if line_idx % stride != 0:
                continue
            keep = selected % num_shards == shard_index
            selected += 1
            if not keep:
                continue
            tokens_str = line.split("|", 1)[0].strip()
            if not tokens_str:
                continue
            tokens = [int(t) for t in tokens_str.split()]
            if truncate_slots is not None:
                tokens = tokens[: ALTERNATING_START + 6 * truncate_slots]
            if len(tokens) <= ALTERNATING_START + 5:
                continue
            line_indices.append(line_idx)
            windows.append(torch.tensor(tokens, dtype=torch.long))
            if max_windows is not None and len(windows) >= max_windows:
                break
    return line_indices, windows


def slot_logit_masks(device: torch.device) -> torch.Tensor:
    """(3, VOCAB_SIZE) additive masks: 0 where legal for the slot, -inf where not."""
    rows = [
        constrain_score_token_logits(torch.zeros(VOCAB_SIZE), slot)
        for slot in range(3)
    ]
    return torch.stack(rows).to(device)


@torch.inference_mode()
def nll_at_positions(model, seqs: torch.Tensor, pred_pos: torch.Tensor,
                     target_pos: torch.Tensor, slot_ids: torch.Tensor,
                     slot_masks: torch.Tensor, chunk: int):
    """Teacher-forced NLL of ``seqs[:, target_pos]`` predicted from ``pred_pos``.

    Returns (nll_u, nll_c): (B, len(target_pos)) float32 tensors of
    unconstrained / slot-constrained NLLs.
    """
    outs_u, outs_c = [], []
    for start in range(0, seqs.shape[0], chunk):
        part = seqs[start:start + chunk]
        logits = model(part, use_cache=False).logits
        pred = logits[:, pred_pos, :].float()          # (b, P, V)
        targets = part[:, target_pos]                   # (b, P)
        nll_u = F.cross_entropy(
            pred.permute(0, 2, 1), targets, reduction="none")
        constrained = pred + slot_masks[slot_ids]       # broadcast (b, P, V)
        nll_c = F.cross_entropy(
            constrained.permute(0, 2, 1), targets, reduction="none")
        outs_u.append(nll_u)
        outs_c.append(nll_c)
        del logits, pred, constrained
    return torch.cat(outs_u), torch.cat(outs_c)


def run_shard(args):
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_base_model(device)
    slot_masks = slot_logit_masks(device)

    line_indices, windows = read_shard_windows(
        args.token_file, args.stride, args.shard_index, args.num_shards,
        args.max_windows, args.truncate_slots)
    print(f"{args.token_file}: shard {args.shard_index}/{args.num_shards} "
          f"stride {args.stride} -> {len(windows)} windows")
    if not windows:
        raise SystemExit("no windows selected")

    length = windows[0].shape[0]
    if any(w.shape[0] != length for w in windows):
        raise SystemExit("windows have differing lengths; batched decode "
                         "requires the uniform 1020-token packed layout")

    # Complete (score triplet, control triplet) slot pairs in this layout.
    positions = [p for p in iter_score_slot_positions(length) if p + 5 < length]
    n_slots = len(positions)
    flat_pos = torch.tensor(
        [p + j for p in positions for j in range(3)], device=device)
    packed_slot_ids = torch.tensor(
        [j for _ in positions for j in range(3)], device=device)
    so_len = 3 * n_slots + 1  # AUTOREGRESS flag + flattened triplets
    so_target_pos = torch.arange(1, so_len, device=device)
    so_pred_pos = so_target_pos - 1
    so_slot_ids = (so_target_pos - 1) % 3
    autoregress_col = torch.full((1, 1), AUTOREGRESS, dtype=torch.long,
                                 device=device)

    per_window = []
    slot_sums = {f"{m}_{a}": [[0.0, 0] for _ in range(3)]
                 for m in METRICS for a in ARMS}
    n_dummy_skipped = 0
    n_nonfinite = 0
    t0 = time.time()

    for start in tqdm(range(0, len(windows), args.decode_batch_size),
                      desc="batches"):
        rows = windows[start:start + args.decode_batch_size]
        idxs = line_indices[start:start + args.decode_batch_size]
        batch = torch.stack(rows).to(device)

        # Body score slots must all hold real notes (dummy RESTs are
        # prefix-only by construction); drop any window violating that.
        real = (batch[:, flat_pos[2::3]] != REST).all(dim=1)
        if not bool(real.all()):
            n_dummy_skipped += int((~real).sum().item())
            batch = batch[real]
            idxs = [i for i, keep in zip(idxs, real.tolist()) if keep]
            if batch.shape[0] == 0:
                continue

        gen_ctx = batched_autoregressive_generate_score(
            model, batch, ALTERNATING_START, str(device),
            constrain_score_tokens=True,
            ground_truth_score_tokens_to_feed=0,
        )

        # --- score-only measurement (primary) ---
        so = {}
        for arm, ctx in (("gt", batch), ("gen", gen_ctx)):
            flat = ctx[:, flat_pos]
            seqs = torch.cat(
                [autoregress_col.expand(flat.shape[0], 1), flat], dim=1)
            nll_u, nll_c = nll_at_positions(
                model, seqs, so_pred_pos, so_target_pos, so_slot_ids,
                slot_masks, args.score_chunk)
            so[arm] = {"u": nll_u, "c": nll_c}

        # --- in-context (packed) measurement (secondary) ---
        pk = {}
        for arm, ctx in (("gt", batch), ("gen", gen_ctx)):
            nll_u, nll_c = nll_at_positions(
                model, ctx, flat_pos - 1, flat_pos, packed_slot_ids,
                slot_masks, args.score_chunk)
            pk[arm] = {"u": nll_u, "c": nll_c}

        all_nlls = {
            f"so_{v}_{arm}": so[arm][v] for arm in ARMS for v in ("c", "u")
        } | {
            f"pk_{v}_{arm}": pk[arm][v] for arm in ARMS for v in ("c", "u")
        }
        finite = torch.stack(
            [t.isfinite().all(dim=1) for t in all_nlls.values()]).all(dim=0)
        n_nonfinite += int((~finite).sum().item())

        for key, t in all_nlls.items():
            tf = t[finite]
            for slot in range(3):
                vals = tf[:, slot::3]
                slot_sums[key][slot][0] += float(vals.sum().item())
                slot_sums[key][slot][1] += vals.numel()

        means = {key: t.mean(dim=1) for key, t in all_nlls.items()}
        for row, (i, ok) in enumerate(zip(idxs, finite.tolist())):
            if not ok:
                continue
            per_window.append(
                {"i": i} | {key: round(float(means[key][row].item()), 6)
                            for key in all_nlls})

    elapsed = time.time() - t0
    print(f"Scored {len(per_window)} windows in {elapsed:.1f}s "
          f"({elapsed / max(len(per_window), 1):.3f} s/window); "
          f"skipped {n_dummy_skipped} dummy-slot, {n_nonfinite} non-finite")

    result = {
        "token_file": args.token_file,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "stride": args.stride,
        "tokens_per_window": 3 * n_slots,
        "n_windows": len(per_window),
        "n_dummy_skipped": n_dummy_skipped,
        "n_nonfinite": n_nonfinite,
        "elapsed_seconds": round(elapsed, 1),
        "slot_sums": slot_sums,
        "windows": per_window,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f)
    print(f"Wrote {out}")
    print_stats([result], label=f"{args.token_file} shard {args.shard_index}")


def _bootstrap_ci(diffs: list[float], rng: random.Random,
                  n_boot: int = 2000) -> tuple[float, float]:
    """95% bootstrap CI of the mean of ``diffs`` (numpy-vectorized when
    available; the pure-Python path is far too slow for 100k+ windows)."""
    n = len(diffs)
    try:
        import numpy as np
        arr = np.asarray(diffs)
        gen = np.random.default_rng(rng.randrange(2**32))
        means = np.empty(n_boot)
        step = max(1, min(n_boot, 512 * 1024 * 1024 // (8 * n)))
        for s in range(0, n_boot, step):
            k = min(step, n_boot - s)
            idx = gen.integers(0, n, size=(k, n))
            means[s:s + k] = arr[idx].mean(axis=1)
        means.sort()
        return float(means[int(0.025 * n_boot)]), float(means[int(0.975 * n_boot)])
    except ImportError:
        boot = []
        for _ in range(n_boot):
            sample = [diffs[rng.randrange(n)] for _ in range(n)]
            boot.append(sum(sample) / n)
        boot.sort()
        return boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot)]


def print_stats(shards: list[dict], label: str):
    windows = [w for s in shards for w in s["windows"]]
    n = len(windows)
    print(f"\n=== {label}: {n} windows "
          f"({sum(s['n_dummy_skipped'] for s in shards)} dummy-skipped, "
          f"{sum(s['n_nonfinite'] for s in shards)} non-finite) ===")
    if n == 0:
        return

    metric_desc = {
        "so_c": "SCORE-ONLY, slot-constrained  [primary]",
        "so_u": "SCORE-ONLY, unconstrained     [primary]",
        "pk_c": "packed in-context, slot-constrained",
        "pk_u": "packed in-context, unconstrained",
    }
    rng = random.Random(0)
    for m in METRICS:
        gt = [w[f"{m}_gt"] for w in windows]
        gen = [w[f"{m}_gen"] for w in windows]
        gt_mean, gen_mean = sum(gt) / n, sum(gen) / n
        diffs = [a - b for a, b in zip(gt, gen)]  # gt - gen, >0 => GT higher
        gt_higher = sum(1 for d in diffs if d > 0)
        # paired bootstrap CI over windows (windows overlap within a
        # performance, so treat the CI as optimistic)
        lo, hi = _bootstrap_ci(diffs, rng)
        gt_med = sorted(gt)[n // 2]
        gen_med = sorted(gen)[n // 2]

        print(f"\n--- {metric_desc[m]} ---")
        print(f"  mean NLL/token : GT {gt_mean:.4f}   gen {gen_mean:.4f}")
        print(f"  PPL(mean NLL)  : GT {math.exp(gt_mean):.3f}   "
              f"gen {math.exp(gen_mean):.3f}")
        print(f"  median win PPL : GT {math.exp(gt_med):.3f}   "
              f"gen {math.exp(gen_med):.3f}")
        print(f"  GT higher than gen in {gt_higher}/{n} windows "
              f"({100 * gt_higher / n:.1f}%)")
        print(f"  mean(GT-gen) NLL diff: {sum(diffs)/n:+.4f} "
              f"(95% bootstrap CI [{lo:+.4f}, {hi:+.4f}])")

        slot_tot = {a: [[0.0, 0] for _ in range(3)] for a in ARMS}
        for s in shards:
            for a in ARMS:
                for slot in range(3):
                    add = s["slot_sums"][f"{m}_{a}"][slot]
                    slot_tot[a][slot][0] += add[0]
                    slot_tot[a][slot][1] += add[1]
        parts = []
        for slot, name in enumerate(SLOT_NAMES):
            g = slot_tot["gt"][slot]
            h = slot_tot["gen"][slot]
            if g[1] and h[1]:
                parts.append(f"{name} GT {math.exp(g[0]/g[1]):.2f}"
                             f"/gen {math.exp(h[0]/h[1]):.2f}")
        print(f"  per-slot PPL   : " + "   ".join(parts))

    m = "so_c"
    gt_mean = sum(w[f"{m}_gt"] for w in windows) / n
    gen_mean = sum(w[f"{m}_gen"] for w in windows) / n
    which = "ORIGINAL (GT)" if gt_mean > gen_mean else "GENERATED"
    print(f"\n>>> {label}: the {which} scores have higher score-only "
          f"perplexity under the untuned base model "
          f"(GT {math.exp(gt_mean):.3f} vs gen {math.exp(gen_mean):.3f}).")


def run_merge(patterns: list[str]):
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat) if any(c in pat for c in "*?[") else [pat])
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit(f"no shard files match {patterns}")
    shards = []
    for p in paths:
        with open(p) as f:
            shards.append(json.load(f))
    by_file: dict[str, list[dict]] = {}
    for s in shards:
        by_file.setdefault(s["token_file"], []).append(s)
    for token_file, group in sorted(by_file.items()):
        print(f"\n################ {token_file} "
              f"({len(group)} shards) ################")
        print_stats(group, label=token_file)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--token-file", default="data/val_paper.txt")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth line before sharding (even coverage)")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="cap windows for this shard (smoke tests)")
    ap.add_argument("--truncate-slots", type=int, default=None,
                    help="truncate windows to N body slot pairs (smoke tests)")
    ap.add_argument("--decode-batch-size", type=int, default=96)
    ap.add_argument("--score-chunk", type=int, default=16,
                    help="rows per teacher-forced scoring forward")
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default=None,
                    help="shard JSON output path (required unless --merge)")
    ap.add_argument("--merge", nargs="*", default=None,
                    help="merge shard JSONs (files or globs) and print stats")
    args = ap.parse_args()

    if args.merge is not None:
        run_merge(args.merge)
        return
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("need 0 <= --shard-index < --num-shards")
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if not args.output:
        raise SystemExit("--output is required when scoring a shard")
    run_shard(args)


if __name__ == "__main__":
    main()
