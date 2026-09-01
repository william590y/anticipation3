"""Train the masked-diffusion drafter (`nbest/diffdraft.py`) by sequence-level KD.

The training signal is the **target model's own greedy rollout**, cached by
`gen_diffdraft_kd.py`, not the ground truth. A drafter is scored on how often its
token equals the target's constrained argmax, and those are far apart here: the
target's AR exact-match accuracy against ground truth is ~16% on onsets, so a
ground-truth-trained drafter's ceiling on agreement would be roughly "both are
right at once". Distilling the teacher's outputs (Kim & Rush 2016) removes that
ceiling -- the teacher's argmax is a deterministic function of the window and is
much more learnable than the truth is.

Objective. Each step samples one block geometry for the whole batch (see
`diffdraft.sample_block_geometry`), corrupts the block with the 50/50
iid / left-to-right mixture, and takes plain cross-entropy on the masked
positions, averaged over tokens.

Note this is deliberately *not* the MDLM NELBO, which weights a t-masked sample
by 1/t. Two reasons. (1) The drafter is never used as a likelihood model -- it is
a proposal distribution whose errors are corrected by verification -- so the
quantity we care about is top-1 agreement at the mask rates the sampler actually
visits, not a bound on log p(x). (2) The 1/t weight has unbounded variance as
t -> 0 and pours capacity into the nearly-clean regime, whereas the step that
decides whether a block can be drafted at all is the *first* one, where the block
is entirely masked (t = 1) and 1/t is at its smallest. Uniform weighting over
t ~ U(0,1) is the MaskGIT-style reweighting and matches the deployment metric.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

from nbest.diffdraft import (
    DEFAULT_BLOCK_SIZES,
    DEFAULT_BLOCK_WEIGHTS,
    MASK_ID,
    N_BODY_SLOTS,
    PACKED_LENGTH,
    block_token_end,
    build_drafter,
    build_training_batch,
    denoise_block,
    sample_block_geometry,
    score_positions,
)

ROLE_NAMES = ("onset", "duration", "pitch")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class PackedWindows(torch.utils.data.Dataset):
    """Line-offset index over one or more packed token files.

    A transformers-free reader on purpose: the drafter shares nothing with
    `train.py`'s TokenizedDataset except the file format, and it must NOT apply
    that class's augmentation -- the KD targets are the teacher's outputs for the
    *unaugmented* window, so jittering the controls would silently decouple the
    input from its label.
    """

    def __init__(self, patterns, expect_length=PACKED_LENGTH):
        self.files = []
        for pattern in patterns:
            self.files.extend(sorted(glob.glob(pattern)))
        if not self.files:
            raise FileNotFoundError(f"no files matched {patterns}")
        self.index = []
        for file_id, path in enumerate(self.files):
            with open(path, "rb") as handle:
                offset = 0
                for raw in handle:
                    if raw.strip():
                        self.index.append((file_id, offset))
                    offset += len(raw)
        self.expect_length = expect_length
        self._handles = {}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        file_id, offset = self.index[i]
        handle = self._handles.get(file_id)
        if handle is None:
            handle = open(self.files[file_id], "rb")
            self._handles[file_id] = handle
        handle.seek(offset)
        text = handle.readline().decode("utf-8").split("|")[0]
        tokens = [int(t) for t in text.split()]
        if len(tokens) != self.expect_length:
            tokens = (tokens + [0] * self.expect_length)[: self.expect_length]
        return torch.tensor(tokens, dtype=torch.long)


# ---------------------------------------------------------------------------
# Validation: standalone drafter quality
# ---------------------------------------------------------------------------


def masked_loss(model, hidden, target_ids, local, masked, generator=None, max_tokens=0):
    """Cross-entropy over the masked block positions only, as a flat (N, vocab) GEMM.

    The head is applied ONLY where the loss reads it. Running it over all of
    `local` and multiplying by the mask afterwards costs a (batch, 3*B, 55029)
    fp32 tensor plus a second one inside `cross_entropy`: at batch 64 with the
    whole 138-slot body as one block that is 5.8 GB each, and it OOM'd a 48 GB
    A6000 (job 463975). Gathering the masked positions first makes the cost track
    the number of tokens actually supervised, which is what it should have been
    all along -- and is the same trick `anticipation/fast_decode.py` uses on the
    decode side.

    `max_tokens` optionally subsamples: an unbiased estimate of the same mean CE
    with a hard ceiling on the transient, for the t -> 1 draws where everything
    in a 414-token block is masked at once.
    """
    rows, cols = masked.nonzero(as_tuple=True)
    if max_tokens and rows.numel() > max_tokens:
        keep = torch.randperm(rows.numel(), generator=generator, device=rows.device)[:max_tokens]
        rows, cols = rows[keep], cols[keep]
    positions = local[cols]
    logits = model.head(hidden[rows, positions, :]).float()
    targets = target_ids[rows, positions]
    return F.cross_entropy(logits, targets), rows.numel()


@torch.no_grad()
def masked_ce(model, windows, generator, ltr_prob, autocast, max_tokens=0):
    committed, block_slots = sample_block_geometry(generator)
    input_ids, target_ids, local, prefix_len, masked = build_training_batch(
        windows, generator, committed, block_slots, ltr_prob=ltr_prob
    )
    with autocast():
        hidden = model(input_ids, prefix_len)
    loss, count = masked_loss(model, hidden, target_ids, local, masked,
                              generator=generator, max_tokens=max_tokens)
    return float(loss) * count, float(count)


@torch.no_grad()
def draft_agreement(model, windows, *, block_slots, steps, order, committed_slots=0,
                    temperature=0.0, autocast=None):
    """Top-1 agreement with the teacher tokens for one real draft configuration.

    Runs the deployment path exactly: encode the committed prefix into a KV
    cache, mask every score token of the block, denoise it in `steps` forwards.
    Reported per triplet role, plus the "prefix run" -- the mean number of
    leading score tokens of the block that all match, which is what actually
    determines the accepted block length under longest-prefix verification and is
    NOT recoverable from the per-token rate (it is dominated by correlations).
    """
    device = windows.device
    prefix_len = block_token_end(committed_slots)
    end = block_token_end(committed_slots + block_slots)
    ctx = autocast if autocast is not None else torch.autocast

    with ctx():
        cache = model.encode_prefix(windows[:, :prefix_len], 0)
        block = windows[:, prefix_len:end].clone()
        local = score_positions(committed_slots, committed_slots + block_slots,
                                device=device) - prefix_len
        block[:, local] = MASK_ID
        filled, _ = denoise_block(model, block, prefix_len, cache, local, steps=steps,
                                  order=order, temperature=temperature)

    truth = windows[:, prefix_len:end][:, local]
    got = filled[:, local]
    match = got == truth
    roles = ((local + prefix_len) % 3)
    per_role = {
        ROLE_NAMES[r]: float(match[:, roles == r].float().mean()) for r in range(3)
    }
    # Longest matching prefix per row, in score tokens.
    ok = match.float()
    run = (ok.cumprod(dim=1)).sum(dim=1)
    return {
        "block_slots": block_slots,
        "steps": steps,
        "order": order,
        "overall": float(match.float().mean()),
        **per_role,
        "prefix_run_tokens": float(run.mean()),
        "prefix_run_slots": float((run / 3).mean()),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", nargs="+",
                        default=["data/diffdraft_kd_train.shard*of*.txt"])
    parser.add_argument("--val-data", nargs="+",
                        default=["data/diffdraft_kd_val.shard*of*.txt"])
    parser.add_argument("--target-checkpoint", default="run_paper_split_v2/checkpoint-2500")
    parser.add_argument("--output-dir", default="run_diffdraft")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-frac", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ltr-prob", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=list(DEFAULT_BLOCK_SIZES))
    parser.add_argument("--block-weights", type=float, nargs="+",
                        default=list(DEFAULT_BLOCK_WEIGHTS))
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--max-loss-tokens", type=int, default=12000,
                        help="cap on supervised positions per step (0 = no cap); bounds the "
                             "(N, 55029) fp32 head+CE transient")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = PackedWindows(args.train_data)
    val_set = PackedWindows(args.val_data)
    print(f"train windows: {len(train_set)}  val windows: {len(val_set)}", flush=True)

    loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=0,
    )

    if args.resume:
        from nbest.diffdraft import load_drafter

        model, blob = load_drafter(args.resume, device=device, dtype=torch.float32)
        model.train()
        init_info = blob.get("init_info")
        print(f"resumed from {args.resume} (step {blob.get('step')})", flush=True)
    else:
        target_state = load_safetensors(
            str(Path(args.target_checkpoint) / "model.safetensors")
        )
        model, init_info = build_drafter(
            n_layer=args.n_layer, target_state=target_state, dropout=args.dropout
        )
        del target_state
        model = model.to(device)
        print(f"init from target: {json.dumps(init_info)}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    n_trunk = n_params - model.wte.weight.numel() - model.wpe.weight.numel()
    print(f"drafter params: {n_params/1e6:.1f}M total, {n_trunk/1e6:.1f}M non-embedding",
          flush=True)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim < 2 else decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * cosine)

    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    def autocast():
        return torch.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda"
                                                                and amp_dtype != torch.float32))

    generator = torch.Generator(device=device).manual_seed(args.seed)
    # Eval configurations: the (block, K, order) triples the decoder will actually
    # be run with in bench/bench_diffdraft.py.
    eval_configs = [
        dict(block_slots=16, steps=1, order="confidence"),
        dict(block_slots=16, steps=2, order="confidence"),
        dict(block_slots=16, steps=4, order="confidence"),
        dict(block_slots=16, steps=8, order="confidence"),
        dict(block_slots=16, steps=3, order="role"),
        dict(block_slots=16, steps=4, order="ltr"),
        dict(block_slots=8, steps=4, order="confidence"),
        dict(block_slots=32, steps=4, order="confidence"),
    ]

    history = []
    csv_path = out_dir / "diffdraft_train.csv"
    step = 0
    started = time.perf_counter()
    running = 0.0
    running_n = 0

    def run_eval(step):
        model.eval()
        ce_sum = ce_n = 0.0
        agree = []
        with torch.no_grad():
            batches = []
            for i, windows in enumerate(val_loader):
                if i >= args.eval_batches:
                    break
                batches.append(windows.to(device, non_blocking=True))
            for windows in batches:
                s, n = masked_ce(model, windows, generator, args.ltr_prob, autocast,
                                 max_tokens=args.max_loss_tokens)
                ce_sum += s
                ce_n += n
            for config in eval_configs:
                rows = [
                    draft_agreement(model, windows, autocast=autocast,
                                    committed_slots=0, **config)
                    for windows in batches[: max(1, args.eval_batches // 2)]
                ]
                merged = {k: (sum(r[k] for r in rows) / len(rows) if isinstance(rows[0][k], float)
                              else rows[0][k]) for k in rows[0]}
                agree.append(merged)
        model.train()
        row = {
            "step": step,
            "val_masked_ce": ce_sum / max(ce_n, 1),
            "minutes": (time.perf_counter() - started) / 60,
            "agreement": agree,
        }
        history.append(row)
        print(f"[eval @ {step}] masked CE {row['val_masked_ce']:.4f}", flush=True)
        for a in agree:
            print(
                f"    B={a['block_slots']:>3} K={a['steps']} {a['order']:<10} "
                f"top1 {a['overall']*100:5.1f}%  onset {a['onset']*100:5.1f}  "
                f"dur {a['duration']*100:5.1f}  pitch {a['pitch']*100:5.1f}  "
                f"prefix-run {a['prefix_run_slots']:.2f} slots",
                flush=True,
            )
        (out_dir / "diffdraft_history.json").write_text(json.dumps(history, indent=1))
        return row

    def save(step, tag):
        path = out_dir / f"diffdraft_{tag}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": model.config.to_dict(),
                "step": step,
                "args": vars(args),
                "init_info": init_info,
            },
            path,
        )
        print(f"saved {path}", flush=True)

    print("starting training", flush=True)
    model.train()
    while step < args.steps:
        for windows in loader:
            if step >= args.steps:
                break
            windows = windows.to(device, non_blocking=True)
            committed, block_slots = sample_block_geometry(
                generator, tuple(args.block_sizes), tuple(args.block_weights)
            )
            input_ids, target_ids, local, prefix_len, masked = build_training_batch(
                windows, generator, committed, block_slots, ltr_prob=args.ltr_prob
            )

            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            with autocast():
                hidden = model(input_ids, prefix_len)
            loss, _ = masked_loss(model, hidden, target_ids, local, masked,
                                  generator=generator, max_tokens=args.max_loss_tokens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            running += float(loss.detach())
            running_n += 1
            step += 1

            if step % 100 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"step {step}/{args.steps}  loss {running/running_n:.4f}  "
                    f"lr {lr_at(step):.2e}  {step/elapsed:.2f} it/s  "
                    f"{elapsed/60:.1f} min",
                    flush=True,
                )
                running = running_n = 0
            if step % args.eval_every == 0 or step == args.steps:
                run_eval(step)
            if step % args.save_every == 0 or step == args.steps:
                save(step, "last")

    run_eval(step)
    save(step, "final")

    import csv as csv_module

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.writer(handle)
        writer.writerow(["step", "val_masked_ce", "minutes"])
        for row in history:
            writer.writerow([row["step"], row["val_masked_ce"], row["minutes"]])
    print(f"wrote {csv_path}", flush=True)
    print(f"total wall clock: {(time.perf_counter() - started)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
