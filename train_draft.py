"""Train the speculative-decoding draft model (see `nbest/speculative.py`).

Design, and why it is this and not a small GPT-2 trained from scratch
--------------------------------------------------------------------
Incremental decode of this model is **kernel-launch bound**, not compute bound:
a 1-token forward at batch 64 does ~1 ms of arithmetic but takes tens of ms of
wall clock, because the cost is ~10 CUDA launches x n_layer plus the Python/HF
dispatch around them.  In that regime the per-forward cost scales with *depth*
and is nearly independent of *width*.  A draft is only useful if its forward is
cheap, so the right axis to cut is layers, and width is free.

That immediately settles the architecture question:

* a narrow GPT-2 trained from scratch (say 4 x 256) saves FLOPs we were not
  paying for, throws away the target's 56M-parameter tied embedding table, and
  starts from random -- it would have to relearn the packed format from scratch
  and would still cost 4 layers' worth of launches;
* a **shallow copy of the target at full width** (K of 24 blocks, same
  ``wte``/``wpe``/``ln_f``/tied head) costs exactly the same K layers' worth of
  launches, but starts as a genuine truncation of the model it has to imitate.

So the draft is the target's blocks at evenly spaced depths (0, 8, 15, 23 for
K=4), with the ``scale_attn_by_inverse_layer_idx`` factor folded into each
copied block's query projection so the copy is function-preserving
(`nbest.speculative.build_shallow_draft`).

Objective
---------
What speculative decoding actually rewards is a *low total-variation distance to
the target's constrained per-slot distribution*, not low perplexity against the
ground truth: acceptance rate at a position is exactly ``1 - TV(p, q)``.  So the
loss is the forward KL ``KL(p_target || q_draft)`` between the two
**constrained, renormalised** distributions at score positions only -- the same
masking `constrain_score_token_logits` applies at decode time, so the draft
never wastes capacity on tokens the decoder cannot emit.  Forward KL upper
bounds TV via Pinsker, and (unlike reverse KL) it penalises the draft for
dropping modes the target keeps, which is what causes rejections.

Data: on-policy matters here
----------------------------
This checkpoint has a huge exposure-bias gap (teacher-forced score CE 0.266 vs
10.76 along its own greedy rollout), so the states a draft will actually be
asked about during speculative decoding look nothing like ground-truth prefixes.
Training only on GT windows would calibrate the draft on states the decoder
never visits.  The target is frozen, so its rollout distribution is fixed and a
pool of its own rollouts can be generated once in the job prologue
(`--onpolicy_windows`, half greedy / half T=1.0) and mixed into every batch
(`--onpolicy_frac`).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from anticipation.vocab import VOCAB_SIZE
from evaluate_muster import load_model
from nbest.speculative import build_shallow_draft
from onpolicy_rollout import (
    _role_constraint_mask,
    rollout_score_slots,
    score_token_positions,
    score_token_roles,
)

TOKEN_TYPE_NAMES = ("onset", "duration", "pitch")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class PackedLineDataset:
    """Line-offset index over one or more packed token files.

    Deliberately *not* `train.py`'s `TokenizedDataset`: distillation wants the
    exact windows the decoder will see, so none of the augmentation (transpose,
    jitter, score masking) applies -- an augmented window is a state the target
    is never asked about at inference.
    """

    def __init__(self, paths, stride=1, limit=None, seed=0):
        self.entries = []
        for path in paths:
            path = str(path)
            offsets = []
            with open(path, "rb") as handle:
                offset = 0
                for raw in handle:
                    if raw.strip():
                        offsets.append(offset)
                    offset += len(raw)
            offsets = offsets[::stride]
            if limit is not None and len(offsets) > limit:
                rng = random.Random(seed)
                offsets = rng.sample(offsets, limit)
            print(f"  {path}: {len(offsets)} windows (stride {stride})", flush=True)
            self.entries.extend((path, o) for o in offsets)
        self.handles = {}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, offset = self.entries[index]
        handle = self.handles.get(path)
        if handle is None:
            handle = open(path, "rb")
            self.handles[path] = handle
        handle.seek(offset)
        line = handle.readline().decode("utf-8")
        if "|" in line:
            line = line.split("|", 1)[0]
        tokens = [int(t) for t in line.split()]
        return torch.tensor(tokens, dtype=torch.long)


def stack_batch(dataset, indices, length):
    rows = []
    for i in indices:
        tokens = dataset[i]
        if tokens.shape[0] < length:
            continue
        rows.append(tokens[:length])
    return torch.stack(rows) if rows else None


# ---------------------------------------------------------------------------
# Constrained KD loss
# ---------------------------------------------------------------------------


def _chunk_metrics(p_logits, q_logits):
    """(kl, top1-agreement, 1 - TV) for one chunk of positions, all (batch, n)."""
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    p = log_p.exp()
    # Both sides carry the same -inf constraint mask, so off-support entries are
    # 0 * (-inf - -inf) = 0 * nan.  Zero them explicitly instead of letting the
    # whole loss go NaN (it silently did, on the first smoke run).
    terms = torch.where(torch.isfinite(log_p), p * (log_p - log_q), torch.zeros_like(p))
    kl = terms.sum(dim=-1)
    with torch.no_grad():
        agree = (p_logits.argmax(dim=-1) == q_logits.argmax(dim=-1)).float()
        # Expected acceptance rate under speculative sampling is exactly 1-TV(p,q).
        accept = 1.0 - 0.5 * (p - log_q.exp()).abs().sum(dim=-1)
    return kl, agree, accept


def kd_step(target, draft, batch, positions, roles, mask3, chunk_size, backward, amp):
    """Chunked forward-KL distillation over the score positions.

    The LM head is the memory bottleneck: 55028 classes x 414 positions x batch
    is gigabytes of float32 logits, and only 414 of the 1020 positions are even
    used.  So the trunk runs once, its output is detached, and the head + loss
    run in position chunks whose gradients are accumulated into the detached
    hidden state; one trunk backward at the end then covers all of them.
    """
    with torch.no_grad(), amp():
        t_hidden = target.transformer(batch, use_cache=False).last_hidden_state
    with amp():
        d_hidden = draft.transformer(batch, use_cache=False).last_hidden_state
    hidden = d_hidden.detach().requires_grad_(True) if backward else d_hidden

    n = positions.numel()
    totals = torch.zeros(3, 3, dtype=torch.float64, device=batch.device)
    counts = torch.zeros(3, dtype=torch.float64, device=batch.device)
    loss_value = 0.0
    for start in range(0, n, chunk_size):
        index = positions[start : start + chunk_size] - 1
        chunk_roles = roles[start : start + chunk_size]
        role_mask = mask3[chunk_roles].unsqueeze(0)
        with torch.no_grad(), amp():
            p_logits = target.lm_head(t_hidden[:, index])
        p_logits = p_logits.float().masked_fill(role_mask, -float("inf"))
        with amp():
            q_logits = draft.lm_head(hidden[:, index])
        q_logits = q_logits.float().masked_fill(role_mask, -float("inf"))
        kl, agree, accept = _chunk_metrics(p_logits, q_logits)
        chunk_loss = kl.mean() * (index.numel() / n)
        if backward:
            chunk_loss.backward()
        loss_value += float(chunk_loss.detach())
        with torch.no_grad():
            for role in range(3):
                selector = (chunk_roles == role).unsqueeze(0).expand_as(kl)
                if selector.any():
                    totals[role, 0] += kl.detach()[selector].sum().double()
                    totals[role, 1] += agree[selector].sum().double()
                    totals[role, 2] += accept[selector].sum().double()
                    counts[role] += selector.sum().double()
    if backward:
        d_hidden.backward(hidden.grad)

    stats = {}
    total_count = float(counts.sum())
    for role in range(3):
        c = float(counts[role]) or 1.0
        stats[f"kl_{TOKEN_TYPE_NAMES[role]}"] = float(totals[role, 0]) / c
        stats[f"agree_{TOKEN_TYPE_NAMES[role]}"] = float(totals[role, 1]) / c
        stats[f"accept_{TOKEN_TYPE_NAMES[role]}"] = float(totals[role, 2]) / c
    stats["agree"] = float(totals[:, 1].sum()) / (total_count or 1.0)
    stats["accept"] = float(totals[:, 2].sum()) / (total_count or 1.0)
    stats["kl"] = float(totals[:, 0].sum()) / (total_count or 1.0)
    return loss_value, stats


# ---------------------------------------------------------------------------
# On-policy pool
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_onpolicy_pool(target, dataset, count, batch_size, length, device, autocast_ctx, seed=0):
    """Roll the frozen target out on `count` windows and keep the rolled sequences.

    Half greedy, half T=1.0: the benchmark measures both regimes and their
    on-policy state distributions differ (greedy rollouts are the ones the eval
    protocol produces; T=1.0 is what sampled decoding visits).
    """
    rng = random.Random(seed)
    order = rng.sample(range(len(dataset)), min(count, len(dataset)))
    pool = []
    start = time.time()
    for i in range(0, len(order), batch_size):
        chunk = order[i : i + batch_size]
        batch = stack_batch(dataset, chunk, length)
        if batch is None:
            continue
        batch = batch.to(device)
        temperature = 0.0 if (i // batch_size) % 2 == 0 else 1.0
        rolled = rollout_score_slots(
            target,
            batch,
            temperature=temperature,
            constrain=True,
            collect_logprobs=False,
            collect_gt_ce=False,
            autocast_ctx=autocast_ctx,
        )["rolled"]
        pool.append(rolled.to("cpu", torch.int32))
        done = min(i + batch_size, len(order))
        if (i // batch_size) % 10 == 0:
            print(
                f"  on-policy pool {done}/{len(order)} ({time.time() - start:.0f}s)",
                flush=True,
            )
    return torch.cat(pool, dim=0) if pool else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_checkpoint", default="run_paper_split_v2/checkpoint-2500")
    parser.add_argument("--draft_layers", type=int, default=4)
    parser.add_argument("--layer_select", default="spaced", choices=["spaced", "first"])
    parser.add_argument(
        "--data_files",
        default="data/train_paper_unfiltered.txt,data/train_paper.txt",
    )
    parser.add_argument("--data_stride", type=int, default=1)
    parser.add_argument("--data_limit", type=int, default=60000, help="windows per file")
    parser.add_argument("--val_files", default="data/val_paper_unfiltered.txt,data/val_paper.txt")
    parser.add_argument("--val_limit", type=int, default=256)
    parser.add_argument("--output_dir", default="run_draft")
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--train_embeddings", action="store_true",
                        help="unfreeze wte/wpe (the tied head starts calibrated; default freezes)")
    parser.add_argument("--onpolicy_windows", type=int, default=4096)
    parser.add_argument("--onpolicy_frac", type=float, default=0.5)
    parser.add_argument("--rollout_batch", type=int, default=32)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--loss_chunk", type=int, default=48,
                        help="score positions per LM-head chunk (memory knob)")
    parser.add_argument("--val_eval_windows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(vars(args), indent=2), flush=True)
    print(f"torch {torch.__version__}", flush=True)

    target, device = load_model(args.target_checkpoint)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)

    draft = build_shallow_draft(target, args.draft_layers, args.layer_select)
    draft = draft.to(device)
    draft.gradient_checkpointing_disable()
    n_params = sum(p.numel() for p in draft.parameters())
    print(f"draft: {args.draft_layers} layers, {n_params / 1e6:.1f}M params", flush=True)

    if not args.train_embeddings:
        draft.transformer.wte.weight.requires_grad_(False)
        draft.transformer.wpe.weight.requires_grad_(False)
    trainable = [p for p in draft.parameters() if p.requires_grad]
    print(f"trainable: {sum(p.numel() for p in trainable) / 1e6:.1f}M params", flush=True)

    autocast_ctx = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if device == "cuda"
        else None
    )
    amp = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext

    train_paths = [p for p in args.data_files.split(",") if p]
    val_paths = [p for p in args.val_files.split(",") if p]
    print("train data:", flush=True)
    train_data = PackedLineDataset(train_paths, args.data_stride, args.data_limit, args.seed)
    print("val data:", flush=True)
    val_data = PackedLineDataset(val_paths, 1, args.val_limit, args.seed + 1)

    length = train_data[0].shape[0]
    positions = score_token_positions(length, device=device)
    roles = score_token_roles(positions)
    mask3 = _role_constraint_mask(VOCAB_SIZE, device)
    print(f"sequence length {length}, {positions.numel()} score positions", flush=True)

    pool = None
    if args.onpolicy_frac > 0 and args.onpolicy_windows > 0:
        print("building on-policy rollout pool from the frozen target...", flush=True)
        pool = build_onpolicy_pool(
            target, train_data, args.onpolicy_windows, args.rollout_batch, length,
            device, autocast_ctx, seed=args.seed,
        )
        print(f"  pool: {tuple(pool.shape)}", flush=True)
        val_pool = build_onpolicy_pool(
            target, val_data, 128, args.rollout_batch, length, device, autocast_ctx,
            seed=args.seed + 1,
        )
    else:
        val_pool = None

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine)

    def sample_batch(rng):
        n_on = int(round(args.batch_size * args.onpolicy_frac)) if pool is not None else 0
        n_gt = args.batch_size - n_on
        rows = []
        if n_gt:
            batch = stack_batch(train_data, [rng.randrange(len(train_data)) for _ in range(n_gt)], length)
            if batch is not None:
                rows.append(batch)
        if n_on:
            idx = torch.randint(0, pool.shape[0], (n_on,))
            rows.append(pool[idx].long())
        return torch.cat(rows, dim=0).to(device)

    @torch.no_grad()
    def evaluate():
        draft.eval()
        out = {}
        for name in ("gt", "onpolicy"):
            if name == "onpolicy" and val_pool is None:
                continue
            totals, count = {}, 0
            for i in range(0, args.val_eval_windows, args.batch_size):
                if name == "gt":
                    hi = min(i + args.batch_size, args.val_eval_windows, len(val_data))
                    if hi <= i:
                        continue
                    batch = stack_batch(val_data, list(range(i, hi)), length)
                    if batch is None:
                        continue
                    batch = batch.to(device)
                else:
                    batch = val_pool[i : i + args.batch_size].long().to(device)
                    if batch.shape[0] == 0:
                        continue
                _, stats = kd_step(
                    target, draft, batch, positions, roles, mask3,
                    args.loss_chunk, backward=False, amp=amp,
                )
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0.0) + value
                count += 1
            for key in totals:
                out[f"val/{name}/{key}"] = totals[key] / max(count, 1)
        draft.train()
        return out

    rng = random.Random(args.seed + 7)
    draft.train()
    history = []
    start = time.time()
    for step in range(args.max_steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        batch = sample_batch(rng)

        loss_value, stats = kd_step(
            target, draft, batch, positions, roles, mask3,
            args.loss_chunk, backward=True, amp=amp,
        )
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == args.max_steps - 1:
            elapsed = time.time() - start
            row = {"step": step, "loss": loss_value, "lr": lr_at(step),
                   "elapsed_s": elapsed, **stats}
            history.append(row)
            print(
                f"step {step:5d} kl {stats['kl']:.4f} "
                f"(on {stats['kl_onset']:.3f} du {stats['kl_duration']:.3f} "
                f"pi {stats['kl_pitch']:.3f}) top1 {stats['agree']:.3f} "
                f"E[accept] {stats['accept']:.3f} lr {lr_at(step):.2e} "
                f"{elapsed:.0f}s",
                flush=True,
            )
        if args.eval_every and step > 0 and step % args.eval_every == 0:
            metrics = evaluate()
            print("  VAL " + json.dumps({k: round(v, 4) for k, v in metrics.items()}), flush=True)
            history.append({"step": step, **metrics})
        if args.save_every and step > 0 and step % args.save_every == 0:
            draft.save_pretrained(output_dir / "checkpoint")
            (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    metrics = evaluate()
    print("FINAL VAL " + json.dumps({k: round(v, 4) for k, v in metrics.items()}), flush=True)
    history.append({"step": args.max_steps, **metrics})
    draft.save_pretrained(output_dir / "final")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"saved to {output_dir / 'final'}  wall {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
