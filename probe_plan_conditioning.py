#!/usr/bin/env python
"""Does the stage-2 LM actually condition on the plan?

The in-training diagnostic ``val/plan_onset_gap`` compares the oracle plan
against the model's own greedily decoded plan. That confounds two things: how
much the plan is worth, and how well the model can predict one -- and under
``placement=front`` the second is known to be broken (plan-code accuracy ~1%).

This script runs the clean control instead, the same one stage 1 used to prove
its codes carry information: reconstruct the window from a *different* window's
plan. Three rollouts on identical windows,

    oracle      the window's own codes
    shuffled    the neighbouring window's codes (roll by 1 in the batch)
    own         the model's own greedy plan

so oracle-minus-shuffled is a paired per-window difference and its standard
error is the standard error of the *difference*, not of either arm.

It also checks the conditioning path directly: whether swapping the plan codes
changes the window's logits at all, both in the training forward
(``forward_plan_batch``, explicit all-ones mask) and in the KV-cached decode
(``plan_autoregressive_generate_score``, no mask). A zero there would mean the
plan is severed and every accuracy number below is meaningless.

    srun -p ellis --gres=gpu:nvidia_rtx_a6000:1 --mem=64G \
        python probe_plan_conditioning.py \
            --checkpoint run_plan_lm_.../checkpoint-2500 \
            --vq_checkpoint run_plan_vq_.../plan_vq_best.pt \
            --num_windows 512
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM

from anticipation.packed_sequence import ALTERNATING_START, iter_score_slot_positions
from anticipation.vocab import REST
from packed_dataset import TokenizedDataset
from plan_lm import PACKED_LENGTH, PlanLayout, plan_autoregressive_generate_score, prepare_plan_model
from train_plan_lm import OnlinePlanEncoder, PlanBatcher, collate_fn, forward_plan_batch

CONDITIONS = ("oracle", "shuffled", "own")


def latest_checkpoint(run_dir):
    """Newest ``checkpoint-<step>`` (or ``final``) under a run directory."""
    run_dir = Path(run_dir)
    if (run_dir / "config.json").exists():
        return run_dir
    candidates = [
        (int(p.name.split("-")[1]), p)
        for p in run_dir.glob("checkpoint-*")
        if p.is_dir() and p.name.split("-")[1].isdigit()
    ]
    if not candidates:
        final = run_dir / "final"
        if final.is_dir():
            return final
        raise FileNotFoundError(f"no checkpoints under {run_dir}")
    return max(candidates)[1]


def score_slot_positions():
    return list(iter_score_slot_positions(PACKED_LENGTH, ALTERNATING_START))


def per_window_hits(generated, reference, positions):
    """Per-window (correct, total) counts for onset / duration / pitch.

    Returns three ``(B,)`` int arrays of hits plus the ``(B,)`` count of real
    score slots, which is shared by all three.
    """
    onset_pos = torch.tensor(positions, device=generated.device)
    real = reference[:, onset_pos + 2] != REST                       # (B, slots)
    hits = []
    for offset in range(3):
        match = generated[:, onset_pos + offset] == reference[:, onset_pos + offset]
        hits.append((match & real).sum(dim=1))
    return hits[0], hits[1], hits[2], real.sum(dim=1)


def paired_summary(a, b, name_a, name_b):
    """Paired per-window difference of two accuracy vectors."""
    diff = a - b
    n = len(diff)
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "comparison": f"{name_a} - {name_b}",
        "n_windows": int(n),
        f"mean_{name_a}": float(a.mean()),
        f"mean_{name_b}": float(b.mean()),
        "mean_diff": mean,
        "se_diff": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
        "t": mean / se if se > 0 else float("nan"),
    }


@torch.no_grad()
def logit_sensitivity(model, batcher, batch, device):
    """Does swapping the plan change the window's logits? Both code paths.

    ``forward``: the training/teacher-forced path, explicit all-ones mask.
    ``cached``: the KV-cached decode path used by every AR metric, no mask.

    Both report the max absolute logit difference at the first body score slot
    under two different plans. Exactly 0.0 is the severed signature.
    """
    layout = batcher.layout
    codes = batcher.codes(batch)
    rolled = codes.roll(shifts=1, dims=0)

    def forward_logits(plan_codes):
        prepared = batcher.build(batch, codes=plan_codes)
        prepared["labels"] = None
        out = model(
            input_ids=prepared["input_ids"],
            attention_mask=prepared["attention_mask"],
            position_ids=prepared["position_ids"],
        )
        return out.logits[:, layout.score_start_idx - 1, :].float()

    def forward_logits_nomask(plan_codes):
        prepared = batcher.build(batch, codes=plan_codes)
        out = model(
            input_ids=prepared["input_ids"],
            position_ids=prepared["position_ids"],
        )
        return out.logits[:, layout.score_start_idx - 1, :].float()

    def cached_logits(plan_codes):
        """First body-score logits through the real AR decode entry point."""
        packed = batch["input_ids"].to(device)
        past, next_logits = None, None
        batch_size = packed.shape[0]
        packed_positions = (
            torch.arange(PACKED_LENGTH, device=device).unsqueeze(0).expand(batch_size, -1)
        )

        def feed(tokens, positions):
            nonlocal past, next_logits
            out = model(
                input_ids=tokens, position_ids=positions, past_key_values=past, use_cache=True
            )
            past = out.past_key_values
            next_logits = out.logits[:, -1, :]

        def column(value):
            return torch.full((batch_size, 1), int(value), dtype=torch.long, device=device)

        def positions_for(values):
            return (
                torch.tensor(values, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        from plan_lm import PLAN_CODE_OFFSET, PLAN_POSITION_BASE

        if layout.placement == "after_prefix":
            feed(packed[:, :ALTERNATING_START], packed_positions[:, :ALTERNATING_START])
        feed(column(layout.plan_bos), positions_for([PLAN_POSITION_BASE]))
        for j in range(layout.num_codes):
            feed(
                (PLAN_CODE_OFFSET + plan_codes[:, j]).unsqueeze(1),
                positions_for([PLAN_POSITION_BASE + 1 + j]),
            )
        if layout.placement == "front":
            feed(packed[:, :ALTERNATING_START], packed_positions[:, :ALTERNATING_START])
        return next_logits.float()

    results = {}
    for name, fn in (
        ("forward_masked", forward_logits),
        ("forward_unmasked", forward_logits_nomask),
        ("cached_decode", cached_logits),
    ):
        a, b = fn(codes), fn(rolled)
        results[name] = {
            "max_abs_logit_delta": float((a - b).abs().max()),
            "mean_abs_logit_delta": float((a - b).abs().mean()),
            "argmax_changed_frac": float((a.argmax(-1) != b.argmax(-1)).float().mean()),
        }
    results["codes_differ_frac"] = float((codes != rolled).float().mean())
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=Path("run_plan_lm_20260814_192509"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Explicit checkpoint dir; defaults to the newest under --run_dir.")
    parser.add_argument("--vq_checkpoint", type=Path, required=True)
    parser.add_argument("--pianobart_checkpoint", type=str, default=None)
    parser.add_argument("--val_file", type=Path, default=Path("data/val_paper.txt"))
    parser.add_argument("--plan_placement", type=str, default="front",
                        choices=["front", "after_prefix"])
    parser.add_argument("--num_windows", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, default=Path("plan_conditioning_probe.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint or latest_checkpoint(args.run_dir)
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")

    plan_encoder = OnlinePlanEncoder(
        args.vq_checkpoint,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        pianobart_checkpoint=args.pianobart_checkpoint,
    )
    layout = PlanLayout(
        num_codes=plan_encoder.num_codes,
        codebook_size=plan_encoder.codebook_size,
        placement=args.plan_placement,
    )
    print(
        f"Plan: {layout.num_codes} codes / {layout.codebook_size} entries, "
        f"placement={layout.placement}, total length {layout.total_length}"
    )
    batcher = PlanBatcher(plan_encoder, layout)

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, use_cache=True
    )
    prepare_plan_model(model, layout, verbose=True)
    model.to(device).eval()

    dataset = TokenizedDataset(args.val_file, is_training=False)
    rng = random.Random(args.seed)
    count = min(args.num_windows, len(dataset))
    indices = rng.sample(range(len(dataset)), count)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"Windows: {count}/{len(dataset)} from {args.val_file} (seed {args.seed})")

    positions = score_slot_positions()
    accumulated = {c: {"onset": [], "duration": [], "pitch": []} for c in CONDITIONS}
    slot_counts = []
    token_divergence = []          # oracle vs shuffled rollout token disagreement
    plan_code_match = [0, 0]
    sensitivity = None

    start = time.time()
    for step, batch in enumerate(loader):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        if input_ids.shape[0] < 2:
            print("Skipping a trailing batch of 1 (the shuffle control needs a partner).")
            continue

        codes = batcher.codes(batch)
        shuffled_codes = codes.roll(shifts=1, dims=0)

        if sensitivity is None:
            sensitivity = logit_sensitivity(model, batcher, batch, device)
            print("Plan -> logit sensitivity (first body score slot):")
            print(json.dumps(sensitivity, indent=2))

        rollouts = {}
        rollouts["oracle"], _ = plan_autoregressive_generate_score(
            model, input_ids, layout, device, codes=codes
        )
        rollouts["shuffled"], _ = plan_autoregressive_generate_score(
            model, input_ids, layout, device, codes=shuffled_codes
        )
        rollouts["own"], own_codes = plan_autoregressive_generate_score(
            model, input_ids, layout, device, generate_plan=True
        )
        plan_code_match[0] += int((own_codes == codes).sum())
        plan_code_match[1] += int(codes.numel())

        totals = None
        for name, generated in rollouts.items():
            onset, duration, pitch, real = per_window_hits(generated, input_ids, positions)
            totals = real
            for key, hits in (("onset", onset), ("duration", duration), ("pitch", pitch)):
                accumulated[name][key].append(hits.cpu().numpy())
        slot_counts.append(totals.cpu().numpy())

        slot_index = torch.tensor(positions, device=device)
        body = torch.cat([slot_index + o for o in range(3)])
        token_divergence.append(
            float(
                (rollouts["oracle"][:, body] != rollouts["shuffled"][:, body]).float().mean()
            )
        )

        done = (step + 1) * args.batch_size
        elapsed = time.time() - start
        print(
            f"[{done}/{count}] {elapsed:.1f}s elapsed "
            f"({elapsed / max(1, done):.2f}s/window)",
            flush=True,
        )

    elapsed = time.time() - start
    slots = np.concatenate(slot_counts).astype(np.float64)
    keep = slots > 0
    accuracy = {
        condition: {
            key: np.concatenate(values).astype(np.float64)[keep] / slots[keep]
            for key, values in per_key.items()
        }
        for condition, per_key in accumulated.items()
    }
    pooled = {
        condition: {
            key: float(np.concatenate(values).sum() / slots.sum())
            for key, values in per_key.items()
        }
        for condition, per_key in accumulated.items()
    }

    report = {
        "checkpoint": str(checkpoint),
        "vq_checkpoint": str(args.vq_checkpoint),
        "val_file": str(args.val_file),
        "placement": layout.placement,
        "n_windows": int(keep.sum()),
        "n_score_notes": int(slots.sum()),
        "seconds": elapsed,
        "seconds_per_window": elapsed / max(1, int(keep.sum())),
        "plan_logit_sensitivity": sensitivity,
        "oracle_vs_shuffled_rollout_token_disagreement": float(np.mean(token_divergence)),
        "greedy_plan_code_accuracy": (
            plan_code_match[0] / plan_code_match[1] if plan_code_match[1] else 0.0
        ),
        "pooled_note_level_accuracy": pooled,
        "macro_window_accuracy": {
            condition: {key: float(values.mean()) for key, values in per_key.items()}
            for condition, per_key in accuracy.items()
        },
        "paired": {},
    }
    for key in ("onset", "duration", "pitch"):
        report["paired"][key] = [
            paired_summary(accuracy["oracle"][key], accuracy["shuffled"][key],
                           "oracle", "shuffled"),
            paired_summary(accuracy["oracle"][key], accuracy["own"][key], "oracle", "own"),
        ]

    args.output.write_text(json.dumps(report, indent=2))
    print("\n" + "=" * 70)
    print(json.dumps(report, indent=2))
    print("=" * 70)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
