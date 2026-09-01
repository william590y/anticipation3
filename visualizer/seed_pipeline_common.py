#!/usr/bin/env python
"""Shared, side-effect-free helpers for the multi-seed visualization pipeline.

The heavy entry points live in separate scripts so SLURM workers load exactly one
model at a time.  This module centralizes the raw-note alignment rule, rollout
construction, and inline metric attachment used by both worker types.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
VISUALIZER_DIR = REPO_ROOT / "visualizer"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VISUALIZER_DIR))

from anticipation.config import CONTEXT_SIZE  # noqa: E402
from compute_f1 import score_notes  # noqa: E402
from compute_sequence_ppl import (  # noqa: E402
    build_packed_tokens,
    control_notes_for_variant,
    gt_notes_for_variant,
    notes_from_pred,
    score_packed_sequence,
    summarize_pair,
)
from precompute_visualizer import (  # noqa: E402
    build_branches_from_slots,
    compact_entropy,
    compact_perplexity,
    filtered_seed_prefix,
    load_lora_model,
    raw_seed_prefix,
    rollout_with_candidates,
    tokens_from_controls,
)
from evaluate_muster import load_model  # noqa: E402


SEED_COUNTS = (1, 2, 3, 4, 5)
EXTRA_SEED_COUNTS = (2, 3, 4, 5)
BACKFILL_GROUPS = (
    "rollouts_lora",
    "rollouts_valloss",
    "rollouts_lora_valloss",
)
DEFAULT_GRPO_CHECKPOINT = "run_grpo_acc_reward/checkpoint-250"


def roll_args(topk_onset=5, topk_dur=4, topk_pitch=8, max_candidates=40):
    return SimpleNamespace(
        topk_onset=int(topk_onset),
        topk_dur=int(topk_dur),
        topk_pitch=int(topk_pitch),
        max_candidates=int(max_candidates),
        slot_progress=False,
    )


def canonical_seed_variant(stream: str, count: int) -> str:
    if stream not in ("filtered", "raw"):
        raise ValueError(f"unknown stream {stream!r}")
    if count not in SEED_COUNTS:
        raise ValueError(f"seed count must be 1..5, got {count}")
    return f"{stream}_seeded" if count == 1 else f"{stream}_seed{count}"


def expected_seed_prefix(ex: dict, stream: str, count: int):
    gt = ex.get("gt_score") or []
    if stream == "filtered":
        return filtered_seed_prefix(gt, count)
    if stream == "raw":
        return raw_seed_prefix(ex.get("raw_notes") or [], gt, count)
    raise ValueError(f"unknown stream {stream!r}")


def legacy_raw_seed_prefix(ex: dict, count: int):
    """The prefix used by the already-running legacy GRPO worker.

    It incorrectly treated raw score slots as filtered score slots.  Returning
    ``None`` for an incomplete prefix mirrors the corrected helper's contract.
    """
    gt = ex.get("gt_score") or []
    notes = list(gt[:count])
    return notes if len(notes) == count and all(note is not None for note in notes) else None


def raw_seed_needs_repair(ex: dict, count: int) -> bool:
    """Whether the legacy and raw-index-aligned seed walks differ."""
    return expected_seed_prefix(ex, "raw", count) != legacy_raw_seed_prefix(ex, count)


def checkpoint_path(value: str | os.PathLike) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


def same_checkpoint(left, right) -> bool:
    return bool(left and right) and checkpoint_path(left) == checkpoint_path(right)


def checkpoint_for_group(payload: dict, group: str):
    field = {
        "rollouts": "checkpoint",
        "rollouts_lora": "lora_checkpoint",
        "rollouts_valloss": "checkpoint_val_loss",
        "rollouts_lora_valloss": "lora_checkpoint_val_loss",
        "rollouts_grpo": "grpo_checkpoint",
        "rollouts_ppo": "ppo_checkpoint",
    }.get(group)
    if field is None:
        raise ValueError(f"unknown rollout group {group!r}")
    return payload.get(field)


def group_is_lora(group: str) -> bool:
    return group in ("rollouts_lora", "rollouts_lora_valloss")


def load_checkpoint_model(checkpoint: str, device: torch.device, *, is_lora=False):
    if is_lora:
        model = load_lora_model(checkpoint)
    else:
        model, _loaded = load_model(checkpoint, config_source=None)
    model.to(device)
    model.eval()
    return model


def window_tokens(ex: dict, stream: str):
    controls = (
        ex.get("raw_notes") or []
        if stream == "raw"
        else ex.get("perf_notes") or []
    )
    return tokens_from_controls(controls, CONTEXT_SIZE - 4)


@torch.inference_mode()
def build_seed_rollout(model, device, ex: dict, stream: str, count: int, args):
    """Compute exactly one correctly aligned seeded rollout."""
    seeds = expected_seed_prefix(ex, stream, count)
    if seeds is None:
        return None

    raw_notes = ex.get("raw_notes") or []
    tokens = window_tokens(ex, stream)
    pred, candidates, perplexity, entropy = rollout_with_candidates(
        model,
        device,
        tokens,
        args.topk_onset,
        args.topk_dur,
        args.topk_pitch,
        args.max_candidates,
        slot_progress=False,
        seed_note=seeds,
    )

    if stream == "raw":
        key_for_slot = lambda slot: slot if slot < len(raw_notes) else None

        def slot_meta(slot):
            return {
                "gt_slot": raw_notes[slot].get("j") if slot < len(raw_notes) else None,
                "raw_index": slot if slot < len(raw_notes) else None,
            }
    else:
        key_for_slot = lambda slot: slot
        slot_meta = lambda slot: {"gt_slot": slot, "filtered_index": slot}

    branches = build_branches_from_slots(
        candidates,
        key_for_slot=key_for_slot,
        slot_meta=slot_meta,
    )
    compact_ent = compact_entropy(entropy)
    return {
        "pred_score": pred,
        "branches": branches,
        "perplexity": compact_perplexity(perplexity),
        "entropy": compact_ent["entropy"],
        "log_entropy": compact_ent["log_entropy"],
    }


def _clean_notes(notes):
    return [
        note
        for note in (notes or [])
        if isinstance(note, dict) and note.get("p") is not None
    ]


@torch.inference_mode()
def attach_inline_metrics(model, device, ex: dict, variant: str, rollout: dict):
    """Attach note F1 and generated/GT sequence PPL to one rollout in place."""
    pred = rollout.get("pred_score") or []
    rollout["f1"] = score_notes(_clean_notes(pred), _clean_notes(ex.get("gt_score")))

    controls = control_notes_for_variant(ex, variant)
    n_slots = len(pred)
    generated_notes = notes_from_pred(pred)
    ground_truth_notes = gt_notes_for_variant(ex, variant, n_slots)
    generated_tokens, generated_positions, _ = build_packed_tokens(
        controls, generated_notes
    )
    gt_tokens, gt_positions, _ = build_packed_tokens(controls, ground_truth_notes)
    generated = score_packed_sequence(
        model,
        device,
        generated_tokens,
        generated_positions,
        n_slots,
    )
    ground_truth = score_packed_sequence(
        model,
        device,
        gt_tokens,
        gt_positions,
        n_slots,
    )
    rollout["sequence_perplexity"] = summarize_pair(generated, ground_truth)
    return rollout


def valid_sequence_perplexity(value) -> bool:
    if not isinstance(value, dict):
        return False
    for key in ("generated", "ground_truth"):
        number = value.get(key)
        if not isinstance(number, (int, float)) or isinstance(number, bool):
            return False
        if not math.isfinite(number) or number <= 0:
            return False
    return True


def valid_f1(value) -> bool:
    if not isinstance(value, dict):
        return False
    for key in ("onset_pitch", "onset_pitch_dur", "onset_pitch_tol1"):
        entry = value.get(key)
        if not isinstance(entry, dict):
            return False
        score = entry.get("f1")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            return False
        if not math.isfinite(score) or not 0 <= score <= 1:
            return False
    return True


def valid_rollout(value, *, require_metrics=True) -> bool:
    if not (
        isinstance(value, dict)
        and isinstance(value.get("pred_score"), list)
        and isinstance(value.get("branches"), dict)
    ):
        return False
    if not require_metrics:
        return True
    return valid_f1(value.get("f1")) and valid_sequence_perplexity(
        value.get("sequence_perplexity")
    )


def unload_model(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
