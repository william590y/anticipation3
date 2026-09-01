#!/usr/bin/env python
"""Small, dependency-free invariants shared by the PPO visualization chain."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


EXPECTED_WINDOWS = 24
# One shard per idle thickstun GPU, so all eight run the rollouts concurrently.
EXPECTED_SHARDS = 8
WINDOWS_PER_SHARD = EXPECTED_WINDOWS // EXPECTED_SHARDS
SEED_COUNTS = (1, 2, 3, 4, 5)
F1_VARIANTS = ("onset_pitch", "onset_pitch_dur", "onset_pitch_tol1")


def load_data_js(path):
    """Load the visualizer assignment while preserving its JavaScript prefix."""
    text = Path(path).read_text(encoding="utf-8")
    left = text.find("{")
    right = text.rfind("}")
    if left < 0 or right < left or not text[:left].lstrip().startswith("window."):
        raise ValueError(f"{path} is not a window.* JSON assignment")
    return json.loads(text[left : right + 1]), text[:left]


def ordered_example_ids(payload):
    examples = payload.get("examples")
    if not isinstance(examples, dict):
        raise ValueError("visualizer payload has no examples object")
    order = list(payload.get("example_order") or examples)
    if len(order) != EXPECTED_WINDOWS:
        raise ValueError(
            f"expected exactly {EXPECTED_WINDOWS} visualizer windows, got {len(order)}"
        )
    if len(set(order)) != len(order) or set(order) != set(examples):
        raise ValueError("example_order is not a one-to-one ordering of examples")
    return order


def dataset_identity(payload):
    """Hash only rollout inputs, so independent GRPO/PPO publishing can coexist."""
    order = ordered_example_ids(payload)
    examples = payload["examples"]
    identity = {
        "example_order": order,
        "examples": {
            eid: {
                "perf_notes": examples[eid].get("perf_notes"),
                "raw_notes": examples[eid].get("raw_notes"),
                "gt_score": examples[eid].get("gt_score"),
            }
            for eid in order
        },
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def has_filtered_seed(ex, count):
    gt = ex.get("gt_score") or []
    return len(gt) >= count and all(gt[i] is not None for i in range(count))


def has_raw_seed(ex, count):
    gt = ex.get("gt_score") or []
    matched = 0
    for raw_note in ex.get("raw_notes") or []:
        j = raw_note.get("j") if isinstance(raw_note, dict) else None
        if (
            isinstance(j, int)
            and not isinstance(j, bool)
            and 0 <= j < len(gt)
            and gt[j] is not None
        ):
            matched += 1
            if matched == count:
                return True
    return False


def expected_variant_states(ex):
    """Expected compute_rollout_set keys and whether each must be non-null."""
    states = {
        "filtered": True,
        "filtered_seeded": has_filtered_seed(ex, 1),
        "raw": bool(ex.get("raw_notes")),
        "raw_seeded": has_raw_seed(ex, 1),
    }
    for count in SEED_COUNTS[1:]:
        if has_filtered_seed(ex, count):
            states[f"filtered_seed{count}"] = True
        if has_raw_seed(ex, count):
            states[f"raw_seed{count}"] = True
    return states


def _finite_number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    value = float(value)
    return math.isfinite(value) and (not positive or value > 0)


def validate_rollout_metrics(rollout, *, context):
    if not isinstance(rollout, dict):
        raise ValueError(f"{context}: rollout is not an object")
    if not isinstance(rollout.get("pred_score"), list):
        raise ValueError(f"{context}: missing pred_score")
    if not isinstance(rollout.get("branches"), dict):
        raise ValueError(f"{context}: missing branches")

    f1 = rollout.get("f1")
    if not isinstance(f1, dict) or set(f1) != set(F1_VARIANTS):
        raise ValueError(f"{context}: missing complete note-F1 metrics")
    for variant in F1_VARIANTS:
        values = f1[variant]
        if not isinstance(values, dict) or not _finite_number(values.get("f1")):
            raise ValueError(f"{context}: invalid {variant} F1 metric")

    pair = rollout.get("sequence_perplexity")
    if not isinstance(pair, dict):
        raise ValueError(f"{context}: missing sequence_perplexity")
    for field in ("generated", "ground_truth"):
        if not _finite_number(pair.get(field), positive=True):
            raise ValueError(f"{context}: invalid sequence_perplexity.{field}")


def validate_rollout_block(block, ex, *, context):
    if not isinstance(block, dict):
        raise ValueError(f"{context}: rollouts_ppo is not an object")
    states = expected_variant_states(ex)
    allowed = set(states)
    unknown = set(block) - allowed
    missing = set(states) - set(block)
    if unknown:
        raise ValueError(f"{context}: unexpected variants {sorted(unknown)}")
    if missing:
        raise ValueError(f"{context}: missing variants {sorted(missing)}")
    for variant, required in states.items():
        rollout = block[variant]
        if required:
            validate_rollout_metrics(rollout, context=f"{context}/{variant}")
        elif rollout is not None:
            raise ValueError(f"{context}/{variant}: expected null unavailable rollout")

