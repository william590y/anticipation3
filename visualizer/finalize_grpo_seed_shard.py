#!/usr/bin/env python
"""Repair, score, and canonicalize one legacy GRPO visualization shard.

The source array was already running when raw-note seed alignment was fixed.  A
worker validates one of its four outputs, dynamically identifies every legacy
raw seed walk whose prefix differs from ``raw_notes[].j`` alignment, recomputes
only those walks, removes redundant ``*_seed1`` aliases, and scores all rollouts
that this shard will add to the visualizer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from atomic_json import atomic_dump_json  # noqa: E402
from compute_sequence_ppl import load_payload  # noqa: E402
from seed_pipeline_common import (  # noqa: E402
    DEFAULT_GRPO_CHECKPOINT,
    SEED_COUNTS,
    attach_inline_metrics,
    build_seed_rollout,
    canonical_seed_variant,
    expected_seed_prefix,
    legacy_raw_seed_prefix,
    load_checkpoint_model,
    roll_args,
    same_checkpoint,
    valid_rollout,
)


NUM_SHARDS = 4


def fail(message):
    raise ValueError(message)


def _source_variant(block, stream, count):
    canonical = canonical_seed_variant(stream, count)
    if count == 1:
        return block.get(canonical) or block.get(f"{stream}_seed1")
    return block.get(canonical)


def validate_source_shard(payload, shard, shard_index, source, expected_grpo):
    if not isinstance(shard, dict):
        fail(f"{source}: shard root is not an object")
    if not same_checkpoint(shard.get("grpo_checkpoint"), expected_grpo):
        fail(
            f"{source}: GRPO checkpoint {shard.get('grpo_checkpoint')!r} does not "
            f"match {expected_grpo!r}"
        )
    try:
        counts = tuple(sorted({int(v) for v in shard.get("seed_counts", [])}))
    except (TypeError, ValueError) as exc:
        fail(f"{source}: invalid seed_counts: {exc}")
    if counts != SEED_COUNTS:
        fail(f"{source}: seed_counts must be {list(SEED_COUNTS)}, got {list(counts)}")
    if "shard_index" in shard and shard["shard_index"] != shard_index:
        fail(f"{source}: metadata shard_index is not {shard_index}")
    if "num_shards" in shard and shard["num_shards"] != NUM_SHARDS:
        fail(f"{source}: metadata num_shards is not {NUM_SHARDS}")

    examples = payload.get("examples")
    order = list(payload.get("example_order") or examples or {})
    expected_ids = order[shard_index::NUM_SHARDS]
    patches = shard.get("examples")
    if not isinstance(patches, dict) or set(patches) != set(expected_ids):
        fail(
            f"{source}: examples must be exactly stride {shard_index}: "
            f"expected {expected_ids}, got {sorted(patches or {})}"
        )
    declared = shard.get("example_order")
    if declared is not None and list(declared) != expected_ids:
        fail(f"{source}: declared example_order does not match stride order")

    for eid in expected_ids:
        patch = patches[eid]
        if not isinstance(patch, dict):
            fail(f"{source}: {eid} patch is not an object")
        ex = examples[eid]
        for group, base_patch in (
            ("rollouts_grpo", False),
            ("rollouts_seed_patch", True),
        ):
            block = patch.get(group)
            if not isinstance(block, dict):
                fail(f"{source}: {eid} is missing {group}")
            if not base_patch:
                for variant in ("filtered", "raw"):
                    if variant == "raw" and not ex.get("raw_notes"):
                        continue
                    if not valid_rollout(block.get(variant), require_metrics=False):
                        fail(f"{source}: {eid}/{group}/{variant} is incomplete")
            for count in SEED_COUNTS:
                for stream in ("filtered", "raw"):
                    if expected_seed_prefix(ex, stream, count) is None:
                        continue
                    rollout = _source_variant(block, stream, count)
                    if not valid_rollout(rollout, require_metrics=False):
                        fail(
                            f"{source}: {eid}/{group} lacks {stream} seed-{count}"
                        )
    return expected_ids


def canonicalize_and_repair(
    block,
    ex,
    model,
    device,
    args,
    *,
    base_patch,
):
    """Canonicalize seed-1 and repair every dynamically divergent raw prefix."""
    repaired = []

    # The source job recomputed a redundant filtered_seed1.  The GRPO block
    # retains its established filtered_seeded spelling; the base patch does not
    # own seed-1 unless a raw repair must override existing data.
    filtered_alias = block.pop("filtered_seed1", None)
    if base_patch:
        block.pop("filtered_seeded", None)
    elif "filtered_seeded" not in block and filtered_alias is not None:
        block["filtered_seeded"] = filtered_alias

    for count in SEED_COUNTS:
        canonical = canonical_seed_variant("raw", count)
        alias = "raw_seed1" if count == 1 else None
        correct = expected_seed_prefix(ex, "raw", count)
        legacy = legacy_raw_seed_prefix(ex, count)

        if correct is None:
            block.pop(canonical, None)
            if alias:
                block.pop(alias, None)
            continue

        if correct != legacy:
            rollout = build_seed_rollout(
                model, device, ex, "raw", count, args
            )
            if rollout is None:
                fail(f"repair unexpectedly unavailable for raw seed-{count}")
            block[canonical] = rollout
            repaired.append(canonical)
        elif count == 1:
            if base_patch:
                # Existing examples[id].rollouts.raw_seeded remains authoritative.
                block.pop(canonical, None)
            elif canonical not in block and block.get(alias) is not None:
                block[canonical] = block[alias]

        if alias:
            block.pop(alias, None)

    # Remove any malformed/unavailable seeded variants rather than publishing a
    # selector entry that cannot be displayed.
    for count in SEED_COUNTS:
        for stream in ("filtered", "raw"):
            variant = canonical_seed_variant(stream, count)
            if base_patch and count == 1 and variant != "raw_seeded":
                block.pop(variant, None)
                continue
            if expected_seed_prefix(ex, stream, count) is None:
                block.pop(variant, None)

    for key in list(block):
        if key.endswith("_seed1") or block[key] is None:
            del block[key]
    return repaired


def score_block(model, device, ex, block, eid, group):
    for variant, rollout in tqdm(
        list(block.items()),
        desc=f"metrics {eid}/{group}",
        leave=False,
    ):
        if not valid_rollout(rollout, require_metrics=False):
            fail(f"{eid}/{group}/{variant}: incomplete rollout before scoring")
        attach_inline_metrics(model, device, ex, variant, rollout)
        if not valid_rollout(rollout, require_metrics=True):
            fail(f"{eid}/{group}/{variant}: inline metric scoring failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--grpo-checkpoint", default=DEFAULT_GRPO_CHECKPOINT)
    ap.add_argument("--base-checkpoint", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--topk-onset", type=int, default=5)
    ap.add_argument("--topk-dur", type=int, default=4)
    ap.add_argument("--topk-pitch", type=int, default=8)
    ap.add_argument("--max-candidates", type=int, default=40)
    args = ap.parse_args()

    if not 0 <= args.shard_index < NUM_SHARDS:
        raise SystemExit(f"--shard-index must be 0..{NUM_SHARDS - 1}")
    payload, _prefix = load_payload(args.data)
    source = Path(args.input)
    try:
        shard = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read complete source shard {source}: {exc}") from exc

    base_checkpoint = args.base_checkpoint or payload.get("checkpoint")
    if not base_checkpoint:
        raise SystemExit("base checkpoint is absent from both CLI and data.js")
    try:
        order = validate_source_shard(
            payload, shard, args.shard_index, source, args.grpo_checkpoint
        )
    except ValueError as exc:
        raise SystemExit(f"refusing to finalize source shard: {exc}") from exc

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rollout_args = roll_args(
        args.topk_onset,
        args.topk_dur,
        args.topk_pitch,
        args.max_candidates,
    )
    repairs = {}
    started = time.perf_counter()

    for group, checkpoint, base_patch in (
        ("rollouts_grpo", args.grpo_checkpoint, False),
        ("rollouts_seed_patch", base_checkpoint, True),
    ):
        print(f"Loading {group} checkpoint {checkpoint} on {device}", flush=True)
        model = load_checkpoint_model(checkpoint, device, is_lora=False)
        for eid in order:
            ex = payload["examples"][eid]
            block = shard["examples"][eid][group]
            repaired = canonicalize_and_repair(
                block,
                ex,
                model,
                device,
                rollout_args,
                base_patch=base_patch,
            )
            if repaired:
                repairs.setdefault(eid, {})[group] = repaired
                print(f"{eid}/{group}: repaired {', '.join(repaired)}", flush=True)
            score_block(model, device, ex, block, eid, group)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    shard.update(
        {
            "format": 3,
            "finalized": True,
            "source_shard": str(source),
            "shard_index": args.shard_index,
            "num_shards": NUM_SHARDS,
            "example_order": order,
            "grpo_checkpoint": args.grpo_checkpoint,
            "base_checkpoint": base_checkpoint,
            "seed_counts": list(SEED_COUNTS),
            "seed_alignment": "raw_notes.j",
            "seed1_aliases": "canonicalized_to_seeded",
            "raw_repairs": repairs,
        }
    )
    atomic_dump_json(args.output, shard)
    print(
        f"Atomically wrote {args.output}; {sum(len(v) for x in repairs.values() for v in x.values())} "
        f"raw variant repair(s), {time.perf_counter() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
