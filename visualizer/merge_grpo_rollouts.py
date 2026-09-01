#!/usr/bin/env python
"""Validate and merge four GRPO / multi-seed shards into ``data.js``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atomic_json import atomic_dump_data_js  # noqa: E402
from compute_sequence_ppl import load_payload  # noqa: E402


EXPECTED_SHARDS = 4
REQUIRED_SEED_COUNTS = (1, 2, 3, 4, 5)


def _fail(message):
    raise ValueError(message)


def _valid_rollout(value):
    return (
        isinstance(value, dict)
        and isinstance(value.get("pred_score"), list)
        and isinstance(value.get("branches"), dict)
    )


def _seed_variant(block, stream, count):
    if count == 1:
        # Legacy shards contain both spellings; current shards keep only the
        # established ``*_seeded`` spelling to avoid duplicating a large rollout.
        return block.get(f"{stream}_seeded") or block.get(f"{stream}_seed1")
    return block.get(f"{stream}_seed{count}")


def _has_n_filtered_gt(ex, count):
    gt = ex.get("gt_score") or []
    return len(gt) >= count and all(gt[i] is not None for i in range(count))


def _has_n_raw_matches(ex, count):
    gt = ex.get("gt_score") or []
    matched = 0
    for note in ex.get("raw_notes") or []:
        j = note.get("j") if isinstance(note, dict) else None
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


def _validate_rollouts(block, ex, path, eid, *, base_patch=False):
    if not isinstance(block, dict):
        _fail(f"{path}: example {eid} has no rollout dictionary")

    if not base_patch:
        if not _valid_rollout(block.get("filtered")):
            _fail(f"{path}: example {eid} has an incomplete filtered GRPO rollout")
        if ex.get("raw_notes") and not _valid_rollout(block.get("raw")):
            _fail(f"{path}: example {eid} has an incomplete raw GRPO rollout")

    for count in REQUIRED_SEED_COUNTS:
        # Base patches intentionally start at two: the base data already owns
        # the legacy one-note variants.
        if not base_patch or count > 1:
            if _has_n_filtered_gt(ex, count) and not _valid_rollout(
                _seed_variant(block, "filtered", count)
            ):
                _fail(
                    f"{path}: example {eid} is missing complete filtered seed-{count} data"
                )
            if _has_n_raw_matches(ex, count) and not _valid_rollout(
                _seed_variant(block, "raw", count)
            ):
                _fail(f"{path}: example {eid} is missing complete raw seed-{count} data")


def validate_shards(payload, shard_paths):
    """Load all shards and reject partial, overlapping, or inconsistent output."""
    if len(shard_paths) != EXPECTED_SHARDS:
        _fail(f"expected exactly {EXPECTED_SHARDS} shard paths, got {len(shard_paths)}")
    resolved = [Path(path).resolve() for path in shard_paths]
    if len(set(resolved)) != EXPECTED_SHARDS:
        _fail("the four shard paths must be distinct")

    examples = payload.get("examples")
    if not isinstance(examples, dict) or not examples:
        _fail("data.js has no examples")
    order = list(payload.get("example_order") or examples)
    if len(order) != len(set(order)) or set(order) != set(examples):
        _fail("data.js example_order is not a one-to-one ordering of examples")
    expected_partitions = [set(order[i::EXPECTED_SHARDS]) for i in range(EXPECTED_SHARDS)]

    shards = []
    checkpoints = []
    seed_count_sets = []
    metadata_indices = []
    alignments = []
    observed_partitions = []
    base_patch_modes = []
    seen_examples = set()

    for path in resolved:
        try:
            shard = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _fail(f"could not read complete JSON from {path}: {exc}")
        if not isinstance(shard, dict):
            _fail(f"{path}: shard root is not an object")

        checkpoint = shard.get("grpo_checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            _fail(f"{path}: missing grpo_checkpoint")
        checkpoints.append(checkpoint)

        try:
            counts = tuple(sorted({int(n) for n in shard.get("seed_counts", [])}))
        except (TypeError, ValueError):
            _fail(f"{path}: invalid seed_counts")
        if counts != REQUIRED_SEED_COUNTS:
            _fail(
                f"{path}: seed_counts must be {list(REQUIRED_SEED_COUNTS)}, got {list(counts)}"
            )
        seed_count_sets.append(counts)
        alignments.append(shard.get("seed_alignment"))

        has_index = "shard_index" in shard or "num_shards" in shard
        if has_index:
            if shard.get("num_shards") != EXPECTED_SHARDS:
                _fail(f"{path}: num_shards must be {EXPECTED_SHARDS}")
            idx = shard.get("shard_index")
            if not isinstance(idx, int) or not 0 <= idx < EXPECTED_SHARDS:
                _fail(f"{path}: invalid shard_index {idx!r}")
            metadata_indices.append(idx)
        else:
            metadata_indices.append(None)

        patches = shard.get("examples")
        if not isinstance(patches, dict):
            _fail(f"{path}: examples is not an object")
        ids = set(patches)
        unknown = ids - set(examples)
        overlap = ids & seen_examples
        if unknown:
            _fail(f"{path}: unknown examples {sorted(unknown)}")
        if overlap:
            _fail(f"{path}: duplicate examples also present in another shard: {sorted(overlap)}")
        seen_examples |= ids
        observed_partitions.append(ids)

        declared_order = shard.get("example_order")
        if declared_order is not None and list(declared_order) != [eid for eid in order if eid in ids]:
            # New writers declare their stride order.  Compare it directly when
            # shard metadata is available below; legacy shards omit this field.
            if metadata_indices[-1] is None or list(declared_order) != order[metadata_indices[-1]::EXPECTED_SHARDS]:
                _fail(f"{path}: example_order does not match the shard's examples")

        patch_modes = set()
        for eid, patch in patches.items():
            if not isinstance(patch, dict):
                _fail(f"{path}: example {eid} patch is not an object")
            ex = examples[eid]
            _validate_rollouts(patch.get("rollouts_grpo"), ex, path, eid)
            has_base = "rollouts_seed_patch" in patch
            patch_modes.add(has_base)
            if has_base:
                _validate_rollouts(
                    patch.get("rollouts_seed_patch"), ex, path, eid, base_patch=True
                )
        if len(patch_modes) > 1:
            _fail(f"{path}: only some examples contain the base seed patch")
        base_patch_modes.append(next(iter(patch_modes), False))
        shards.append((path, shard))

    if len(set(checkpoints)) != 1:
        _fail(f"inconsistent GRPO checkpoints across shards: {sorted(set(checkpoints))}")
    if len(set(seed_count_sets)) != 1:
        _fail("inconsistent seed_counts across shards")
    if len(set(alignments)) != 1:
        _fail("mixed legacy/current seed-alignment formats across shards")
    if len(set(base_patch_modes)) != 1:
        _fail("only some shards contain base-model seed patches")
    if seen_examples != set(examples):
        missing = set(examples) - seen_examples
        _fail(f"shards do not cover every data.js example; missing {sorted(missing)}")

    has_metadata = [idx is not None for idx in metadata_indices]
    if any(has_metadata) and not all(has_metadata):
        _fail("mixed legacy/current shard metadata")
    if all(has_metadata):
        if sorted(metadata_indices) != list(range(EXPECTED_SHARDS)):
            _fail(f"shard indices must be exactly 0..{EXPECTED_SHARDS - 1}")
        for (path, _), idx, ids in zip(shards, metadata_indices, observed_partitions):
            if ids != expected_partitions[idx]:
                _fail(f"{path}: example coverage is incomplete for shard index {idx}")
    else:
        # Legacy shards predate explicit indices.  Exact stride-partition matching
        # still proves that all four complete workers are represented.
        unmatched = list(expected_partitions)
        for (path, _), ids in zip(shards, observed_partitions):
            try:
                match = next(i for i, expected in enumerate(unmatched) if ids == expected)
            except StopIteration:
                _fail(f"{path}: legacy shard is not one complete 4-way stride partition")
            unmatched.pop(match)

    return {
        "shards": shards,
        "grpo_checkpoint": checkpoints[0],
        "seed_counts": list(seed_count_sets[0]),
        "seed_alignment": alignments[0],
        "base_seed_patch": base_patch_modes[0],
    }


def _drop_duplicate_seed1_aliases(block):
    """Remove only byte-equivalent legacy aliases; reject divergent aliases."""
    for stream in ("filtered", "raw"):
        legacy = f"{stream}_seeded"
        alias = f"{stream}_seed1"
        if legacy not in block or alias not in block:
            continue
        if block[legacy] != block[alias]:
            _fail(f"legacy aliases {legacy} and {alias} contain different rollouts")
        del block[alias]


def merge_validated(payload, validated):
    added_grpo = patched = 0
    for _path, shard in validated["shards"]:
        for eid, patch in shard["examples"].items():
            ex = payload["examples"][eid]
            grpo = patch["rollouts_grpo"]
            _drop_duplicate_seed1_aliases(grpo)
            ex["rollouts_grpo"] = grpo
            added_grpo += 1
            if "rollouts_seed_patch" in patch:
                base_patch = patch["rollouts_seed_patch"]
                _drop_duplicate_seed1_aliases(base_patch)
                ex["rollouts"].update(base_patch)
                patched += 1

    checkpoint = validated["grpo_checkpoint"]
    payload["grpo_checkpoint"] = checkpoint
    payload["seed_counts"] = validated["seed_counts"]
    sets = payload.setdefault("checkpoint_sets", {})
    sets["grpo"] = {"label": "GRPO (best val reward)", "checkpoint": checkpoint}
    return added_grpo, patched


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload, prefix = load_payload(args.data)
    try:
        validated = validate_shards(payload, args.shards)
        added_grpo, patched = merge_validated(payload, validated)
    except ValueError as exc:
        raise SystemExit(f"refusing to merge GRPO shards: {exc}") from exc

    print(f"rollouts_grpo on {added_grpo} windows; base seed-patch on {patched}")
    print(
        f"grpo_checkpoint={validated['grpo_checkpoint']}  "
        f"seed_counts={validated['seed_counts']}"
    )
    if validated["seed_alignment"] is None:
        print(
            "WARNING: accepting complete legacy shards without seed_alignment metadata; "
            "raw-note alignment cannot be certified from shard metadata"
        )
    if args.dry_run:
        print("dry-run: not writing")
        return

    out = Path(args.data)
    atomic_dump_data_js(out, prefix, payload)
    print(f"Atomically wrote {out}")


if __name__ == "__main__":
    main()
