#!/usr/bin/env python
"""Validate and atomically publish finalized GRPO and multi-seed shards."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from compute_sequence_ppl import load_payload  # noqa: E402
from seed_pipeline_common import (  # noqa: E402
    BACKFILL_GROUPS,
    DEFAULT_GRPO_CHECKPOINT,
    EXTRA_SEED_COUNTS,
    SEED_COUNTS,
    canonical_seed_variant,
    checkpoint_for_group,
    expected_seed_prefix,
    raw_seed_needs_repair,
    same_checkpoint,
    valid_rollout,
)


NUM_SHARDS = 4
GROUP_TAGS = {
    "rollouts_lora": "lora",
    "rollouts_valloss": "valloss",
    "rollouts_lora_valloss": "lora_valloss",
}


def fail(message):
    raise ValueError(message)


def ordered_examples(payload, *, expected_count=24):
    examples = payload.get("examples")
    if not isinstance(examples, dict):
        fail("data.js examples is not an object")
    order = list(payload.get("example_order") or examples)
    if len(order) != expected_count:
        fail(f"expected exactly {expected_count} windows, found {len(order)}")
    if len(order) != len(set(order)) or set(order) != set(examples):
        fail("example_order is not a one-to-one ordering of examples")
    return order


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read complete JSON shard {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{path}: shard root is not an object")
    return value


def validate_seed_block(ex, block, counts, path, *, require_plain=False):
    if not isinstance(block, dict):
        fail(f"{path}: rollout block is not an object")
    if any(name.endswith("_seed1") for name in block):
        fail(f"{path}: redundant *_seed1 alias survived finalization")
    if require_plain:
        if not valid_rollout(block.get("filtered")):
            fail(f"{path}: filtered rollout or inline metrics are incomplete")
        if ex.get("raw_notes") and not valid_rollout(block.get("raw")):
            fail(f"{path}: raw rollout or inline metrics are incomplete")
    for count in counts:
        for stream in ("filtered", "raw"):
            if expected_seed_prefix(ex, stream, count) is None:
                continue
            variant = canonical_seed_variant(stream, count)
            if not valid_rollout(block.get(variant)):
                fail(f"{path}: {variant} or inline metrics are incomplete")


def merge_grpo(payload, order, shard_dir, expected_grpo):
    seen = set()
    base_checkpoint = payload.get("checkpoint")
    if not base_checkpoint:
        fail("data.js has no base checkpoint")
    for index in range(NUM_SHARDS):
        path = Path(shard_dir) / f"shard_0{index}.json"
        shard = read_json(path)
        if shard.get("finalized") is not True:
            fail(f"{path}: shard was not finalized")
        if shard.get("shard_index") != index or shard.get("num_shards") != NUM_SHARDS:
            fail(f"{path}: invalid shard index metadata")
        if shard.get("seed_alignment") != "raw_notes.j":
            fail(f"{path}: raw seed alignment is not certified")
        if tuple(shard.get("seed_counts") or []) != SEED_COUNTS:
            fail(f"{path}: seed counts are not exactly 1..5")
        if not same_checkpoint(shard.get("grpo_checkpoint"), expected_grpo):
            fail(f"{path}: unexpected GRPO checkpoint")
        if not same_checkpoint(shard.get("base_checkpoint"), base_checkpoint):
            fail(f"{path}: base checkpoint differs from current data.js")
        expected_ids = order[index::NUM_SHARDS]
        if list(shard.get("example_order") or []) != expected_ids:
            fail(f"{path}: example_order is not the expected stride")
        patches = shard.get("examples")
        if not isinstance(patches, dict) or set(patches) != set(expected_ids):
            fail(f"{path}: example coverage is incomplete")
        for eid in expected_ids:
            ex = payload["examples"][eid]
            patch = patches[eid]
            grpo = patch.get("rollouts_grpo")
            base = patch.get("rollouts_seed_patch")
            validate_seed_block(
                ex, grpo, SEED_COUNTS, f"{path}:{eid}/rollouts_grpo", require_plain=True
            )
            validate_seed_block(
                ex, base, EXTRA_SEED_COUNTS, f"{path}:{eid}/rollouts_seed_patch"
            )
            # A seed-1 entry in the base patch is allowed only when dynamic
            # inspection proved the legacy raw walk needed replacement.
            if "raw_seeded" in base:
                if not raw_seed_needs_repair(ex, 1) or not valid_rollout(base["raw_seeded"]):
                    fail(f"{path}:{eid}: unexpected base raw_seeded override")
            ex["rollouts_grpo"] = grpo
            existing_base = ex.get("rollouts")
            if not isinstance(existing_base, dict):
                fail(f"data.js:{eid} has no base rollout block")
            existing_base.update(base)
            seen.add(eid)
    if seen != set(order):
        fail("finalized GRPO shards do not cover all 24 windows")


def merge_backfill(payload, order, shard_dir):
    for group in BACKFILL_GROUPS:
        checkpoint = checkpoint_for_group(payload, group)
        if not checkpoint:
            fail(f"data.js has no checkpoint identity for {group}")
        seen = set()
        tag = GROUP_TAGS[group]
        for index in range(NUM_SHARDS):
            path = Path(shard_dir) / f"{tag}_shard_0{index}.json"
            shard = read_json(path)
            if shard.get("group") != group:
                fail(f"{path}: group metadata is not {group}")
            if shard.get("shard_index") != index or shard.get("num_shards") != NUM_SHARDS:
                fail(f"{path}: invalid shard index metadata")
            if shard.get("seed_alignment") != "raw_notes.j":
                fail(f"{path}: raw seed alignment is not certified")
            if tuple(shard.get("seed_counts") or []) != EXTRA_SEED_COUNTS:
                fail(f"{path}: seed counts are not exactly 2..5")
            if not same_checkpoint(shard.get("checkpoint"), checkpoint):
                fail(f"{path}: checkpoint differs from current data.js")
            expected_ids = order[index::NUM_SHARDS]
            if list(shard.get("example_order") or []) != expected_ids:
                fail(f"{path}: example_order is not the expected stride")
            patches = shard.get("examples")
            if not isinstance(patches, dict) or set(patches) != set(expected_ids):
                fail(f"{path}: example coverage is incomplete")
            for eid in expected_ids:
                ex = payload["examples"][eid]
                patch = patches[eid]
                if set(patch) != {group}:
                    fail(f"{path}:{eid}: patch contains unexpected groups")
                block = patch[group]
                validate_seed_block(ex, block, EXTRA_SEED_COUNTS, f"{path}:{eid}/{group}")
                current = ex.get(group)
                if not isinstance(current, dict):
                    fail(f"data.js:{eid} has no existing {group} block")
                current.update(block)
                seen.add(eid)
        if seen != set(order):
            fail(f"{group} backfill does not cover all 24 windows")


def validate_final_payload(payload, expected_grpo):
    order = ordered_examples(payload)
    if not same_checkpoint(payload.get("grpo_checkpoint"), expected_grpo):
        fail("published GRPO checkpoint identity is wrong")
    if tuple(payload.get("seed_counts") or []) != SEED_COUNTS:
        fail("published seed_counts are not exactly 1..5")
    if payload.get("seed_alignment") != "raw_notes.j":
        fail("published seed_alignment is not raw_notes.j")
    for eid in order:
        ex = payload["examples"][eid]
        for group in ("rollouts", *BACKFILL_GROUPS, "rollouts_grpo"):
            validate_seed_block(
                ex,
                ex.get(group),
                SEED_COUNTS,
                f"published:{eid}/{group}",
                require_plain=True,
            )
    sets = payload.get("checkpoint_sets")
    if not isinstance(sets, dict):
        fail("checkpoint_sets is missing")
    if not same_checkpoint((sets.get("grpo") or {}).get("checkpoint"), expected_grpo):
        fail("checkpoint_sets.grpo identity is wrong")
    return order


def stage_data_js(path, prefix, payload):
    path = Path(path)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.seed-publish.", suffix=".tmp"
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(prefix)
            json.dump(payload, handle, allow_nan=False)
            handle.write(";\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            os.chmod(temp, path.stat().st_mode & 0o777)
        return temp
    except BaseException:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise


def replace_and_fsync(temp, destination):
    destination = Path(destination)
    os.replace(temp, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def bump_html_cachebuster(html_path, cache_version):
    """Atomically point the HTML at the newly published data.js generation."""
    html_path = Path(html_path)
    text = html_path.read_text(encoding="utf-8")
    pattern = re.compile(r'(?P<prefix>src=["\']data\.js\?v=)[^"\']+(?P<quote>["\'])')
    replacement = rf"\g<prefix>{cache_version}\g<quote>"
    updated, count = pattern.subn(replacement, text)
    if count != 1:
        fail(f"{html_path}: expected exactly one data.js?v= script tag, found {count}")
    fd, temp_name = tempfile.mkstemp(
        dir=html_path.parent,
        prefix=f".{html_path.name}.cache-bump.",
        suffix=".tmp",
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, html_path.stat().st_mode & 0o777)
        replace_and_fsync(temp, html_path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--grpo-shards", default="visualizer/grpo_final_shards")
    ap.add_argument("--backfill-shards", default="visualizer/seed_backfill_shards")
    ap.add_argument("--grpo-checkpoint", default=DEFAULT_GRPO_CHECKPOINT)
    ap.add_argument("--html", default="visualizer/visualizer.html")
    ap.add_argument("--cache-version", default="20260814020000")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    destination = Path(args.data)
    lock_path = destination.with_name(destination.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(f"Acquired exclusive publish lock {lock_path}", flush=True)
        payload, prefix = load_payload(destination)
        order = ordered_examples(payload)
        merge_grpo(payload, order, args.grpo_shards, args.grpo_checkpoint)
        merge_backfill(payload, order, args.backfill_shards)

        payload["grpo_checkpoint"] = args.grpo_checkpoint
        payload["seed_counts"] = list(SEED_COUNTS)
        payload["seed_alignment"] = "raw_notes.j"
        checkpoint_sets = payload.setdefault("checkpoint_sets", {})
        checkpoint_sets["pitch_ar"] = {
            "label": "best pitch AR",
            "checkpoint": payload.get("checkpoint"),
            "lora_checkpoint": payload.get("lora_checkpoint"),
        }
        checkpoint_sets["val_loss"] = {
            "label": "best val loss",
            "checkpoint": payload.get("checkpoint_val_loss"),
            "lora_checkpoint": payload.get("lora_checkpoint_val_loss"),
        }
        checkpoint_sets["grpo"] = {
            "label": "GRPO (best val reward)",
            "checkpoint": args.grpo_checkpoint,
        }
        payload["seed_pipeline"] = {
            "published_at_utc": datetime.now(timezone.utc).isoformat(),
            "grpo_shards": NUM_SHARDS,
            "backfill_tasks": len(BACKFILL_GROUPS) * NUM_SHARDS,
            "seed_counts": list(SEED_COUNTS),
            "seed_alignment": "raw_notes.j",
            "inline_metrics": ["f1", "sequence_perplexity"],
        }
        validate_final_payload(payload, args.grpo_checkpoint)
        if args.dry_run:
            print("Dry run passed; data.js was not rewritten", flush=True)
            return

        temp = stage_data_js(destination, prefix, payload)
        print(f"Staged merged data at {temp}; validating serialized JSON", flush=True)
        del payload
        gc.collect()
        staged_payload, staged_prefix = load_payload(temp)
        if staged_prefix != prefix:
            fail("staged data.js prefix changed")
        validate_final_payload(staged_payload, args.grpo_checkpoint)
        del staged_payload
        gc.collect()
        replace_and_fsync(temp, destination)
        temp = None
        print(f"Atomically published and directory-fsynced {destination}", flush=True)
        bump_html_cachebuster(args.html, args.cache_version)
        print(
            f"Atomically bumped {args.html} to data.js?v={args.cache_version}",
            flush=True,
        )

    if temp is not None:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"refusing to publish seed pipeline: {exc}") from exc
