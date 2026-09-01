#!/usr/bin/env python
"""Validate exactly four PPO shards and atomically publish them into data.js."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from atomic_json import atomic_dump_data_js
from ppo_pipeline_common import (
    EXPECTED_SHARDS,
    EXPECTED_WINDOWS,
    SEED_COUNTS,
    dataset_identity,
    load_data_js,
    ordered_example_ids,
    validate_rollout_block,
)
from select_ppo_best import validate_selected_manifest


SHARD_SCHEMA = "anticipation3.ppo-rollout-shard.v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read complete {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} {path} is not an object")
    return value


def exact_shard_paths(shard_dir):
    shard_dir = Path(shard_dir).resolve()
    expected = [shard_dir / f"shard_{index:02d}.json" for index in range(EXPECTED_SHARDS)]
    observed = sorted(shard_dir.glob("shard_*.json")) if shard_dir.is_dir() else []
    if observed != expected:
        raise ValueError(
            f"expected exactly {[path.name for path in expected]} in {shard_dir}; "
            f"found {[path.name for path in observed]}"
        )
    return expected


def _same_checkpoint(shard, manifest):
    try:
        shard_checkpoint = Path(shard.get("checkpoint", "")).resolve()
        manifest_checkpoint = Path(manifest["checkpoint"]).resolve()
    except (TypeError, OSError):
        return False
    return (
        shard_checkpoint == manifest_checkpoint
        and shard.get("checkpoint_id") == manifest["checkpoint_id"]
        and shard.get("train_job_id") == manifest["train_job_id"]
        and shard.get("best_step") == manifest["best_step"]
        and shard.get("val_reward") == manifest["val_reward"]
    )


def validate_and_merge(payload, shard_paths, manifest, manifest_path):
    order = ordered_example_ids(payload)
    if len(order) != EXPECTED_WINDOWS:
        raise ValueError(f"expected exactly {EXPECTED_WINDOWS} windows")
    current_identity = dataset_identity(payload)
    manifest_path = Path(manifest_path).resolve()
    manifest_digest = file_sha256(manifest_path)
    seen = set()

    for expected_index, path in enumerate(shard_paths):
        shard = load_json(path, "PPO shard")
        if shard.get("schema") != SHARD_SCHEMA:
            raise ValueError(f"{path}: unexpected shard schema")
        if shard.get("shard_index") != expected_index or shard.get("num_shards") != EXPECTED_SHARDS:
            raise ValueError(f"{path}: invalid shard index/count metadata")
        if shard.get("seed_counts") != list(SEED_COUNTS):
            raise ValueError(f"{path}: seed counts are not exactly {list(SEED_COUNTS)}")
        if not _same_checkpoint(shard, manifest):
            raise ValueError(f"{path}: checkpoint identity does not match selected PPO model")
        try:
            shard_manifest = Path(shard.get("manifest", "")).resolve()
        except (TypeError, OSError) as exc:
            raise ValueError(f"{path}: invalid manifest path") from exc
        if shard_manifest != manifest_path or shard.get("manifest_sha256") != manifest_digest:
            raise ValueError(f"{path}: selection manifest identity changed")
        if shard.get("source_data_identity") != current_identity:
            raise ValueError(f"{path}: visualizer rollout inputs changed since precompute")

        expected_order = order[expected_index::EXPECTED_SHARDS]
        if shard.get("example_order") != expected_order:
            raise ValueError(f"{path}: shard does not contain its exact stride partition")
        patches = shard.get("examples")
        if not isinstance(patches, dict) or set(patches) != set(expected_order):
            raise ValueError(f"{path}: examples do not match declared partition")
        if seen & set(patches):
            raise ValueError(f"{path}: duplicate example coverage")
        seen.update(patches)

        # Mutating this in-memory payload is safe: no target is replaced unless
        # every later shard also validates successfully.
        for eid in expected_order:
            patch = patches[eid]
            if not isinstance(patch, dict) or set(patch) != {"rollouts_ppo"}:
                raise ValueError(f"{path}: {eid} has an invalid patch wrapper")
            block = patch["rollouts_ppo"]
            validate_rollout_block(block, payload["examples"][eid], context=f"{path}/{eid}")
            payload["examples"][eid]["rollouts_ppo"] = block

    if seen != set(order) or len(seen) != EXPECTED_WINDOWS:
        raise ValueError(
            f"{EXPECTED_SHARDS} shards do not cover exactly all {EXPECTED_WINDOWS} windows"
        )

    checkpoint_display = manifest.get("checkpoint_display") or manifest["checkpoint"]
    payload["ppo_checkpoint"] = checkpoint_display
    payload["ppo_checkpoint_id"] = manifest["checkpoint_id"]
    payload["ppo_best_val_reward"] = manifest["val_reward"]
    payload["ppo_best_step"] = manifest["best_step"]
    payload["ppo_train_job_id"] = manifest["train_job_id"]
    payload["ppo_selection_manifest"] = str(manifest_path)
    payload["ppo_published_at"] = datetime.now(timezone.utc).isoformat()
    payload["seed_counts"] = list(SEED_COUNTS)
    payload.setdefault("checkpoint_sets", {})["ppo"] = {
        "label": (
            f"PPO (best val reward {manifest['val_reward']:.4f}, "
            f"step {manifest['best_step']})"
        ),
        "checkpoint": checkpoint_display,
        "checkpoint_id": manifest["checkpoint_id"],
        "val_reward": manifest["val_reward"],
        "step": manifest["best_step"],
    }
    return payload


def fsync_directory(path):
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_cachebuster(html_path):
    html_path = Path(html_path).resolve()
    try:
        source = html_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read visualizer HTML {html_path}: {exc}") from exc
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    pattern = re.compile(
        r"(?P<prefix>src=[\"']data\.js\?v=)[^\"']+(?P<quote>[\"'])"
    )
    updated, count = pattern.subn(
        lambda match: match.group("prefix") + version + match.group("quote"),
        source,
        count=1,
    )
    if count != 1:
        raise ValueError(f"expected exactly one data.js cachebuster script tag in {html_path}")
    return html_path, updated, version


def atomic_write_text(path, text):
    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="visualizer/data.js")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--lock", default=None)
    parser.add_argument("--html", default="visualizer/visualizer.html")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    lock_path = Path(args.lock).resolve() if args.lock else data_path.with_suffix(data_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shard_paths = exact_shard_paths(args.shard_dir)
        # Full hashes are checked once, immediately before publication.  Shards
        # therefore cannot silently point at a checkpoint modified after rollout.
        manifest = validate_selected_manifest(args.manifest, verify_hashes=True)
        with lock_path.open("a+", encoding="utf-8") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            print(f"Acquired exclusive publisher lock {lock_path}", flush=True)
            # Load only now, under the lock: any earlier GRPO publisher is retained.
            payload, prefix = load_data_js(data_path)
            validate_and_merge(payload, shard_paths, manifest, args.manifest)
            html_path, updated_html, cache_version = prepare_cachebuster(args.html)
            print(
                f"Validated {EXPECTED_SHARDS} PPO shards / {EXPECTED_WINDOWS} windows "
                f"for {manifest['checkpoint_id']}",
                flush=True,
            )
            if args.dry_run:
                print(
                    f"dry-run: latest data.js validated but not replaced; HTML "
                    f"cachebuster would become {cache_version}"
                )
                return
            atomic_dump_data_js(data_path, prefix, payload)
            # Only advertise the new asset after its os.replace has succeeded.
            atomic_write_text(html_path, updated_html)
            fsync_directory(data_path.parent)
            print(
                f"Atomically published PPO into {data_path}; bumped {html_path.name} "
                f"data.js?v={cache_version}",
                flush=True,
            )
    except ValueError as exc:
        raise SystemExit(f"refusing to publish PPO visualization: {exc}") from exc


if __name__ == "__main__":
    main()
