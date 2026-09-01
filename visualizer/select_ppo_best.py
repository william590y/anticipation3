#!/usr/bin/env python
"""Validate PPO's exact best-validation checkpoint and publish its manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from atomic_json import atomic_dump_json


MANIFEST_SCHEMA = "anticipation3.ppo-best-val.v1"


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} {path} is not a JSON object")
    return value


def _validate_safetensors_header(path):
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors file: {path}")
        header_length = int.from_bytes(raw_length, "little")
        if header_length <= 0 or header_length > min(size - 8, 64 * 1024 * 1024):
            raise ValueError(f"invalid safetensors header length in {path}")
        try:
            header = json.loads(stream.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid safetensors header in {path}: {exc}") from exc
    tensors = [key for key in header if key != "__metadata__"] if isinstance(header, dict) else []
    if not tensors:
        raise ValueError(f"safetensors file has no tensors: {path}")


def _weight_paths(checkpoint):
    indexes = sorted(checkpoint.glob("*.index.json"))
    if indexes:
        names = set()
        for index in indexes:
            parsed = _load_json(index, "weight index")
            weight_map = parsed.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"weight index has no weight_map: {index}")
            names.update(weight_map.values())
        paths = [checkpoint / name for name in sorted(names)]
    else:
        paths = sorted(checkpoint.glob("*.safetensors"))
        paths += sorted(checkpoint.glob("pytorch_model*.bin"))
    if not paths:
        raise ValueError(f"no model weights found in {checkpoint}")
    checkpoint_root = checkpoint.resolve()
    for path in paths:
        resolved = path.resolve()
        try:
            resolved.relative_to(checkpoint_root)
        except ValueError as exc:
            raise ValueError(f"weight path escapes checkpoint: {path}") from exc
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"missing or empty model weight: {path}")
        if path.suffix == ".safetensors":
            _validate_safetensors_header(path)
    return paths


def checkpoint_id(config_entry, model_entries):
    digest = hashlib.sha256()
    for entry in [config_entry, *sorted(model_entries, key=lambda item: item["name"])]:
        digest.update(entry["name"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry["sha256"].encode("ascii"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def inventory_checkpoint(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    config_path = checkpoint / "config.json"
    config = _load_json(config_path, "model config")
    if not config or not isinstance(config.get("model_type"), str):
        raise ValueError(f"invalid Hugging Face model config: {config_path}")
    config_entry = {
        "name": "config.json",
        "size": config_path.stat().st_size,
        "sha256": sha256_file(config_path),
    }
    model_entries = []
    for path in _weight_paths(checkpoint):
        model_entries.append(
            {
                "name": path.relative_to(checkpoint).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return config_entry, model_entries, checkpoint_id(config_entry, model_entries)


def _relative_display(path, repo_root):
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def create_manifest(run_dir, train_job_id, hyperparameters=None):
    run_dir = Path(run_dir).resolve()
    metadata_path = run_dir / "best_val_reward.json"
    metadata = _load_json(metadata_path, "best-validation metadata")

    step = metadata.get("step")
    reward = metadata.get("val_reward")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError(f"invalid best-validation step: {step!r}")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise ValueError(f"invalid best-validation reward: {reward!r}")

    expected_checkpoint = (run_dir / "best-val-reward").resolve()
    declared = metadata.get("checkpoint")
    if not isinstance(declared, str) or not declared:
        raise ValueError("best_val_reward.json has no checkpoint")
    declared_checkpoint = Path(declared)
    if not declared_checkpoint.is_absolute():
        declared_checkpoint = Path.cwd() / declared_checkpoint
    if declared_checkpoint.resolve() != expected_checkpoint:
        raise ValueError(
            "best_val_reward.json checkpoint does not name "
            f"{expected_checkpoint}: {declared!r}"
        )
    if not expected_checkpoint.is_dir():
        raise ValueError(f"best checkpoint directory is missing: {expected_checkpoint}")

    config_entry, model_entries, identity = inventory_checkpoint(expected_checkpoint)
    repo_root = Path(__file__).resolve().parent.parent
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "train_job_id": str(train_job_id),
        "best_step": step,
        "val_reward": float(reward),
        "checkpoint": str(expected_checkpoint),
        "checkpoint_display": _relative_display(expected_checkpoint, repo_root),
        "checkpoint_id": identity,
        "config_file": config_entry,
        "model_files": model_entries,
        "best_metadata": {
            "path": str(metadata_path),
            "size": metadata_path.stat().st_size,
            "sha256": sha256_file(metadata_path),
        },
    }
    if hyperparameters is not None:
        manifest["hyperparameters"] = hyperparameters
    return manifest


def validate_selected_manifest(path, *, verify_hashes=False):
    manifest_path = Path(path).resolve()
    manifest = _load_json(manifest_path, "PPO selection manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unexpected PPO manifest schema in {manifest_path}")

    checkpoint = Path(manifest.get("checkpoint", "")).resolve()
    run_dir = Path(manifest.get("run_dir", "")).resolve()
    if checkpoint != (run_dir / "best-val-reward").resolve() or not checkpoint.is_dir():
        raise ValueError("manifest checkpoint is not run_dir/best-val-reward")
    step = manifest.get("best_step")
    reward = manifest.get("val_reward")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("manifest has an invalid best_step")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise ValueError("manifest has an invalid val_reward")
    if not isinstance(manifest.get("train_job_id"), str) or not manifest["train_job_id"]:
        raise ValueError("manifest has no train_job_id")
    hyperparameters = manifest.get("hyperparameters")
    required_hparams = {"learning_rate", "ppo_epochs", "gamma", "gae_lambda", "target_kl"}
    if not isinstance(hyperparameters, dict) or set(hyperparameters) != required_hparams:
        raise ValueError("manifest has no exact production hyperparameters")
    for name, value in hyperparameters.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"manifest hyperparameter {name} is invalid")
    if not isinstance(hyperparameters["ppo_epochs"], int) or hyperparameters["ppo_epochs"] <= 0:
        raise ValueError("manifest ppo_epochs must be a positive integer")
    if hyperparameters["learning_rate"] <= 0 or hyperparameters["target_kl"] <= 0:
        raise ValueError("manifest learning_rate and target_kl must be positive")
    for name in ("gamma", "gae_lambda"):
        if not 0 <= hyperparameters[name] <= 1:
            raise ValueError(f"manifest {name} must lie in [0, 1]")

    config_entry = manifest.get("config_file")
    model_entries = manifest.get("model_files")
    if not isinstance(config_entry, dict) or not isinstance(model_entries, list) or not model_entries:
        raise ValueError("manifest has no checkpoint file inventory")
    entries = [config_entry, *model_entries]
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError("invalid checkpoint inventory entry")
        file_path = checkpoint / entry["name"]
        if not file_path.is_file() or file_path.stat().st_size != entry.get("size"):
            raise ValueError(f"checkpoint file changed or vanished: {file_path}")
        if verify_hashes and sha256_file(file_path) != entry.get("sha256"):
            raise ValueError(f"checkpoint file hash changed: {file_path}")
    if checkpoint_id(config_entry, model_entries) != manifest.get("checkpoint_id"):
        raise ValueError("manifest checkpoint_id does not match its file inventory")

    metadata = manifest.get("best_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("manifest has no best_metadata identity")
    metadata_path = Path(metadata.get("path", ""))
    if metadata_path.resolve() != (run_dir / "best_val_reward.json").resolve():
        raise ValueError("manifest best_metadata path is inconsistent")
    if not metadata_path.is_file() or metadata_path.stat().st_size != metadata.get("size"):
        raise ValueError("best_val_reward.json changed or vanished")
    if verify_hashes and sha256_file(metadata_path) != metadata.get("sha256"):
        raise ValueError("best_val_reward.json hash changed")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-job-id", default=os.environ.get("PPO_TRAIN_JOB_ID"))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("PPO_LR", "3e-7")))
    parser.add_argument("--ppo-epochs", type=int, default=int(os.environ.get("PPO_EPOCHS", "2")))
    parser.add_argument("--gamma", type=float, default=float(os.environ.get("PPO_GAMMA", "1.0")))
    parser.add_argument("--gae-lambda", type=float, default=float(os.environ.get("PPO_GAE_LAMBDA", "0.95")))
    parser.add_argument("--target-kl", type=float, default=float(os.environ.get("PPO_TARGET_KL", "0.02")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.train_job_id:
        raise SystemExit("--train-job-id or PPO_TRAIN_JOB_ID is required")

    try:
        manifest = create_manifest(
            args.run_dir,
            args.train_job_id,
            {
                "learning_rate": args.learning_rate,
                "ppo_epochs": args.ppo_epochs,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "target_kl": args.target_kl,
            },
        )
    except ValueError as exc:
        raise SystemExit(f"refusing to select PPO checkpoint: {exc}") from exc
    print(
        f"Selected PPO step {manifest['best_step']} with val REWARD "
        f"{manifest['val_reward']:.6f}: {manifest['checkpoint']}"
    )
    print(f"Checkpoint identity: {manifest['checkpoint_id']}")
    if args.dry_run:
        print("dry-run: manifest not written")
        return
    atomic_dump_json(args.output, manifest)
    print(f"Atomically wrote {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
