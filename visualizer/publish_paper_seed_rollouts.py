#!/usr/bin/env python
"""Validate exactly four Paper 1/2 shards and atomically publish ``data.js``."""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent

from compute_f1 import score_notes  # noqa: E402
from compute_sequence_ppl import load_payload  # noqa: E402


NUM_SHARDS = 4
EXPECTED_WINDOWS = 24
SEED_COUNTS = (1, 2, 3, 4, 5)
PAPER_GROUPS = ("rollouts_paper1", "rollouts_paper2")
INPUT_PROTOCOL = "official full-piece chunk replay on original unfiltered performance MIDI"
SEEDED_CONDITIONING = (
    "autoregressive aligned GT note in paper-native timing/quantization"
)
MODEL_SPECS = {
    "paper1": {
        "identity": "Zeng+ joint-APT-EPR (ICLR 2026 released weights)",
        "checkpoint": "external/weights/joint_apt_epr.ckpt",
        "chunk": 256,
        "overlap": 64,
    },
    "paper2": {
        "identity": "Beyer & Dai MIDI2ScoreTransformer (ISMIR 2024 released weights)",
        "checkpoint": "external/weights/MIDI2ScoreTF.ckpt",
        "chunk": 512,
        "overlap": 64,
    },
}


def fail(message: str) -> None:
    raise ValueError(message)


def variant_name(count: int) -> str:
    return "filtered_seeded" if count == 1 else f"filtered_seed{count}"


def ordered_examples(payload: dict) -> list[str]:
    examples = payload.get("examples")
    if not isinstance(examples, dict):
        fail("data.js examples is not an object")
    order = list(payload.get("example_order") or examples)
    if len(order) != EXPECTED_WINDOWS:
        fail(f"expected exactly {EXPECTED_WINDOWS} windows, found {len(order)}")
    if len(order) != len(set(order)) or set(order) != set(examples):
        fail("example_order is not a one-to-one ordering of examples")
    return order


def semantic_input_identity(payload: dict) -> str:
    """Hash only paper-inference inputs, so independent publishers can coexist."""
    order = ordered_examples(payload)
    examples = payload["examples"]
    semantic = {
        "example_order": order,
        "examples": {
            eid: {
                "piece": examples[eid].get("piece"),
                "perf_notes": examples[eid].get("perf_notes"),
                "raw_notes": examples[eid].get("raw_notes"),
                "gt_score": examples[eid].get("gt_score"),
            }
            for eid in order
        },
    }
    encoded = json.dumps(
        semantic,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(path: Path) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "sha256": sha256_file(path)}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read complete shard {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{path}: shard root is not an object")
    return value


def exact_shard_paths(shard_dir: Path) -> list[Path]:
    shard_dir = Path(shard_dir).resolve()
    expected = [shard_dir / f"shard_{index:02d}.json" for index in range(NUM_SHARDS)]
    observed = sorted(shard_dir.iterdir()) if shard_dir.is_dir() else []
    if observed != expected:
        fail(
            f"expected exactly {[path.name for path in expected]} in {shard_dir}; "
            f"found {[path.name for path in observed]}"
        )
    return expected


def validate_note(note, path: str, *, seeded: bool | None = None) -> None:
    if not isinstance(note, dict):
        fail(f"{path}: note is not an object")
    for key in ("t", "d", "p"):
        value = note.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            fail(f"{path}: {key} is not an integer")
    if note["t"] < 0 or note["d"] < 1 or not 0 <= note["p"] <= 127:
        fail(f"{path}: invalid note triplet {note}")
    if seeded is not None and bool(note.get("seeded")) is not seeded:
        fail(f"{path}: expected seeded={seeded}")


def _zero_force_certified(rollout: dict) -> bool:
    baseline = rollout.get("baseline_certification")
    if (
        isinstance(baseline, dict)
        and baseline.get("canonical_official_infer") is True
        and baseline.get("explicit_pad_mask_applied") is True
    ):
        return True
    for name in ("inference_certification", "seed_certification"):
        cert = rollout.get(name)
        if isinstance(cert, dict) and cert.get("zero_force_matches_official_infer") is True:
            return True
    return rollout.get("zero_force_matches_official_infer") is True


def validate_rollout(rollout, gt: list, count: int, path: str) -> None:
    if not isinstance(rollout, dict):
        fail(f"{path}: rollout is not an object")
    if rollout.get("seed_count") != count:
        fail(f"{path}: seed_count is not {count}")
    if rollout.get("input_protocol") != INPUT_PROTOCOL:
        fail(f"{path}: external-paper input protocol changed")
    expected_conditioning = (
        SEEDED_CONDITIONING if count else "unseeded canonical official infer"
    )
    if rollout.get("conditioning") != expected_conditioning:
        fail(f"{path}: conditioning metadata changed")
    if not _zero_force_certified(rollout):
        fail(f"{path}: official-infer zero-force parity is not certified")

    pred = rollout.get("pred_score")
    if not isinstance(pred, list) or not pred or len(pred) < count:
        fail(f"{path}: prediction is empty or shorter than its seed prefix")
    for index, note in enumerate(pred):
        validate_note(note, f"{path}/pred_score/{index}")
    if count == 0:
        if any(bool(note.get("seeded")) for note in pred):
            fail(f"{path}: corrected unseeded baseline contains seeded notes")
        cert = rollout.get("baseline_certification")
        if not isinstance(cert, dict):
            fail(f"{path}: corrected baseline certification is missing")
        for field in ("selected_input_rows", "valid_rows", "padded_rows"):
            value = cert.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                fail(f"{path}: invalid baseline certification field {field}")
        if cert["valid_rows"] + cert["padded_rows"] != cert["selected_input_rows"]:
            fail(f"{path}: baseline pad accounting is inconsistent")
        if rollout.get("padded_notes") != cert["padded_rows"]:
            fail(f"{path}: rollout and certification padded-note counts differ")
    else:
        cert = rollout.get("seed_certification")
        if not isinstance(cert, dict):
            fail(f"{path}: seed certification is missing")
        if cert.get("common_triplet_only") is not True:
            fail(f"{path}: intervention is not certified as common-triplet-only")
        if cert.get("complete_notation_rows") is not False:
            fail(f"{path}: intervention incorrectly claims complete notation rows")
        if cert.get("explicit_pad_mask_applied") is not True:
            fail(f"{path}: explicit pad masking is not certified")
        for field in (
            "native_forced_token_rows_exact",
            "display_seed_triplets_exact_gt",
            "cache_to_visual_gt_exact",
        ):
            if cert.get(field) is not True:
                fail(f"{path}: missing seed certification {field}")
        if cert.get("native_target_representation") != (
            "original uniquely aligned MusicXML token row"
        ):
            fail(f"{path}: native GT target representation changed")
        if cert.get("certified_slots") != list(range(count)):
            fail(f"{path}: certified seed slots are incomplete")
        forced = set(cert.get("forced_streams") or ())
        if forced != {"offset", "downbeat", "duration", "pitch", "pad"}:
            fail(f"{path}: unexpected forced streams {sorted(forced)}")
        for field in (
            "project_performance_indices",
            "paper_input_positions",
            "score_midi_indices",
            "project_score_midi_indices",
            "musicxml_token_rows",
        ):
            values = cert.get(field)
            if (
                not isinstance(values, list)
                or len(values) != count
                or len(set(values)) != count
            ):
                fail(f"{path}: {field} is not a distinct {count}-row mapping")
        mappings = cert.get("mapping_certifications")
        required_mapping_fields = {
            "cache_builder_triplet_exact",
            "cache_window_gt_triplet_exact",
            "raw_score_index_within_10ms_token_rounding",
            "absolute_tick_onset_pitch_duration_unique",
            "project_score_order_uniquely_mapped",
            "paper_score_order_uniquely_mapped",
            "musicxml_second_parse_row_identity_exact",
            "native_xml_core_unmodified",
            "native_xml_bucket_roundtrip",
        }
        if not isinstance(mappings, list) or len(mappings) != count:
            fail(f"{path}: native GT mapping certifications are incomplete")
        for mapping in mappings:
            if not isinstance(mapping, dict) or any(
                mapping.get(field) is not True for field in required_mapping_fields
            ):
                fail(f"{path}: native GT mapping certification failed")
        input_mappings = cert.get("input_mapping_certifications")
        if (
            not isinstance(input_mappings, list)
            or len(input_mappings) != count
            or any(
                not isinstance(mapping, dict)
                or mapping.get("paper_input_order_uniquely_mapped") is not True
                for mapping in input_mappings
            )
        ):
            fail(f"{path}: paper input mapping certification failed")
        intervention_rows = cert.get("intervention_rows")
        if (
            not isinstance(intervention_rows, list)
            or len(intervention_rows) != count
            or any(
                not isinstance(row, dict)
                or row.get("native_core_token_assignment_exact") is not True
                or row.get("native_pad_assignment_exact") is not True
                for row in intervention_rows
            )
        ):
            fail(f"{path}: exact native token intervention is not certified")
        for index in range(count):
            validate_note(pred[index], f"{path}/pred_score/{index}", seeded=True)
            expected = gt[index] if index < len(gt) else None
            if not isinstance(expected, dict):
                fail(f"{path}: ground-truth slot {index} is unavailable")
            if any(pred[index][key] != expected[key] for key in ("t", "d", "p")):
                fail(f"{path}: seeded slot {index} is not the exact visualizer GT triplet")

    ratio = rollout.get("span_ratio")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        fail(f"{path}: span_ratio is not numeric")
    if not math.isfinite(float(ratio)) or ratio <= 0:
        fail(f"{path}: invalid span_ratio")
    if not isinstance(rollout.get("span_ok"), bool):
        fail(f"{path}: span_ok is not boolean")


def validate_model_metadata(shard: dict, path: Path, checkpoints: dict[str, dict]) -> None:
    models = shard.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_SPECS):
        fail(f"{path}: model metadata is incomplete")
    for kind, spec in MODEL_SPECS.items():
        meta = models[kind]
        if not isinstance(meta, dict) or meta.get("identity") != spec["identity"]:
            fail(f"{path}: {kind} identity differs")
        if meta.get("chunk") != spec["chunk"] or meta.get("overlap") != spec["overlap"]:
            fail(f"{path}: {kind} official chunk/overlap metadata differs")
        observed = meta.get("checkpoint")
        expected = checkpoints[kind]
        if not isinstance(observed, dict):
            fail(f"{path}: {kind} checkpoint identity is not an object")
        try:
            same_path = Path(observed.get("path", "")).resolve() == Path(expected["path"])
        except (TypeError, OSError):
            same_path = False
        if (
            not same_path
            or observed.get("size") != expected["size"]
            or observed.get("sha256") != expected["sha256"]
        ):
            fail(f"{path}: {kind} checkpoint path/size/SHA-256 differs")


def validate_and_merge(
    payload: dict,
    shard_paths: list[Path],
    checkpoints: dict[str, dict],
) -> dict:
    order = ordered_examples(payload)
    source_identity = semantic_input_identity(payload)
    expected_variants = {"filtered", *(variant_name(count) for count in SEED_COUNTS)}
    seen: set[str] = set()

    for index, path in enumerate(shard_paths):
        shard = read_json(path)
        if shard.get("format") != 3:
            fail(f"{path}: unsupported paper-seed shard format")
        if shard.get("diagnostic_only") is not False:
            fail(f"{path}: diagnostic-only or unspecified shard cannot be published")
        if shard.get("shard_index") != index or shard.get("num_shards") != NUM_SHARDS:
            fail(f"{path}: invalid shard index/count metadata")
        if shard.get("seed_counts") != list(SEED_COUNTS):
            fail(f"{path}: seed counts are not exactly 1..5")
        if shard.get("input_protocol") != INPUT_PROTOCOL:
            fail(f"{path}: wrong paper input protocol")
        if shard.get("conditioning") != SEEDED_CONDITIONING:
            fail(f"{path}: wrong seeded conditioning mode")
        if shard.get("source_data_identity") != source_identity:
            fail(f"{path}: semantic visualizer inputs changed since precompute")
        validate_model_metadata(shard, path, checkpoints)

        expected_ids = order[index::NUM_SHARDS]
        if shard.get("example_order") != expected_ids:
            fail(f"{path}: example_order is not the exact stride partition")
        patches = shard.get("examples")
        if not isinstance(patches, dict) or set(patches) != set(expected_ids):
            fail(f"{path}: example coverage is incomplete")
        if seen & set(patches):
            fail(f"{path}: duplicate example coverage")
        seen.update(patches)

        for eid in expected_ids:
            patch = patches[eid]
            if not isinstance(patch, dict) or set(patch) != set(PAPER_GROUPS):
                fail(f"{path}:{eid}: paper groups are incomplete or contain extras")
            gt = payload["examples"][eid].get("gt_score") or []
            for group in PAPER_GROUPS:
                block = patch[group]
                if not isinstance(block, dict) or set(block) != expected_variants:
                    fail(f"{path}:{eid}/{group}: expected corrected filtered plus seed 1..5")
                validate_rollout(block["filtered"], gt, 0, f"{path}:{eid}/{group}/filtered")
                block["filtered"]["f1"] = score_notes(block["filtered"]["pred_score"], gt)
                for count in SEED_COUNTS:
                    name = variant_name(count)
                    validate_rollout(block[name], gt, count, f"{path}:{eid}/{group}/{name}")
                    block[name]["f1"] = score_notes(block[name]["pred_score"], gt)
                # Replace the historical block wholesale: notably, this installs
                # the corrected pad-filtered zero-force baseline as `filtered`.
                payload["examples"][eid][group] = block

    if seen != set(order) or len(seen) != EXPECTED_WINDOWS:
        fail("four shards do not cover exactly all 24 windows")

    now = datetime.now(timezone.utc).isoformat()
    payload["paper_seed_counts"] = list(SEED_COUNTS)
    payload["paper_seed_pipeline"] = {
        "published_at_utc": now,
        "shards": NUM_SHARDS,
        "models": list(MODEL_SPECS),
        "checkpoints": checkpoints,
        "source_data_identity": source_identity,
        "seed_counts": list(SEED_COUNTS),
        "input_protocol": INPUT_PROTOCOL,
        "conditioning": SEEDED_CONDITIONING,
        "corrected_zero_force_filtered": True,
    }
    return payload


def validate_published(payload: dict, checkpoints: dict[str, dict]) -> None:
    order = ordered_examples(payload)
    if payload.get("paper_seed_counts") != list(SEED_COUNTS):
        fail("paper_seed_counts are not exactly 1..5")
    metadata = payload.get("paper_seed_pipeline")
    if not isinstance(metadata, dict):
        fail("paper_seed_pipeline metadata is missing")
    if metadata.get("checkpoints") != checkpoints:
        fail("published checkpoint path/size/SHA-256 identities differ")
    if metadata.get("source_data_identity") != semantic_input_identity(payload):
        fail("published semantic source identity differs")
    if metadata.get("corrected_zero_force_filtered") is not True:
        fail("corrected zero-force paper baselines are not certified")
    expected_variants = {"filtered", *(variant_name(count) for count in SEED_COUNTS)}
    for eid in order:
        ex = payload["examples"][eid]
        gt = ex.get("gt_score") or []
        for group in PAPER_GROUPS:
            block = ex.get(group)
            if not isinstance(block, dict) or set(block) != expected_variants:
                fail(f"published:{eid}/{group}: rollout variants are incomplete")
            validate_rollout(block["filtered"], gt, 0, f"published:{eid}/{group}/filtered")
            for count in SEED_COUNTS:
                name = variant_name(count)
                validate_rollout(block[name], gt, count, f"published:{eid}/{group}/{name}")
            for name in expected_variants:
                expected_f1 = score_notes(block[name]["pred_score"], gt)
                if block[name].get("f1") != expected_f1:
                    fail(f"published:{eid}/{group}/{name}: note F1 was not recomputed")


def stage_text(destination: Path, text: str, tag: str) -> Path:
    destination = Path(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.{tag}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists():
            os.chmod(temporary, destination.stat().st_mode & 0o777)
        return temporary
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def stage_data_js(destination: Path, prefix: str, payload: dict) -> Path:
    destination = Path(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.paper-publish.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(prefix)
            json.dump(payload, stream, allow_nan=False)
            stream.write(";\n")
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists():
            os.chmod(temporary, destination.stat().st_mode & 0o777)
        return temporary
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def prepare_html_cachebuster(html_path: Path) -> tuple[str, str]:
    html_path = Path(html_path)
    text = html_path.read_text(encoding="utf-8")
    if "paperSeededRollout" not in text:
        fail(f"{html_path}: paper seeded-rollout UI hook is missing")
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    pattern = re.compile(r'(?P<prefix>src=["\']data\.js\?v=)[^"\']+(?P<quote>["\'])')
    updated, count = pattern.subn(
        lambda match: match.group("prefix") + version + match.group("quote"),
        text,
        count=1,
    )
    if count != 1:
        fail(f"{html_path}: expected exactly one data.js cachebuster, found {count}")
    return updated, version


def replace_and_fsync(temporary: Path, destination: Path) -> None:
    os.replace(temporary, destination)
    descriptor = os.open(Path(destination).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="visualizer/data.js")
    parser.add_argument("--shards", default="visualizer/paper_seed_shards")
    parser.add_argument("--html", default="visualizer/visualizer.html")
    parser.add_argument("--lock", default=None)
    parser.add_argument("--paper1-ckpt", default=MODEL_SPECS["paper1"]["checkpoint"])
    parser.add_argument("--paper2-ckpt", default=MODEL_SPECS["paper2"]["checkpoint"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    destination = Path(args.data).resolve()
    html_path = Path(args.html).resolve()
    shard_paths = exact_shard_paths(args.shards)
    checkpoint_paths = {
        "paper1": Path(args.paper1_ckpt).resolve(),
        "paper2": Path(args.paper2_ckpt).resolve(),
    }
    for kind, checkpoint in checkpoint_paths.items():
        if not checkpoint.is_file():
            fail(f"missing {kind} checkpoint: {checkpoint}")
    # Hash each large released checkpoint once, immediately before validation.
    checkpoints = {kind: checkpoint_identity(path) for kind, path in checkpoint_paths.items()}

    lock_path = Path(args.lock).resolve() if args.lock else destination.with_name(destination.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    staged_data = None
    staged_html = None
    try:
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            print(f"Acquired exclusive publish lock {lock_path}", flush=True)
            # Load only under the lock, preserving any earlier GRPO/PPO/paper publish.
            payload, prefix = load_payload(destination)
            validate_and_merge(payload, shard_paths, checkpoints)
            validate_published(payload, checkpoints)
            updated_html, cache_version = prepare_html_cachebuster(html_path)
            print("Validated exactly four Paper 1/2 shards covering 24 windows", flush=True)
            if args.dry_run:
                print(
                    f"Dry run passed; latest data.js was not replaced; cachebuster would be {cache_version}",
                    flush=True,
                )
                return

            # Prepare and fsync both complete replacements before either public path
            # changes. Validate a fresh parse of the staged data before publication.
            staged_data = stage_data_js(destination, prefix, payload)
            staged_html = stage_text(html_path, updated_html, "paper-cache")
            del payload
            gc.collect()
            staged_payload, staged_prefix = load_payload(staged_data)
            if staged_prefix != prefix:
                fail("staged data.js assignment prefix changed")
            validate_published(staged_payload, checkpoints)
            del staged_payload
            gc.collect()

            # Publish data first; only then advertise its new cache key in HTML.
            replace_and_fsync(staged_data, destination)
            staged_data = None
            replace_and_fsync(staged_html, html_path)
            staged_html = None
            print(
                f"Atomically published corrected/unseeded and seed1..5 Paper rollouts; "
                f"data.js?v={cache_version}",
                flush=True,
            )
    finally:
        for temporary in (staged_data, staged_html):
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"refusing to publish paper seeds: {exc}") from exc
