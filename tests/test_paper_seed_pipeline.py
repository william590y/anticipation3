from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
VIS = REPO / "visualizer"
sys.path.insert(0, str(VIS))

from compute_f1 import score_notes  # noqa: E402
from publish_paper_seed_rollouts import (  # noqa: E402
    INPUT_PROTOCOL,
    MODEL_SPECS,
    SEEDED_CONDITIONING,
    exact_shard_paths,
    semantic_input_identity,
    validate_and_merge,
    validate_published,
    variant_name,
)


def note(index: int, *, seeded: bool = False) -> dict:
    value = {"t": index * 2, "d": 2, "p": 60 + index}
    if seeded:
        value["seeded"] = True
    return value


def baseline_rollout(gt: list[dict]) -> dict:
    return {
        "source": "paper",
        "input_protocol": INPUT_PROTOCOL,
        "conditioning": "unseeded canonical official infer",
        "seed_count": 0,
        "pred_score": copy.deepcopy(gt),
        "matched_notes": len(gt),
        "padded_notes": 0,
        "span_ratio": 1.0,
        "span_ok": True,
        "baseline_certification": {
            "canonical_official_infer": True,
            "explicit_pad_mask_applied": True,
            "selected_input_rows": len(gt),
            "valid_rows": len(gt),
            "padded_rows": 0,
            "legacy_comparison": {},
        },
    }


def seeded_rollout(gt: list[dict], count: int) -> dict:
    pred = [note(index, seeded=index < count) for index in range(len(gt))]
    return {
        "source": "paper",
        "input_protocol": INPUT_PROTOCOL,
        "conditioning": SEEDED_CONDITIONING,
        "seed_count": count,
        "pred_score": pred,
        "matched_notes": len(pred),
        "padded_notes": 0,
        "span_ratio": 1.0,
        "span_ok": True,
        "seed_certification": {
            "complete_notation_rows": False,
            "common_triplet_only": True,
            "explicit_pad_mask_applied": True,
            "native_forced_token_rows_exact": True,
            "display_seed_triplets_exact_gt": True,
            "cache_to_visual_gt_exact": True,
            "native_target_representation": (
                "original uniquely aligned MusicXML token row"
            ),
            "certified_slots": list(range(count)),
            "forced_streams": ["downbeat", "duration", "offset", "pad", "pitch"],
            "zero_force_matches_official_infer": True,
            "project_performance_indices": list(range(count)),
            "paper_input_positions": list(range(count)),
            "score_midi_indices": list(range(count)),
            "project_score_midi_indices": list(range(count)),
            "musicxml_token_rows": list(range(count)),
            "input_mapping_certifications": [
                {"paper_input_order_uniquely_mapped": True} for _ in range(count)
            ],
            "mapping_certifications": [
                {
                    "cache_builder_triplet_exact": True,
                    "cache_window_gt_triplet_exact": True,
                    "raw_score_index_within_10ms_token_rounding": True,
                    "absolute_tick_onset_pitch_duration_unique": True,
                    "project_score_order_uniquely_mapped": True,
                    "paper_score_order_uniquely_mapped": True,
                    "musicxml_second_parse_row_identity_exact": True,
                    "native_xml_core_unmodified": True,
                    "native_xml_bucket_roundtrip": True,
                }
                for _ in range(count)
            ],
            "intervention_rows": [
                {
                    "native_core_token_assignment_exact": True,
                    "native_pad_assignment_exact": True,
                }
                for _ in range(count)
            ],
        },
    }


def payload() -> dict:
    order = [f"window-{index:02d}" for index in range(24)]
    examples = {}
    for index, eid in enumerate(order):
        gt = [note(slot) for slot in range(8)]
        examples[eid] = {
            "piece": f"Composer/Piece/{index}.mid",
            "perf_notes": [{"t": slot, "d": 2, "p": 60 + slot} for slot in range(8)],
            "raw_notes": [
                {"t": slot, "d": 2, "p": 60 + slot, "j": slot, "r": slot}
                for slot in range(8)
            ],
            "gt_score": gt,
            "rollouts_paper1": {"filtered": {"historical": True}},
            "rollouts_paper2": {"filtered": {"historical": True}},
        }
    return {"example_order": order, "examples": examples}


def checkpoint_identities(tmp_path: Path) -> dict[str, dict]:
    result = {}
    for kind in MODEL_SPECS:
        path = (tmp_path / f"{kind}.ckpt").resolve()
        path.write_bytes(kind.encode("ascii"))
        result[kind] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": f"digest-{kind}",
        }
    return result


def write_shards(tmp_path: Path, data: dict, checkpoints: dict[str, dict]) -> list[Path]:
    order = data["example_order"]
    source_identity = semantic_input_identity(data)
    for shard_index in range(4):
        ids = order[shard_index::4]
        shard = {
            "format": 3,
            "shard_index": shard_index,
            "num_shards": 4,
            "diagnostic_only": False,
            "seed_counts": [1, 2, 3, 4, 5],
            "input_protocol": INPUT_PROTOCOL,
            "conditioning": SEEDED_CONDITIONING,
            "source_data_identity": source_identity,
            "example_order": ids,
            "models": {
                kind: {
                    "identity": spec["identity"],
                    "checkpoint": checkpoints[kind],
                    "chunk": spec["chunk"],
                    "overlap": spec["overlap"],
                }
                for kind, spec in MODEL_SPECS.items()
            },
            "examples": {},
        }
        for eid in ids:
            gt = data["examples"][eid]["gt_score"]
            block = {"filtered": baseline_rollout(gt)}
            for count in range(1, 6):
                block[variant_name(count)] = seeded_rollout(gt, count)
            shard["examples"][eid] = {
                "rollouts_paper1": copy.deepcopy(block),
                "rollouts_paper2": copy.deepcopy(block),
            }
        (tmp_path / f"shard_{shard_index:02d}.json").write_text(
            json.dumps(shard), encoding="utf-8"
        )
    return exact_shard_paths(tmp_path)


def test_merge_replaces_corrected_baseline_and_recomputes_every_f1(tmp_path):
    data = payload()
    checkpoints = checkpoint_identities(tmp_path)
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    paths = write_shards(shard_dir, data, checkpoints)

    validate_and_merge(data, paths, checkpoints)
    validate_published(data, checkpoints)

    for example in data["examples"].values():
        for group in ("rollouts_paper1", "rollouts_paper2"):
            block = example[group]
            assert set(block) == {
                "filtered",
                "filtered_seeded",
                "filtered_seed2",
                "filtered_seed3",
                "filtered_seed4",
                "filtered_seed5",
            }
            assert "historical" not in block["filtered"]
            for rollout in block.values():
                assert rollout["f1"] == score_notes(
                    rollout["pred_score"], example["gt_score"]
                )


def test_semantic_identity_ignores_rollouts_but_covers_piece_perf_raw_and_gt():
    data = payload()
    original = semantic_input_identity(data)
    data["examples"][data["example_order"][0]]["rollouts_paper1"] = {"changed": True}
    assert semantic_input_identity(data) == original
    for field in ("piece", "perf_notes", "raw_notes", "gt_score"):
        changed = copy.deepcopy(data)
        changed["examples"][changed["example_order"][0]][field] = {"changed": field}
        assert semantic_input_identity(changed) != original


def test_exact_shard_directory_rejects_extras(tmp_path):
    data = payload()
    checkpoints = checkpoint_identities(tmp_path)
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    write_shards(shard_dir, data, checkpoints)
    (shard_dir / "stale.json").write_text("{}", encoding="utf-8")
    try:
        exact_shard_paths(shard_dir)
    except ValueError as exc:
        assert "expected exactly" in str(exc)
    else:
        raise AssertionError("publisher accepted an extra shard-directory file")


def test_visualizer_paper_seed_lookup_is_exact_and_marks_seeded_notes():
    html = (VIS / "visualizer.html").read_text(encoding="utf-8")
    assert "function paperSeededRollout(block)" in html
    assert "return n ? seededVariant(block, 'filtered', n) : (block.filtered || null);" in html
    assert "block[variant] || block.filtered" not in html
    assert "availableSeedCounts(ex?.rollouts_paper1, 'filtered')" in html
    assert "availableSeedCounts(ex?.rollouts_paper2, 'filtered')" in html
    assert "if (n.seeded) setBaseStroke(e, css('--sel'), 1.6, '2 2');" in html
    assert "aligned GT note seed" in html
    assert "original paper-native timing/quantization" in html
    assert "fixed unfiltered input" in html
