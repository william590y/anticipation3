"""Focused regression tests for RL rollout visualization plumbing."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

VISUALIZER_DIR = Path(__file__).resolve().parent
REPO_ROOT = VISUALIZER_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VISUALIZER_DIR))

from atomic_json import atomic_dump_data_js  # noqa: E402
from compute_sequence_ppl import notes_from_pred  # noqa: E402
from merge_grpo_rollouts import merge_validated, validate_shards  # noqa: E402
from precompute_visualizer import raw_seed_prefix  # noqa: E402


def _rollout():
    return {"pred_score": [], "branches": {}}


def _full_rollouts(*, legacy_seed1=True):
    block = {"filtered": _rollout(), "raw": _rollout()}
    for stream in ("filtered", "raw"):
        block[f"{stream}_seeded"] = _rollout()
        if legacy_seed1:
            block[f"{stream}_seed1"] = copy.deepcopy(block[f"{stream}_seeded"])
        for count in range(2, 6):
            block[f"{stream}_seed{count}"] = _rollout()
    return block


def _base_seed_patch():
    return {
        f"{stream}_seed{count}": _rollout()
        for stream in ("filtered", "raw")
        for count in range(2, 6)
    }


class SeedAlignmentTest(unittest.TestCase):
    def test_raw_seed_prefix_uses_j_and_leaves_unmatched_slots_unseeded(self):
        gt = [{"p": 60}, {"p": 61}, {"p": 62}]
        raw = [
            {"j": None},
            {"j": 1},
            {"j": None},
            {"j": 0},
            {"j": 2},
        ]
        self.assertEqual(raw_seed_prefix(raw, gt, 2), [None, gt[1], None, gt[0]])
        self.assertEqual(raw_seed_prefix(raw, gt, 3), [None, gt[1], None, gt[0], gt[2]])
        self.assertIsNone(raw_seed_prefix(raw[:3], gt, 2))

    def test_notes_from_pred_preserves_one_output_per_input_slot(self):
        pred = [{"p": 60}, None, {"p": None}, {"p": 61}]
        self.assertEqual(notes_from_pred(pred), [pred[0], None, None, pred[3]])
        self.assertEqual(len(notes_from_pred(pred)), len(pred))


class GrpoMergeTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.order = [f"w{i}" for i in range(8)]
        self.payload = {
            "example_order": list(self.order),
            "examples": {
                eid: {
                    "gt_score": [{"p": 60 + i} for i in range(5)],
                    "raw_notes": [{"j": i} for i in range(5)],
                    "rollouts": {},
                }
                for eid in self.order
            },
        }

    def tearDown(self):
        self.tempdir.cleanup()

    def _legacy_shards(self):
        paths = []
        for index in range(4):
            shard = {
                "grpo_checkpoint": "run_grpo/checkpoint-best",
                "seed_counts": [1, 2, 3, 4, 5],
                "examples": {
                    eid: {
                        "rollouts_grpo": _full_rollouts(),
                        "rollouts_seed_patch": _base_seed_patch(),
                    }
                    for eid in self.order[index::4]
                },
            }
            path = self.root / f"shard_{index:02d}.json"
            path.write_text(json.dumps(shard), encoding="utf-8")
            paths.append(path)
        return paths

    def test_accepts_complete_legacy_four_shard_format_and_deduplicates_seed1(self):
        validated = validate_shards(self.payload, self._legacy_shards())
        added, patched = merge_validated(self.payload, validated)
        self.assertEqual((added, patched), (len(self.order), len(self.order)))
        self.assertEqual(self.payload["grpo_checkpoint"], "run_grpo/checkpoint-best")
        for ex in self.payload["examples"].values():
            self.assertIn("rollouts_grpo", ex)
            self.assertNotIn("filtered_seed1", ex["rollouts_grpo"])
            self.assertNotIn("raw_seed1", ex["rollouts_grpo"])
            self.assertIn("filtered_seeded", ex["rollouts_grpo"])

    def test_rejects_incomplete_coverage(self):
        paths = self._legacy_shards()
        shard = json.loads(paths[0].read_text(encoding="utf-8"))
        del shard["examples"][self.order[0]]
        paths[0].write_text(json.dumps(shard), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "do not cover every data.js example"):
            validate_shards(self.payload, paths)

    def test_data_js_writer_replaces_complete_assignment(self):
        path = self.root / "data.js"
        path.write_text("old", encoding="utf-8")
        atomic_dump_data_js(path, "window.VISUALIZER_DATA = ", {"ok": True})
        self.assertEqual(path.read_text(encoding="utf-8"),
                         'window.VISUALIZER_DATA = {"ok": true};\n')
        self.assertEqual(list(self.root.glob(".data.js.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
