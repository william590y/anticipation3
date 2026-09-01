"""Tests for 40k masked visualizer keys, merge protection, and batched AR identity."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

VISUALIZER_DIR = Path(__file__).resolve().parent
REPO_ROOT = VISUALIZER_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VISUALIZER_DIR))

from anticipation.packed_sequence import (  # noqa: E402
    ALTERNATING_START,
    PREFIX_CONTROLS,
    dummy_rest_triplet,
)
from anticipation.vocab import (  # noqa: E402
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    TIME_OFFSET,
    VOCAB_SIZE,
)
from atomic_json import atomic_dump_data_js  # noqa: E402
from compute_f1 import iter_rollout_groups  # noqa: E402
from fast_rollout import batched_rollout_with_candidates  # noqa: E402
from merge_masked_40k_rollouts import PROTECTED_GROUPS  # noqa: E402
from precompute_masked_40k_rollouts import (  # noqa: E402
    PROTECTED_GROUPS as PRE_PROTECTED,
    chunk_for_rank,
    parse_ckpt_groups,
    plan_units,
)
from precompute_visualizer import encode_control_triplet, rollout_with_candidates  # noqa: E402


class ShardPlanTest(unittest.TestCase):
    def test_six_ranks_split_both_checkpoints_without_mixing_per_rank(self):
        order = [f"val-{i:02d}" for i in range(1, 13)] + [
            f"test-{i:02d}" for i in range(1, 13)
        ]
        ckpts = [
            ("ckpt-7500", "rollouts_masked_40k"),
            ("ckpt-40000", "rollouts_masked_40k_final"),
        ]
        units = plan_units(order, ckpts)
        self.assertEqual(len(units), 48)
        chunks = [chunk_for_rank(units, r, 6) for r in range(6)]
        self.assertTrue(all(len(c) == 8 for c in chunks))
        self.assertEqual(sum(len(c) for c in chunks), 48)
        # Consecutive assignment: ranks 0-2 are 7500 only, 3-5 are 40000 only.
        for rank, chunk in enumerate(chunks):
            groups = {g for _, g, _ in chunk}
            ckpt_names = {c for c, _, _ in chunk}
            self.assertEqual(len(groups), 1)
            self.assertEqual(len(ckpt_names), 1)
            if rank < 3:
                self.assertEqual(groups, {"rollouts_masked_40k"})
            else:
                self.assertEqual(groups, {"rollouts_masked_40k_final"})
        self.assertNotIn("rollouts_masked", {g for _, g, _ in units})

    def test_five_ranks_cover_all_48_units(self):
        order = [f"val-{i:02d}" for i in range(1, 13)] + [
            f"test-{i:02d}" for i in range(1, 13)
        ]
        ckpts = [
            ("ckpt-7500", "rollouts_masked_40k"),
            ("ckpt-40000", "rollouts_masked_40k_final"),
        ]
        units = plan_units(order, ckpts)
        chunks = [chunk_for_rank(units, r, 5) for r in range(5)]
        self.assertEqual([u for c in chunks for u in c], units)
        self.assertEqual(sorted(len(c) for c in chunks), [9, 9, 10, 10, 10])

    def test_parse_ckpt_groups_refuses_20k_key(self):
        with self.assertRaises(SystemExit):
            parse_ckpt_groups(["run/x:rollouts_masked"])
        self.assertIn("rollouts_masked", PRE_PROTECTED)
        self.assertIn("rollouts_masked", PROTECTED_GROUPS)


class MergeProtectTest(unittest.TestCase):
    def test_merge_attaches_40k_and_leaves_20k_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "data.js"
            payload = {
                "format": 4,
                "example_order": ["val-01"],
                "checkpoint_masked": "run_paper_split_v2_masked/checkpoint-20000",
                "examples": {
                    "val-01": {
                        "gt_score": [{"t": 0, "d": 1, "p": 60}],
                        "rollouts_masked": {
                            "raw": {"pred_score": [{"t": 0, "d": 1, "p": 60}]}
                        },
                    }
                },
            }
            atomic_dump_data_js(data_path, "window.VISUALIZER_DATA = ", payload)
            shard = {
                "checkpoints": {
                    "rollouts_masked_40k": "run_paper_split_v2_masked_40k/checkpoint-7500",
                    "rollouts_masked": "SHOULD_NOT_APPLY",
                },
                "examples": {
                    "val-01": {
                        "rollouts_masked_40k": {
                            "filtered": {"pred_score": [{"t": 1, "d": 2, "p": 61}]},
                            "raw": {"pred_score": [{"t": 1, "d": 2, "p": 61}]},
                        },
                        "rollouts_masked": {
                            "raw": {"pred_score": [{"t": 99, "d": 99, "p": 1}]}
                        },
                    }
                },
            }
            shard_path = root / "shard_00.json"
            shard_path.write_text(json.dumps(shard), encoding="utf-8")
            sys.argv = [
                "merge_masked_40k_rollouts.py",
                "--data", str(data_path),
                "--shards", str(shard_path),
            ]
            from merge_masked_40k_rollouts import main
            main()
            text = data_path.read_text(encoding="utf-8")
            out = json.loads(text[text.index("{"): text.rindex("}") + 1])
            ex = out["examples"]["val-01"]
            self.assertEqual(ex["rollouts_masked"]["raw"]["pred_score"][0]["p"], 60)
            self.assertEqual(ex["rollouts_masked_40k"]["raw"]["pred_score"][0]["p"], 61)
            self.assertEqual(
                out["checkpoint_masked"],
                "run_paper_split_v2_masked/checkpoint-20000",
            )
            self.assertEqual(
                out["checkpoint_masked_40k"],
                "run_paper_split_v2_masked_40k/checkpoint-7500",
            )

    def test_compute_f1_discovers_40k_groups(self):
        example = {
            "rollouts_masked": {"raw": {"pred_score": []}},
            "rollouts_masked_40k": {"raw": {"pred_score": []}},
            "rollouts_masked_40k_final": {"filtered": {"pred_score": []}},
        }
        groups = [g for g, _, _ in iter_rollout_groups(example)]
        self.assertEqual(
            groups,
            [
                "rollouts_masked",
                "rollouts_masked_40k",
                "rollouts_masked_40k_final",
            ],
        )


class HtmlHookTest(unittest.TestCase):
    def test_html_has_40k_panels_and_f1_rows(self):
        html = (VISUALIZER_DIR / "visualizer.html").read_text(encoding="utf-8")
        for needle in (
            'id="masked_40kPanel"',
            'id="masked_40k_finalPanel"',
            'id="masked_40kRoll"',
            'id="masked_40k_finalRoll"',
            'id="f1Mask40kA"',
            'id="f1Mask40kFinalA"',
            'id="f1PieceMask40kA"',
            'id="f1PieceMask40kFinalA"',
            "drawPaperPanel('masked_40k')",
            "drawPaperPanel('masked_40k_final')",
            "rollouts_masked_40k",
            "rollouts_masked_40k_final",
            "ours (masked 20k)",
            "ours (masked 40k @ 7.5k)",
            "ours (masked 40k final)",
        ):
            self.assertIn(needle, html)
        # 20k panel still present and not replaced.
        self.assertIn('id="maskedPanel"', html)
        self.assertIn("rollouts_masked?.raw", html)


class TinyLM(torch.nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB_SIZE, dim)
        self.lm_head = torch.nn.Linear(dim, VOCAB_SIZE, bias=False)
        self.heads = 1
        self.dim = dim

    def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        h = self.embed(input_ids)
        logits = self.lm_head(h)
        bsz, seq = input_ids.shape
        k = h.view(bsz, seq, self.heads, self.dim).transpose(1, 2)
        v = k
        if past_key_values is not None:
            pk, pv = past_key_values[0]
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        return SimpleNamespace(logits=logits, past_key_values=((k, v),))


def _short_packed(n_slots=2):
    tokens = []
    for i in range(PREFIX_CONTROLS):
        tokens.extend(encode_control_triplet((i, 4, 60 + (i % 12))))
        tokens.extend(dummy_rest_triplet(0))
    for s in range(n_slots):
        tokens.extend([TIME_OFFSET + s, DUR_OFFSET + 4, NOTE_OFFSET + 60])
        tokens.extend(encode_control_triplet((10 + s, 4, 64)))
    assert len(tokens) == ALTERNATING_START + 6 * n_slots
    return tokens


class BatchedIdentityTest(unittest.TestCase):
    def test_batched_greedy_matches_sequential(self):
        torch.manual_seed(0)
        model = TinyLM()
        model.eval()
        tokens = _short_packed(2)
        seed = {"t": 3, "d": 5, "p": 67}
        kwargs = dict(topk_onset=2, topk_dur=2, topk_pitch=2, max_candidates=8)
        seq_plain = rollout_with_candidates(
            model, "cpu", tokens, seed_note=None, **kwargs
        )
        seq_seed = rollout_with_candidates(
            model, "cpu", tokens, seed_note=seed, **kwargs
        )
        batched = batched_rollout_with_candidates(
            model, "cpu", [tokens, tokens], [None, seed], **kwargs
        )
        self.assertEqual(seq_plain[0], batched[0][0])
        self.assertEqual(seq_seed[0], batched[1][0])


if __name__ == "__main__":
    unittest.main()
