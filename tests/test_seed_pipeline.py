from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from finalize_grpo_seed_shard import canonicalize_and_repair  # noqa: E402
from publish_seed_pipeline import bump_html_cachebuster, ordered_examples  # noqa: E402
from seed_pipeline_common import (  # noqa: E402
    canonical_seed_variant,
    expected_seed_prefix,
    legacy_raw_seed_prefix,
    raw_seed_needs_repair,
    valid_rollout,
)


def note(index):
    return {"t": index, "d": 1, "p": 60 + index}


def metric_rollout():
    f1_entry = {"f1": 0.5}
    return {
        "pred_score": [note(0)],
        "branches": {},
        "f1": {
            "onset_pitch": dict(f1_entry),
            "onset_pitch_dur": dict(f1_entry),
            "onset_pitch_tol1": dict(f1_entry),
        },
        "sequence_perplexity": {"generated": 2.0, "ground_truth": 3.0},
    }


def test_raw_seed_prefix_uses_j_and_detects_legacy_shift():
    gt = [note(i) for i in range(5)]
    ex = {
        "gt_score": gt,
        "raw_notes": [
            {"j": 0},
            {"j": 1},
            {"j": None},
            {"j": 2},
            {"j": 3},
            {"j": 4},
        ],
    }
    assert expected_seed_prefix(ex, "raw", 2) == gt[:2]
    assert not raw_seed_needs_repair(ex, 2)
    assert expected_seed_prefix(ex, "raw", 3) == [gt[0], gt[1], None, gt[2]]
    assert legacy_raw_seed_prefix(ex, 3) == gt[:3]
    assert raw_seed_needs_repair(ex, 3)


def test_seed1_aliases_canonicalize_without_unneeded_recompute():
    gt = [note(i) for i in range(5)]
    ex = {"gt_score": gt, "raw_notes": [{"j": i} for i in range(5)]}
    filtered = {"pred_score": [], "branches": {}}
    raw = {"pred_score": [], "branches": {}}
    block = {
        "filtered_seeded": filtered,
        "filtered_seed1": {"pred_score": [], "branches": {}},
        "raw_seeded": raw,
        "raw_seed1": {"pred_score": [], "branches": {}},
    }
    repaired = canonicalize_and_repair(
        block,
        ex,
        model=None,
        device=None,
        args=SimpleNamespace(),
        base_patch=False,
    )
    assert repaired == []
    assert block == {"filtered_seeded": filtered, "raw_seeded": raw}


def test_metric_rollout_and_canonical_names():
    assert valid_rollout(metric_rollout())
    assert canonical_seed_variant("filtered", 1) == "filtered_seeded"
    assert canonical_seed_variant("raw", 5) == "raw_seed5"


def test_exact_window_order_validation():
    ids = [f"window-{i:02d}" for i in range(24)]
    payload = {"example_order": ids, "examples": {key: {} for key in ids}}
    assert ordered_examples(payload) == ids
    payload["example_order"] = ids[:-1]
    try:
        ordered_examples(payload)
    except ValueError as exc:
        assert "exactly 24" in str(exc)
    else:
        raise AssertionError("23-window payload should be rejected")


def test_cachebuster_bump_is_atomic_and_numeric(tmp_path):
    html = tmp_path / "visualizer.html"
    html.write_text('<script src="data.js?v=202608111806"></script>\n', encoding="utf-8")
    bump_html_cachebuster(html, "20260814020000")
    assert html.read_text(encoding="utf-8") == (
        '<script src="data.js?v=20260814020000"></script>\n'
    )
