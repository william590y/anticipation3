#!/usr/bin/env python
"""Build a data.js-shaped payload for the TEST-SET windows, so the two
reference papers can be scored on the same windows as our selectors.

run_paper_models.py is driven by `payload["examples"]` and needs exactly three
fields per example -- `piece`, `gt_score`, `perf_notes` -- then transcribes each
PIECE ONCE (it caches on `piece`) and slices every window out of that. With
~1,181 windows spanning only 14 test works, the cost is 14 transcriptions, not
1,181, which is what makes this comparison affordable at all.

Nothing here touches the papers' decode path: the four traps documented in
CLAUDE.md (shared representation, bucketed pitch, per-repo output shape, <PAD>)
all live in run_paper_models.py and are reused unmodified.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from f1_reward import score_triplet_to_note            # noqa: E402
from onpolicy_rollout import score_token_positions     # noqa: E402


def _pv():
    spec = importlib.util.spec_from_file_location(
        "precompute_visualizer", ROOT / "visualizer" / "precompute_visualizer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", default="nbest_data/test9_stride150.pt")
    ap.add_argument("--token-file", default="data/test_paper.txt")
    ap.add_argument("--out", default="visualizer/data_testset.js")
    a = ap.parse_args()

    pv = _pv()
    d = torch.load(ROOT / a.shard, map_location="cpu", weights_only=False)
    flat_pos = score_token_positions(d["window_tokens"].shape[1])
    lines_wanted = {int(x) for x in d["window_line_idx"].tolist()}
    lines = {}
    with open(ROOT / a.token_file, encoding="utf-8") as fh:
        for i, raw in enumerate(fh):
            if i in lines_wanted:
                lines[i] = raw.strip()
                if len(lines) == len(lines_wanted):
                    break

    # Key prefix follows the token file: val windows must be val-*, both for
    # honesty in the viz and because downstream code splits on the prefix.
    prefix = "val" if "val" in Path(a.token_file).name else "test"
    pieces = pv._load_cache_pieces()
    examples, order = {}, []
    stats = defaultdict(int)
    for wi in range(d["window_tokens"].shape[0]):
        li = int(d["window_line_idx"][wi])
        toks = [int(t) for t in lines[li].split("|")[0].split()]
        # Same guard as eval_test_selectors: a mismatched token file resolves
        # every index to a DIFFERENT window and attributes it confidently.
        if toks[:d["window_tokens"].shape[1]] != d["window_tokens"][wi].tolist():
            raise SystemExit(f"--token-file does not match the shard at line {li}")
        ctl = pv.extract_window_controls(toks)
        pc, _ = pv.locate_window(pieces, ctl)
        if pc is None:
            stats["unlocated"] += 1
            continue
        gt = []
        for k in range(len(flat_pos) // 3):
            v = [int(toks[int(flat_pos[3 * k + j])]) for j in range(3)]
            n = score_triplet_to_note(*v)
            if n is not None:
                gt.append({"t": int(n[0]), "d": int(n[1]), "p": int(n[2])})
        if not gt:
            stats["no_gt"] += 1
            continue
        key = f"{prefix}-{wi:05d}"
        examples[key] = {
            "split": prefix,
            "piece": pc["piece_id"].split("asap-dataset-master/")[-1],
            "gt_score": gt,
            "perf_notes": [{"t": int(t), "d": int(dd), "p": int(p)}
                           for (t, dd, p) in ctl],
            "split": "test",
            "source_line_index": li,
        }
        order.append(key)
        stats["ok"] += 1

    payload = {"format": 4, "checkpoint": d.get("checkpoint", "?"),
               "beat_seconds": 0.5, "example_order": order,
               "examples": examples}
    out = ROOT / a.out
    out.write_text("window.VISUALIZER_DATA = " + json.dumps(payload) + ";",
                   encoding="utf-8")
    print(f"wrote {out}  ({stats['ok']} windows, "
          f"{len({e['piece'].rsplit('/',1)[0] for e in examples.values()})} works, "
          f"{stats['unlocated']} unlocated, {stats['no_gt']} without GT)")


if __name__ == "__main__":
    main()
