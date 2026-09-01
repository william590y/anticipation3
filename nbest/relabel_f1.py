#!/usr/bin/env python
"""Relabel N-best shards with the TABLE's F1, removing the train/eval mismatch.

Two F1 implementations exist in this repo and they are not the same function:

  * `f1_reward.final_f1` -- matches predictions to GT in EMISSION order. Used
    to label candidates in `nbest/generate_nbest.py`, so it is what every
    reranker was trained to rank by, and what every holdout number reports.
  * `visualizer/compute_f1.score_notes` -- matches greedily in ONSET order,
    each prediction taking the closest unused GT note of that pitch. Used for
    the F1 table, i.e. the number every result is finally judged on.

Measured on the visualizer pools they differ by +0.63 points on average and up
to 4.87 on a single window, so a reranker trained on the first is optimizing a
neighbour of the metric it is scored by. This script recomputes `cand_f1` with
the table's semantics so training and evaluation agree.

No GPU and no regeneration: the shards already hold `cand_tokens` and the
window's own GT inside `window_tokens`, so this is pure re-scoring.

  python -m nbest.relabel_f1 --shards 'nbest_data/unf32_train_shard*.pt'
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from compute_f1 import score_notes  # noqa: E402
from nbest.generate_nbest import flat_notes  # noqa: E402
from onpolicy_rollout import score_token_positions  # noqa: E402


def as_dicts(triplets):
    # flat_notes returns None for a slot that decoded to a non-note (REST /
    # out-of-range); compute_f1 counts only real notes, and dropping them here
    # matches how the table builds its `pred_score` lists.
    return [{"t": int(n[0]), "d": int(n[1]), "p": int(n[2])}
            for n in triplets if n is not None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--suffix", default="_tblf1",
                    help="written alongside the input, with this suffix")
    ap.add_argument("--criterion", default="onset_pitch_tol1",
                    choices=["onset_pitch", "onset_pitch_dur",
                             "onset_pitch_tol1"])
    a = ap.parse_args()

    paths = sorted(glob.glob(a.shards)) if any(c in a.shards for c in "*?[") \
        else [a.shards]
    if not paths:
        raise SystemExit(f"no shards match {a.shards}")

    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        positions = score_token_positions(d["window_tokens"].shape[1])
        row_of_line = {int(l): i
                       for i, l in enumerate(d["window_line_idx"].tolist())}
        gt_cache: dict[int, list] = {}
        old = d["cand_f1"].clone()
        new = torch.empty_like(old)
        for i in range(d["cand_tokens"].shape[0]):
            line = int(d["cand_line_idx"][i])
            if line not in gt_cache:
                row = row_of_line[line]
                gt_flat = d["window_tokens"][row].long()[positions]
                gt_cache[line] = as_dicts(flat_notes(gt_flat))
            pred = as_dicts(flat_notes(d["cand_tokens"][i].long()))
            new[i] = score_notes(pred, gt_cache[line])[a.criterion]["f1"]
        d["cand_f1_emission_order"] = old      # keep the old labels for audit
        d["cand_f1"] = new
        d["f1_metric"] = f"compute_f1.score_notes:{a.criterion}"
        delta = (new - old)
        out = Path(p).with_name(Path(p).stem + a.suffix + ".pt")
        torch.save(d, out)
        print(f"{Path(p).name}: mean {old.mean():.4f} -> {new.mean():.4f} "
              f"(delta {delta.mean():+.4f}, max |delta| {delta.abs().max():.4f}, "
              f"changed {(delta.abs() > 1e-6).float().mean() * 100:.1f}%) -> "
              f"{out.name}", flush=True)


if __name__ == "__main__":
    main()
