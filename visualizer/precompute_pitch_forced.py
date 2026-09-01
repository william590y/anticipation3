#!/usr/bin/env python
"""Pitch-forced rollouts for the 24 paper-viz windows (FILTERED variant only).

Rebuilds each window's packed tokens from its manifest (file, line_index),
runs nbest.pitch_forcing_eval.rollout_pitch_forced under the checkpoint the
viz's 'ours' row displays (data.js `checkpoint`), and merges the result into
data.js as `rollouts_pitch_forced.filtered` with per-window forcing stats.

Filtered-only is principled, not lazy: forcing needs the slot-aligned GT pitch,
and the slot-k<->control-k pairing holds only for the filtered tokenisation --
the raw repack breaks it (CLAUDE.md, token format section).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nbest.pitch_forcing_eval import rollout_pitch_forced   # noqa: E402
from onpolicy_rollout import score_token_positions                    # noqa: E402
from f1_reward import score_triplet_to_note                           # noqa: E402
from evaluate_muster import load_model                                # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "visualizer/data.js"))
    ap.add_argument("--manifest", default=str(ROOT / "visualizer/paper_windows.json"))
    ap.add_argument("--out", default=str(ROOT / "visualizer/rerank_feat_shards/pitch_forced.json"))
    a = ap.parse_args()

    txt = Path(a.data).read_text(encoding="utf-8")
    payload = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    ckpt = payload["checkpoint"]
    wins = json.load(open(a.manifest))["windows"]
    files = {"val": ROOT / "data/val_paper.txt", "validation": ROOT / "data/val_paper.txt",
             "test": ROOT / "data/test_paper.txt"}
    wanted = {}
    for w in wins:
        wanted.setdefault(w["split"], {})[w["line_index"]] = w["key"]
    rows = {}
    for split, idx in wanted.items():
        with open(files[split], encoding="utf-8") as fh:
            for i, raw in enumerate(fh):
                if i in idx:
                    toks = [int(t) for t in raw.split("|")[0].split()]
                    rows[idx[i]] = toks
                    if len(rows) == sum(len(v) for v in wanted.values()):
                        break
    keys = sorted(rows)
    W = torch.tensor([rows[k] for k in keys], dtype=torch.long)
    print(f"{len(keys)} windows under {ckpt}", flush=True)
    model, device = load_model(str(ROOT / ckpt))
    model.eval()
    rolled, forced, nvalid = rollout_pitch_forced(model, W.to(device))
    rolled = rolled.cpu()
    flat = score_token_positions(W.shape[1])
    out = {"checkpoint": ckpt, "examples": {}}
    for j, k in enumerate(keys):
        toks = rolled[j][flat].tolist()
        pred = [score_triplet_to_note(toks[3*i], toks[3*i+1], toks[3*i+2])
                for i in range(len(toks)//3)]
        pred = [None if n is None else {"t": int(n[0]), "d": int(n[1]), "p": int(n[2])}
                for n in pred]
        out["examples"][k] = {
            "pred_score": pred,
            "pitch_forced": int(forced[j]), "n_slots": int(nvalid[j]),
        }
        print(f"  {k}: forced {int(forced[j])}/{int(nvalid[j])} pitches")
    Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
