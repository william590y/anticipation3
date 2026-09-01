#!/usr/bin/env python
"""Assemble data_fullsong.js for the full-rollout viewer (fullsong.html).

Per performance: GT + the three systems' whole-song note lists (10ms-bin
units, each re-anchored to its own min onset -- the same convention the
whole-song eval scores under), plus that eval's four metrics when available.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def norm(notes):
    if not notes:
        return notes
    m = min(n["t"] for n in notes)
    return [[n["t"] - m, n["d"], n["p"]] for n in notes]

def main() -> None:
    ev = {}
    evp = ROOT / "nbest_data/fullsong_eval.json"
    if evp.exists():
        for r in json.load(open(evp))["records"]:
            ev[r["key"]] = r
    out, order = {}, []
    for split in ("test", "val"):
        for f in sorted((ROOT / "fullsong_rollouts" / split / "slide").glob("*.json")):
            r = json.load(open(f))
            key = f"{split}/{r['perf_path'].lstrip('./')}"
            rec = {"split": split, "perf": r["perf_path"].lstrip("./"),
                   "work": "/".join(r["perf_path"].lstrip("./").split("/")[:-1]),
                   "boundaries": r["gen_stats"]["num_window_resets"],
                   "overlap": r.get("window_overlap", 69),
                   "gt": norm(r["gt"]), "ours": norm(r["pred"])}
            pf = ROOT / "fullsong_rollouts" / split / "slide_pforce" / f.name
            if pf.exists():
                pr = json.load(open(pf))
                rec["ours_pf"] = norm(pr["pred"])
                rec["pf_forced"] = pr["gen_stats"].get("pitch_forced", 0)
            for short, kind in (("zeng", "paper1"), ("beyer", "paper2")):
                pf = ROOT / "fullsong_papers" / split / kind / (f.stem + ".json")
                if pf.exists():
                    pr = json.load(open(pf))
                    fq = 50.0 / float(pr["qpb"])
                    rec[short] = norm([{"t": int(round(n["on_q"] * fq)),
                                        "d": max(1, int(round(n["dur_q"] * fq))),
                                        "p": n["p"]} for n in pr["notes"]])
            e = ev.get(key)
            if e:
                rec["f1"] = {m: {n: e.get(f"{n}_{m}") for n in
                                 ("ours", "beyer", "zeng")}
                             for m in ("base", "smax", "ioi", "ioi_smax")}
            out[key] = rec
            order.append(key)
    (ROOT / "visualizer/data_fullsong.js").write_text(
        "window.FULLSONG_DATA = " + json.dumps(
            {"order": order, "perfs": out}) + ";", encoding="utf-8")
    print(f"wrote data_fullsong.js: {len(order)} performances "
          f"({sum(1 for k in out if 'beyer' in out[k])} with beyer, "
          f"{sum(1 for k in out if 'zeng' in out[k])} with zeng, "
          f"{sum(1 for k in out if 'f1' in out[k])} with eval)")

if __name__ == "__main__":
    main()
