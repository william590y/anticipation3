#!/usr/bin/env python
"""Re-derive the paper models' `pred_score` from their stored `pred_quarters`.

CPU-only. The GPU work is transcription; converting quarters to our bin grid is
arithmetic, so a change to `quarters_per_annotated_beat` needs only this, not a
rerun of the models. Every window is scored -- there is no span guard; the only
scaling applied is quarters -> bins, fixed by the time signature.

Reads each window's stored quarter-valued output, re-converts with the CURRENT
quarters_per_annotated_beat and rewrites `pred_score`, recording `span_ratio`
as a descriptive statistic. Run compute_f1.py afterwards.

    python visualizer/rebin_paper_rollouts.py --data visualizer/data.js
"""
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_paper_models import (BINS_PER_BEAT, SPAN_TOLERANCE, to_bins,
                              quarters_per_annotated_beat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    txt = Path(args.data).read_text(encoding="utf-8")
    payload = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    prefix = txt[: txt.index("{")]

    changed = restored = 0
    for key, ex in payload["examples"].items():
        gt = [n for n in (ex.get("gt_score") or []) if n]
        gt_span = max((n["t"] for n in gt), default=0) or 1
        for kind in ("paper1", "paper2"):
            roll = (ex.get(f"rollouts_{kind}") or {}).get("filtered")
            if not roll or not roll.get("pred_quarters"):
                continue
            qpb = quarters_per_annotated_beat(ex["piece"])
            if qpb is None:
                continue
            pred = []
            for n in roll["pred_quarters"]:
                t, d = to_bins(n["on"], n["dur"], qpb)
                pred.append({"t": t, "d": d, "p": n["p"]})
            ratio = (max((n["t"] for n in pred), default=0) or 1) / gt_span
            ok = 1 / SPAN_TOLERANCE <= ratio <= SPAN_TOLERANCE

            if roll.get("pred_score") is None:
                restored += 1              # was withheld by the old span guard
            elif pred != roll["pred_score"]:
                changed += 1
            roll["quarters_per_beat"] = qpb
            roll["span_ratio"] = round(ratio, 4)
            roll["span_ok"] = bool(ok)     # descriptive only; drives a UI label
            roll["pred_score"] = pred
            print(f"  {key:9s} {kind}: qpb={qpb:g} ratio={ratio:.3f} "
                  f"{'ok' if ok else 'span disagrees (scored, flagged)'}")

    print(f"\n{changed} rollouts rebinned, {restored} recovered from the old span guard; "
          f"all windows scored")
    if args.dry_run:
        print("(dry run -- not written)")
        return
    with Path(args.data).open("w", encoding="utf-8") as fh:
        fh.write(prefix)
        json.dump(payload, fh)
        fh.write(";\n")
    print(f"Wrote {args.data}")


if __name__ == "__main__":
    main()
