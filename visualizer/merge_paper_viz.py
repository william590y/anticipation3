#!/usr/bin/env python
"""Merge the validation-split and test-split precompute outputs into one data.js.

precompute_visualizer.py --only-indices keys its examples by line index within a
single token file, but the paper-split visualizer draws 12 windows from
data/val_paper.txt and 12 from data/test_paper.txt -- two files whose line
indices collide. This re-keys each example to its stable manifest key
("val-01" ... "test-12"), stamps the split on it, and writes a single payload
with an explicit example_order (validation windows first, then test).

Non-integer keys are deliberate: JS reorders plain-integer object keys ascending
regardless of insertion order, so string keys keep the panel order stable.
"""
import argparse
import json
from pathlib import Path


def load_payload(path):
    txt = Path(path).read_text(encoding="utf-8")
    return json.loads(txt[txt.index("{"): txt.rindex("}") + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="visualizer/paper_windows.json")
    ap.add_argument("--val-raw", default="visualizer/data_val_raw.js")
    ap.add_argument("--test-raw", default="visualizer/data_test_raw.js")
    ap.add_argument("--output", default="visualizer/data.js")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())["windows"]
    raw = {"validation": load_payload(args.val_raw),
           "test": load_payload(args.test_raw)}

    merged, order = {}, []
    for w in manifest:
        src = raw[w["split"]]
        ex = src["examples"].get(str(w["line_index"]))
        if ex is None:
            raise SystemExit(
                f"{w['key']}: line {w['line_index']} missing from the "
                f"{w['split']} precompute output -- rerun that precompute step")
        # Guard against a stale manifest: the precompute output records the piece
        # it actually located, which must agree with what selection recorded.
        if ex.get("piece") and ex["piece"] != w["piece"]:
            raise SystemExit(
                f"{w['key']}: piece mismatch -- manifest says {w['piece']!r} but "
                f"precompute located {ex['piece']!r}. Re-run select_paper_windows.py.")
        ex["split"] = w["split"]
        ex["source_line_index"] = w["line_index"]
        merged[w["key"]] = ex
        order.append(w["key"])

    ref = raw["validation"]
    payload = {
        "format": ref.get("format", 4),
        "checkpoint": ref.get("checkpoint"),
        "lora_checkpoint": ref.get("lora_checkpoint"),
        "split_manifest": args.manifest,
        "val_file": ref.get("test_file"),
        "test_file": raw["test"].get("test_file"),
        "beat_seconds": 0.5,
        "seed": ref.get("seed"),
        "logits_conditioning": ref.get("logits_conditioning", "autoregressive_rollout"),
        "example_order": order,
        "examples": merged,
    }

    out = Path(args.output)
    with out.open("w", encoding="utf-8") as fh:
        fh.write("window.VISUALIZER_DATA = ")
        json.dump(payload, fh)
        fh.write(";\n")
    n_val = sum(1 for w in manifest if w["split"] == "validation")
    print(f"Wrote {out}: {len(merged)} windows "
          f"({n_val} validation + {len(merged)-n_val} test)")


if __name__ == "__main__":
    main()
