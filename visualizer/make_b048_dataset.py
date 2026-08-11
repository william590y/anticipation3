#!/usr/bin/env python
"""Turn the raw 0.48s precompute output into the final visualizer/data_b048.js.

precompute_visualizer.py --only-indices writes examples keyed by the b048 test-file
LINE INDEX. The 0.5s viz (visualizer/data.js) keys the *same* pieces/windows by their
0.5s line indices. To let the UI pair them 1:1 across the dataset toggle, this re-keys
each b048 example to its 0.5s window key (via the control-signature match in
window_map.json), restores the 0.5s example_order, and stamps beat_seconds=0.48 so the
grid/engraving code can draw the 0.48s beat spacing.

It also asserts each re-keyed example's piece matches the 0.5s example's piece for that
key -- a guard against a stale/incorrect window_map.
"""
import argparse
import json
from pathlib import Path


def load_payload(path):
    txt = Path(path).read_text(encoding="utf-8")
    return json.loads(txt[txt.index("{"): txt.rindex("}") + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="visualizer/data_b048_raw.js",
                    help="precompute output keyed by b048 line index")
    ap.add_argument("--ref", default="visualizer/data.js",
                    help="the 0.5s viz data.js (source of window keys + display order)")
    ap.add_argument("--window-map", required=True,
                    help="JSON mapping 0.5s window key -> b048 line index")
    ap.add_argument("--beat-seconds", type=float, default=0.48)
    ap.add_argument("--output", default="visualizer/data_b048.js")
    ap.add_argument("--raw-scores", default=None,
                    help="optional scores_xml sidecar keyed by b048 line index (from "
                         "precompute_scores_xml.py on the raw file); re-keyed to the 0.5s "
                         "window ids and written to --output-scores as VISUALIZER_SCORES_B048")
    ap.add_argument("--output-scores", default="visualizer/scores_xml_b048.js")
    args = ap.parse_args()

    raw = load_payload(args.raw)
    ref = load_payload(args.ref)
    wmap = json.loads(Path(args.window_map).read_text())  # "05key" -> b048_idx
    inv = {str(v): str(k) for k, v in wmap.items()}       # b048_idx -> "05key"

    raw_examples = raw["examples"]
    ref_examples = ref["examples"]
    order = ref.get("example_order") or list(ref_examples.keys())

    remapped = {}
    for b048_key, ex in raw_examples.items():
        key05 = inv.get(str(b048_key))
        if key05 is None:
            raise ValueError(f"raw example {b048_key} has no entry in window_map")
        ref_piece = ref_examples[key05]["piece"]
        if ex.get("piece") != ref_piece:
            raise ValueError(
                f"piece mismatch for 0.5 key {key05}: b048 line {b048_key} is "
                f"{ex.get('piece')!r} but 0.5 viz has {ref_piece!r} -- window_map is wrong")
        remapped[key05] = ex

    missing = [k for k in order if k not in remapped]
    if missing:
        raise ValueError(f"missing b048 rollouts for 0.5 window key(s): {missing}")

    payload = {
        "format": raw.get("format", 4),
        "checkpoint": raw.get("checkpoint"),
        "lora_checkpoint": raw.get("lora_checkpoint"),
        "test_file": raw.get("test_file"),
        "cache_dir": "data/asap_aligned_stream_cache_b048",
        "beat_seconds": args.beat_seconds,
        "source_line_indices": {k: wmap[k] for k in order},
        "seed": raw.get("seed"),
        "logits_conditioning": raw.get("logits_conditioning", "autoregressive_rollout"),
        "example_order": order,
        "examples": remapped,
    }

    out = Path(args.output)
    with out.open("w", encoding="utf-8") as fh:
        fh.write("window.VISUALIZER_DATA_B048 = ")
        json.dump(payload, fh)
        fh.write(";\n")
    print(f"Wrote {out} with {len(remapped)} windows (beat_seconds={args.beat_seconds}), "
          f"keyed to the 0.5s window ids, order mirrors {args.ref}.")

    # Re-key the real-XML sidecar the same way (b048 line idx -> 0.5s window id).
    if args.raw_scores:
        sc = load_payload(args.raw_scores)
        sc_windows = sc.get("windows", {})
        remapped_sc = {}
        for b048_key, entry in sc_windows.items():
            key05 = inv.get(str(b048_key))
            if key05 is None:
                raise ValueError(f"raw scores window {b048_key} has no entry in window_map")
            remapped_sc[key05] = entry
        sc_out = {
            "format": sc.get("format", 2),
            "bins_per_beat": sc.get("bins_per_beat"),
            "windows": remapped_sc,
        }
        outsc = Path(args.output_scores)
        with outsc.open("w", encoding="utf-8") as fh:
            fh.write("window.VISUALIZER_SCORES_B048 = ")
            json.dump(sc_out, fh)
            fh.write(";\n")
        n_real = sum(1 for v in remapped_sc.values() if v)
        print(f"Wrote {outsc} with {len(remapped_sc)} windows ({n_real} with real XML, "
              f"bins_per_beat={sc_out['bins_per_beat']}), keyed to the 0.5s window ids.")


if __name__ == "__main__":
    main()
