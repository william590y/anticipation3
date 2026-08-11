#!/usr/bin/env python
"""Pick the visualizer's windows from the paper-split validation and test sets.

Chooses `--per-split` windows from each of data/val_paper.txt and
data/test_paper.txt, each from a DISTINCT source piece (so the 24 panels show 24
different pieces rather than 24 slices of the same sonata), and writes a JSON
manifest consumed by the precompute step:

    {"windows": [{"key": "val-01", "split": "validation",
                  "line_index": 12345, "piece": "Bach/Fugue/bwv_848/Lee01M.mid"}, ...]}

Piece attribution reuses precompute_visualizer's exact control-token matching
against the aligned-stream cache (`locate_window`), which is the only reliable
way to map a tokenized line back to its source performance: tokenize_split uses
imap_unordered, so line order does not follow piece order.
"""
import argparse
import importlib.util
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_precompute_module():
    """Import precompute_visualizer.py (hyphen-free name needed for import)."""
    spec = importlib.util.spec_from_file_location(
        "precompute_visualizer", REPO / "visualizer" / "precompute_visualizer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_lines_at(path, wanted_indices):
    """Stream the token file once, returning {line_index: raw_line} for the wanted set."""
    wanted = set(wanted_indices)
    out = {}
    with open(path, "r", encoding="utf-8") as handle:
        for idx, raw in enumerate(handle):
            if idx in wanted:
                out[idx] = raw.strip()
                if len(out) == len(wanted):
                    break
    return out


def count_lines(path):
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def pick_for_split(pv, pieces, token_file, split_name, per_split, candidates, seed, prefix):
    """Sample candidate lines, attribute each to a piece, keep the first
    `per_split` that land on distinct pieces."""
    total = count_lines(token_file)
    rng = random.Random(seed)
    cand_idx = rng.sample(range(total), min(candidates, total))
    lines = read_lines_at(token_file, cand_idx)

    chosen, seen_pieces = [], set()
    for idx in cand_idx:
        if len(chosen) >= per_split:
            break
        raw = lines.get(idx)
        if not raw:
            continue
        tokens = [int(t) for t in raw.split("|")[0].split()]
        window_controls = pv.extract_window_controls(tokens)
        piece, start = pv.locate_window(pieces, window_controls)
        if piece is None:
            continue
        piece_id = piece["piece_id"].split("asap-dataset-master/")[-1]
        # De-duplicate on the SCORE (the musical work = the ASAP folder), not the
        # performance file: the paper splits hold only 16 (val) / 14 (test) unique
        # scores but many performances each, so deduping on the .mid path alone
        # would fill the panel with N pianists playing the same sonata.
        work = piece_id.rsplit("/", 1)[0]
        if work in seen_pieces:
            continue
        seen_pieces.add(work)
        chosen.append({
            "key": f"{prefix}-{len(chosen)+1:02d}",
            "split": split_name,
            "line_index": idx,
            "piece": piece_id,
            "work": work,
        })
    if len(chosen) < per_split:
        raise SystemExit(
            f"only found {len(chosen)}/{per_split} distinct-piece windows for {split_name}; "
            f"raise --candidates (tried {len(cand_idx)})"
        )
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-file", default="data/val_paper.txt")
    ap.add_argument("--test-file", default="data/test_paper.txt")
    ap.add_argument("--cache-dir", default="data/asap_aligned_stream_cache")
    ap.add_argument("--per-split", type=int, default=12)
    ap.add_argument("--candidates", type=int, default=400,
                    help="Candidate lines sampled per split before distinct-piece filtering.")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output", default="visualizer/paper_windows.json")
    args = ap.parse_args()

    pv = _load_precompute_module()
    print(f"Loading aligned-stream cache from {args.cache_dir} ...")
    pieces = pv._load_cache_pieces(cache_dir=args.cache_dir)
    print(f"  {len(pieces)} cached pieces")

    windows = []
    windows += pick_for_split(pv, pieces, args.val_file, "validation",
                              args.per_split, args.candidates, args.seed, "val")
    windows += pick_for_split(pv, pieces, args.test_file, "test",
                              args.per_split, args.candidates, args.seed + 1, "test")

    Path(args.output).write_text(json.dumps({"windows": windows}, indent=2))
    print(f"\nWrote {args.output} with {len(windows)} windows")
    for w in windows:
        print(f"  {w['key']:8s} {w['split']:10s} line {w['line_index']:>7d}  {w['piece']}")


if __name__ == "__main__":
    main()
