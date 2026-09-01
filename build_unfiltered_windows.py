#!/usr/bin/env python
"""Rebuild packed windows with UNFILTERED (mistakes-included) performance controls.

`data/train_paper.txt` conditions on the *filtered* performance stream: notes the
aligner could not match to a score note were dropped at tokenization time, which
is what makes score slot k pair exactly with control k. This script re-emits the
same windows with the raw stream instead -- every performance note the pianist
actually played, extra notes and all -- so the model has to find its aligned
control at a shifted, content-dependent index rather than a fixed one.

Output is byte-compatible with the tokenizer's own format (space-separated token
ids, ` | ` terminator), including the real ground-truth score triplets in the
score slots, so `train.py`'s dataset and `posttrain_common` read it unchanged.

Per window:
  1. decode the packed line into its 170 filtered controls + 138 score triplets
  2. locate the window in the aligned-stream cache (pitch-bytes probe, then an
     exact re-anchored verification -- the same test the visualizer uses)
  3. take every raw note spanning those filtered notes
  4. repack: 32 (raw control, dummy rest) prefix pairs, then 138
     (ground-truth score triplet, raw control) body pairs

Windows that cannot be located exactly are skipped and counted, never guessed at.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from anticipation.config import CONTEXT_SIZE  # noqa: E402
from anticipation.packed_sequence import (  # noqa: E402
    ALTERNATING_START,
    PREFIX_CONTROLS,
    dummy_rest_triplet,
    extract_packed_components,
)
from precompute_visualizer import (  # noqa: E402
    _load_cache_pieces,
    build_raw_window,
    encode_control_triplet,
    locate_window,
)

PACKED_LEN = CONTEXT_SIZE - 4
BODY_SLOTS = (PACKED_LEN - ALTERNATING_START) // 6
TOTAL_CONTROLS = PREFIX_CONTROLS + BODY_SLOTS


def parse_line(line):
    body = line.split("|", 1)[0]
    return [int(tok) for tok in body.split()]


def repack_unfiltered(raw_notes, score_triplets):
    """32 (raw control, dummy rest) prefix pairs + 138 (score, raw control) pairs."""
    packed = []
    for i in range(PREFIX_CONTROLS):
        packed.extend(encode_control_triplet((raw_notes[i]["t"], raw_notes[i]["d"], raw_notes[i]["p"])))
        packed.extend(dummy_rest_triplet(0))
    for s in range(BODY_SLOTS):
        note = raw_notes[s + PREFIX_CONTROLS]
        packed.extend(score_triplets[s])
        packed.extend(encode_control_triplet((note["t"], note["d"], note["p"])))
    if len(packed) != PACKED_LEN:
        raise ValueError(f"repacked length {len(packed)} != {PACKED_LEN}")
    return packed


def convert_line(line, pieces, stats):
    tokens = parse_line(line)
    if len(tokens) != PACKED_LEN:
        stats["bad_length"] += 1
        return None
    controls, score_triplets = extract_packed_components(tokens)
    if len(controls) != TOTAL_CONTROLS or len(score_triplets) != BODY_SLOTS:
        stats["incomplete"] += 1
        return None

    piece, start = locate_window(pieces, controls)
    if piece is None:
        stats["unlocated"] += 1
        return None

    raw_notes = build_raw_window(piece, start, TOTAL_CONTROLS)
    if len(raw_notes) < TOTAL_CONTROLS:
        # Should not happen (raw >= filtered), but never emit a short window.
        stats["short_raw"] += 1
        return None

    # How much extra material the unfiltered stream carries, and how far the
    # aligned control for the last score slot has been pushed.
    stats["raw_notes_sum"] += len(raw_notes)
    matched_in_context = sum(
        1 for note in raw_notes[:TOTAL_CONTROLS] if note.get("j") is not None
    )
    stats["matched_in_context_sum"] += matched_in_context
    stats["slots_with_aligned_control"] += min(matched_in_context, BODY_SLOTS)

    packed = repack_unfiltered(raw_notes, score_triplets)
    stats["converted"] += 1
    return " ".join(str(t) for t in packed) + " | "


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-windows", type=int, default=0,
                    help="Stop after this many converted windows (0 = no limit).")
    ap.add_argument("--stride", type=int, default=1,
                    help="Keep every Nth input line; sliding windows overlap heavily.")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--cache-dir", default="data/asap_aligned_stream_cache")
    ap.add_argument("--report-every", type=int, default=2000)
    args = ap.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard settings")
    random.seed(args.seed + args.shard_index)

    started = time.perf_counter()
    pieces = _load_cache_pieces(args.cache_dir)
    print(f"loaded {len(pieces)} cached pieces in {time.perf_counter() - started:.1f}s",
          flush=True)
    if not pieces:
        raise SystemExit(f"no aligned-stream cache under {args.cache_dir}")

    stats = {k: 0 for k in (
        "bad_length", "incomplete", "unlocated", "short_raw", "converted",
        "raw_notes_sum", "matched_in_context_sum", "slots_with_aligned_control",
    )}
    seen = 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with open(args.input) as source, open(output, "w") as sink:
        for lineno, line in enumerate(source):
            if lineno % args.stride:
                continue
            if (lineno // args.stride) % args.num_shards != args.shard_index:
                continue
            seen += 1
            converted = convert_line(line, pieces, stats)
            if converted is not None:
                sink.write(converted + "\n")
            if args.max_windows and stats["converted"] >= args.max_windows:
                break
            if args.report_every and seen % args.report_every == 0:
                rate = stats["converted"] / max(seen, 1)
                print(
                    f"  seen {seen} kept {stats['converted']} ({100 * rate:.1f}%) "
                    f"{time.perf_counter() - started:.0f}s",
                    flush=True,
                )

    converted = stats["converted"]
    print(f"\nconverted {converted} / {seen} candidate windows -> {output}")
    for key in ("bad_length", "incomplete", "unlocated", "short_raw"):
        if stats[key]:
            print(f"  skipped {key}: {stats[key]}")
    if converted:
        print(
            f"  raw notes spanning the window: "
            f"{stats['raw_notes_sum'] / converted:.1f} (filtered: {TOTAL_CONTROLS})"
        )
        print(
            f"  matched notes among the {TOTAL_CONTROLS} controls kept: "
            f"{stats['matched_in_context_sum'] / converted:.1f}"
        )
        print(
            f"  score slots whose aligned control is still in context: "
            f"{stats['slots_with_aligned_control'] / converted:.1f} / {BODY_SLOTS}"
        )


if __name__ == "__main__":
    main()
