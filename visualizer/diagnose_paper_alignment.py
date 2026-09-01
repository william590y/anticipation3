#!/usr/bin/env python
"""Inspect current align_tokens2 rows against one aligned-stream cache file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from alignment import align_tokens2, load_annotation_file  # noqa: E402
from anticipation.asap_aligned_stream import (  # noqa: E402
    build_full_normalized_score_triplets,
    normalize_score_triplet_to_fixed_beat,
)
from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET  # noqa: E402


def compact(row):
    if row is None or row[0] is None:
        return None
    return {
        "t": int(row[0] - TIME_OFFSET),
        "d": int(row[1] - DUR_OFFSET),
        "p": int(row[2] - NOTE_OFFSET),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    payload = json.loads(Path(args.cache).read_text(encoding="utf-8"))
    fingerprint = payload["fingerprint"]
    perf = fingerprint["perf_midi"]["path"]
    score = fingerprint["score_midi"]["path"]
    perf_beats = fingerprint["perf_beats"]["path"]
    score_beats = fingerprint["score_beats"]["path"]
    aligned = align_tokens2(
        perf, score, perf_beats, score_beats, skip_Nones=False
    )
    normalized = build_full_normalized_score_triplets(score, score_beats)
    beat_times = [row[0] for row in load_annotation_file(score_beats)]

    mismatches = 0
    for _control, perf_index, score_triplet, score_index in aligned:
        cached = payload["items"][perf_index].get("score")
        direct = (
            None
            if score_triplet[0] is None
            else normalize_score_triplet_to_fixed_beat(score_triplet, beat_times)
        )
        indexed = None if score_index is None else normalized[score_index]
        if cached != direct or cached != indexed:
            print(
                json.dumps(
                    {
                        "perf_index": perf_index,
                        "score_index": score_index,
                        "cached": compact(cached),
                        "direct": compact(direct),
                        "indexed": compact(indexed),
                        "cache_equals_direct": cached == direct,
                        "cache_equals_indexed": cached == indexed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            mismatches += 1
            if mismatches >= args.limit:
                break
    print(f"reported_mismatches={mismatches}", flush=True)


if __name__ == "__main__":
    main()
