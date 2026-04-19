"""
Light sanity check for alignment.align_tokens2:
  - two consecutive calls return equal results (process cache)
  - optional wall-clock timing when four ASAP-side paths are provided

Example:
  python scripts/benchmark_alignment.py \\
    --perf path/to/performance.mid \\
    --score path/to/score.mid \\
    --perf-ann path/to/performance_annotation.txt \\
    --score-ann path/to/score_annotation.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    p = argparse.ArgumentParser()
    p.add_argument("--perf", default="")
    p.add_argument("--score", default="")
    p.add_argument("--perf-ann", dest="perf_ann", default="")
    p.add_argument("--score-ann", dest="score_ann", default="")
    p.add_argument("--repeat", type=int, default=2)
    args = p.parse_args()
    paths = [args.perf, args.score, args.perf_ann, args.score_ann]
    if not all(paths):
        print("skip: provide --perf --score --perf-ann --score-ann to run timing + parity")
        return 0

    for path in paths:
        if not os.path.isfile(path):
            print(f"error: missing file {path}")
            return 1

    from alignment import align_tokens2

    outs = []
    t0 = time.perf_counter()
    for _ in range(max(1, args.repeat)):
        outs.append(align_tokens2(paths[0], paths[1], paths[2], paths[3]))
    elapsed = time.perf_counter() - t0
    if outs[0] != outs[-1]:
        print("error: first and last align_tokens2 outputs differ (cache bug?)")
        return 2
    print(f"align_tokens2 ok: {len(outs[0])} matches, {args.repeat} call(s) in {elapsed:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
