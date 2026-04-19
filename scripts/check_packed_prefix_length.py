#!/usr/bin/env python3
"""Check that tokenized lines have the packed structural prefix (length ALTERNATING_START)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from anticipation.config import CONTEXT_SIZE, EVENT_SIZE
from anticipation.packed_sequence import ALTERNATING_START, PREFIX_CONTROLS

PACKED_BODY = CONTEXT_SIZE - 4


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            f"Verify the first token column has length >= {ALTERNATING_START} "
            f"(prefix = {PREFIX_CONTROLS} control triplets + {PREFIX_CONTROLS} dummy score triplets "
            f"= {PREFIX_CONTROLS} * 2 * {EVENT_SIZE} = {ALTERNATING_START})."
        )
    )
    p.add_argument(
        "path",
        nargs="?",
        default="data/train_normalized.txt",
        help="Token file (space-separated tokens, optional ' | ' suffix)",
    )
    p.add_argument("--num-lines", type=int, default=8, help="How many non-empty lines to check from the top")
    p.add_argument(
        "--require-packed-length",
        action="store_true",
        help=f"Also require exactly {PACKED_BODY} tokens before '|' (CONTEXT_SIZE-4)",
    )
    args = p.parse_args()

    path = Path(args.path)
    if not path.is_file():
        print(f"error: not a file: {path}", file=sys.stderr)
        return 1

    print(
        f"PREFIX_CONTROLS={PREFIX_CONTROLS}, EVENT_SIZE={EVENT_SIZE} "
        f"=> ALTERNATING_START={ALTERNATING_START} (token positions before alternating body)"
    )
    if args.require_packed_length:
        print(f"Also checking full packed body length == {PACKED_BODY}")

    bad = 0
    checked = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if checked >= args.num_lines:
                break
            s = line.strip()
            if not s:
                continue
            toks = s.split("|", 1)[0].split()
            n = len(toks)
            ok_prefix = n >= ALTERNATING_START
            ok_len = (n == PACKED_BODY) if args.require_packed_length else True
            status = "ok" if (ok_prefix and ok_len) else "FAIL"
            if not (ok_prefix and ok_len):
                bad += 1
            print(f"line {checked}: n_tokens={n} prefix>={ALTERNATING_START} {ok_prefix} {status}")
            checked += 1

    if checked == 0:
        print("error: no non-empty lines read", file=sys.stderr)
        return 2
    if bad:
        print(f"summary: {bad}/{checked} failed", file=sys.stderr)
        return 3
    print(f"summary: all {checked} line(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
