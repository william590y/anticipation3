#!/usr/bin/env python3
"""Split visualizer/data.js into a slim first-paint payload + per-window branches.

The 467MB data.js is almost entirely per-slot candidate lists (`branches`).
Serving that as a blocking <script> over HTTP fails (connection resets, hung
first paint, empty piano rolls). Notes (gt_score / pred_score / perf_notes)
are a few hundred KB; this peels branches into data_ex/{id}.js so the
visualizer can draw immediately and lazy-load candidates.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PREFIX = "window.VISUALIZER_DATA = "
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
ROLLOUT_KEYS = (
    "rollouts",
    "rollouts_lora",
    "rollouts_masked",
    "rollouts_masked_40k",
    "rollouts_masked_40k_final",
)


def peel_branches(example: dict) -> dict:
    packed: dict = {}
    if example.get("branches"):
        packed["branches"] = example["branches"]
        example["branches"] = {}
    for key in ROLLOUT_KEYS:
        block = example.get(key)
        if not isinstance(block, dict):
            continue
        streams = {}
        for stream, payload in block.items():
            if isinstance(payload, dict) and payload.get("branches"):
                streams[stream] = {"branches": payload["branches"]}
                payload["branches"] = {}
        if streams:
            packed[key] = streams
    return packed


def atomic_write_text(path: Path, body: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--slim", default="visualizer/data_slim.js")
    ap.add_argument("--ex-dir", default="visualizer/data_ex")
    args = ap.parse_args()

    src = Path(args.data)
    slim_path = Path(args.slim)
    ex_dir = Path(args.ex_dir)
    text = src.read_text(encoding="utf-8")
    if not text.startswith(PREFIX):
        raise SystemExit(f"unexpected prefix in {src}")
    payload = json.loads(text[len(PREFIX) :].rstrip().rstrip(";"))
    examples = payload.get("examples") or {}
    print(f"loaded {src}  examples={len(examples)}", flush=True)

    ex_dir.mkdir(parents=True, exist_ok=True)
    for eid, example in examples.items():
        if not SAFE_ID.match(str(eid)):
            raise SystemExit(f"refusing unsafe example id {eid!r}")
        packed = peel_branches(example)
        body = (
            "window.VISUALIZER_BRANCHES = window.VISUALIZER_BRANCHES || {};\n"
            f"window.VISUALIZER_BRANCHES[{json.dumps(eid)}] = "
            + json.dumps(packed, separators=(",", ":"))
            + ";\n"
        )
        atomic_write_text(ex_dir / f"{eid}.js", body)
        print(f"  wrote {eid}  {len(body):,} bytes", flush=True)
        del packed

    slim_body = PREFIX + json.dumps(payload, separators=(",", ":")) + ";\n"
    atomic_write_text(slim_path, slim_body)
    print(f"wrote {slim_path}  {len(slim_body):,} bytes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
