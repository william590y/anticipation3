#!/usr/bin/env python
"""CPU audit of all Paper 1/2 input, cache, and GT-to-MusicXML mappings."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "visualizer")]

from compute_sequence_ppl import load_payload  # noqa: E402
from precompute_visualizer import locate_window  # noqa: E402
import precompute_paper_seed_rollouts as worker  # noqa: E402


def load_tokenizer(kind: str):
    repo = REPO / worker.MODEL_SPECS[kind]["repo"]
    worker._add_repo_to_path(
        repo, "midi2scoretransformer" if kind == "paper2" else None
    )
    roots = tuple(
        str((REPO / "external" / name).resolve())
        for name in ("paper1_joint_apt_epr", "paper2_midi2score")
    )
    for name, cached in list(sys.modules.items()):
        if (getattr(cached, "__file__", None) or "").startswith(roots):
            del sys.modules[name]
    import tokenizer

    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="visualizer/data.js")
    parser.add_argument("--cache-dir", default="data/asap_aligned_stream_cache")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()

    payload, _prefix = load_payload(REPO / args.data)
    full_order = list(payload.get("example_order") or payload["examples"])
    if len(full_order) != 24:
        raise ValueError(f"expected 24 windows, found {len(full_order)}")
    if args.shard_index is None:
        order = full_order
    else:
        if args.num_shards != 4 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("mapping audit shards require index 0..3 of four")
        order = full_order[args.shard_index :: args.num_shards]
        if len(order) != 6:
            raise ValueError(f"expected six windows in audit shard, found {len(order)}")
    pieces = worker.load_cache_pieces(REPO / args.cache_dir)
    located = {}
    for eid in order:
        ex = payload["examples"][eid]
        controls = [(note["t"], note["d"], note["p"]) for note in ex["perf_notes"]]
        piece, start = locate_window(pieces, controls)
        qpb = worker.quarters_per_annotated_beat(str(ex.get("piece") or ""))
        if piece is None or qpb is None or not math.isfinite(qpb) or qpb <= 0:
            raise ValueError(f"{eid}: cannot locate source or meter")
        located[eid] = piece, start, qpb

    summary = {}
    for kind in ("paper1", "paper2"):
        tokenizer = load_tokenizer(kind)
        maximum = 0
        windows = []
        for eid in order:
            ex = payload["examples"][eid]
            piece, start, qpb = located[eid]
            target = piece["filtered_to_raw"][start : start + len(ex["gt_score"])]
            mapped, _input_certifications = worker.map_raw_input_positions(
                kind, tokenizer, piece, target
            )
            gt_tokens, xml_rows, _score_rows, _project_rows, certifications = (
                worker.prepare_gt_rows(
                    kind,
                    tokenizer,
                    piece,
                    target,
                    ex,
                    torch.device("cpu"),
                )
            )
            errors = [
                max(
                    cert["native_xml_onset_quarter_error"],
                    cert["native_xml_duration_quarter_error"],
                )
                for cert in certifications
            ]
            maximum = max(maximum, *errors)
            windows.append(
                {
                    "window": eid,
                    "input_positions": mapped[:5],
                    "xml_rows": xml_rows,
                    "native_xml_quarter_errors": errors,
                }
            )
            del gt_tokens
            gc.collect()
            print(
                f"{kind} {eid}: input={mapped[:5]} xml={xml_rows} errors={errors}",
                flush=True,
            )
        summary[kind] = {
            "max_native_xml_quarter_error": maximum,
            "windows": windows,
        }
        del tokenizer
        gc.collect()
    result = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards if args.shard_index is not None else 1,
        "example_order": order,
        "models": summary,
    }
    print("AUDIT_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
