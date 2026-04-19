"""
Fresh tokenize one ASAP piece (opening window, start_idx=0), then compare
autoregressive rollout: inference.py packed path vs evaluate_muster_asap.py.

Requires: asap-dataset-master, metadata.csv, checkpoint with config+weights.

Usage (from repo root):
  python scripts/compare_opening_rollout.py --checkpoint checkpoint-20000 --piece-index 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from anticipation import ops
from anticipation.asap_aligned_stream import build_performance_anchored_stream
from anticipation.config import CONTEXT_SIZE
from anticipation.packed_sequence import (
    PREFIX_CONTROLS,
    dummy_rest_triplet,
    extract_packed_components,
)
from anticipation.vocab import ADUR_OFFSET, ANOTE_OFFSET, ATIME_OFFSET, TIME_OFFSET

from evaluate_muster import load_model
from evaluate_muster_asap import (
    ASAP_META_CSV,
    ASAP_PATH,
    autoregressive_generate_from_controls,
    preprocess_asap_piece,
)
from inference import ALTERNATING_START, autoregressive_generate_score


def _strip_control_offsets(control_triplet):
    return [
        control_triplet[0] - ATIME_OFFSET,
        control_triplet[1] - ADUR_OFFSET,
        control_triplet[2] - ANOTE_OFFSET,
    ]


def _build_real_score_suffix_min(score_triplets):
    suffix_min = [0] * len(score_triplets)
    has_real = [False] * len(score_triplets)
    current = None
    for idx in range(len(score_triplets) - 1, -1, -1):
        score_triplet = score_triplets[idx]
        if score_triplet is not None:
            score_time_units = score_triplet[0] - TIME_OFFSET
            current = score_time_units if current is None else min(current, score_time_units)
        if current is not None:
            suffix_min[idx] = current
            has_real[idx] = True
    return suffix_min, has_real


def build_opening_packed_tokens(filegroup, prefix_controls=None):
    """
    Opening window only (start_idx=0), mirroring tokenize-asap-sliding.py
    without importing that module (its top-level spawns a huge Pool on import).
    """
    if prefix_controls is None:
        prefix_controls = PREFIX_CONTROLS
    file1, file2, file3, file4 = filegroup
    aligned_items = build_performance_anchored_stream(file1, file2, file3, file4)
    if len(aligned_items) < 20:
        return None, f"only {len(aligned_items)} perf notes (need >= 20)"

    num_items = len(aligned_items)
    k = min(prefix_controls, num_items)
    start_idx = 0
    raw_perf_triplets = [_strip_control_offsets(item["control"]) for item in aligned_items]
    global_score_triplets = [item["score"] for item in aligned_items]
    score_suffix_min_times, score_suffix_has_real = _build_real_score_suffix_min(global_score_triplets)

    remaining = num_items - start_idx
    if remaining < k:
        return None, "not enough notes for prefix"

    interleaved_tokens = []
    perf_anchor = raw_perf_triplets[start_idx][0]
    min_score_time_units = 0
    if score_suffix_has_real[start_idx]:
        min_score_time_units = score_suffix_min_times[start_idx]

    for i in range(k):
        perf_triplet = raw_perf_triplets[start_idx + i]
        local_perf_time = perf_triplet[0] - perf_anchor
        interleaved_tokens.extend(
            [
                local_perf_time + ATIME_OFFSET,
                perf_triplet[1] + ADUR_OFFSET,
                perf_triplet[2] + ANOTE_OFFSET,
            ]
        )
        interleaved_tokens.extend(dummy_rest_triplet(0))

    for i in range(remaining):
        item_idx = start_idx + i
        perf_triplet = raw_perf_triplets[item_idx]
        local_perf_time = perf_triplet[0] - perf_anchor
        score_triplet = global_score_triplets[item_idx]
        if score_triplet is None:
            interleaved_tokens.extend(dummy_rest_triplet(0))
        else:
            interleaved_tokens.extend(
                [
                    score_triplet[0] - min_score_time_units,
                    score_triplet[1],
                    score_triplet[2],
                ]
            )
        ii = i + k
        if ii < remaining:
            perf_triplet = raw_perf_triplets[start_idx + ii]
            interleaved_tokens.extend(
                [
                    (perf_triplet[0] - perf_anchor) + ATIME_OFFSET,
                    perf_triplet[1] + ADUR_OFFSET,
                    perf_triplet[2] + ANOTE_OFFSET,
                ]
            )

    max_body = CONTEXT_SIZE - 4
    if len(interleaved_tokens) < max_body:
        return None, "sequence shorter than packed length"
    interleaved_tokens = interleaved_tokens[:max_body]
    return [max(0, int(t)) for t in interleaved_tokens], None


def filegroup_for_row(row) -> tuple[str, str, str, str] | None:
    file1 = os.path.join(ASAP_PATH, row["midi_performance"])
    file2 = os.path.join(ASAP_PATH, row["midi_score"])
    file3 = os.path.join(ASAP_PATH, row["performance_annotations"])
    file4 = os.path.join(ASAP_PATH, row["midi_score_annotations"])
    if all(os.path.exists(f) for f in (file1, file2, file3, file4)):
        return (file1, file2, file3, file4)
    return None


def piece_info_for_row(row):
    perf_midi = os.path.join(ASAP_PATH, row["midi_performance"])
    score_midi = os.path.join(ASAP_PATH, row["midi_score"])
    score_beats = os.path.join(ASAP_PATH, row["midi_score_annotations"])
    score_xml = None
    if "xml_score" in row.index and isinstance(row["xml_score"], str) and row["xml_score"]:
        cand = os.path.join(ASAP_PATH, row["xml_score"])
        if os.path.exists(cand):
            score_xml = cand
    return {
        "perf_path": row["midi_performance"],
        "perf_midi": perf_midi,
        "score_midi": score_midi,
        "score_beats": score_beats,
        "score_xml": score_xml,
    }


def _feeds(tr):
    return [x for x in tr if x.get("event") == "feed"]


def _run_scan(rows_ok, checkpoint: str, config_source: str, n_pieces: int) -> None:
    ckpt = ROOT / checkpoint
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)
    print(f"Loading model {ckpt} (once) for scan of {n_pieces} pieces...")
    model, device = load_model(str(ckpt), config_source=config_source)

    token_mismatches: list[tuple[int, int, int, int]] = []
    triplet_mismatches: list[tuple[int, int, list, list]] = []
    skipped = 0
    ok = 0
    n = min(n_pieces, len(rows_ok))

    for idx in range(n):
        row, filegroup = rows_ok[idx]
        packed, err = build_opening_packed_tokens(filegroup)
        if packed is None:
            skipped += 1
            continue
        pre = preprocess_asap_piece(piece_info_for_row(row), gt_score_source="midi")
        if pre.get("error"):
            skipped += 1
            continue
        controls = pre["control_triplets"]
        gt_scores = pre["gt_score_triplets"]
        body_slots = (len(packed) - ALTERNATING_START) // 6

        trace_i: list = []
        trace_a: list = []
        pred_ctx, _ = autoregressive_generate_score(
            model,
            packed,
            ALTERNATING_START,
            device,
            constrain_score_tokens=True,
            ground_truth_score_tokens_to_feed=0,
            rollout_trace=trace_i,
        )
        pred_asap, _ = autoregressive_generate_from_controls(
            model,
            controls,
            gt_scores,
            device,
            temperature=0.0,
            ground_truth_score_notes_to_feed=0,
            rollout_trace=trace_a,
            max_notes=body_slots,
        )

        fi, fa = _feeds(trace_i), _feeds(trace_a)
        has_tok_mis = False
        if len(fi) != len(fa):
            token_mismatches.append((idx, -1, len(fi), len(fa)))
            has_tok_mis = True
        else:
            for j, (a, b) in enumerate(zip(fi, fa)):
                if a["token"] != b["token"]:
                    token_mismatches.append((idx, j, a["token"], b["token"]))
                    has_tok_mis = True
                    break

        if not has_tok_mis:
            _, pred_score_inf = extract_packed_components(
                pred_ctx, ALTERNATING_START, include_dummy_score=True
            )
            m = min(len(pred_score_inf), len(pred_asap), body_slots)
            trip_mis = False
            for t in range(m):
                if [int(x) for x in pred_score_inf[t]] != [int(x) for x in pred_asap[t]]:
                    triplet_mismatches.append((idx, t, pred_score_inf[t], pred_asap[t]))
                    trip_mis = True
                    break
            if not trip_mis:
                ok += 1

    print(f"Scan complete: n={n}, ok={ok}, skipped={skipped}")
    print(f"  Token feed mismatches: {len(token_mismatches)}")
    for rec in token_mismatches[:15]:
        print(f"    piece_idx={rec[0]} ... {rec}")
    if len(token_mismatches) > 15:
        print(f"    ... and {len(token_mismatches) - 15} more")
    print(f"  Predicted triplet mismatches (given matching feeds): {len(triplet_mismatches)}")
    for rec in triplet_mismatches[:15]:
        print(f"    piece_idx={rec[0]} triplet={rec[1]} inf={rec[2]} asap={rec[3]}")
    if len(triplet_mismatches) > 15:
        print(f"    ... and {len(triplet_mismatches) - 15} more")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoint-20000")
    parser.add_argument(
        "--config-source",
        type=str,
        default="checkpoint-20000",
        help="Fallback when checkpoint has only safetensors",
    )
    parser.add_argument(
        "--piece-index",
        type=int,
        default=0,
        help="Index into rows that have all four tokenization files (sorted by perf path)",
    )
    parser.add_argument(
        "--max-trace-steps",
        type=int,
        default=5000,
        help="Stop comparing after this many feed events (safety)",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="",
        help="Write full traces and diff to this path",
    )
    parser.add_argument(
        "--scan-pieces",
        type=int,
        default=0,
        help="If >0, run aligned (0/0) comparison on piece indices 0..N-1 and summarize",
    )
    parser.add_argument(
        "--show-defaults-diff",
        action="store_true",
        help="On piece-index only: also run default teacher forcing (1 token vs 1 note) and report first feed mismatch",
    )
    args = parser.parse_args()

    if not os.path.exists(ASAP_META_CSV):
        print(f"Missing {ASAP_META_CSV}")
        sys.exit(1)

    df = pd.read_csv(ASAP_META_CSV)
    rows_ok = []
    for _, row in df.iterrows():
        fg = filegroup_for_row(row)
        if fg is not None:
            rows_ok.append((row, fg))

    if not rows_ok:
        print("No pieces with all four MIDI/annotation paths for sliding tokenization.")
        sys.exit(1)

    rows_ok.sort(key=lambda x: str(x[0]["midi_performance"]))
    if args.piece_index < 0 or args.piece_index >= len(rows_ok):
        print(f"--piece-index out of range 0..{len(rows_ok) - 1}")
        sys.exit(1)

    if args.scan_pieces > 0:
        _run_scan(
            rows_ok,
            args.checkpoint,
            args.config_source,
            args.scan_pieces,
        )
        return

    row, filegroup = rows_ok[args.piece_index]
    piece_name = row["midi_performance"]
    print(f"Piece [{args.piece_index}]: {piece_name}")

    packed_tokens, tok_err = build_opening_packed_tokens(filegroup)
    if packed_tokens is None:
        print(f"Opening tokenization failed: {tok_err}")
        sys.exit(1)
    print(f"  Opening packed length: {len(packed_tokens)} tokens")

    piece_info = piece_info_for_row(row)
    pre = preprocess_asap_piece(piece_info, gt_score_source="midi")
    if pre.get("error"):
        print(f"preprocess_asap_piece failed: {pre['error']}")
        sys.exit(1)
    controls = pre["control_triplets"]
    gt_scores = pre["gt_score_triplets"]
    print(f"  Controls: {len(controls)}, gt_score triplets: {len(gt_scores)}")

    ckpt = ROOT / args.checkpoint
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    print(f"Loading model {ckpt} ...")
    model, device = load_model(str(ckpt), config_source=args.config_source)

    trace_inf: list = []
    trace_asap: list = []

    print("Running inference.py packed rollout (ground_truth_score_tokens_to_feed=0)...")
    pred_ctx, _fed = autoregressive_generate_score(
        model,
        packed_tokens,
        ALTERNATING_START,
        device,
        constrain_score_tokens=True,
        ground_truth_score_tokens_to_feed=0,
        rollout_trace=trace_inf,
    )

    body_slots = (len(packed_tokens) - ALTERNATING_START) // 6
    print(
        "Running evaluate_muster_asap rollout for same body length as opening packed "
        f"({body_slots} perf-note cycles, ground_truth_score_notes_to_feed=0)..."
    )
    pred_asap, _stats = autoregressive_generate_from_controls(
        model,
        controls,
        gt_scores,
        device,
        temperature=0.0,
        ground_truth_score_notes_to_feed=0,
        rollout_trace=trace_asap,
        max_notes=body_slots,
    )

    fi = _feeds(trace_inf)
    fa = _feeds(trace_asap)
    print(f"  Trace feed events: inference={len(fi)}, asap={len(fa)}")

    mismatch = None
    limit = min(len(fi), len(fa), args.max_trace_steps)
    for i in range(limit):
        a, b = fi[i], fa[i]
        if a["token"] != b["token"]:
            mismatch = (i, a, b)
            break

    if mismatch is None:
        if len(fi) != len(fa):
            print(
                f"All compared feed tokens match ({limit}), but trace lengths differ: "
                f"{len(fi)} vs {len(fa)}"
            )
        else:
            print(f"All {len(fi)} feed events match token-by-token.")
    else:
        idx, a, b = mismatch
        print(f"First mismatch at feed index {idx}:")
        print(f"  inference: token={a['token']} n={a.get('n')}")
        print(f"  asap:      token={b['token']} n={b.get('n')}")

    # Prefix alignment (packed opening vs asap header before any body feeds)
    header_len = ALTERNATING_START
    if len(packed_tokens) >= header_len:
        pref_packed = packed_tokens[:header_len]
        # asap header is first rollout segment until first feed after_prime
        # Rebuild header only from preprocess path:
        from evaluate_muster_asap import initialize_generation_window

        h, _, _, _ = initialize_generation_window(controls, 0)
        if list(pref_packed) != list(h):
            print("Prefix token mismatch (packed line vs asap header):")
            for j in range(min(len(pref_packed), len(h), header_len + 20)):
                if j >= len(pref_packed) or j >= len(h) or pref_packed[j] != h[j]:
                    print(f"  first diff at prefix index {j}: packed={pref_packed[j] if j < len(pref_packed) else 'NA'} asap={h[j] if j < len(h) else 'NA'}")
                    break
        else:
            print(f"Prefix ({header_len} tokens) matches exactly.")

    _, pred_score_inf = extract_packed_components(
        pred_ctx, ALTERNATING_START, include_dummy_score=True
    )
    m = min(len(pred_score_inf), len(pred_asap), body_slots)
    trip_diff = None
    for t in range(m):
        if [int(x) for x in pred_score_inf[t]] != [int(x) for x in pred_asap[t]]:
            trip_diff = (t, pred_score_inf[t], pred_asap[t])
            break
    if trip_diff is None:
        print(f"Predicted score triplets match for first {m} body notes.")
    else:
        t, a, b = trip_diff
        print(f"Predicted triplet mismatch at body index {t}: inference={a} asap={b}")

    if args.show_defaults_diff:
        print(
            "\n--show-defaults-diff: default teacher forcing "
            "(inference 1 token vs asap 1 note), same horizon..."
        )
        ti2: list = []
        ta2: list = []
        autoregressive_generate_score(
            model,
            packed_tokens,
            ALTERNATING_START,
            device,
            constrain_score_tokens=True,
            ground_truth_score_tokens_to_feed=1,
            rollout_trace=ti2,
        )
        autoregressive_generate_from_controls(
            model,
            controls,
            gt_scores,
            device,
            temperature=0.0,
            ground_truth_score_notes_to_feed=1,
            rollout_trace=ta2,
            max_notes=body_slots,
        )
        fi2, fa2 = _feeds(ti2), _feeds(ta2)
        d = None
        for j in range(min(len(fi2), len(fa2), 50)):
            if fi2[j]["token"] != fa2[j]["token"]:
                d = (j, fi2[j]["token"], fa2[j]["token"])
                break
        if d is None:
            print(f"  First 50 feeds: identical (len inf={len(fi2)} asap={len(fa2)})")
        else:
            print(f"  First feed mismatch at index {d[0]}: inf={d[1]} asap={d[2]}")

    payload = {
        "piece": piece_name,
        "piece_index": args.piece_index,
        "checkpoint": str(ckpt),
        "mismatch_feed_index": mismatch[0] if mismatch else None,
        "trace_inference": trace_inf[: args.max_trace_steps + 50],
        "trace_asap": trace_asap[: args.max_trace_steps + 50],
    }
    if args.out_json:
        out_path = ROOT / args.out_json if not os.path.isabs(args.out_json) else Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
