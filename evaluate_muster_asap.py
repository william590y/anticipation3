"""
Evaluate MUSTER metrics on freshly tokenized ASAP-only pieces.

Tokenizes directly from the ASAP dataset (using the same algorithm as
tokenize-combined-dtw.py for ASAP) and runs the MUSTER evaluation pipeline.
Only uses pieces from the ASAP dataset, filtering out ATEPP.

Usage:
    python evaluate_muster_asap.py --checkpoint checkpoint-1750 --num-pieces 30
"""

import os
import sys
import json
import random
import warnings
import argparse
import numpy as np
import torch
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

import pandas as pd
warnings.filterwarnings('ignore', category=UserWarning)

from anticipation.config import *
from anticipation.vocab import *
from anticipation import ops
from alignment import align_tokens2, load_annotation_file

# Import evaluation utilities from evaluate_muster
from evaluate_muster import (
    load_model,
    autoregressive_generate_score,
    extract_components,
    normalize_triplet_times,
    triplets_to_musicxml,
    triplets_to_events,
    save_midi,
    run_muster_evaluation,
    print_muster_summary,
    check_muster_installation,
    ALTERNATING_START,
    OUTPUT_BASE,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ASAP_PATH            = 'asap-dataset-master'
ASAP_META_CSV        = os.path.join(ASAP_PATH, 'metadata.csv')
SPLIT_FILE           = 'data/combined_split.txt'
DEFAULT_CHECKPOINT   = 'checkpoint-1750'
DEFAULT_NUM_PIECES   = 30
RANDOM_SEED          = 42
NUM_WORKERS          = 32
TARGET_BEAT_INTERVAL = 0.5   # seconds per normalized beat


# ---------------------------------------------------------------------------
# ASAP tokenisation (ported from tokenize-combined-dtw.py, ASAP branch only)
# ---------------------------------------------------------------------------

def _build_sequences(normalized_matched_tuples, prefix_controls=33):
    """Build 1024-token sequences using the sliding-window algorithm."""
    sequences = []
    k = min(prefix_controls, len(normalized_matched_tuples))

    for start_idx in range(len(normalized_matched_tuples)):
        subset = normalized_matched_tuples[start_idx:]
        if len(subset) < k:
            break

        # Performance triplets (remove offsets, normalise to t=0)
        perf_triplets = [
            [m[0][0] - ATIME_OFFSET, m[0][1] - ADUR_OFFSET, m[0][2] - ANOTE_OFFSET]
            for m in subset
        ]
        if perf_triplets:
            perf_min = min(t[0] for t in perf_triplets)
            perf_triplets = [[t[0] - perf_min, t[1], t[2]] for t in perf_triplets]

        # Score triplets (already beat-normalised), shift to t=0
        score_triplets = [m[2] for m in subset]
        score_times = [t[0] - TIME_OFFSET for t in score_triplets if t[0] is not None]
        score_min = min(score_times) if score_times else 0
        score_triplets = [
            [t[0] - score_min, t[1], t[2]] if t[0] is not None else t
            for t in score_triplets
        ]

        interleaved_tokens = []

        # Prefix: k control + rest pairs
        for i in range(k):
            pt = perf_triplets[i]
            interleaved_tokens.extend([
                pt[0] + ATIME_OFFSET,
                pt[1] + ADUR_OFFSET,
                pt[2] + ANOTE_OFFSET,
            ])
            cc_time = max(0, pt[0])
            interleaved_tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])

        # Body: alternate score / control
        for i in range(len(subset)):
            st = score_triplets[i]
            if st[0] is not None:
                interleaved_tokens.extend(st)
            ii = i + k
            if ii < len(subset):
                pt = perf_triplets[ii]
                interleaved_tokens.extend([
                    pt[0] + ATIME_OFFSET,
                    pt[1] + ADUR_OFFSET,
                    pt[2] + ANOTE_OFFSET,
                ])

        # Prepend 3 SEPs
        interleaved_tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]

        max_body = EVENT_SIZE * M
        if len(interleaved_tokens) < max_body:
            break
        interleaved_tokens = interleaved_tokens[:max_body]

        if ops.max_time(interleaved_tokens, seconds=False) >= MAX_TIME:
            continue

        sequence = [ANTICIPATE] + interleaved_tokens
        assert len(sequence) == CONTEXT_SIZE, \
            f"Expected {CONTEXT_SIZE} tokens, got {len(sequence)}"
        sequences.append(sequence)

    return sequences


def tokenize_asap_piece(filegroup):
    """
    Worker: tokenize one ASAP piece.
    filegroup = ('asap', perf_midi, score_midi, perf_beats, score_beats)
    Returns list of token sequences (each a list of ints), or [].
    """
    _, perf_midi, score_midi, perf_beats, score_beats = filegroup

    try:
        matched_tuples = align_tokens2(
            perf_midi, score_midi, perf_beats, score_beats, skip_Nones=True
        )
        if len(matched_tuples) < 20:
            return []

        score_annotations = load_annotation_file(score_beats)
        score_beat_times  = [a[0] for a in score_annotations]

        normalized = []
        for match in matched_tuples:
            perf_triplet  = match[0]
            score_triplet = match[2]

            if score_triplet[0] is not None:
                orig_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
                orig_dur_sec  = (score_triplet[1] - DUR_OFFSET)  / TIME_RESOLUTION
                pitch         = score_triplet[2]

                norm_time_sec = 0.0
                time_scale    = 1.0

                if score_beat_times and len(score_beat_times) >= 2:
                    if orig_time_sec < score_beat_times[0]:
                        beat_dur   = score_beat_times[1] - score_beat_times[0]
                        progress   = ((orig_time_sec - score_beat_times[0]) / beat_dur
                                      if beat_dur > 0 else 0)
                        time_scale = TARGET_BEAT_INTERVAL / beat_dur if beat_dur > 0 else 1.0
                        norm_time_sec = progress * TARGET_BEAT_INTERVAL
                    else:
                        found = False
                        for i in range(len(score_beat_times) - 1):
                            if score_beat_times[i] <= orig_time_sec <= score_beat_times[i + 1]:
                                beat_dur   = score_beat_times[i + 1] - score_beat_times[i]
                                progress   = ((orig_time_sec - score_beat_times[i]) / beat_dur
                                              if beat_dur > 0 else 0)
                                time_scale = TARGET_BEAT_INTERVAL / beat_dur if beat_dur > 0 else 1.0
                                norm_time_sec = (i * TARGET_BEAT_INTERVAL
                                                 + progress * TARGET_BEAT_INTERVAL)
                                found = True
                                break
                        if not found:
                            last_dur = (score_beat_times[-1] - score_beat_times[-2]
                                        if len(score_beat_times) >= 2 else 1.0)
                            progress   = ((orig_time_sec - score_beat_times[-1]) / last_dur
                                          if last_dur > 0 else 0)
                            time_scale = TARGET_BEAT_INTERVAL / last_dur if last_dur > 0 else 1.0
                            norm_time_sec = ((len(score_beat_times) - 1) * TARGET_BEAT_INTERVAL
                                             + progress * TARGET_BEAT_INTERVAL)
                else:
                    norm_time_sec = orig_time_sec - (score_beat_times[0]
                                                     if score_beat_times else 0)

                norm_time_units = max(0, round(norm_time_sec * TIME_RESOLUTION))
                norm_dur_units  = max(0, round(orig_dur_sec * time_scale * TIME_RESOLUTION))
                normalized_score = [
                    norm_time_units + TIME_OFFSET,
                    norm_dur_units  + DUR_OFFSET,
                    pitch,
                ]
            else:
                normalized_score = score_triplet

            normalized.append([perf_triplet, match[1], normalized_score, match[3]])

        return _build_sequences(normalized, prefix_controls=33)

    except Exception:
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_asap_metadata():
    """Load ASAP metadata and return list of valid piece dicts."""
    if not os.path.exists(ASAP_META_CSV):
        print(f"ERROR: ASAP metadata not found: {ASAP_META_CSV}")
        sys.exit(1)

    df = pd.read_csv(ASAP_META_CSV)
    pieces = []
    for _, row in df.iterrows():
        perf_midi   = os.path.join(ASAP_PATH, row['midi_performance'])
        score_midi  = os.path.join(ASAP_PATH, row['midi_score'])
        perf_beats  = os.path.join(ASAP_PATH, row['performance_annotations'])
        score_beats = os.path.join(ASAP_PATH, row['midi_score_annotations'])
        if all(os.path.exists(f) for f in [perf_midi, score_midi, perf_beats, score_beats]):
            pieces.append({
                'filegroup': ('asap', perf_midi, score_midi, perf_beats, score_beats),
                'perf_path': row['midi_performance'],
            })
    return pieces


def load_asap_test_perfs(split_file):
    """
    Parse combined_split.txt and return a set of ASAP test-split
    performance relative paths (e.g. 'Bach/Fugue/bwv_848/Denisova06M.mid').
    Returns None if the split file cannot be read.
    """
    if not os.path.exists(split_file):
        return None

    test_perfs = set()
    in_test = False
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if '=== TEST PIECES ===' in line:
                in_test = True
                continue
            if line.startswith('==='):
                in_test = False
                continue
            if in_test and line and not line.startswith('#'):
                test_perfs.add(line.lstrip('./'))
    return test_perfs or None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_asap_muster(checkpoint_path, piece_infos, output_dir,
                          forced=False, forced_max_attempts=1000,
                          beam_size=1, temperature=0.0):
    """Run model + MUSTER on freshly tokenized ASAP sequences."""
    model, device = load_model(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    aggregate_metrics = {
        'pitch_error_rate': [], 'missing_note_rate': [], 'extra_note_rate': [],
        'onset_time_error_rate': [], 'offset_time_error_rate': [],
        'mean_error_rate': [], 'voice_error_rate': [],
        'mean_error_rate_with_voice': [],
    }
    per_sequence_metrics = []
    num_successful = 0
    num_failed = 0

    for piece_info in tqdm(piece_infos, desc='Evaluating'):
        sequences  = piece_info['sequences']
        piece_name = piece_info['perf_path']

        if not sequences:
            num_failed += 1
            continue

        # Use the first window from this piece
        tokens = sequences[0]

        if len(tokens) <= ALTERNATING_START:
            num_failed += 1
            continue

        gt_perf, gt_score = extract_components(tokens, ALTERNATING_START)
        if len(gt_score) < 5:
            num_failed += 1
            continue

        try:
            predicted_tokens, gen_stats = autoregressive_generate_score(
                model, tokens, ALTERNATING_START, device,
                forced=forced, forced_max_attempts=forced_max_attempts,
                beam_size=beam_size, temperature=temperature,
            )
        except Exception as e:
            print(f"  {piece_name}: generation failed - {e}")
            num_failed += 1
            continue

        _, pred_score = extract_components(predicted_tokens, ALTERNATING_START)
        if len(pred_score) < 3:
            num_failed += 1
            continue

        safe_name = piece_name.replace('/', '_').replace('\\', '_')
        seq_dir = Path(output_dir) / safe_name
        os.makedirs(seq_dir, exist_ok=True)

        gt_norm   = normalize_triplet_times(gt_score)
        pred_norm = normalize_triplet_times(pred_score)

        save_midi(triplets_to_events(gt_norm),   str(seq_dir / 'ground_truth_score.mid'))
        save_midi(triplets_to_events(pred_norm),  str(seq_dir / 'output_score.mid'))

        gt_xml   = seq_dir / 'ground_truth_score.xml'
        pred_xml = seq_dir / 'output_score.xml'

        if not triplets_to_musicxml(gt_norm,   str(gt_xml)):
            num_failed += 1
            continue
        if not triplets_to_musicxml(pred_norm, str(pred_xml)):
            num_failed += 1
            continue

        work_dir = seq_dir / 'muster_work'
        os.makedirs(work_dir, exist_ok=True)

        metrics = run_muster_evaluation(gt_xml, pred_xml, safe_name, work_dir)

        if metrics:
            metrics['piece'] = piece_name
            metrics['num_gt_notes']   = len(gt_score)
            metrics['num_pred_notes'] = len(pred_score)
            per_sequence_metrics.append(metrics)

            for key in aggregate_metrics:
                if key in metrics:
                    aggregate_metrics[key].append(metrics[key])

            num_successful += 1
            with open(seq_dir / 'muster_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
        else:
            num_failed += 1

    final = {
        'num_sequences_evaluated': num_successful,
        'num_sequences_failed':    num_failed,
    }
    for key, vals in aggregate_metrics.items():
        if vals:
            final[f'{key}_mean'] = float(np.mean(vals))
            final[f'{key}_std']  = float(np.std(vals))
            final[f'{key}_min']  = float(np.min(vals))
            final[f'{key}_max']  = float(np.max(vals))

    with open(Path(output_dir) / 'aggregate_muster_stats.json', 'w') as f:
        json.dump(final, f, indent=2)
    with open(Path(output_dir) / 'per_sequence_muster_stats.json', 'w') as f:
        json.dump(per_sequence_metrics, f, indent=2)

    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MUSTER on freshly tokenized ASAP-only pieces'
    )
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--num-pieces', type=int, default=DEFAULT_NUM_PIECES,
                        help='Number of ASAP pieces to sample (default: 30)')
    parser.add_argument('--split-file', default=SPLIT_FILE,
                        help='Path to combined_split.txt for test-split filtering')
    parser.add_argument('--all-pieces', action='store_true',
                        help='Use all ASAP pieces (train+test), not just test split')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                        help=f'Tokenisation worker count (default: {NUM_WORKERS})')
    parser.add_argument('--forced', action='store_true')
    parser.add_argument('--forced-max-attempts', type=int, default=1000)
    parser.add_argument('--beam', type=int, default=1, metavar='BEAM_SIZE')
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()

    if args.forced and args.beam > 1:
        print('ERROR: --forced and --beam are mutually exclusive.')
        sys.exit(1)

    check_muster_installation()

    # ---- Load metadata ----
    print(f"Loading ASAP metadata from {ASAP_META_CSV}...")
    all_pieces = load_asap_metadata()
    print(f"  {len(all_pieces)} valid ASAP pieces found")

    # ---- Filter to test split ----
    if not args.all_pieces:
        test_perfs = load_asap_test_perfs(args.split_file)
        if test_perfs:
            filtered = [p for p in all_pieces if p['perf_path'] in test_perfs]
            print(f"  {len(filtered)} in TEST split of {args.split_file}")
            all_pieces = filtered if filtered else all_pieces
        else:
            print(f"  Warning: split file not found or unreadable; using all ASAP pieces")

    # ---- Sample ----
    random.seed(RANDOM_SEED)
    sampled = (random.sample(all_pieces, args.num_pieces)
               if args.num_pieces < len(all_pieces) else all_pieces)
    print(f"  Sampled {len(sampled)} pieces\n")

    # ---- Tokenise in parallel ----
    print(f"Tokenising with {args.workers} workers...")
    filegroups = [p['filegroup'] for p in sampled]

    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(tokenize_asap_piece, filegroups),
            total=len(filegroups), desc='Tokenising'
        ))

    piece_infos = []
    n_ok = n_fail = 0
    for meta, seqs in zip(sampled, results):
        if seqs:
            piece_infos.append({**meta, 'sequences': seqs})
            n_ok += 1
        else:
            n_fail += 1

    print(f"  Tokenised: {n_ok} ok, {n_fail} failed\n")

    if not piece_infos:
        print("ERROR: no sequences produced. Check ASAP dataset paths.")
        sys.exit(1)

    # ---- Output directory ----
    mode   = ('_forced' if args.forced
               else (f'_beam{args.beam}' if args.beam > 1 else ''))
    temp   = f'_temp{args.temperature}' if args.temperature > 0 else ''
    subdir = f'{args.checkpoint}_asap{mode}{temp}'
    output_dir = str(Path(OUTPUT_BASE) / subdir)
    os.makedirs(output_dir, exist_ok=True)

    with open(Path(output_dir) / 'sampled_pieces.json', 'w') as f:
        json.dump([p['perf_path'] for p in piece_infos], f, indent=2)

    # ---- Evaluate ----
    print("Running model + MUSTER evaluation...")
    stats = evaluate_asap_muster(
        args.checkpoint, piece_infos, output_dir,
        forced=args.forced, forced_max_attempts=args.forced_max_attempts,
        beam_size=args.beam, temperature=args.temperature,
    )

    print_muster_summary(args.checkpoint, stats)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
