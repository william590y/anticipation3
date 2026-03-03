"""
Evaluate MUSTER on freshly tokenized ASAP-only test pieces.

Reads combined_split.txt to find ASAP test pieces, re-tokenizes them using
the same algorithm as tokenize-combined-dtw.py (beat-annotation alignment +
beat-normalised score times), then runs the MUSTER evaluation pipeline.

Usage:
    python evaluate_muster_asap.py [options]

Options mirror evaluate_muster.py (--checkpoint, --num-examples, --forced,
--beam, --temperature, --forced-max-attempts).
"""

import os
import sys
import re
import json
import random
import argparse
import tempfile
import shutil
import warnings
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

import pandas as pd
import torch
from transformers import AutoModelForCausalLM

from anticipation.config import *
from anticipation.vocab import *
from anticipation import ops
from alignment import align_tokens2, load_annotation_file

warnings.filterwarnings('ignore', category=UserWarning)

# ── Configuration ────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = 'checkpoint-1750'
ASAP_PATH         = 'asap-dataset-master'
ASAP_META_CSV     = os.path.join(ASAP_PATH, 'metadata.csv')
SPLIT_FILE        = 'data/combined_split.txt'
OUTPUT_BASE       = 'muster_evaluation_results'

NUM_EXAMPLES  = 30
RANDOM_SEED   = 42
K_PREFIX      = 33
ALTERNATING_START = 4 + K_PREFIX * 6   # = 202

NUM_WORKERS   = 32   # tokenisation parallelism

TARGET_BEAT_INTERVAL = 0.5              # normalise score to 0.5 s / beat


# ── ASAP identifier check ────────────────────────────────────────────────────
def _is_asap(path_str: str) -> bool:
    """ATEPP pieces have all-digit filenames (e.g. 00458.mid); ASAP do not."""
    stem = os.path.splitext(os.path.basename(path_str))[0]
    return not stem.isdigit()


# ── Tokenisation (extracted from tokenize-combined-dtw.py) ──────────────────
def _build_sequences(normalized_matched_tuples, prefix_controls=33):
    sequences = []
    k = min(prefix_controls, len(normalized_matched_tuples))

    for start_idx in range(len(normalized_matched_tuples)):
        subset = normalized_matched_tuples[start_idx:]
        if len(subset) < k:
            break

        perf_triplets = [
            [m[0][0] - ATIME_OFFSET, m[0][1] - ADUR_OFFSET, m[0][2] - ANOTE_OFFSET]
            for m in subset
        ]
        if perf_triplets:
            perf_min = min(t[0] for t in perf_triplets)
            perf_triplets = [[t[0] - perf_min, t[1], t[2]] for t in perf_triplets]

        score_triplets = [m[2] for m in subset]
        score_times = [t[0] - TIME_OFFSET for t in score_triplets if t[0] is not None]
        score_min = min(score_times) if score_times else 0
        score_triplets = [
            [t[0] - score_min, t[1], t[2]] if t[0] is not None else t
            for t in score_triplets
        ]

        interleaved = []
        for i in range(k):
            pt = perf_triplets[i]
            interleaved.extend([pt[0] + ATIME_OFFSET, pt[1] + ADUR_OFFSET, pt[2] + ANOTE_OFFSET])
            cc_time = max(0, pt[0])
            interleaved.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])

        for i in range(len(subset)):
            st = score_triplets[i]
            if st[0] is not None:
                interleaved.extend(st)
            ii = i + k
            if ii < len(subset):
                pt = perf_triplets[ii]
                interleaved.extend([pt[0] + ATIME_OFFSET, pt[1] + ADUR_OFFSET, pt[2] + ANOTE_OFFSET])

        interleaved[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]

        max_body = EVENT_SIZE * M
        if len(interleaved) < max_body:
            break
        interleaved = interleaved[:max_body]

        if ops.max_time(interleaved, seconds=False) >= MAX_TIME:
            continue

        sequence = [ANTICIPATE] + interleaved
        assert len(sequence) == CONTEXT_SIZE
        sequences.append(' '.join(str(t) for t in sequence) + ' | ')

    return sequences


def _tokenize_asap(filegroup, prefix_controls=33):
    """Tokenise one ASAP (perf_midi, score_midi, perf_beats, score_beats) tuple."""
    _, perf_midi, score_midi, perf_beats, score_beats = filegroup
    try:
        matched = align_tokens2(perf_midi, score_midi, perf_beats, score_beats, skip_Nones=True)
        if len(matched) < 20:
            return []

        score_annotations = load_annotation_file(score_beats)
        score_beat_times  = [a[0] for a in score_annotations]

        normalised = []
        for match in matched:
            perf_triplet  = match[0]
            score_triplet = match[2]

            if score_triplet[0] is not None:
                orig_t   = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
                orig_dur = (score_triplet[1] - DUR_OFFSET)  / TIME_RESOLUTION
                pitch    = score_triplet[2]

                norm_t = 0.0
                scale  = 1.0

                if score_beat_times and len(score_beat_times) >= 2:
                    if orig_t < score_beat_times[0]:
                        bd = score_beat_times[1] - score_beat_times[0]
                        scale = TARGET_BEAT_INTERVAL / bd if bd > 0 else 1.0
                        prog  = (orig_t - score_beat_times[0]) / bd if bd > 0 else 0
                        norm_t = prog * TARGET_BEAT_INTERVAL
                    else:
                        found = False
                        for i in range(len(score_beat_times) - 1):
                            if score_beat_times[i] <= orig_t <= score_beat_times[i + 1]:
                                bd = score_beat_times[i + 1] - score_beat_times[i]
                                scale  = TARGET_BEAT_INTERVAL / bd if bd > 0 else 1.0
                                prog   = (orig_t - score_beat_times[i]) / bd if bd > 0 else 0
                                norm_t = i * TARGET_BEAT_INTERVAL + prog * TARGET_BEAT_INTERVAL
                                found  = True
                                break
                        if not found:
                            bd = (score_beat_times[-1] - score_beat_times[-2]
                                  if len(score_beat_times) >= 2 else 1.0)
                            scale  = TARGET_BEAT_INTERVAL / bd if bd > 0 else 1.0
                            prog   = (orig_t - score_beat_times[-1]) / bd if bd > 0 else 0
                            norm_t = (len(score_beat_times) - 1) * TARGET_BEAT_INTERVAL + prog * TARGET_BEAT_INTERVAL
                else:
                    norm_t = orig_t - (score_beat_times[0] if score_beat_times else 0)

                norm_t   = max(0, round(norm_t   * TIME_RESOLUTION))
                norm_dur = max(0, round(orig_dur * scale * TIME_RESOLUTION))
                norm_score = [norm_t + TIME_OFFSET, norm_dur + DUR_OFFSET, pitch]
            else:
                norm_score = score_triplet

            normalised.append([perf_triplet, match[1], norm_score, match[3]])

        return _build_sequences(normalised, prefix_controls)
    except Exception:
        return []


def _worker(fg):
    seqs = _tokenize_asap(fg)
    return (seqs, len(seqs))


# ── MUSTER pipeline (imported from evaluate_muster) ──────────────────────────
# Import the heavy lifting — safe because evaluate_muster guards argparse with
# if __name__ == '__main__'
from evaluate_muster import (
    check_muster_installation,
    load_model,
    parse_sequence,
    extract_components,
    autoregressive_generate_score,
    normalize_triplet_times,
    triplets_to_events,
    triplets_to_musicxml,
    save_midi,
    run_muster_evaluation,
    print_muster_summary,
)


def evaluate_asap_muster(checkpoint_path, test_lines, original_identifiers,
                         output_dir,
                         forced=False, forced_max_attempts=1000,
                         beam_size=1, temperature=0.0):
    model, device = load_model(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    aggregate_metrics = {
        'pitch_error_rate': [], 'missing_note_rate': [], 'extra_note_rate': [],
        'onset_time_error_rate': [], 'offset_time_error_rate': [],
        'mean_error_rate': [], 'voice_error_rate': [],
        'mean_error_rate_with_voice': [],
    }
    if forced:
        aggregate_metrics['forced_total_triplet_attempts'] = []
        aggregate_metrics['forced_positions_forced']       = []
    if beam_size > 1:
        aggregate_metrics['beam_log_prob'] = []

    per_sequence_metrics = []
    num_ok = num_fail = 0

    for line, ident in tqdm(zip(test_lines, original_identifiers),
                            total=len(test_lines),
                            desc=f'Evaluating {checkpoint_path}'):
        tokens = parse_sequence(line)
        if len(tokens) <= ALTERNATING_START:
            num_fail += 1
            continue

        _, gt_score = extract_components(tokens, ALTERNATING_START)
        if len(gt_score) < 5:
            num_fail += 1
            continue

        try:
            pred_tokens, gen_stats = autoregressive_generate_score(
                model, tokens, ALTERNATING_START, device,
                forced=forced, forced_max_attempts=forced_max_attempts,
                beam_size=beam_size, temperature=temperature,
            )
        except Exception as e:
            print(f'  {ident}: generation failed — {e}')
            num_fail += 1
            continue

        _, pred_score = extract_components(pred_tokens, ALTERNATING_START)
        if len(pred_score) < 3:
            num_fail += 1
            continue

        safe_ident = re.sub(r'[^\w]', '_', str(ident))
        seq_dir = Path(output_dir) / safe_ident
        os.makedirs(seq_dir, exist_ok=True)

        gt_norm   = normalize_triplet_times(gt_score)
        pred_norm = normalize_triplet_times(pred_score)
        save_midi(triplets_to_events(gt_norm),   str(seq_dir / 'ground_truth_score.mid'))
        save_midi(triplets_to_events(pred_norm), str(seq_dir / 'output_score.mid'))

        gt_xml   = seq_dir / 'ground_truth_score.xml'
        pred_xml = seq_dir / 'output_score.xml'
        if not triplets_to_musicxml(gt_norm,   str(gt_xml)):   num_fail += 1;  continue
        if not triplets_to_musicxml(pred_norm, str(pred_xml)): num_fail += 1;  continue

        work_dir = seq_dir / 'muster_work'
        os.makedirs(work_dir, exist_ok=True)
        metrics = run_muster_evaluation(gt_xml, pred_xml, safe_ident, work_dir)

        if metrics:
            metrics['piece'] = str(ident)
            metrics['num_gt_notes']   = len(gt_score)
            metrics['num_pred_notes'] = len(pred_score)
            if forced:
                metrics['forced_total_triplet_attempts'] = gen_stats['total_triplet_attempts']
                metrics['forced_positions_forced']       = gen_stats['positions_forced']
            if beam_size > 1:
                metrics['beam_log_prob'] = gen_stats.get('beam_log_prob', 0.0)
            per_sequence_metrics.append(metrics)

            for key in aggregate_metrics:
                if key in metrics:
                    aggregate_metrics[key].append(metrics[key])
            if forced:
                aggregate_metrics['forced_total_triplet_attempts'].append(gen_stats['total_triplet_attempts'])
                aggregate_metrics['forced_positions_forced'].append(gen_stats['positions_forced'])
            if beam_size > 1:
                aggregate_metrics['beam_log_prob'].append(gen_stats.get('beam_log_prob', 0.0))

            num_ok += 1
            with open(seq_dir / 'muster_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
        else:
            num_fail += 1

    final = {'num_sequences_evaluated': num_ok, 'num_sequences_failed': num_fail}
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


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='MUSTER eval on freshly tokenized ASAP test pieces')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--num-examples', type=int, default=NUM_EXAMPLES)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                        help='Parallel workers for tokenisation (default: 32)')
    parser.add_argument('--forced', action='store_true')
    parser.add_argument('--forced-max-attempts', type=int, default=1000)
    parser.add_argument('--beam', type=int, default=1, metavar='BEAM_SIZE')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--split-file', default=SPLIT_FILE)
    parser.add_argument('--asap-path', default=ASAP_PATH)
    args = parser.parse_args()

    if args.forced and args.beam > 1:
        print('ERROR: --forced and --beam are mutually exclusive.')
        sys.exit(1)

    # ── 1. Read test ASAP piece paths from split file ────────────────────────
    print(f'Reading split file: {args.split_file}')
    in_test = False
    asap_test_paths = []        # relative paths as they appear in split file
    with open(args.split_file) as f:
        for line in f:
            line = line.strip()
            if line == '=== TEST PIECES ===':
                in_test = True
                continue
            if not in_test or not line or line.startswith('#'):
                continue
            rel = line.lstrip('./')
            if _is_asap(rel):
                asap_test_paths.append(rel)

    print(f'Found {len(asap_test_paths)} ASAP test pieces in split file')

    # ── 2. Load ASAP metadata and build lookup ───────────────────────────────
    if not os.path.exists(args.asap_path):
        print(f'ERROR: ASAP path not found: {args.asap_path}')
        sys.exit(1)
    if not os.path.exists(ASAP_META_CSV.replace(ASAP_PATH, args.asap_path)):
        print(f'ERROR: ASAP metadata CSV not found')
        sys.exit(1)

    meta_csv = ASAP_META_CSV.replace(ASAP_PATH, args.asap_path)
    df = pd.read_csv(meta_csv)

    # Build lookup: midi_performance (relative) → full paths
    lookup = {}
    for _, row in df.iterrows():
        key = str(row['midi_performance'])
        lookup[key] = (
            os.path.join(args.asap_path, row['midi_performance']),
            os.path.join(args.asap_path, row['midi_score']),
            os.path.join(args.asap_path, row['performance_annotations']),
            os.path.join(args.asap_path, row['midi_score_annotations']),
        )

    # Resolve paths that are in split file AND in metadata
    available = []
    for p in asap_test_paths:
        if p in lookup:
            paths = lookup[p]
            if all(os.path.exists(x) for x in paths):
                available.append((p, paths))
    print(f'{len(available)} ASAP test pieces resolved with all files present')

    if len(available) == 0:
        print('ERROR: No pieces available for tokenisation.')
        sys.exit(1)

    # ── 3. Sample ─────────────────────────────────────────────────────────────
    random.seed(args.seed)
    n = min(args.num_examples, len(available))
    sampled = random.sample(available, n)
    print(f'Sampled {n} pieces (seed={args.seed})')

    # ── 4. Tokenise in parallel ───────────────────────────────────────────────
    filegroups = [('asap', *paths) for _, paths in sampled]
    piece_names = [name for name, _ in sampled]

    print(f'Tokenising {n} pieces with {args.workers} workers…')
    all_sequences = []          # list of (piece_name, sequence_str)
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap(_worker, filegroups), total=n, desc='Tokenising'))

    for (seqs, count), name in zip(results, piece_names):
        for seq in seqs:
            all_sequences.append((name, seq))

    print(f'Generated {len(all_sequences)} sequences from {n} pieces')
    if len(all_sequences) == 0:
        print('ERROR: No sequences generated — check ASAP dataset and alignment.')
        sys.exit(1)

    # ── 5. Output directory ───────────────────────────────────────────────────
    temp_suffix = f'_temp{args.temperature}' if args.temperature > 0 else ''
    mode_suffix = ('_forced' if args.forced
                   else (f'_beam{args.beam}' if args.beam > 1 else ''))
    subdir = f'{args.checkpoint}_asap{mode_suffix}{temp_suffix}'
    output_dir = Path(OUTPUT_BASE) / subdir
    os.makedirs(output_dir, exist_ok=True)

    # Save piece list
    with open(output_dir / 'pieces_evaluated.json', 'w') as f:
        json.dump({'seed': args.seed, 'pieces': piece_names,
                   'total_sequences': len(all_sequences)}, f, indent=2)

    # ── 6. MUSTER check ───────────────────────────────────────────────────────
    check_muster_installation()

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    mode_tag = (' [FORCED]' if args.forced
                else (f' [BEAM={args.beam}]' if args.beam > 1 else ''))
    temp_tag = f' [TEMP={args.temperature}]' if args.temperature > 0 else ''
    print('=' * 80)
    print(f'MUSTER EVAL (ASAP only){mode_tag}{temp_tag}')
    print('=' * 80)
    print(f'Checkpoint : {args.checkpoint}')
    print(f'Pieces     : {n}')
    print(f'Sequences  : {len(all_sequences)}')
    print()

    lines       = [seq for _, seq in all_sequences]
    identifiers = [name for name, _ in all_sequences]

    stats = evaluate_asap_muster(
        args.checkpoint, lines, identifiers, str(output_dir),
        forced=args.forced,
        forced_max_attempts=args.forced_max_attempts,
        beam_size=args.beam,
        temperature=args.temperature,
    )

    print_muster_summary(args.checkpoint, stats)
    print('=' * 80)
    print(f'Results saved to: {output_dir}')


if __name__ == '__main__':
    main()
