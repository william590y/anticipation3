"""
Evaluate MUSTER metrics on ASAP pieces using the model's fixed interleaving format.

This evaluation path is designed for fairer comparison against chunked PM2S systems:
1. Performance note streams are loaded raw from MIDI with no note-pair filtering.
2. Ground-truth score notes keep the full true score note set, normalized to a
   fixed 60 BPM score timeline.
3. No inverse timing map back to the original ASAP score tempo is applied.
4. The model still sees the interleaving pattern it was trained on:
   prefix controls + rests, then alternating generated score / future control.
5. When the window fills, we rebuild the interleaving from the kept half and
   re-prime the KV cache from scratch so positional state is not reused across
   shifts.

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
from anticipation.convert import midi_to_events
from alignment import load_annotation_file

from evaluate_muster import (
    load_model,
    normalize_triplet_times,
    triplets_to_musicxml,
    triplets_to_events,
    save_midi,
    run_muster_evaluation,
    print_muster_summary,
    check_muster_installation,
    OUTPUT_BASE,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ASAP_PATH              = 'asap-dataset-master'
ASAP_META_CSV          = os.path.join(ASAP_PATH, 'metadata.csv')
SPLIT_FILE             = 'data/combined_split2.txt'
PIECE_CACHE_FILE       = 'data/test_asap_muster_cache2.jsonl'
DEFAULT_CHECKPOINT     = 'checkpoint-1750'
DEFAULT_NUM_PIECES     = 50
RANDOM_SEED            = 42
NUM_WORKERS            = os.cpu_count()
PACKED_SEQUENCE_LENGTH = CONTEXT_SIZE - 4
DEFAULT_PREFIX_CONTROLS = 33
TARGET_BEAT_INTERVAL   = 1.0


# ---------------------------------------------------------------------------
# Raw ASAP loading
# ---------------------------------------------------------------------------

def _events_to_triplets(events):
    """Convert a flat event token stream into integer triplets."""
    assert len(events) % 3 == 0
    triplets = []
    for i in range(0, len(events), 3):
        triplets.append([
            int(round(events[i])),
            int(round(events[i + 1])),
            int(round(events[i + 2])),
        ])
    return triplets


def _score_events_to_controls(score_triplets):
    """Map raw note triplets to control-space triplets."""
    controls = []
    for time_tok, dur_tok, note_tok in score_triplets:
        controls.append([
            ATIME_OFFSET + (time_tok - TIME_OFFSET),
            ADUR_OFFSET + (dur_tok - DUR_OFFSET),
            ANOTE_OFFSET + (note_tok - NOTE_OFFSET),
        ])
    return controls


def _normalize_single_score_triplet(score_triplet, score_beat_times):
    original_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
    original_duration_sec = (score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION
    pitch = score_triplet[2]

    normalized_time_sec = 0.0
    time_scale_factor = 1.0

    if score_beat_times and len(score_beat_times) >= 2:
        if original_time_sec < score_beat_times[0]:
            beat_duration = score_beat_times[1] - score_beat_times[0]
            if beat_duration > 0:
                progress = (original_time_sec - score_beat_times[0]) / beat_duration
                time_scale_factor = TARGET_BEAT_INTERVAL / beat_duration
            else:
                progress = 0.0
                time_scale_factor = 1.0
            normalized_time_sec = progress * TARGET_BEAT_INTERVAL
        else:
            found = False
            for i in range(len(score_beat_times) - 1):
                if score_beat_times[i] <= original_time_sec <= score_beat_times[i + 1]:
                    beat_duration = score_beat_times[i + 1] - score_beat_times[i]
                    if beat_duration > 0:
                        progress = (original_time_sec - score_beat_times[i]) / beat_duration
                        time_scale_factor = TARGET_BEAT_INTERVAL / beat_duration
                    else:
                        progress = 0.0
                        time_scale_factor = 1.0
                    normalized_time_sec = (
                        i * TARGET_BEAT_INTERVAL + progress * TARGET_BEAT_INTERVAL
                    )
                    found = True
                    break

            if not found:
                last_beat_idx = len(score_beat_times) - 1
                if len(score_beat_times) >= 2:
                    last_beat_duration = score_beat_times[-1] - score_beat_times[-2]
                else:
                    last_beat_duration = 1.0

                if last_beat_duration > 0:
                    progress = (original_time_sec - score_beat_times[-1]) / last_beat_duration
                    time_scale_factor = TARGET_BEAT_INTERVAL / last_beat_duration
                else:
                    progress = 0.0
                    time_scale_factor = 1.0
                normalized_time_sec = (
                    last_beat_idx * TARGET_BEAT_INTERVAL + progress * TARGET_BEAT_INTERVAL
                )
    else:
        normalized_time_sec = original_time_sec - (score_beat_times[0] if score_beat_times else 0.0)

    normalized_duration_sec = original_duration_sec * time_scale_factor
    normalized_time_units = max(0, round(normalized_time_sec * TIME_RESOLUTION))
    normalized_duration_units = max(0, round(normalized_duration_sec * TIME_RESOLUTION))
    return [
        normalized_time_units + TIME_OFFSET,
        normalized_duration_units + DUR_OFFSET,
        pitch,
    ]


def load_asap_piece(filegroup):
    """
    Worker: load one ASAP piece as raw performance and score note streams.

    filegroup = ('asap', perf_midi, score_midi, perf_beats, score_beats)
    Returns a piece dict, or None on failure.
    """
    _, perf_midi, score_midi, _perf_beats, score_beats = filegroup

    try:
        perf_triplets = _events_to_triplets(midi_to_events(perf_midi, quantize=False))
        score_triplets = _events_to_triplets(midi_to_events(score_midi, quantize=False))
        score_annotations = load_annotation_file(score_beats)
        score_beat_times = [anno[0] for anno in score_annotations]
    except Exception:
        return None

    if len(perf_triplets) < DEFAULT_PREFIX_CONTROLS or len(score_triplets) < 5:
        return None

    normalized_score_triplets = [
        _normalize_single_score_triplet(score_triplet, score_beat_times)
        for score_triplet in score_triplets
    ]

    return {
        'perf_triplets': _score_events_to_controls(perf_triplets),
        'raw_score_triplets': score_triplets,
        'normalized_score_triplets': normalized_score_triplets,
        'score_beat_times': score_beat_times,
    }


# ---------------------------------------------------------------------------
# Sliding interleaving helpers
# ---------------------------------------------------------------------------

def _shift_control_triplet(ctrl_triplet, control_origin):
    raw_time = ctrl_triplet[0] - ATIME_OFFSET
    return [
        ATIME_OFFSET + max(0, raw_time - control_origin),
        ctrl_triplet[1],
        ctrl_triplet[2],
    ]


def _shift_score_triplet(score_triplet, score_origin):
    raw_time = score_triplet[0] - TIME_OFFSET
    return [
        TIME_OFFSET + max(0, raw_time - score_origin),
        score_triplet[1],
        score_triplet[2],
    ]


def _masked_score_logits(logits, slot_idx, min_time_tok=None):
    """
    Restrict logits to valid score-token bands for a given slot.
    """
    logits = logits.clone()
    logits[CONTROL_OFFSET:SPECIAL_OFFSET] = -float('inf')
    logits[SPECIAL_OFFSET:] = -float('inf')

    if slot_idx == 0:
        logits[DUR_OFFSET:DUR_OFFSET + MAX_DUR] = -float('inf')
        logits[NOTE_OFFSET:NOTE_OFFSET + MAX_NOTE] = -float('inf')
        if min_time_tok is not None and min_time_tok > TIME_OFFSET:
            upper = min(min_time_tok, TIME_OFFSET + MAX_TIME)
            logits[TIME_OFFSET:upper] = -float('inf')
    elif slot_idx == 1:
        logits[TIME_OFFSET:TIME_OFFSET + MAX_TIME] = -float('inf')
        logits[NOTE_OFFSET:NOTE_OFFSET + MAX_NOTE] = -float('inf')
    else:
        logits[TIME_OFFSET:TIME_OFFSET + MAX_TIME] = -float('inf')
        logits[DUR_OFFSET:DUR_OFFSET + MAX_DUR] = -float('inf')

    return logits


def _safe_sample_from_logits(logits):
    """Sample from logits, guarding against all-masked / NaN distributions."""
    logits = logits.detach().float().cpu()
    finite_mask = torch.isfinite(logits)
    if not finite_mask.any():
        return None

    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all():
        return None

    prob_sum = probs.sum()
    if not torch.isfinite(prob_sum) or prob_sum.item() <= 0:
        return None

    return torch.multinomial(probs, 1).item()


def _sanitize_score_triplet(score_triplet):
    """
    Clamp a score triplet into the valid event token ranges for export.
    """
    time_tok = min(max(int(score_triplet[0]), TIME_OFFSET), TIME_OFFSET + MAX_TIME - 1)
    dur_units = min(max(int(score_triplet[1]) - DUR_OFFSET, 0), MAX_DUR - 1)
    dur_tok = DUR_OFFSET + dur_units
    note_tok = int(score_triplet[2])
    if note_tok == REST:
        return [time_tok, dur_tok, REST]
    note_tok = min(max(note_tok, NOTE_OFFSET), NOTE_OFFSET + MAX_NOTE - 1)
    return [time_tok, dur_tok, note_tok]


def _prepare_export_triplets(score_triplets):
    prepared = []
    for triplet in score_triplets:
        clean_triplet = _sanitize_score_triplet(triplet)
        if clean_triplet[2] == REST:
            continue
        prepared.append(clean_triplet)
    return prepared


def _build_interleaved_context(perf_triplets, score_triplets, generated_score_triplets,
                               window_start, step_idx, prefix_controls=DEFAULT_PREFIX_CONTROLS):
    """
    Rebuild the active interleaved window.

    Prefix controls are rebuilt from the current control window start. Generated
    score triplets already emitted in this window are shifted to the current
    score origin but never tempo-scaled.
    """
    if window_start < len(perf_triplets):
        control_origin = perf_triplets[window_start][0] - ATIME_OFFSET
    elif perf_triplets:
        control_origin = perf_triplets[-1][0] - ATIME_OFFSET
    else:
        control_origin = 0

    if window_start < len(score_triplets):
        score_origin = score_triplets[window_start][0] - TIME_OFFSET
    elif generated_score_triplets:
        score_origin = generated_score_triplets[-1][0] - TIME_OFFSET
    else:
        score_origin = 0

    interleaved = []
    remaining_controls = max(0, len(perf_triplets) - window_start)
    k = min(prefix_controls, remaining_controls)

    for i in range(k):
        ctrl = _shift_control_triplet(perf_triplets[window_start + i], control_origin)
        ctrl_time = ctrl[0] - ATIME_OFFSET
        interleaved.extend(ctrl)
        interleaved.extend([TIME_OFFSET + ctrl_time, DUR_OFFSET + 0, REST])

    for score_idx in range(window_start, step_idx):
        if score_idx < len(generated_score_triplets):
            interleaved.extend(
                _shift_score_triplet(generated_score_triplets[score_idx], score_origin)
            )

        ctrl_idx = score_idx + k
        if ctrl_idx < len(perf_triplets):
            interleaved.extend(
                _shift_control_triplet(perf_triplets[ctrl_idx], control_origin)
            )

    return interleaved, score_origin, k


def autoregressive_generate_interleaved_raw(
    model, perf_triplets, normalized_score_triplets, device,
    prefix_controls=DEFAULT_PREFIX_CONTROLS, forced=False,
    forced_max_attempts=1000, beam_size=1, temperature=0.0
):
    """
    Generate raw-timeline score triplets using the fixed interleaving format.

    Decode in normalized model space, with one emitted score note per control.
    """
    vocab_size = model.config.vocab_size
    pred_score_triplets = []
    stats = {
        'num_slides': 0,
        'total_triplet_attempts': 0,
        'positions_forced': 0,
        'beam_log_prob': 0.0,
        'num_controls': len(perf_triplets),
        'num_gt_notes': len(normalized_score_triplets),
    }

    past = None
    next_logits = None
    window_start = 0
    context, score_origin, k = _build_interleaved_context(
        perf_triplets, normalized_score_triplets, pred_score_triplets, window_start, 0,
        prefix_controls=prefix_controls,
    )

    def _clamp(toks):
        return [min(max(int(t), 0), vocab_size - 1) for t in toks]

    def _prime():
        nonlocal past, next_logits
        if not context:
            past = None
            next_logits = None
            return
        with torch.no_grad():
            out = model(torch.tensor([_clamp(context)], device=device), use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[0, -1, :]

    def _ensure_primed():
        if next_logits is None:
            _prime()

    def _feed(new_toks):
        nonlocal past, next_logits
        if not new_toks:
            return
        if past is None or next_logits is None:
            _prime()
            return
        with torch.no_grad():
            out = model(
                torch.tensor([_clamp(new_toks)], device=device),
                past_key_values=past,
                use_cache=True,
            )
        past = out.past_key_values
        next_logits = out.logits[0, -1, :]

    def _greedy_next():
        raise RuntimeError("_greedy_next should not be called directly")

    def _decode_next(slot_idx, min_time_tok=None, sample=False):
        _ensure_primed()
        logits = _masked_score_logits(
            next_logits, slot_idx, min_time_tok=min_time_tok
        )
        if temperature > 0:
            logits = logits / temperature
        elif sample:
            pass
        else:
            tok = logits.argmax().item()
            context.append(tok)
            _feed([tok])
            return tok

        tok = _safe_sample_from_logits(logits)
        if tok is None and slot_idx == 0 and min_time_tok is not None:
            relaxed_logits = _masked_score_logits(next_logits, slot_idx, min_time_tok=None)
            if temperature > 0:
                relaxed_logits = relaxed_logits / temperature
            tok = _safe_sample_from_logits(relaxed_logits)
        if tok is None:
            tok = logits.argmax().item()
        context.append(tok)
        _feed([tok])
        return tok

    def _sample_next():
        raise RuntimeError("_sample_next should not be called directly")

    num_steps = len(perf_triplets)
    for step_idx in range(num_steps):
        if beam_size > 1:
            beams = [(list(context), 0.0)]
            beam_min_time_tok = TIME_OFFSET
            if pred_score_triplets and window_start < len(pred_score_triplets):
                beam_min_time_tok = max(
                    TIME_OFFSET,
                    TIME_OFFSET + (
                        pred_score_triplets[-1][0] - TIME_OFFSET - score_origin
                    ),
                )
            for _slot in range(3):
                candidates = []
                for ctx_b, lp in beams:
                    with torch.no_grad():
                        logits_b = model(torch.tensor([_clamp(ctx_b)], device=device)).logits[0, -1, :]
                        logits_b = _masked_score_logits(
                            logits_b,
                            _slot,
                            min_time_tok=beam_min_time_tok if _slot == 0 else None,
                        )
                        if temperature > 0:
                            logits_b = logits_b / temperature
                        log_probs = torch.log_softmax(logits_b, dim=-1)
                    top_lps, top_toks = torch.topk(log_probs, beam_size)
                    for tok, tlp in zip(top_toks.tolist(), top_lps.tolist()):
                        candidates.append((ctx_b + [tok], lp + tlp))
                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_size]

            best_ctx, best_lp = max(beams, key=lambda x: x[1])
            rel_triplet = best_ctx[-3:]
            context[:] = best_ctx
            past = None
            next_logits = None
            stats['beam_log_prob'] += best_lp

        elif forced:
            matched = False
            gt_pitch = (
                normalized_score_triplets[step_idx][2]
                if step_idx < len(normalized_score_triplets)
                else None
            )
            min_time_tok = TIME_OFFSET
            if pred_score_triplets:
                min_time_tok = max(
                    TIME_OFFSET,
                    TIME_OFFSET + (
                        pred_score_triplets[-1][0] - TIME_OFFSET - score_origin
                    ),
                )
            for _attempt in range(forced_max_attempts):
                stats['total_triplet_attempts'] += 1
                ctx_before = list(context)
                past_before = past
                logits_before = next_logits

                rel_triplet = [
                    _decode_next(0, min_time_tok=min_time_tok, sample=True),
                    _decode_next(1, sample=True),
                    _decode_next(2, sample=True),
                ]
                if gt_pitch is None or rel_triplet[2] == gt_pitch:
                    matched = True
                    break

                context[:] = ctx_before
                past = past_before
                next_logits = logits_before

            if not matched:
                if gt_pitch is not None and len(context) >= 3:
                    context[-1] = gt_pitch
                    rel_triplet = context[-3:]
                else:
                    fallback_pitch = rel_triplet[2] if gt_pitch is None else gt_pitch
                    rel_triplet = [TIME_OFFSET, DUR_OFFSET, fallback_pitch]
                    context.extend(rel_triplet)
                if gt_pitch is not None:
                    past = None
                    next_logits = None
                    stats['positions_forced'] += 1

        else:
            min_time_tok = TIME_OFFSET
            if pred_score_triplets:
                min_time_tok = max(
                    TIME_OFFSET,
                    TIME_OFFSET + (
                        pred_score_triplets[-1][0] - TIME_OFFSET - score_origin
                    ),
                )
            rel_triplet = [
                _decode_next(0, min_time_tok=min_time_tok),
                _decode_next(1),
                _decode_next(2),
            ]

        pred_score_triplets.append([
            TIME_OFFSET + max(0, score_origin + (rel_triplet[0] - TIME_OFFSET)),
            rel_triplet[1],
            rel_triplet[2],
        ])

        future_ctrl_idx = step_idx + k
        if future_ctrl_idx < len(perf_triplets):
            if window_start < len(perf_triplets):
                control_origin = perf_triplets[window_start][0] - ATIME_OFFSET
            else:
                control_origin = 0
            shifted_ctrl = _shift_control_triplet(perf_triplets[future_ctrl_idx], control_origin)
            context.extend(shifted_ctrl)
            _feed(shifted_ctrl)

        while len(context) >= PACKED_SEQUENCE_LENGTH:
            generated_in_window = step_idx + 1 - window_start
            if generated_in_window <= 1:
                break

            window_start += max(1, generated_in_window // 2)
            context, score_origin, k = _build_interleaved_context(
                perf_triplets, normalized_score_triplets, pred_score_triplets, window_start, step_idx + 1,
                prefix_controls=prefix_controls,
            )
            past = None
            next_logits = None
            stats['num_slides'] += 1

    return pred_score_triplets, stats


# ---------------------------------------------------------------------------
# ASAP metadata
# ---------------------------------------------------------------------------

def load_asap_metadata():
    """Load ASAP metadata and return list of valid piece dicts."""
    if not os.path.exists(ASAP_META_CSV):
        print(f"ERROR: ASAP metadata not found: {ASAP_META_CSV}")
        sys.exit(1)

    df = pd.read_csv(ASAP_META_CSV)
    pieces = []
    for _, row in df.iterrows():
        perf_midi = os.path.join(ASAP_PATH, row['midi_performance'])
        score_midi = os.path.join(ASAP_PATH, row['midi_score'])
        perf_beats = os.path.join(ASAP_PATH, row['performance_annotations'])
        score_beats = os.path.join(ASAP_PATH, row['midi_score_annotations'])
        if all(os.path.exists(f) for f in [perf_midi, score_midi, perf_beats, score_beats]):
            pieces.append({
                'filegroup': ('asap', perf_midi, score_midi, perf_beats, score_beats),
                'perf_path': row['midi_performance'],
            })
    return pieces


def load_asap_test_perfs(split_file):
    """
    Parse combined_split.txt and return ASAP test-split performance paths.
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


def load_cached_asap_pieces(cache_file):
    if not os.path.exists(cache_file):
        return None

    pieces = []
    try:
        with open(cache_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                required = {
                    'perf_path',
                    'perf_triplets',
                    'raw_score_triplets',
                    'normalized_score_triplets',
                    'score_beat_times',
                }
                if not required.issubset(record):
                    return None
                pieces.append(record)
    except Exception:
        return None

    return pieces or None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_asap_muster(checkpoint_path, piece_infos, output_dir,
                         forced=False, forced_max_attempts=1000,
                         beam_size=1, temperature=0.0):
    """Run model + MUSTER with normalized 60 BPM decode and export space."""
    model, device = load_model(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    aggregate_metrics = {
        'pitch_error_rate': [],
        'missing_note_rate': [],
        'extra_note_rate': [],
        'onset_time_error_rate': [],
        'offset_time_error_rate': [],
        'mean_error_rate': [],
        'voice_error_rate': [],
        'mean_error_rate_with_voice': [],
        'num_slides': [],
    }
    per_sequence_metrics = []
    num_successful = 0
    num_failed = 0

    for piece_info in tqdm(piece_infos, desc='Evaluating'):
        perf_triplets = piece_info['perf_triplets']
        normalized_score = piece_info['normalized_score_triplets']
        piece_name = piece_info['perf_path']

        if not perf_triplets or not normalized_score:
            num_failed += 1
            continue

        try:
            pred_score_norm, gen_stats = autoregressive_generate_interleaved_raw(
                model, perf_triplets, normalized_score, device,
                prefix_controls=DEFAULT_PREFIX_CONTROLS,
                forced=forced,
                forced_max_attempts=forced_max_attempts,
                beam_size=beam_size,
                temperature=temperature,
            )
        except Exception as e:
            print(f"  {piece_name}: generation failed - {e}")
            num_failed += 1
            continue

        if len(pred_score_norm) < 3:
            num_failed += 1
            continue

        pred_score = [_sanitize_score_triplet(triplet) for triplet in pred_score_norm]

        safe_name = piece_name.replace('/', '_').replace('\\', '_')
        seq_dir = Path(output_dir) / safe_name
        os.makedirs(seq_dir, exist_ok=True)

        gt_export = _prepare_export_triplets(normalized_score)
        pred_export = _prepare_export_triplets(pred_score)

        if not gt_export or not pred_export:
            num_failed += 1
            continue

        gt_norm = normalize_triplet_times(gt_export)
        pred_norm = normalize_triplet_times(pred_export)

        if gen_stats.get('num_slides', 0):
            print(
                f"  {piece_name}: {gen_stats['num_slides']} context slides, "
                f"{len(pred_norm)} predicted notes"
            )

        save_midi(triplets_to_events(gt_norm), str(seq_dir / 'ground_truth_score.mid'))
        save_midi(triplets_to_events(pred_norm), str(seq_dir / 'output_score.mid'))

        gt_xml = seq_dir / 'ground_truth_score.xml'
        pred_xml = seq_dir / 'output_score.xml'

        if not triplets_to_musicxml(gt_norm, str(gt_xml)):
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
            metrics['num_gt_notes'] = len(gt_export)
            metrics['num_pred_notes'] = len(pred_export)
            metrics['num_controls'] = len(perf_triplets)
            metrics['num_slides'] = gen_stats.get('num_slides', 0)
            if forced:
                metrics['forced_total_triplet_attempts'] = gen_stats['total_triplet_attempts']
                metrics['forced_positions_forced'] = gen_stats['positions_forced']
            if beam_size > 1:
                metrics['beam_log_prob'] = gen_stats.get('beam_log_prob', 0.0)

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
        'num_sequences_failed': num_failed,
    }
    for key, vals in aggregate_metrics.items():
        if vals:
            final[f'{key}_mean'] = float(np.mean(vals))
            final[f'{key}_std'] = float(np.std(vals))
            final[f'{key}_min'] = float(np.min(vals))
            final[f'{key}_max'] = float(np.max(vals))

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
        description='Evaluate MUSTER on ASAP pieces in normalized 60 BPM score space'
    )
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--num-pieces', type=int, default=DEFAULT_NUM_PIECES,
                        help='Number of ASAP pieces to sample (default: 30)')
    parser.add_argument('--split-file', default=SPLIT_FILE,
                        help='Path to combined_split.txt for test-split filtering')
    parser.add_argument('--piece-cache', default=PIECE_CACHE_FILE,
                        help='Precomputed JSONL cache for ASAP test pieces')
    parser.add_argument('--all-pieces', action='store_true',
                        help='Use all ASAP pieces (train+test), not just test split')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help=f'Random seed for piece sampling/shuffling (default: {RANDOM_SEED})')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                        help=f'Loader worker count (default: {NUM_WORKERS})')
    parser.add_argument(
        '--forced',
        action='store_true',
        help='Keep regenerating each score triplet until its pitch matches ground truth',
    )
    parser.add_argument('--forced-max-attempts', type=int, default=1000)
    parser.add_argument('--beam', type=int, default=1, metavar='BEAM_SIZE')
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()

    if args.forced and args.beam > 1:
        print('ERROR: --forced and --beam are mutually exclusive.')
        sys.exit(1)

    check_muster_installation()
    piece_infos = None

    if not args.all_pieces:
        cached_pieces = load_cached_asap_pieces(args.piece_cache)
        if cached_pieces is not None:
            piece_infos = sorted(cached_pieces, key=lambda p: p['perf_path'])
            print(f"Loaded {len(piece_infos)} cached ASAP test pieces from {args.piece_cache}")

    if piece_infos is None:
        print(f"Loading ASAP metadata from {ASAP_META_CSV}...")
        all_pieces = load_asap_metadata()
        print(f"  {len(all_pieces)} valid ASAP pieces found")

        if not args.all_pieces:
            test_perfs = load_asap_test_perfs(args.split_file)
            if test_perfs:
                filtered = [p for p in all_pieces if p['perf_path'] in test_perfs]
                print(f"  {len(filtered)} in TEST split of {args.split_file}")
                all_pieces = filtered if filtered else all_pieces
            else:
                print("  Warning: split file not found or unreadable; using all ASAP pieces")

        all_pieces = sorted(all_pieces, key=lambda p: p['perf_path'])
        rng = random.Random(args.seed)
        sampled = (
            rng.sample(all_pieces, args.num_pieces)
            if args.num_pieces < len(all_pieces) else rng.sample(all_pieces, len(all_pieces))
        )
        print(f"  Sampled {len(sampled)} pieces\n")

        print(f"Loading raw ASAP note streams with {args.workers} workers...")
        filegroups = [p['filegroup'] for p in sampled]
        with Pool(processes=args.workers) as pool:
            results = list(tqdm(
                pool.imap(load_asap_piece, filegroups),
                total=len(filegroups),
                desc='Loading',
            ))

        piece_infos = []
        n_ok = 0
        n_fail = 0
        for meta, piece in zip(sampled, results):
            if piece:
                piece_infos.append({**meta, **piece})
                n_ok += 1
            else:
                n_fail += 1

        print(f"  Loaded: {n_ok} ok, {n_fail} failed\n")
    else:
        rng = random.Random(args.seed)
        sampled = (
            rng.sample(piece_infos, args.num_pieces)
            if args.num_pieces < len(piece_infos) else rng.sample(piece_infos, len(piece_infos))
        )
        piece_infos = sampled
        print(f"  Sampled {len(piece_infos)} cached pieces\n")

    if not piece_infos:
        print("ERROR: no pieces could be loaded. Check ASAP dataset paths.")
        sys.exit(1)

    mode = '_forced' if args.forced else (f'_beam{args.beam}' if args.beam > 1 else '')
    temp = f'_temp{args.temperature}' if args.temperature > 0 else ''
    subdir = f'{args.checkpoint}_asap_full_raw_interleaved{mode}{temp}'
    output_dir = str(Path(OUTPUT_BASE) / subdir)
    os.makedirs(output_dir, exist_ok=True)

    with open(Path(output_dir) / 'sampled_pieces.json', 'w') as f:
        json.dump([p['perf_path'] for p in piece_infos], f, indent=2)

    print("Running model + MUSTER evaluation...")
    stats = evaluate_asap_muster(
        args.checkpoint,
        piece_infos,
        output_dir,
        forced=args.forced,
        forced_max_attempts=args.forced_max_attempts,
        beam_size=args.beam,
        temperature=args.temperature,
    )

    print_muster_summary(args.checkpoint, stats)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
