"""
Fair MUSTER evaluation on ASAP pieces using performance-only conditioning.

This evaluation path computes preprocessing artifacts directly from local ASAP
files, caches those artifacts on disk, and generates score triplets conditioned
only on the performance MIDI stream. Ground-truth score triplets come from the
full score MIDI normalized to a fixed 0.5 seconds per beat.

Usage:
    python evaluate_muster_asap.py --checkpoint checkpoint-1750 --num-pieces 30
"""

import argparse
import hashlib
import json
import os
import random
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from anticipation.config import *
from anticipation.convert import midi_to_events
from anticipation.vocab import *
from alignment import load_annotation_file
from evaluate_muster import (
    OUTPUT_BASE,
    check_muster_installation,
    load_model,
    normalize_triplet_times,
    print_muster_summary,
    run_muster_evaluation,
    save_midi,
    triplets_to_events,
    triplets_to_musicxml,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ASAP_PATH = "asap-dataset-master"
ASAP_META_CSV = os.path.join(ASAP_PATH, "metadata.csv")
SPLIT_FILE = "data/combined_split.txt"
CACHE_DIR = Path("data") / "asap_muster_cache"
PREPROCESS_VERSION = "fair_asap_muster_v1"
DEFAULT_CHECKPOINT = "checkpoint-1750"
DEFAULT_NUM_PIECES = 30
RANDOM_SEED = 42
NUM_WORKERS = os.cpu_count()
TARGET_BEAT_INTERVAL = 0.5
PREFIX_CONTROLS = 33
PACKED_SEQUENCE_LENGTH = CONTEXT_SIZE - 4
SCORE_SLOT_RANGES = (
    (TIME_OFFSET, DUR_OFFSET),
    (DUR_OFFSET, NOTE_OFFSET),
    (NOTE_OFFSET, REST),
)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def event_tokens_to_triplets(events):
    """Convert a flat event token list to triplets."""
    return [events[i:i + 3] for i in range(0, len(events), 3) if i + 2 < len(events)]


def normalize_control_triplets(control_triplets):
    """Shift control triplets so the first onset is at time zero."""
    if not control_triplets:
        return []

    control_triplets = sorted(control_triplets, key=lambda t: (t[0], t[2], t[1]))
    min_time = min(t[0] - ATIME_OFFSET for t in control_triplets)
    return [[t[0] - min_time, t[1], t[2]] for t in control_triplets]


def normalize_score_triplets_to_fixed_beat(
    raw_score_triplets,
    score_beat_times,
    target_beat_interval=TARGET_BEAT_INTERVAL,
):
    """Normalize score timings so each beat spans `target_beat_interval` seconds."""
    normalized = []

    for score_triplet in raw_score_triplets:
        orig_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
        orig_dur_sec = (score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION
        pitch = int(round(score_triplet[2]))

        norm_time_sec = 0.0
        time_scale = 1.0

        if score_beat_times and len(score_beat_times) >= 2:
            if orig_time_sec < score_beat_times[0]:
                beat_dur = score_beat_times[1] - score_beat_times[0]
                progress = (
                    (orig_time_sec - score_beat_times[0]) / beat_dur
                    if beat_dur > 0
                    else 0.0
                )
                time_scale = target_beat_interval / beat_dur if beat_dur > 0 else 1.0
                norm_time_sec = progress * target_beat_interval
            else:
                found = False
                for i in range(len(score_beat_times) - 1):
                    if score_beat_times[i] <= orig_time_sec <= score_beat_times[i + 1]:
                        beat_dur = score_beat_times[i + 1] - score_beat_times[i]
                        progress = (
                            (orig_time_sec - score_beat_times[i]) / beat_dur
                            if beat_dur > 0
                            else 0.0
                        )
                        time_scale = target_beat_interval / beat_dur if beat_dur > 0 else 1.0
                        norm_time_sec = i * target_beat_interval + progress * target_beat_interval
                        found = True
                        break

                if not found:
                    last_dur = (
                        score_beat_times[-1] - score_beat_times[-2]
                        if len(score_beat_times) >= 2
                        else 1.0
                    )
                    progress = (
                        (orig_time_sec - score_beat_times[-1]) / last_dur
                        if last_dur > 0
                        else 0.0
                    )
                    time_scale = target_beat_interval / last_dur if last_dur > 0 else 1.0
                    norm_time_sec = (
                        (len(score_beat_times) - 1) * target_beat_interval
                        + progress * target_beat_interval
                    )
        else:
            norm_time_sec = orig_time_sec - (score_beat_times[0] if score_beat_times else 0.0)

        norm_time_units = max(0, round(norm_time_sec * TIME_RESOLUTION))
        norm_dur_units = max(1, round(orig_dur_sec * time_scale * TIME_RESOLUTION))
        normalized.append([
            norm_time_units + TIME_OFFSET,
            norm_dur_units + DUR_OFFSET,
            pitch,
        ])

    normalized.sort(key=lambda t: (t[0], t[2], t[1]))
    return normalize_triplet_times(normalized)


def build_performance_control_triplets(perf_midi):
    """Convert a full performance MIDI into normalized control triplets."""
    perf_events = event_tokens_to_triplets(midi_to_events(perf_midi, quantize=False))
    controls = []

    for time_tok, dur_tok, pitch_tok in perf_events:
        time_units = max(0, round(time_tok - TIME_OFFSET))
        dur_units = max(0, round(dur_tok - DUR_OFFSET))
        pitch_units = int(round(pitch_tok - NOTE_OFFSET))
        controls.append([
            ATIME_OFFSET + time_units,
            ADUR_OFFSET + dur_units,
            ANOTE_OFFSET + pitch_units,
        ])

    return normalize_control_triplets(controls)


def build_full_normalized_score_triplets(score_midi, score_beats):
    """Build the full normalized GT score directly from the score MIDI."""
    raw_score_triplets = event_tokens_to_triplets(midi_to_events(score_midi, quantize=False))
    score_annotations = load_annotation_file(score_beats)
    score_beat_times = [annotation[0] for annotation in score_annotations]
    return normalize_score_triplets_to_fixed_beat(raw_score_triplets, score_beat_times)


def build_prefix_header(control_triplets, prefix_controls=PREFIX_CONTROLS):
    """Construct the fixed control+rest prefix used during training."""
    k = min(prefix_controls, len(control_triplets))
    header = []

    for control_triplet in control_triplets[:k]:
        header.extend(control_triplet)
        ctrl_time = max(0, control_triplet[0] - ATIME_OFFSET)
        header.extend([TIME_OFFSET + ctrl_time, DUR_OFFSET + 0, REST])

    return header, k


def control_time_units(control_triplet):
    """Decode a control-triplet onset into plain time units."""
    return max(0, control_triplet[0] - ATIME_OFFSET)


def localize_control_triplet(control_triplet, time_offset):
    """Shift a control triplet into the current local window timeline."""
    local_time = max(0, control_time_units(control_triplet) - time_offset)
    return [ATIME_OFFSET + local_time, control_triplet[1], control_triplet[2]]


def initialize_generation_window(
    control_triplets,
    window_start_idx,
    prefix_controls=PREFIX_CONTROLS,
):
    """
    Build a fresh generation window starting at `window_start_idx`.

    Returns:
        header: serialized prefix tokens for the window
        prefix_count: number of controls used in the prefix
        future_idx: next control index to append after the prefix
        time_offset: absolute time origin of this window
    """
    if window_start_idx >= len(control_triplets):
        return [], 0, len(control_triplets), 0

    time_offset = control_time_units(control_triplet=control_triplets[window_start_idx])
    localized_prefix_controls = [
        localize_control_triplet(control_triplet, time_offset)
        for control_triplet in control_triplets[window_start_idx:window_start_idx + prefix_controls]
    ]
    header, prefix_count = build_prefix_header(
        localized_prefix_controls,
        prefix_controls=prefix_controls,
    )
    future_idx = window_start_idx + prefix_count
    return header, prefix_count, future_idx, time_offset


def _file_fingerprint(path):
    stat = os.stat(path)
    return {
        "path": str(Path(path).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_piece_fingerprint(piece_info):
    """Fingerprint all inputs that affect preprocessing output."""
    return {
        "version": PREPROCESS_VERSION,
        "target_beat_interval": TARGET_BEAT_INTERVAL,
        "prefix_controls": PREFIX_CONTROLS,
        "perf_midi": _file_fingerprint(piece_info["perf_midi"]),
        "score_midi": _file_fingerprint(piece_info["score_midi"]),
        "score_beats": _file_fingerprint(piece_info["score_beats"]),
    }


def cache_path_for_piece(piece_info):
    """Stable cache path for a performance."""
    key = hashlib.sha1(piece_info["perf_path"].encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / f"{key}.json"


def load_piece_cache(cache_path):
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def write_piece_cache(cache_path, payload):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp_path, cache_path)


def preprocess_asap_piece(piece_info):
    """
    Build or load cached preprocessing artifacts for one ASAP performance.

    Returns the original piece info plus:
        control_triplets
        gt_score_triplets
        cache_hit
        cache_path
        error (optional)
    """
    cache_path = cache_path_for_piece(piece_info)
    fingerprint = build_piece_fingerprint(piece_info)
    cached = load_piece_cache(cache_path)

    if cached and cached.get("fingerprint") == fingerprint:
        return {
            **piece_info,
            "control_triplets": cached.get("control_triplets", []),
            "gt_score_triplets": cached.get("gt_score_triplets", []),
            "cache_hit": True,
            "cache_path": str(cache_path),
        }

    try:
        control_triplets = build_performance_control_triplets(piece_info["perf_midi"])
        gt_score_triplets = build_full_normalized_score_triplets(
            piece_info["score_midi"],
            piece_info["score_beats"],
        )
        if not control_triplets:
            raise ValueError("no performance control triplets found")
        if not gt_score_triplets:
            raise ValueError("no normalized score triplets found")

        write_piece_cache(cache_path, {
            "fingerprint": fingerprint,
            "control_triplets": control_triplets,
            "gt_score_triplets": gt_score_triplets,
        })

        return {
            **piece_info,
            "control_triplets": control_triplets,
            "gt_score_triplets": gt_score_triplets,
            "cache_hit": False,
            "cache_path": str(cache_path),
        }
    except Exception as exc:
        return {
            **piece_info,
            "cache_hit": False,
            "cache_path": str(cache_path),
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# ASAP metadata / split loading
# ---------------------------------------------------------------------------

def load_asap_metadata():
    """Load ASAP metadata and keep only files needed by the fair evaluator."""
    if not os.path.exists(ASAP_META_CSV):
        print(f"ERROR: ASAP metadata not found: {ASAP_META_CSV}")
        sys.exit(1)

    df = pd.read_csv(ASAP_META_CSV)
    pieces = []
    for _, row in df.iterrows():
        perf_midi = os.path.join(ASAP_PATH, row["midi_performance"])
        score_midi = os.path.join(ASAP_PATH, row["midi_score"])
        score_beats = os.path.join(ASAP_PATH, row["midi_score_annotations"])
        if all(os.path.exists(path) for path in [perf_midi, score_midi, score_beats]):
            pieces.append({
                "perf_path": row["midi_performance"],
                "perf_midi": perf_midi,
                "score_midi": score_midi,
                "score_beats": score_beats,
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
    with open(split_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if "=== TEST PIECES ===" in line:
                in_test = True
                continue
            if line.startswith("==="):
                in_test = False
                continue
            if in_test and line and not line.startswith("#"):
                test_perfs.add(line.lstrip("./"))
    return test_perfs or None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def autoregressive_generate_from_controls(
    model,
    control_triplets,
    device,
    prefix_controls=PREFIX_CONTROLS,
    beam_size=1,
    temperature=0.0,
    overlap_windows=True,
):
    """
    Generate score triplets for an entire piece using performance-only controls.

    The prefix is built from control+rest pairs, after which generation alternates
    between one generated score triplet and the next performance control triplet.
    No ground-truth score tokens are ever inserted into the sequence.
    """
    window_mode = "half_overlap" if overlap_windows else "no_overlap"
    if not control_triplets:
        return [], {
            "num_slides": 0,
            "num_window_resets": 0,
            "beam_log_prob": 0.0,
            "num_controls_used": 0,
            "total_performance_notes": 0,
            "prefix_controls_used": 0,
            "score_start_idx": 0,
            "window_mode": window_mode,
        }

    vocab_size = model.config.vocab_size
    header, prefix_count, future_idx, time_offset = initialize_generation_window(
        control_triplets,
        window_start_idx=0,
        prefix_controls=prefix_controls,
    )
    score_start_idx = len(header)
    context = list(header)
    pred_score_triplets = []
    stats = {
        "num_slides": 0,
        "num_window_resets": 0,
        "beam_log_prob": 0.0,
        "num_controls_used": len(control_triplets),
        "total_performance_notes": len(control_triplets),
        "prefix_controls_used": prefix_count,
        "score_start_idx": score_start_idx,
        "window_mode": window_mode,
    }

    past = None
    next_logits = None
    note_idx = 0

    def _clamp(tokens):
        return [min(max(int(token), 0), vocab_size - 1) for token in tokens]

    def _prime():
        nonlocal past, next_logits
        with torch.no_grad():
            out = model(torch.tensor([_clamp(context)], device=device), use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[0, -1, :]

    def _feed(new_tokens):
        nonlocal past, next_logits
        with torch.no_grad():
            out = model(
                torch.tensor([_clamp(new_tokens)], device=device),
                past_key_values=past,
                use_cache=True,
            )
        past = out.past_key_values
        next_logits = out.logits[0, -1, :]

    def _ensure_primed():
        if past is None:
            _prime()

    def _get_logits():
        _ensure_primed()
        return next_logits

    def _slot_logits(logits, slot_idx):
        start, end = SCORE_SLOT_RANGES[slot_idx]
        return logits[start:end], start

    def _decode_from_logits(logits, slot_idx, sample):
        slot_logits, start = _slot_logits(logits, slot_idx)
        if temperature > 0:
            slot_logits = slot_logits / temperature
        if sample:
            token = torch.multinomial(torch.softmax(slot_logits, dim=-1), 1).item()
        else:
            token = slot_logits.argmax().item()
        return start + token

    def _decode_next(slot_idx, sample):
        logits = _get_logits()
        token = _decode_from_logits(logits, slot_idx, sample=sample)
        context.append(token)
        _feed([token])
        return token

    def _assert_valid_score_triplet(time_tok, dur_tok, pitch_tok):
        time_ok = TIME_OFFSET <= time_tok < DUR_OFFSET
        dur_ok = DUR_OFFSET <= dur_tok < NOTE_OFFSET
        pitch_ok = NOTE_OFFSET <= pitch_tok < REST
        if not (time_ok and dur_ok and pitch_ok):
            raise ValueError(
                "decoder produced invalid score triplet "
                f"({time_tok}, {dur_tok}, {pitch_tok})"
            )

    def _renormalize_alt_times(alt_tokens):
        """
        Re-normalize alternating section times to start at zero after a slide.
        """
        if not alt_tokens:
            return alt_tokens, 0

        raw_times = []
        for i in range(0, len(alt_tokens) - 2, 6):
            if alt_tokens[i] < CONTROL_OFFSET:
                raw_times.append(alt_tokens[i] - TIME_OFFSET)
            if i + 3 < len(alt_tokens) and alt_tokens[i + 3] >= CONTROL_OFFSET:
                raw_times.append(alt_tokens[i + 3] - ATIME_OFFSET)

        if not raw_times:
            return alt_tokens, 0

        time_shift = min(raw_times)
        if time_shift <= 0:
            return alt_tokens, 0

        new_alt = list(alt_tokens)
        for i in range(0, len(new_alt) - 2, 6):
            if new_alt[i] < CONTROL_OFFSET:
                new_alt[i] = max(TIME_OFFSET, new_alt[i] - time_shift)
            if i + 3 < len(new_alt) and new_alt[i + 3] >= CONTROL_OFFSET:
                new_alt[i + 3] = max(ATIME_OFFSET, new_alt[i + 3] - time_shift)

        return new_alt, time_shift

    num_controls = len(control_triplets)
    while note_idx < num_controls:
        if beam_size > 1:
            beams = [(list(context), 0.0)]
            for slot_idx in range(3):
                candidates = []
                for beam_context, log_prob in beams:
                    with torch.no_grad():
                        logits = model(
                            torch.tensor([_clamp(beam_context)], device=device)
                        ).logits[0, -1, :]
                    slot_logits, start = _slot_logits(logits, slot_idx)
                    if temperature > 0:
                        slot_logits = slot_logits / temperature
                    beam_log_probs = torch.log_softmax(slot_logits, dim=-1)
                    top_k = min(beam_size, int(beam_log_probs.shape[0]))
                    top_log_probs, top_tokens = torch.topk(beam_log_probs, top_k)
                    for token, token_log_prob in zip(top_tokens.tolist(), top_log_probs.tolist()):
                        candidates.append(
                            (beam_context + [start + token], log_prob + token_log_prob)
                        )
                candidates.sort(key=lambda item: item[1], reverse=True)
                beams = candidates[:beam_size]

            best_context, best_log_prob = max(beams, key=lambda item: item[1])
            _assert_valid_score_triplet(
                best_context[-3],
                best_context[-2],
                best_context[-1],
            )
            stats["beam_log_prob"] += best_log_prob
            pred_score_triplets.append([
                best_context[-3] + time_offset,
                best_context[-2],
                best_context[-1],
            ])
            context[:] = best_context
            past = None
            next_logits = None
        else:
            sample = temperature > 0
            token_time = _decode_next(0, sample=sample)
            token_dur = _decode_next(1, sample=sample)
            token_pitch = _decode_next(2, sample=sample)
            _assert_valid_score_triplet(token_time, token_dur, token_pitch)
            pred_score_triplets.append([
                token_time + time_offset,
                token_dur,
                token_pitch,
            ])
        note_idx += 1

        if future_idx < num_controls:
            control_triplet = localize_control_triplet(control_triplets[future_idx], time_offset)
            context.extend(control_triplet)
            if past is not None:
                _feed(control_triplet)
            future_idx += 1

        if len(context) >= PACKED_SEQUENCE_LENGTH and note_idx < num_controls:
            if not overlap_windows:
                header, prefix_count, future_idx, time_offset = initialize_generation_window(
                    control_triplets,
                    window_start_idx=note_idx,
                    prefix_controls=prefix_controls,
                )
                context = list(header)
                score_start_idx = len(header)
                past = None
                next_logits = None
                stats["num_slides"] += 1
                stats["num_window_resets"] += 1
                continue

            alternating = context[score_start_idx:]
            half = (len(alternating) // 2) // 6 * 6
            if half > 0:
                remaining, time_shift = _renormalize_alt_times(alternating[half:])
                context = header + remaining
                time_offset += time_shift
                past = None
                next_logits = None
                stats["num_slides"] += 1

    return pred_score_triplets, stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def print_piece_muster_metrics(piece_name, metrics, gen_stats):
    """Print a compact MUSTER summary for one evaluated piece."""
    summary = [
        f"MER={metrics['mean_error_rate']:.2f}%",
        f"PER={metrics['pitch_error_rate']:.2f}%",
        f"MNR={metrics['missing_note_rate']:.2f}%",
        f"ENR={metrics['extra_note_rate']:.2f}%",
        f"OTER={metrics['onset_time_error_rate']:.2f}%",
        f"OFTER={metrics['offset_time_error_rate']:.2f}%",
    ]
    if "voice_error_rate" in metrics:
        summary.append(f"VER={metrics['voice_error_rate']:.2f}%")
    if "mean_error_rate_with_voice" in metrics:
        summary.append(f"MER+V={metrics['mean_error_rate_with_voice']:.2f}%")
    summary.append(f"slides={gen_stats['num_slides']}")
    summary.append(f"window_mode={gen_stats['window_mode']}")
    tqdm.write(
        f"[MUSTER] {piece_name}: " + ", ".join(summary),
        file=sys.stderr,
    )


def evaluate_asap_muster(
    checkpoint_path,
    piece_infos,
    output_dir,
    beam_size=1,
    temperature=0.0,
    overlap_windows=True,
):
    """Run model + MUSTER on full-piece ASAP sequences using fair conditioning."""
    model, device = load_model(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    aggregate_metrics = {
        "pitch_error_rate": [],
        "missing_note_rate": [],
        "extra_note_rate": [],
        "onset_time_error_rate": [],
        "offset_time_error_rate": [],
        "mean_error_rate": [],
        "voice_error_rate": [],
        "mean_error_rate_with_voice": [],
    }
    per_sequence_metrics = []
    num_successful = 0
    num_failed = 0
    num_cache_hits = 0
    num_cache_misses = 0

    for piece_info in tqdm(piece_infos, desc="Evaluating"):
        control_triplets = piece_info["control_triplets"]
        gt_score_triplets = piece_info["gt_score_triplets"]
        piece_name = piece_info["perf_path"]

        if piece_info.get("cache_hit"):
            num_cache_hits += 1
        else:
            num_cache_misses += 1

        if len(control_triplets) < 1 or len(gt_score_triplets) < 5:
            num_failed += 1
            continue

        try:
            pred_score_triplets, gen_stats = autoregressive_generate_from_controls(
                model,
                control_triplets,
                device,
                prefix_controls=PREFIX_CONTROLS,
                beam_size=beam_size,
                temperature=temperature,
                overlap_windows=overlap_windows,
            )
        except Exception as exc:
            print(f"  {piece_name}: generation failed - {exc}")
            num_failed += 1
            continue

        if len(pred_score_triplets) < 3:
            num_failed += 1
            continue

        safe_name = piece_name.replace("/", "_").replace("\\", "_")
        seq_dir = Path(output_dir) / safe_name
        os.makedirs(seq_dir, exist_ok=True)

        gt_norm = normalize_triplet_times(gt_score_triplets)
        pred_norm = normalize_triplet_times(pred_score_triplets)

        if gen_stats.get("num_slides", 0):
            print(
                f"  {piece_name}: {gen_stats['num_slides']} context slides, "
                f"{len(pred_norm)} predicted notes"
            )

        save_midi(triplets_to_events(gt_norm), str(seq_dir / "ground_truth_score.mid"))
        save_midi(triplets_to_events(pred_norm), str(seq_dir / "output_score.mid"))

        gt_xml = seq_dir / "ground_truth_score.xml"
        pred_xml = seq_dir / "output_score.xml"

        if not triplets_to_musicxml(gt_norm, str(gt_xml), beat_seconds=TARGET_BEAT_INTERVAL):
            num_failed += 1
            continue
        if not triplets_to_musicxml(pred_norm, str(pred_xml), beat_seconds=TARGET_BEAT_INTERVAL):
            num_failed += 1
            continue

        work_dir = seq_dir / "muster_work"
        os.makedirs(work_dir, exist_ok=True)

        metrics = run_muster_evaluation(gt_xml, pred_xml, safe_name, work_dir)
        if not metrics:
            num_failed += 1
            continue

        metrics["piece"] = piece_name
        metrics["num_gt_notes"] = len(gt_score_triplets)
        metrics["num_pred_notes"] = len(pred_score_triplets)
        metrics["total_performance_notes"] = len(control_triplets)
        metrics["num_controls_used"] = gen_stats["num_controls_used"]
        metrics["prefix_controls_used"] = gen_stats["prefix_controls_used"]
        metrics["score_start_idx"] = gen_stats["score_start_idx"]
        metrics["num_slides"] = gen_stats["num_slides"]
        metrics["num_window_resets"] = gen_stats["num_window_resets"]
        metrics["beam_log_prob"] = gen_stats["beam_log_prob"]
        metrics["cache_hit"] = piece_info["cache_hit"]
        metrics["cache_path"] = piece_info["cache_path"]
        metrics["evaluation_protocol"] = "fair_control_driven"
        metrics["all_performance_notes_used"] = (
            gen_stats["num_controls_used"] == len(control_triplets)
        )
        metrics["gt_score_beat_interval_sec"] = TARGET_BEAT_INTERVAL
        metrics["window_mode"] = gen_stats["window_mode"]
        print_piece_muster_metrics(piece_name, metrics, gen_stats)

        per_sequence_metrics.append(metrics)
        for key in aggregate_metrics:
            if key in metrics:
                aggregate_metrics[key].append(metrics[key])

        num_successful += 1
        with open(seq_dir / "muster_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    final = {
        "evaluation_protocol": "fair_control_driven",
        "gt_score_beat_interval_sec": TARGET_BEAT_INTERVAL,
        "all_performance_notes_used": True,
        "window_mode": "half_overlap" if overlap_windows else "no_overlap",
        "num_sequences_evaluated": num_successful,
        "num_sequences_failed": num_failed,
        "num_cache_hits": num_cache_hits,
        "num_cache_misses": num_cache_misses,
    }
    for key, values in aggregate_metrics.items():
        if values:
            final[f"{key}_mean"] = float(np.mean(values))
            final[f"{key}_std"] = float(np.std(values))
            final[f"{key}_min"] = float(np.min(values))
            final[f"{key}_max"] = float(np.max(values))

    with open(Path(output_dir) / "aggregate_muster_stats.json", "w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2)
    with open(Path(output_dir) / "per_sequence_muster_stats.json", "w", encoding="utf-8") as handle:
        json.dump(per_sequence_metrics, handle, indent=2)

    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MUSTER on ASAP pieces with fair performance-only conditioning"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--num-pieces",
        type=int,
        default=DEFAULT_NUM_PIECES,
        help="Number of ASAP pieces to sample (default: 30)",
    )
    parser.add_argument(
        "--split-file",
        default=SPLIT_FILE,
        help="Path to combined_split.txt for test-split filtering",
    )
    parser.add_argument(
        "--all-pieces",
        action="store_true",
        help="Use all ASAP pieces (train+test), not just test split",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Preprocessing worker count (default: {NUM_WORKERS})",
    )
    parser.add_argument("--beam", type=int, default=1, metavar="BEAM_SIZE")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--no-overlap",
        action="store_true",
        help="Use disjoint control windows instead of the default half-overlap slides",
    )
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    check_muster_installation()

    print(f"Loading ASAP metadata from {ASAP_META_CSV}...")
    all_pieces = load_asap_metadata()
    print(f"  {len(all_pieces)} valid ASAP pieces found")

    if not args.all_pieces:
        test_perfs = load_asap_test_perfs(args.split_file)
        if test_perfs:
            filtered = [piece for piece in all_pieces if piece["perf_path"] in test_perfs]
            print(f"  {len(filtered)} in TEST split of {args.split_file}")
            all_pieces = filtered if filtered else all_pieces
        else:
            print(f"  Warning: split file not found or unreadable; using all ASAP pieces")

    random.seed(RANDOM_SEED)
    sampled = (
        random.sample(all_pieces, args.num_pieces)
        if args.num_pieces < len(all_pieces)
        else all_pieces
    )
    print(f"  Sampled {len(sampled)} pieces\n")

    print(
        f"Preprocessing with {args.workers} workers "
        f"(fresh from ASAP files; cache at {CACHE_DIR})..."
    )
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(preprocess_asap_piece, sampled),
            total=len(sampled),
            desc="Preprocessing",
        ))

    piece_infos = []
    num_ok = 0
    num_failed = 0
    for result in results:
        if result.get("error"):
            print(f"  {result['perf_path']}: preprocessing failed - {result['error']}")
            num_failed += 1
            continue
        piece_infos.append(result)
        num_ok += 1

    print(f"  Preprocessed: {num_ok} ok, {num_failed} failed\n")

    if not piece_infos:
        print("ERROR: no ASAP pieces were successfully preprocessed.")
        sys.exit(1)

    mode_suffix = f"_beam{args.beam}" if args.beam > 1 else ""
    temp_suffix = f"_temp{args.temperature}" if args.temperature > 0 else ""
    overlap_suffix = "_nooverlap" if args.no_overlap else ""
    subdir = f"{args.checkpoint}_asap_fair{mode_suffix}{temp_suffix}{overlap_suffix}"
    output_dir = str(Path(OUTPUT_BASE) / subdir)
    os.makedirs(output_dir, exist_ok=True)

    with open(Path(output_dir) / "sampled_pieces.json", "w", encoding="utf-8") as handle:
        json.dump([
            {
                "piece": piece["perf_path"],
                "cache_hit": piece["cache_hit"],
                "cache_path": piece["cache_path"],
            }
            for piece in piece_infos
        ], handle, indent=2)

    print("Running fair ASAP model + MUSTER evaluation...")
    stats = evaluate_asap_muster(
        args.checkpoint,
        piece_infos,
        output_dir,
        beam_size=args.beam,
        temperature=args.temperature,
        overlap_windows=not args.no_overlap,
    )

    print_muster_summary(args.checkpoint, stats)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
