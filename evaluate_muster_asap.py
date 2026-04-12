"""
Fair MUSTER evaluation on ASAP pieces using performance-only conditioning.
"""

from __future__ import annotations

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

from anticipation.config import CONTEXT_SIZE, EVENT_SIZE, TIME_RESOLUTION
from anticipation.convert import midi_to_events
from anticipation.vocab import (
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    CONTROL_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    REST,
    TIME_OFFSET,
)
from alignment import load_annotation_file
from evaluate_muster import (
    OUTPUT_BASE,
    check_muster_installation,
    guess_default_checkpoint,
    load_model,
    normalize_triplet_times,
    print_muster_summary,
    run_muster_evaluation,
    save_midi,
    triplets_to_events,
    triplets_to_musicxml,
)


warnings.filterwarnings("ignore", category=UserWarning)

ASAP_PATH = "asap-dataset-master"
ASAP_META_CSV = os.path.join(ASAP_PATH, "metadata.csv")
SPLIT_FILE = "data/normalized_split.txt"
CACHE_DIR = Path("data") / "asap_muster_cache"
PREPROCESS_VERSION = "fair_asap_muster_v2"
DEFAULT_CHECKPOINT = "checkpoint-2000"
DEFAULT_NUM_PIECES = 30
RANDOM_SEED = 42
NUM_WORKERS = os.cpu_count() or 1
TARGET_BEAT_INTERVAL = 0.5
PREFIX_CONTROLS = 33
PACKED_SEQUENCE_LENGTH = CONTEXT_SIZE - 4


def event_tokens_to_triplets(events):
    return [events[i : i + 3] for i in range(0, len(events), 3) if i + 2 < len(events)]


def normalize_control_triplets(control_triplets):
    if not control_triplets:
        return []
    min_time = min(t[0] - ATIME_OFFSET for t in control_triplets)
    return [[t[0] - min_time, t[1], t[2]] for t in control_triplets]


def normalize_score_triplets_to_fixed_beat(
    raw_score_triplets,
    score_beat_times,
    target_beat_interval=TARGET_BEAT_INTERVAL,
):
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
                    (orig_time_sec - score_beat_times[0]) / beat_dur if beat_dur > 0 else 0.0
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
                        (orig_time_sec - score_beat_times[-1]) / last_dur if last_dur > 0 else 0.0
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
        normalized.append(
            [norm_time_units + TIME_OFFSET, norm_dur_units + DUR_OFFSET, pitch]
        )

    normalized.sort(key=lambda t: (t[0], t[2], t[1]))
    return normalize_triplet_times(normalized)


def build_performance_control_triplets(perf_midi):
    perf_events = event_tokens_to_triplets(midi_to_events(perf_midi, quantize=False))
    controls = []
    for time_tok, dur_tok, pitch_tok in perf_events:
        time_units = max(0, round(time_tok - TIME_OFFSET))
        dur_units = max(0, round(dur_tok - DUR_OFFSET))
        pitch_units = int(round(pitch_tok - NOTE_OFFSET))
        controls.append(
            [
                ATIME_OFFSET + time_units,
                ADUR_OFFSET + dur_units,
                ANOTE_OFFSET + pitch_units,
            ]
        )
    return normalize_control_triplets(controls)


def build_full_normalized_score_triplets(score_midi, score_beats):
    raw_score_triplets = event_tokens_to_triplets(midi_to_events(score_midi, quantize=False))
    score_annotations = load_annotation_file(score_beats)
    score_beat_times = [annotation[0] for annotation in score_annotations]
    return normalize_score_triplets_to_fixed_beat(raw_score_triplets, score_beat_times)


def build_prefix_header(control_triplets, prefix_controls=PREFIX_CONTROLS):
    k = min(prefix_controls, len(control_triplets))
    header = []
    for control_triplet in control_triplets[:k]:
        header.extend(control_triplet)
        ctrl_time = max(0, control_triplet[0] - ATIME_OFFSET)
        header.extend([TIME_OFFSET + ctrl_time, DUR_OFFSET + 0, REST])
    return header, k


def control_time_units(control_triplet):
    return max(0, control_triplet[0] - ATIME_OFFSET)


def localize_control_triplet(control_triplet, time_offset):
    local_time = max(0, control_time_units(control_triplet) - time_offset)
    return [ATIME_OFFSET + local_time, control_triplet[1], control_triplet[2]]


def initialize_generation_window(control_triplets, window_start_idx):
    if window_start_idx >= len(control_triplets):
        return [], 0, len(control_triplets), 0

    time_offset = control_time_units(control_triplets[window_start_idx])
    localized_prefix_controls = [
        localize_control_triplet(control_triplet, time_offset)
        for control_triplet in control_triplets[window_start_idx : window_start_idx + PREFIX_CONTROLS]
    ]
    header, prefix_count = build_prefix_header(localized_prefix_controls)
    future_idx = window_start_idx + prefix_count
    return header, prefix_count, future_idx, time_offset


def max_predicted_score_onset_units(pred_score_triplets):
    if not pred_score_triplets:
        return 0
    return max(max(0, triplet[0] - TIME_OFFSET) for triplet in pred_score_triplets)


def _file_fingerprint(path):
    stat = os.stat(path)
    return {
        "path": str(Path(path).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_piece_fingerprint(piece_info):
    return {
        "version": PREPROCESS_VERSION,
        "target_beat_interval": TARGET_BEAT_INTERVAL,
        "prefix_controls": PREFIX_CONTROLS,
        "perf_midi": _file_fingerprint(piece_info["perf_midi"]),
        "score_midi": _file_fingerprint(piece_info["score_midi"]),
        "score_beats": _file_fingerprint(piece_info["score_beats"]),
    }


def cache_path_for_piece(piece_info):
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

        write_piece_cache(
            cache_path,
            {
                "fingerprint": fingerprint,
                "control_triplets": control_triplets,
                "gt_score_triplets": gt_score_triplets,
            },
        )

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


def load_asap_metadata():
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
            pieces.append(
                {
                    "perf_path": row["midi_performance"],
                    "perf_midi": perf_midi,
                    "score_midi": score_midi,
                    "score_beats": score_beats,
                }
            )
    return pieces


def load_asap_test_perfs(split_file):
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


def autoregressive_generate_from_controls(
    model,
    control_triplets,
    device,
    temperature=0.0,
):
    if not control_triplets:
        return [], {
            "num_window_resets": 0,
            "num_controls_used": 0,
            "total_performance_notes": 0,
            "prefix_controls_used": 0,
            "score_start_idx": 0,
            "window_mode": "reset",
            "window_segments": [],
        }

    vocab_size = model.config.vocab_size
    header, prefix_count, future_idx, control_time_offset = initialize_generation_window(
        control_triplets,
        window_start_idx=0,
    )
    score_start_idx = len(header)
    context = list(header)
    pred_score_triplets = []
    stats = {
        "num_window_resets": 0,
        "num_controls_used": len(control_triplets),
        "total_performance_notes": len(control_triplets),
        "prefix_controls_used": prefix_count,
        "score_start_idx": score_start_idx,
        "window_mode": "reset",
        "window_segments": [
            {
                "window_index": 0,
                "control_start_idx": 0,
                "control_end_idx": None,
                "pred_start_idx": 0,
                "pred_end_idx": None,
                "prefix_controls_used": prefix_count,
                "control_time_offset": control_time_offset,
            }
        ],
    }

    past = None
    next_logits = None
    note_idx = 0
    score_time_offset = 0

    def clamp_tokens(tokens):
        return [min(max(int(token), 0), vocab_size - 1) for token in tokens]

    def prime():
        nonlocal past, next_logits
        with torch.no_grad():
            out = model(torch.tensor([clamp_tokens(context)], device=device), use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[0, -1, :]

    def feed(new_tokens):
        nonlocal past, next_logits
        with torch.no_grad():
            out = model(
                torch.tensor([clamp_tokens(new_tokens)], device=device),
                past_key_values=past,
                use_cache=True,
            )
        past = out.past_key_values
        next_logits = out.logits[0, -1, :]

    def ensure_primed():
        if past is None:
            prime()

    def decode_slot(start, end):
        ensure_primed()
        logits = next_logits[start:end]
        if temperature > 0:
            logits = logits / temperature
            rel_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        else:
            rel_token = logits.argmax().item()
        token = start + rel_token
        context.append(token)
        feed([token])
        return token

    while note_idx < len(control_triplets):
        time_tok = decode_slot(TIME_OFFSET, DUR_OFFSET)
        dur_tok = decode_slot(DUR_OFFSET, NOTE_OFFSET)
        pitch_tok = decode_slot(NOTE_OFFSET, REST)
        pred_score_triplets.append(
            [time_tok + score_time_offset, dur_tok, pitch_tok]
        )
        note_idx += 1

        if future_idx < len(control_triplets):
            control_triplet = localize_control_triplet(
                control_triplets[future_idx],
                control_time_offset,
            )
            context.extend(control_triplet)
            if past is not None:
                feed(control_triplet)
            future_idx += 1

        if len(context) >= PACKED_SEQUENCE_LENGTH and note_idx < len(control_triplets):
            stats["window_segments"][-1]["control_end_idx"] = note_idx
            stats["window_segments"][-1]["pred_end_idx"] = len(pred_score_triplets)
            header, prefix_count, future_idx, control_time_offset = initialize_generation_window(
                control_triplets,
                window_start_idx=note_idx,
            )
            # Each training window uses a fresh local score timeline, so stitching new
            # windows to the previous predicted note end is overly sensitive to duration
            # errors. Re-anchor using the latest predicted onset instead.
            score_time_offset = max_predicted_score_onset_units(pred_score_triplets)
            context = list(header)
            score_start_idx = len(header)
            past = None
            next_logits = None
            stats["num_window_resets"] += 1
            stats["window_segments"].append(
                {
                    "window_index": len(stats["window_segments"]),
                    "control_start_idx": note_idx,
                    "control_end_idx": None,
                    "pred_start_idx": len(pred_score_triplets),
                    "pred_end_idx": None,
                    "prefix_controls_used": prefix_count,
                    "control_time_offset": control_time_offset,
                }
            )

    stats["window_segments"][-1]["control_end_idx"] = note_idx
    stats["window_segments"][-1]["pred_end_idx"] = len(pred_score_triplets)
    return pred_score_triplets, stats


def format_muster_summary(metrics):
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
    return ", ".join(summary)


def print_piece_muster_metrics(piece_name, metrics, gen_stats):
    summary = [format_muster_summary(metrics)]
    summary.append(f"resets={gen_stats['num_window_resets']}")
    message = f"[MUSTER] {piece_name}: " + ", ".join(summary)
    if sys.stderr.isatty():
        tqdm.write(message, file=sys.stderr)
    else:
        print(message, file=sys.stderr, flush=True)


def print_window_muster_metrics(piece_name, window_metrics):
    message = (
        f"[MUSTER][window {window_metrics['window_index']:03d}] {piece_name}: "
        f"controls={window_metrics['control_start_idx']}:{window_metrics['control_end_idx']}, "
        f"gt_notes={window_metrics['num_gt_notes']}, "
        f"pred_notes={window_metrics['num_pred_notes']}, "
        f"{format_muster_summary(window_metrics)}"
    )
    if sys.stderr.isatty():
        tqdm.write(message, file=sys.stderr)
    else:
        print(message, file=sys.stderr, flush=True)


def print_window_muster_skip(piece_name, window_metrics, reason):
    message = (
        f"[MUSTER][window {window_metrics['window_index']:03d}] {piece_name}: "
        f"controls={window_metrics['control_start_idx']}:{window_metrics['control_end_idx']}, "
        f"gt_notes={window_metrics['num_gt_notes']}, "
        f"pred_notes={window_metrics['num_pred_notes']}, "
        f"skipped ({reason})"
    )
    if sys.stderr.isatty():
        tqdm.write(message, file=sys.stderr)
    else:
        print(message, file=sys.stderr, flush=True)


def evaluate_triplet_slice_with_muster(
    gt_triplets,
    pred_triplets,
    seq_dir,
    output_prefix,
):
    gt_norm = normalize_triplet_times(gt_triplets)
    pred_norm = normalize_triplet_times(pred_triplets)

    save_midi(triplets_to_events(gt_norm), str(seq_dir / "ground_truth_score.mid"))
    save_midi(triplets_to_events(pred_norm), str(seq_dir / "output_score.mid"))

    gt_xml = seq_dir / "ground_truth_score.xml"
    pred_xml = seq_dir / "output_score.xml"
    if not triplets_to_musicxml(gt_norm, str(gt_xml), beat_seconds=TARGET_BEAT_INTERVAL):
        return None
    if not triplets_to_musicxml(pred_norm, str(pred_xml), beat_seconds=TARGET_BEAT_INTERVAL):
        return None

    work_dir = seq_dir / "muster_work"
    os.makedirs(work_dir, exist_ok=True)
    return run_muster_evaluation(gt_xml, pred_xml, output_prefix, work_dir)


def evaluate_asap_muster(
    checkpoint_path,
    piece_infos,
    output_dir,
    config_source,
    temperature=0.0,
):
    model, device = load_model(checkpoint_path, config_source=config_source)
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
                temperature=temperature,
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

        metrics = evaluate_triplet_slice_with_muster(
            gt_score_triplets,
            pred_score_triplets,
            seq_dir,
            safe_name,
        )
        if not metrics:
            num_failed += 1
            continue

        window_metrics = []
        window_metrics_failed = 0
        windows_dir = seq_dir / "windows"
        for segment in gen_stats.get("window_segments", []):
            gt_window = gt_score_triplets[
                segment["control_start_idx"] : segment["control_end_idx"]
            ]
            pred_window = pred_score_triplets[
                segment["pred_start_idx"] : segment["pred_end_idx"]
            ]
            window_record = {
                "window_index": segment["window_index"],
                "control_start_idx": segment["control_start_idx"],
                "control_end_idx": segment["control_end_idx"],
                "pred_start_idx": segment["pred_start_idx"],
                "pred_end_idx": segment["pred_end_idx"],
                "prefix_controls_used": segment["prefix_controls_used"],
                "control_time_offset": segment["control_time_offset"],
                "num_gt_notes": len(gt_window),
                "num_pred_notes": len(pred_window),
            }
            if len(gt_window) < 3 or len(pred_window) < 3:
                window_record["status"] = "skipped_too_short"
                window_metrics.append(window_record)
                print_window_muster_skip(piece_name, window_record, "too few notes")
                continue

            window_dir = windows_dir / f"window_{segment['window_index']:03d}"
            os.makedirs(window_dir, exist_ok=True)
            window_safe_name = f"{safe_name}_window_{segment['window_index']:03d}"
            window_result = evaluate_triplet_slice_with_muster(
                gt_window,
                pred_window,
                window_dir,
                window_safe_name,
            )
            if not window_result:
                window_record["status"] = "failed"
                window_metrics_failed += 1
                window_metrics.append(window_record)
                print_window_muster_skip(piece_name, window_record, "muster failed")
                continue

            window_record.update(window_result)
            window_record["status"] = "ok"
            window_metrics.append(window_record)
            print_window_muster_metrics(piece_name, window_record)

        metrics["piece"] = piece_name
        metrics["num_gt_notes"] = len(gt_score_triplets)
        metrics["num_pred_notes"] = len(pred_score_triplets)
        metrics["total_performance_notes"] = len(control_triplets)
        metrics["num_controls_used"] = gen_stats["num_controls_used"]
        metrics["prefix_controls_used"] = gen_stats["prefix_controls_used"]
        metrics["score_start_idx"] = gen_stats["score_start_idx"]
        metrics["num_window_resets"] = gen_stats["num_window_resets"]
        metrics["cache_hit"] = piece_info["cache_hit"]
        metrics["cache_path"] = piece_info["cache_path"]
        metrics["evaluation_protocol"] = "fair_control_driven"
        metrics["all_performance_notes_used"] = (
            gen_stats["num_controls_used"] == len(control_triplets)
        )
        metrics["gt_score_beat_interval_sec"] = TARGET_BEAT_INTERVAL
        metrics["window_mode"] = gen_stats["window_mode"]
        metrics["num_generation_windows"] = len(gen_stats.get("window_segments", []))
        metrics["num_window_muster_successful"] = sum(
            1 for window in window_metrics if window.get("status") == "ok"
        )
        metrics["num_window_muster_failed"] = window_metrics_failed
        metrics["num_window_muster_skipped"] = sum(
            1 for window in window_metrics if window.get("status") == "skipped_too_short"
        )
        print_piece_muster_metrics(piece_name, metrics, gen_stats)

        per_sequence_metrics.append(metrics)
        for key in aggregate_metrics:
            if key in metrics:
                aggregate_metrics[key].append(metrics[key])

        num_successful += 1
        with open(seq_dir / "muster_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        with open(seq_dir / "window_muster_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(window_metrics, handle, indent=2)

    final = {
        "evaluation_protocol": "fair_control_driven",
        "gt_score_beat_interval_sec": TARGET_BEAT_INTERVAL,
        "all_performance_notes_used": True,
        "window_mode": "reset",
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MUSTER on ASAP pieces with fair performance-only conditioning"
    )
    parser.add_argument("--checkpoint", default=guess_default_checkpoint())
    parser.add_argument(
        "--config-source",
        default=DEFAULT_CHECKPOINT,
        help="Fallback config source for checkpoints with only model.safetensors",
    )
    parser.add_argument(
        "--num-pieces",
        type=int,
        default=DEFAULT_NUM_PIECES,
        help="Number of ASAP pieces to sample (default: 30)",
    )
    parser.add_argument(
        "--split-file",
        default=SPLIT_FILE,
        help="Path to normalized_split.txt for test-split filtering",
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
    parser.add_argument("--temperature", type=float, default=0.0)
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
            print("  Warning: split file not found or unreadable; using all ASAP pieces")

    random.seed(RANDOM_SEED)
    sampled = (
        random.sample(all_pieces, args.num_pieces)
        if args.num_pieces < len(all_pieces)
        else all_pieces
    )
    print(f"  Sampled {len(sampled)} pieces\n")

    print(f"Preprocessing with {args.workers} workers (cache at {CACHE_DIR})...")
    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(preprocess_asap_piece, sampled),
                total=len(sampled),
                desc="Preprocessing",
            )
        )

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

    temp_suffix = f"_temp{args.temperature}" if args.temperature > 0 else ""
    subdir = f"{Path(args.checkpoint).name}_asap_fair{temp_suffix}"
    output_dir = str(Path(OUTPUT_BASE) / subdir)
    os.makedirs(output_dir, exist_ok=True)

    with open(Path(output_dir) / "sampled_pieces.json", "w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "piece": piece["perf_path"],
                    "cache_hit": piece["cache_hit"],
                    "cache_path": piece["cache_path"],
                }
                for piece in piece_infos
            ],
            handle,
            indent=2,
        )

    print("Running fair ASAP model + MUSTER evaluation...")
    stats = evaluate_asap_muster(
        args.checkpoint,
        piece_infos,
        output_dir,
        config_source=args.config_source,
        temperature=args.temperature,
    )

    print_muster_summary(args.checkpoint, stats)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
