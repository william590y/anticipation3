"""
Fair MUSTER evaluation on ASAP pieces using performance-only conditioning.
"""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import os
import random
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from anticipation.asap_aligned_stream import (
    build_full_normalized_score_triplets as build_full_raw_score_triplets,
    build_full_normalized_score_triplets_from_xml as build_full_xml_score_triplets,
    build_raw_performance_control_triplets,
)
from anticipation.config import CONTEXT_SIZE, TIME_RESOLUTION
from anticipation.packed_sequence import PREFIX_CONTROLS, dummy_rest_triplet
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
PREPROCESS_VERSION = "fair_asap_muster_v8_prefix32_window_zero"
DEFAULT_CHECKPOINT = "checkpoint-2000"
DEFAULT_NUM_PIECES = 30
RANDOM_SEED = 42
NUM_WORKERS = os.cpu_count() or 1
TARGET_BEAT_INTERVAL = 0.5
PACKED_SEQUENCE_LENGTH = CONTEXT_SIZE - 4
SCORE_TOKEN_TYPES = ("time", "dur", "pitch")
SCORE_TOKEN_PLOT_COLORS = {
    "time": "tab:blue",
    "dur": "tab:orange",
    "pitch": "tab:green",
}
SCORE_TOKEN_PERPLEXITY_PLOT = "generated_score_token_perplexity.png"
SCORE_TOKEN_PERPLEXITY_TRACE = "generated_score_token_perplexity.json"


def normalize_control_triplets(control_triplets):
    if not control_triplets:
        return []
    min_time = min(t[0] - ATIME_OFFSET for t in control_triplets)
    return [[t[0] - min_time, t[1], t[2]] for t in control_triplets]


def build_performance_control_triplets(perf_midi):
    return normalize_control_triplets(build_raw_performance_control_triplets(perf_midi))

def build_full_xml_gt_score_triplets(score_xml):
    try:
        return build_full_xml_score_triplets(
            score_xml,
            target_beat_interval=TARGET_BEAT_INTERVAL,
            require_exact_grid=True,
        ), "xml_score_exact"
    except ValueError:
        return build_full_xml_score_triplets(
            score_xml,
            target_beat_interval=TARGET_BEAT_INTERVAL,
        ), "xml_score_rounded_to_grid"


def build_full_normalized_score_triplets(
    score_midi,
    score_beats,
    score_xml=None,
    score_source="midi",
):
    if score_source == "midi":
        return build_full_raw_score_triplets(
            score_midi,
            score_beats,
            target_beat_interval=TARGET_BEAT_INTERVAL,
        ), "midi_score_annotations"

    if score_source == "xml":
        if not score_xml or not os.path.exists(score_xml):
            raise FileNotFoundError("requested XML GT score source, but xml_score.musicxml is missing")
        return build_full_xml_gt_score_triplets(score_xml)

    if score_source == "auto":
        if score_xml and os.path.exists(score_xml):
            return build_full_xml_gt_score_triplets(score_xml)
        return build_full_raw_score_triplets(
            score_midi,
            score_beats,
            target_beat_interval=TARGET_BEAT_INTERVAL,
        ), "midi_score_annotations"

    raise ValueError(f"unknown GT score source: {score_source}")


def build_prefix_header(control_triplets, prefix_controls=PREFIX_CONTROLS):
    k = min(prefix_controls, len(control_triplets))
    header = []
    for control_triplet in control_triplets[:k]:
        header.extend(control_triplet)
        header.extend(dummy_rest_triplet(0))
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


def min_real_score_onset_units_suffix(gt_score_triplets, start_note_idx):
    """
    Minimum onset (in score time units above TIME_OFFSET) over real score notes
    from start_note_idx onward. Matches tokenize-asap-sliding.py's
    score_suffix_min_times / min_score_time_units convention for body scores.
    """
    if not gt_score_triplets or start_note_idx >= len(gt_score_triplets):
        return 0
    onsets = []
    for t in gt_score_triplets[start_note_idx:]:
        if len(t) < 3:
            continue
        if int(t[2]) == REST:
            continue
        onsets.append(int(t[0]) - TIME_OFFSET)
    return min(onsets) if onsets else 0


def empty_score_token_perplexity_trace():
    return {
        "note_index": [],
        "onset_bins": [],
        "onset_seconds": [],
        "time": [],
        "dur": [],
        "pitch": [],
    }


def summarize_score_token_perplexity_trace(trace):
    summary = {"num_generated_notes": len(trace.get("time", []))}
    for token_type in SCORE_TOKEN_TYPES:
        values = trace.get(token_type, [])
        if not values:
            continue
        summary[f"{token_type}_mean"] = float(np.mean(values))
        summary[f"{token_type}_std"] = float(np.std(values))
        summary[f"{token_type}_min"] = float(np.min(values))
        summary[f"{token_type}_max"] = float(np.max(values))
    return summary


def save_score_token_perplexity_artifacts(seq_dir, piece_name, trace):
    if not trace.get("time"):
        return None

    summary = summarize_score_token_perplexity_trace(trace)
    payload = {
        "piece": piece_name,
        "summary": summary,
        "trace": trace,
    }
    with open(seq_dir / SCORE_TOKEN_PERPLEXITY_TRACE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    use_onset_axis = len(set(trace["onset_seconds"])) > 1
    x_values = np.asarray(
        trace["onset_seconds"] if use_onset_axis else trace["note_index"],
        dtype=float,
    )
    x_label = "Predicted score onset (s)" if use_onset_axis else "Generated score note index"
    if use_onset_axis:
        x_order = np.argsort(x_values, kind="stable")
        x_plot = x_values[x_order]
    else:
        x_order = None
        x_plot = x_values

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for token_type in SCORE_TOKEN_TYPES:
        y_values = np.asarray(trace[token_type], dtype=float)
        if x_order is not None:
            y_plot = y_values[x_order]
        else:
            y_plot = y_values

        ax.scatter(
            x_plot,
            y_plot,
            s=10,
            alpha=0.2,
            color=SCORE_TOKEN_PLOT_COLORS[token_type],
        )
        ax.plot(
            x_plot,
            y_plot,
            linewidth=1.4,
            alpha=0.9,
            label=token_type,
            color=SCORE_TOKEN_PLOT_COLORS[token_type],
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Perplexity")
    ax.set_title(f"Generated score-token perplexity: {Path(piece_name).name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(seq_dir / SCORE_TOKEN_PERPLEXITY_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summary


def _file_fingerprint(path):
    stat = os.stat(path)
    return {
        "path": str(Path(path).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_piece_fingerprint(piece_info, gt_score_source):
    fingerprint = {
        "version": PREPROCESS_VERSION,
        "requested_gt_score_source": gt_score_source,
        "target_beat_interval": TARGET_BEAT_INTERVAL,
        "prefix_controls": PREFIX_CONTROLS,
        "perf_midi": _file_fingerprint(piece_info["perf_midi"]),
        "score_midi": _file_fingerprint(piece_info["score_midi"]),
        "score_beats": _file_fingerprint(piece_info["score_beats"]),
    }
    if (
        gt_score_source in {"xml", "auto"}
        and piece_info.get("score_xml")
        and os.path.exists(piece_info["score_xml"])
    ):
        fingerprint["score_xml"] = _file_fingerprint(piece_info["score_xml"])
    return fingerprint


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


def preprocess_asap_piece(piece_info, gt_score_source="midi"):
    cache_path = cache_path_for_piece(piece_info)
    fingerprint = build_piece_fingerprint(piece_info, gt_score_source)
    cached = load_piece_cache(cache_path)

    if cached and cached.get("fingerprint") == fingerprint:
        return {
            **piece_info,
            "control_triplets": cached.get("control_triplets", []),
            "gt_score_triplets": cached.get("gt_score_triplets", []),
            "gt_score_source": cached.get("gt_score_source", "unknown"),
            "requested_gt_score_source": gt_score_source,
            "cache_hit": True,
            "cache_path": str(cache_path),
        }

    try:
        control_triplets = build_performance_control_triplets(piece_info["perf_midi"])
        gt_score_triplets, gt_score_source = build_full_normalized_score_triplets(
            piece_info["score_midi"],
            piece_info["score_beats"],
            score_xml=piece_info.get("score_xml"),
            score_source=gt_score_source,
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
                "gt_score_source": gt_score_source,
            },
        )

        return {
            **piece_info,
            "control_triplets": control_triplets,
            "gt_score_triplets": gt_score_triplets,
            "gt_score_source": gt_score_source,
            "requested_gt_score_source": gt_score_source,
            "cache_hit": False,
            "cache_path": str(cache_path),
        }
    except Exception as exc:
        return {
            **piece_info,
            "requested_gt_score_source": gt_score_source,
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
        score_xml = None
        if "xml_score" in row.index and isinstance(row["xml_score"], str) and row["xml_score"]:
            score_xml = os.path.join(ASAP_PATH, row["xml_score"])
        if all(os.path.exists(path) for path in [perf_midi, score_midi, score_beats]):
            pieces.append(
                {
                    "perf_path": row["midi_performance"],
                    "perf_midi": perf_midi,
                    "score_midi": score_midi,
                    "score_beats": score_beats,
                    "score_xml": score_xml if score_xml and os.path.exists(score_xml) else None,
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
    gt_score_triplets,
    device,
    temperature=0.0,
    ground_truth_score_notes_to_feed=0,
    rollout_trace: list | None = None,
    max_notes: int | None = None,
):
    if not control_triplets:
        return [], {
            "num_window_resets": 0,
            "num_controls_used": 0,
            "total_performance_notes": 0,
            "prefix_controls_used": 0,
            "score_start_idx": 0,
            "ground_truth_score_notes_fed": 0,
            "generated_score_note_count": 0,
            "score_token_perplexity_trace": empty_score_token_perplexity_trace(),
            "score_token_perplexity_summary": {"num_generated_notes": 0},
            "window_mode": "reset",
        }
    if ground_truth_score_notes_to_feed < 0:
        raise ValueError("--ground-truth-score-notes-to-feed must be non-negative")

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
        "ground_truth_score_notes_fed": 0,
        "generated_score_note_count": 0,
        "score_token_perplexity_trace": empty_score_token_perplexity_trace(),
        "score_token_perplexity_summary": {},
        "window_mode": "reset",
    }

    past = None
    next_logits = None
    note_idx = 0
    score_time_offset = 0
    min_score_time_units = min_real_score_onset_units_suffix(gt_score_triplets, 0)

    def clamp_tokens(tokens):
        return [min(max(int(token), 0), vocab_size - 1) for token in tokens]

    def prime():
        nonlocal past, next_logits
        with torch.no_grad():
            out = model(torch.tensor([clamp_tokens(context)], device=device), use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[0, -1, :]
        if rollout_trace is not None:
            rollout_trace.append(
                {"source": "asap_controls", "event": "after_prime", "n": len(context)}
            )

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
        if rollout_trace is not None:
            base_n = len(context) - len(new_tokens)
            for i, tok in enumerate(new_tokens):
                rollout_trace.append(
                    {
                        "source": "asap_controls",
                        "event": "feed",
                        "token": int(tok),
                        "n": base_n + i + 1,
                    }
                )

    def ensure_primed():
        if past is None:
            prime()

    def decode_slot(start, end):
        ensure_primed()
        logits = next_logits[start:end]
        if temperature > 0:
            logits = logits / temperature
        log_probs = F.log_softmax(logits, dim=-1)
        if temperature > 0:
            rel_token = torch.multinomial(log_probs.exp(), 1).item()
        else:
            rel_token = logits.argmax().item()
        log_prob = log_probs[rel_token].item()
        token = start + rel_token
        context.append(token)
        feed([token])
        return token, log_prob

    while note_idx < len(control_triplets):
        if max_notes is not None and note_idx >= max_notes:
            break
        use_ground_truth_note = (
            note_idx < ground_truth_score_notes_to_feed
            and note_idx < len(gt_score_triplets)
            and stats["num_window_resets"] == 0
        )
        if use_ground_truth_note:
            gt_triplet = gt_score_triplets[note_idx]
            # Match tokenize-asap-sliding.py: body score times use score[0] - min_suffix
            # (min real onset in units from this note index onward). Keeps teacher-forced
            # tokens in-distribution vs bare full-score onset ids when score_time_offset==0.
            raw_time_tok = int(gt_triplet[0]) - score_time_offset
            shifted_time_tok = raw_time_tok - min_score_time_units
            time_tok = min(max(shifted_time_tok, TIME_OFFSET), DUR_OFFSET - 1)
            dur_tok = min(max(int(gt_triplet[1]), DUR_OFFSET), NOTE_OFFSET - 1)
            pitch_tok = min(max(int(gt_triplet[2]), NOTE_OFFSET), CONTROL_OFFSET - 1)

            context.extend([time_tok, dur_tok, pitch_tok])
            ensure_primed()
            feed([time_tok, dur_tok, pitch_tok])
            stats["ground_truth_score_notes_fed"] += 1
            pred_score_triplets.append(
                [int(gt_triplet[0]), int(gt_triplet[1]), int(gt_triplet[2])]
            )
        else:
            time_tok, time_log_prob = decode_slot(TIME_OFFSET, DUR_OFFSET)
            dur_tok, dur_log_prob = decode_slot(DUR_OFFSET, NOTE_OFFSET)
            pitch_tok, pitch_log_prob = decode_slot(NOTE_OFFSET, CONTROL_OFFSET)

            onset_bins = max(0, int(time_tok + score_time_offset - TIME_OFFSET))
            trace = stats["score_token_perplexity_trace"]
            trace["note_index"].append(int(note_idx))
            trace["onset_bins"].append(onset_bins)
            trace["onset_seconds"].append(float(onset_bins / TIME_RESOLUTION))
            trace["time"].append(float(np.exp(-time_log_prob)))
            trace["dur"].append(float(np.exp(-dur_log_prob)))
            trace["pitch"].append(float(np.exp(-pitch_log_prob)))
            stats["generated_score_note_count"] += 1
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
            header, prefix_count, future_idx, control_time_offset = initialize_generation_window(
                control_triplets,
                window_start_idx=note_idx,
            )
            # Each training window uses a fresh local score timeline, so stitching new
            # windows to the previous predicted note end is overly sensitive to duration
            # errors. Re-anchor using the latest predicted onset instead.
            score_time_offset = max_predicted_score_onset_units(pred_score_triplets)
            min_score_time_units = min_real_score_onset_units_suffix(
                gt_score_triplets, note_idx
            )
            context = list(header)
            score_start_idx = len(header)
            past = None
            next_logits = None
            stats["num_window_resets"] += 1
    stats["score_token_perplexity_summary"] = summarize_score_token_perplexity_trace(
        stats["score_token_perplexity_trace"]
    )
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

def evaluate_triplet_slice_with_muster(
    gt_triplets,
    pred_triplets,
    seq_dir,
    output_prefix,
):
    gt_export = [triplet for triplet in gt_triplets if triplet[2] != REST]
    pred_export = [triplet for triplet in pred_triplets if triplet[2] != REST]
    if not gt_export or not pred_export:
        return None

    gt_norm = normalize_triplet_times(gt_export)
    pred_norm = normalize_triplet_times(pred_export)

    save_midi(triplets_to_events(gt_norm), str(seq_dir / "ground_truth_score.mid"))
    save_midi(triplets_to_events(pred_norm), str(seq_dir / "output_score.mid"))

    gt_xml = seq_dir / "ground_truth_score.xml"
    pred_xml = seq_dir / "output_score.xml"
    # Export directly from the normalized triplet grid so MUSTER sees the same
    # onset/duration bins we evaluated, without an extra MIDI round-trip.
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
    ground_truth_score_notes_to_feed=0,
    requested_gt_score_source="midi",
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
                gt_score_triplets,
                device,
                temperature=temperature,
                ground_truth_score_notes_to_feed=ground_truth_score_notes_to_feed,
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

        metrics["piece"] = piece_name
        metrics["num_gt_notes"] = len(gt_score_triplets)
        metrics["num_pred_notes"] = sum(1 for triplet in pred_score_triplets if triplet[2] != REST)
        metrics["num_pred_score_slots"] = len(pred_score_triplets)
        metrics["num_pred_dummy_slots"] = sum(
            1 for triplet in pred_score_triplets if triplet[2] == REST
        )
        metrics["total_performance_notes"] = len(control_triplets)
        metrics["num_controls_used"] = gen_stats["num_controls_used"]
        metrics["prefix_controls_used"] = gen_stats["prefix_controls_used"]
        metrics["score_start_idx"] = gen_stats["score_start_idx"]
        metrics["num_window_resets"] = gen_stats["num_window_resets"]
        metrics["ground_truth_score_notes_fed"] = gen_stats["ground_truth_score_notes_fed"]
        metrics["generated_score_note_count"] = gen_stats["generated_score_note_count"]
        metrics["generated_score_token_perplexity_summary"] = gen_stats[
            "score_token_perplexity_summary"
        ]
        metrics["cache_hit"] = piece_info["cache_hit"]
        metrics["cache_path"] = piece_info["cache_path"]
        metrics["evaluation_protocol"] = "raw_score_control_driven"
        metrics["all_performance_notes_used"] = (
            gen_stats["num_controls_used"] == len(control_triplets)
        )
        metrics["gt_score_beat_interval_sec"] = TARGET_BEAT_INTERVAL
        metrics["gt_score_source"] = piece_info.get("gt_score_source", "unknown")
        metrics["requested_gt_score_source"] = piece_info.get(
            "requested_gt_score_source",
            requested_gt_score_source,
        )
        metrics["window_mode"] = gen_stats["window_mode"]
        if save_score_token_perplexity_artifacts(
            seq_dir,
            piece_name,
            gen_stats["score_token_perplexity_trace"],
        ):
            metrics["score_token_perplexity_plot"] = SCORE_TOKEN_PERPLEXITY_PLOT
            metrics["score_token_perplexity_trace"] = SCORE_TOKEN_PERPLEXITY_TRACE
        print_piece_muster_metrics(piece_name, metrics, gen_stats)

        per_sequence_metrics.append(metrics)
        for key in aggregate_metrics:
            if key in metrics:
                aggregate_metrics[key].append(metrics[key])

        num_successful += 1
        with open(seq_dir / "muster_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

    final = {
        "evaluation_protocol": "raw_score_control_driven",
        "gt_score_beat_interval_sec": TARGET_BEAT_INTERVAL,
        "requested_gt_score_source": requested_gt_score_source,
        "all_performance_notes_used": True,
        "window_mode": "reset",
        "ground_truth_score_notes_to_feed": ground_truth_score_notes_to_feed,
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
        description="Evaluate MUSTER on ASAP pieces with raw-score full-piece conditioning from performance controls"
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
    parser.add_argument(
        "--gt-score-source",
        choices=("midi", "xml", "auto"),
        default="midi",
        help="GT score source before MusicXML export for MUSTER (default: midi)",
    )
    parser.add_argument(
        "--ground-truth-score-notes-to-feed",
        type=int,
        default=1,
        help="Teacher-force this many initial GT score notes in the first window only (default: 1)",
    )
    args = parser.parse_args()
    if args.ground_truth_score_notes_to_feed < 0:
        raise ValueError("--ground-truth-score-notes-to-feed must be non-negative")

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
    preprocess_fn = partial(preprocess_asap_piece, gt_score_source=args.gt_score_source)
    with Pool(processes=args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(preprocess_fn, sampled),
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
    gt_suffix = (
        f"_gt{args.ground_truth_score_notes_to_feed}"
        if args.ground_truth_score_notes_to_feed > 0
        else ""
    )
    score_source_suffix = f"_score-{args.gt_score_source}"
    subdir = f"{Path(args.checkpoint).name}_asap_fair{score_source_suffix}{temp_suffix}{gt_suffix}"
    output_dir = str(Path(OUTPUT_BASE) / subdir)
    os.makedirs(output_dir, exist_ok=True)

    with open(Path(output_dir) / "sampled_pieces.json", "w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "piece": piece["perf_path"],
                    "cache_hit": piece["cache_hit"],
                    "cache_path": piece["cache_path"],
                    "gt_score_source": piece.get("gt_score_source", "unknown"),
                    "requested_gt_score_source": piece.get(
                        "requested_gt_score_source",
                        args.gt_score_source,
                    ),
                }
                for piece in piece_infos
            ],
            handle,
            indent=2,
        )

    print("Running ASAP model + raw-score MUSTER evaluation...")
    stats = evaluate_asap_muster(
        args.checkpoint,
        piece_infos,
        output_dir,
        config_source=args.config_source,
        temperature=args.temperature,
        ground_truth_score_notes_to_feed=args.ground_truth_score_notes_to_feed,
        requested_gt_score_source=args.gt_score_source,
    )

    print_muster_summary(args.checkpoint, stats)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
