"""Shared ASAP preprocessing helpers for performance-anchored tokenization."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

from anticipation.config import TIME_RESOLUTION
from anticipation.convert import midi_to_events
from anticipation.vocab import ADUR_OFFSET, ANOTE_OFFSET, ATIME_OFFSET, DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET
from alignment import align_tokens2, load_annotation_file


TARGET_BEAT_INTERVAL = 0.5
STREAM_CACHE_DIR = Path("data") / "asap_aligned_stream_cache"
# Bump when aligned-stream semantics change (forces cache miss vs data/asap_aligned_stream_cache).
# v3: score durations stay fractional through beat normalization (round once at the
# end) instead of being pre-quantized to 10ms at the piece's original tempo.
STREAM_PREPROCESS_VERSION = "performance_anchored_v3_fractional_durations"


def _path_str(pathlike) -> str:
    return str(Path(pathlike))


def event_tokens_to_triplets(events):
    return [events[i : i + 3] for i in range(0, len(events), 3) if i + 2 < len(events)]


def _bins_per_quarter(target_beat_interval):
    bins_per_quarter = target_beat_interval * TIME_RESOLUTION
    rounded = round(bins_per_quarter)
    if abs(bins_per_quarter - rounded) > 1e-9:
        raise ValueError(
            "target_beat_interval does not map to an integer number of bins per quarter note"
        )
    return int(rounded)


def _quarter_length_to_bins(value, bins_per_quarter):
    return Fraction(value).limit_denominator() * bins_per_quarter


def build_raw_performance_control_triplets(perf_midi):
    perf_events = event_tokens_to_triplets(midi_to_events(_path_str(perf_midi), quantize=False))
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
    return controls


def normalize_score_triplet_to_fixed_beat(
    score_triplet,
    score_beat_times,
    target_beat_interval=TARGET_BEAT_INTERVAL,
):
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
            for idx in range(len(score_beat_times) - 1):
                left = score_beat_times[idx]
                right = score_beat_times[idx + 1]
                if left <= orig_time_sec <= right:
                    beat_dur = right - left
                    progress = (
                        (orig_time_sec - left) / beat_dur if beat_dur > 0 else 0.0
                    )
                    time_scale = target_beat_interval / beat_dur if beat_dur > 0 else 1.0
                    norm_time_sec = idx * target_beat_interval + progress * target_beat_interval
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

    norm_time_units = round(norm_time_sec * TIME_RESOLUTION)
    norm_dur_units = max(0, round(orig_dur_sec * time_scale * TIME_RESOLUTION))
    return [norm_time_units + TIME_OFFSET, norm_dur_units + DUR_OFFSET, pitch]


def normalize_full_score_triplets_to_fixed_beat(
    raw_score_triplets,
    score_beat_times,
    target_beat_interval=TARGET_BEAT_INTERVAL,
):
    return [
        normalize_score_triplet_to_fixed_beat(
            score_triplet,
            score_beat_times,
            target_beat_interval=target_beat_interval,
        )
        for score_triplet in raw_score_triplets
    ]


def build_full_normalized_score_triplets(
    score_midi,
    score_beats,
    target_beat_interval=TARGET_BEAT_INTERVAL,
):
    raw_score_triplets = event_tokens_to_triplets(midi_to_events(_path_str(score_midi), quantize=False))
    score_annotations = load_annotation_file(_path_str(score_beats))
    score_beat_times = [annotation[0] for annotation in score_annotations]
    return normalize_full_score_triplets_to_fixed_beat(
        raw_score_triplets,
        score_beat_times,
        target_beat_interval=target_beat_interval,
    )


def build_full_normalized_score_triplets_from_xml(
    score_xml,
    target_beat_interval=TARGET_BEAT_INTERVAL,
    require_exact_grid=False,
):
    """
    Convert ASAP MusicXML notes to the fixed-beat score grid.

    When ``require_exact_grid`` is True, this raises if any note onset/duration cannot
    be represented exactly on the current token grid.
    """
    from music21 import chord, converter

    bins_per_quarter = _bins_per_quarter(target_beat_interval)
    score = converter.parse(_path_str(score_xml))
    triplets = []

    for element in score.flatten().notes:
        if getattr(element.duration, "isGrace", False):
            continue

        onset_bins = _quarter_length_to_bins(element.offset, bins_per_quarter)
        dur_bins = _quarter_length_to_bins(element.quarterLength, bins_per_quarter)

        if require_exact_grid and (
            onset_bins.denominator != 1 or dur_bins.denominator != 1
        ):
            raise ValueError(
                f"MusicXML note grid is not exactly representable at {bins_per_quarter} bins/quarter"
            )

        onset_units = int(round(float(onset_bins)))
        dur_units = max(0, int(round(float(dur_bins))))
        if dur_units <= 0:
            continue

        pitches = element.pitches if isinstance(element, chord.Chord) else [element.pitch]
        for pitch in pitches:
            triplets.append(
                [TIME_OFFSET + onset_units, DUR_OFFSET + dur_units, NOTE_OFFSET + int(pitch.midi)]
            )

    triplets.sort(key=lambda triplet: (triplet[0], triplet[2], triplet[1]))
    return triplets


def _file_fingerprint(path):
    path_str = _path_str(path)
    stat = os.stat(path_str)
    return {
        "path": str(Path(path_str).resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_stream_fingerprint(
    perf_midi,
    score_midi,
    perf_beats,
    score_beats,
    target_beat_interval=TARGET_BEAT_INTERVAL,
):
    return {
        "version": STREAM_PREPROCESS_VERSION,
        "target_beat_interval": target_beat_interval,
        "perf_midi": _file_fingerprint(perf_midi),
        "score_midi": _file_fingerprint(score_midi),
        "perf_beats": _file_fingerprint(perf_beats),
        "score_beats": _file_fingerprint(score_beats),
    }


def cache_path_for_piece(perf_midi, cache_dir=STREAM_CACHE_DIR):
    key = hashlib.sha1(str(Path(perf_midi).resolve()).encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"{key}.json"


def load_stream_cache(cache_path):
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def write_stream_cache(cache_path, payload):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    os.replace(tmp_path, cache_path)


def build_performance_anchored_stream(
    perf_midi,
    score_midi,
    perf_beats,
    score_beats,
    target_beat_interval=TARGET_BEAT_INTERVAL,
    cache_dir=STREAM_CACHE_DIR,
):
    cache_path = cache_path_for_piece(perf_midi, cache_dir=cache_dir)
    fingerprint = build_stream_fingerprint(
        perf_midi,
        score_midi,
        perf_beats,
        score_beats,
        target_beat_interval=target_beat_interval,
    )
    cached = load_stream_cache(cache_path)
    if cached and cached.get("fingerprint") == fingerprint:
        return cached.get("items", [])

    raw_controls = build_raw_performance_control_triplets(perf_midi)
    matched_tuples = align_tokens2(
        _path_str(perf_midi),
        _path_str(score_midi),
        _path_str(perf_beats),
        _path_str(score_beats),
        skip_Nones=False,
    )
    score_annotations = load_annotation_file(_path_str(score_beats))
    score_beat_times = [annotation[0] for annotation in score_annotations]

    score_by_perf_idx = {}
    for _, perf_idx, score_triplet, _ in matched_tuples:
        if score_triplet[0] is None:
            score_by_perf_idx[perf_idx] = None
        else:
            score_by_perf_idx[perf_idx] = normalize_score_triplet_to_fixed_beat(
                score_triplet,
                score_beat_times,
                target_beat_interval=target_beat_interval,
            )

    items = [
        {
            "control": control_triplet,
            "score": score_by_perf_idx.get(perf_idx),
        }
        for perf_idx, control_triplet in enumerate(raw_controls)
    ]

    write_stream_cache(
        cache_path,
        {
            "fingerprint": fingerprint,
            "items": items,
        },
    )
    return items
