import json
import os
import shutil
import time
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

import mido
import torch

import evaluate_muster_asap as ema
from anticipation.config import TIME_RESOLUTION
from anticipation.convert import midi_to_events
from anticipation.vocab import (
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    CONTROL_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    REST,
    SPECIAL_OFFSET,
    TIME_OFFSET,
)


def write_midi(path, notes, bpm=60, ticks_per_beat=480):
    """Write a simple single-track MIDI file from (start_beats, dur_beats, pitch)."""
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))

    events = []
    for start_beats, dur_beats, pitch in notes:
        start_tick = round(start_beats * ticks_per_beat)
        end_tick = round((start_beats + dur_beats) * ticks_per_beat)
        events.append((start_tick, True, pitch))
        events.append((end_tick, False, pitch))

    events.sort(key=lambda item: (item[0], 0 if not item[1] else 1, item[2]))
    current_tick = 0
    for tick, is_note_on, pitch in events:
        delta = tick - current_tick
        current_tick = tick
        if is_note_on:
            track.append(mido.Message("note_on", note=pitch, velocity=64, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))

    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(path)


def write_beats(path, beat_times):
    with open(path, "w", encoding="utf-8") as handle:
        for beat_time in beat_times:
            handle.write(f"{beat_time}\t0\tbeat\n")


class WorkspaceTempDir:
    """Create scratch directories inside the repo workspace."""

    def __enter__(self):
        self.path = Path(os.getcwd()) / "test_scratch" / uuid4().hex
        self.path.mkdir(parents=True, exist_ok=False)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb):
        shutil.rmtree(self.path, ignore_errors=True)
        return False


class FakeModel:
    """Minimal autoregressive model stub for control-driven generation tests."""

    def __init__(self):
        self.config = SimpleNamespace(vocab_size=SPECIAL_OFFSET + 32)

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        vocab_size = self.config.vocab_size
        last_token = int(input_ids[0, -1].item())

        if last_token == REST or last_token >= NOTE_OFFSET:
            next_token = TIME_OFFSET + 0
        elif last_token < DUR_OFFSET:
            next_token = DUR_OFFSET + 50
        else:
            next_token = NOTE_OFFSET + 60

        logits = torch.full((1, input_ids.shape[1], vocab_size), -1e9)
        logits[0, -1, next_token] = 0.0

        cached_length = 0 if past_key_values is None else int(past_key_values)
        return SimpleNamespace(
            logits=logits,
            past_key_values=cached_length + int(input_ids.shape[1]),
        )


class EvaluateMusterAsapTests(unittest.TestCase):
    def make_piece(self, root, perf_notes, score_notes, score_bpm=60):
        perf_midi = root / "performance.mid"
        score_midi = root / "score.mid"
        score_beats = root / "score_annotations.txt"

        write_midi(perf_midi, perf_notes, bpm=60)
        write_midi(score_midi, score_notes, bpm=score_bpm)
        max_beat = max(int(start + dur) for start, dur, _ in score_notes) + 2
        write_beats(score_beats, list(range(max_beat)))

        return {
            "perf_path": "Synthetic/Piece/performance.mid",
            "perf_midi": str(perf_midi),
            "score_midi": str(score_midi),
            "score_beats": str(score_beats),
        }

    def test_preprocess_keeps_all_performance_notes(self):
        with WorkspaceTempDir() as tmpdir:
            root = Path(tmpdir)
            piece = self.make_piece(
                root,
                perf_notes=[(0, 1, 60), (1, 1, 62), (2, 1, 64)],
                score_notes=[(0, 1, 60), (1, 1, 62)],
            )
            cache_dir = root / "cache"

            with mock.patch.object(ema, "CACHE_DIR", cache_dir):
                result = ema.preprocess_asap_piece(piece)

            self.assertNotIn("error", result)
            self.assertEqual(len(result["control_triplets"]), 3)
            self.assertEqual(len(result["gt_score_triplets"]), 2)
            self.assertFalse(result["cache_hit"])

    def test_full_score_ground_truth_uses_full_score_midi(self):
        with WorkspaceTempDir() as tmpdir:
            root = Path(tmpdir)
            piece = self.make_piece(
                root,
                perf_notes=[(0, 1, 60)],
                score_notes=[(0, 1, 60), (1, 1, 64), (2, 1, 67)],
            )
            cache_dir = root / "cache"

            with mock.patch.object(ema, "CACHE_DIR", cache_dir):
                result = ema.preprocess_asap_piece(piece)

            self.assertNotIn("error", result)
            self.assertEqual(len(result["gt_score_triplets"]), 3)

    def test_cache_reuses_and_invalidates_when_source_changes(self):
        with WorkspaceTempDir() as tmpdir:
            root = Path(tmpdir)
            piece = self.make_piece(
                root,
                perf_notes=[(0, 1, 60), (1, 1, 62)],
                score_notes=[(0, 1, 60), (1, 1, 62)],
            )
            cache_dir = root / "cache"

            with mock.patch.object(ema, "CACHE_DIR", cache_dir):
                first = ema.preprocess_asap_piece(piece)
                second = ema.preprocess_asap_piece(piece)
                self.assertFalse(first["cache_hit"])
                self.assertTrue(second["cache_hit"])

                before = os.stat(piece["perf_midi"]).st_mtime_ns
                time.sleep(0.02)
                os.utime(piece["perf_midi"], None)
                after = os.stat(piece["perf_midi"]).st_mtime_ns
                self.assertNotEqual(before, after)

                third = ema.preprocess_asap_piece(piece)
                self.assertFalse(third["cache_hit"])

    def test_generator_consumes_all_controls_without_ground_truth_scores(self):
        controls = [
            [ATIME_OFFSET + 0, ADUR_OFFSET + 20, ANOTE_OFFSET + 60],
            [ATIME_OFFSET + 50, ADUR_OFFSET + 20, ANOTE_OFFSET + 62],
            [ATIME_OFFSET + 100, ADUR_OFFSET + 20, ANOTE_OFFSET + 64],
        ]

        pred_score, stats = ema.autoregressive_generate_from_controls(
            FakeModel(),
            controls,
            device="cpu",
            prefix_controls=2,
            beam_size=1,
            temperature=0.0,
        )

        self.assertEqual(len(pred_score), len(controls))
        self.assertEqual(stats["num_controls_used"], len(controls))
        self.assertEqual(stats["total_performance_notes"], len(controls))
        self.assertGreater(stats["score_start_idx"], 0)

    def test_score_normalization_and_export_use_half_second_beats(self):
        with WorkspaceTempDir() as tmpdir:
            root = Path(tmpdir)
            score_midi = root / "score.mid"
            score_beats = root / "score_annotations.txt"
            xml_path = root / "score.xml"

            write_midi(score_midi, [(0, 1, 60)], bpm=60)
            write_beats(score_beats, [0, 1, 2])

            raw_triplets = ema.event_tokens_to_triplets(midi_to_events(str(score_midi), quantize=False))
            normalized = ema.normalize_score_triplets_to_fixed_beat(
                raw_triplets,
                [0, 1, 2],
                target_beat_interval=ema.TARGET_BEAT_INTERVAL,
            )

            expected_bins = round(TIME_RESOLUTION * ema.TARGET_BEAT_INTERVAL)
            self.assertEqual(normalized[0][1] - DUR_OFFSET, expected_bins)
            self.assertTrue(
                ema.triplets_to_musicxml(
                    normalized,
                    str(xml_path),
                    beat_seconds=ema.TARGET_BEAT_INTERVAL,
                )
            )

            tree = ET.parse(xml_path)
            first_duration = int(tree.find(".//note/duration").text)
            self.assertEqual(first_duration, expected_bins)

    def test_end_to_end_smoke_reports_fair_protocol_metadata(self):
        with WorkspaceTempDir() as tmpdir:
            root = Path(tmpdir)
            piece = self.make_piece(
                root,
                perf_notes=[(0, 1, 60), (1, 1, 62), (2, 1, 64), (3, 1, 65), (4, 1, 67)],
                score_notes=[(0, 1, 60), (1, 1, 64), (2, 1, 67), (3, 1, 69), (4, 1, 71)],
            )
            cache_dir = root / "cache"
            output_dir = root / "results"

            with mock.patch.object(ema, "CACHE_DIR", cache_dir):
                piece_info = ema.preprocess_asap_piece(piece)

            fake_metrics = {
                "pitch_error_rate": 0.1,
                "missing_note_rate": 0.2,
                "extra_note_rate": 0.3,
                "onset_time_error_rate": 0.4,
                "offset_time_error_rate": 0.5,
                "mean_error_rate": 0.6,
                "voice_error_rate": 0.7,
                "mean_error_rate_with_voice": 0.8,
            }

            with mock.patch.object(ema, "load_model", return_value=(FakeModel(), "cpu")):
                with mock.patch.object(ema, "save_midi", return_value=None):
                    with mock.patch.object(ema, "run_muster_evaluation", return_value=fake_metrics):
                        stats = ema.evaluate_asap_muster(
                            "dummy-checkpoint",
                            [piece_info],
                            str(output_dir),
                            beam_size=1,
                            temperature=0.0,
                        )

            self.assertEqual(stats["num_sequences_evaluated"], 1)
            self.assertEqual(stats["evaluation_protocol"], "fair_control_driven")
            self.assertEqual(stats["gt_score_beat_interval_sec"], ema.TARGET_BEAT_INTERVAL)

            with open(output_dir / "per_sequence_muster_stats.json", "r", encoding="utf-8") as handle:
                per_sequence = json.load(handle)

            self.assertEqual(len(per_sequence), 1)
            metrics = per_sequence[0]
            self.assertEqual(metrics["evaluation_protocol"], "fair_control_driven")
            self.assertEqual(metrics["gt_score_beat_interval_sec"], ema.TARGET_BEAT_INTERVAL)
            self.assertEqual(metrics["num_controls_used"], metrics["total_performance_notes"])
            self.assertEqual(metrics["num_gt_notes"], len(piece_info["gt_score_triplets"]))


if __name__ == "__main__":
    unittest.main()
