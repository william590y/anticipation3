"""
Shared MUSTER helpers for packed-format evaluation.

This module provides:
  - checkpoint loading
  - MUSTER installation / compilation checks
  - MIDI / MusicXML export helpers
  - MUSTER pipeline execution
  - summary printing
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoConfig, AutoModelForCausalLM

from anticipation.config import TIME_RESOLUTION
from anticipation.convert import events_to_midi
from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, REST, TIME_OFFSET


OUTPUT_BASE = "muster_evaluation_results"
DEFAULT_CONFIG_SOURCE = "checkpoint-2000"
MUSTER_DIR = Path(__file__).parent / "MUSTER"
MUSTER_PROGRAMS = MUSTER_DIR / "Programs"
IS_WINDOWS = sys.platform.startswith("win")
EXE_EXT = ".exe" if IS_WINDOWS else ""


def resolve_muster_program_path(name: str) -> Path:
    candidates = [MUSTER_PROGRAMS / f"{name}{EXE_EXT}"]
    if EXE_EXT:
        candidates.append(MUSTER_PROGRAMS / name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def guess_default_checkpoint() -> str:
    candidates = []
    for candidate in (
        Path("checkpoint-2000"),
        Path("checkpoint-3500"),
        Path("hf-ckpt-3500") / "checkpoint-3500",
    ):
        config_path = candidate / "config.json"
        weight_path = candidate / "model.safetensors"
        if config_path.exists():
            candidates.append((config_path.stat().st_mtime, str(candidate)))
        elif weight_path.exists():
            candidates.append((weight_path.stat().st_mtime, str(candidate)))

    if not candidates:
        return DEFAULT_CONFIG_SOURCE

    candidates.sort(reverse=True)
    return candidates[0][1]


def get_muster_exe(name: str) -> str:
    return str(resolve_muster_program_path(name))


def compile_muster_linux() -> bool:
    print("Compiling MUSTER programs for Linux...")
    os.makedirs(MUSTER_PROGRAMS, exist_ok=True)

    code_dir = MUSTER_DIR / "Code"
    if not code_dir.exists():
        print(f"ERROR: MUSTER source directory not found: {code_dir}")
        print("The repository is missing the MUSTER/Code sources needed for Linux compilation.")
        return False

    compilations = [
        ("Fmt3xToSpr_v220118.cpp", "Fmt3xToSpr"),
        ("ScoreMatchEvaluation_VoicePlus_v220118.cpp", "ScoreMatchEvaluation_VoicePlus"),
        ("MusicXMLToFmt3x_v170104.cpp", "MusicXMLToFmt3x"),
        ("MusicXMLToHMM_v170104.cpp", "MusicXMLToHMM"),
        ("ScorePerfmMatcher_v170503.cpp", "ScorePerfmMatcher"),
        ("ErrorDetection_v170503.cpp", "ErrorDetection"),
        ("RealignmentMOHMM_v190326.cpp", "RealignmentMOHMM"),
    ]

    for src, name in compilations:
        src_path = code_dir / src
        out_path = MUSTER_PROGRAMS / name
        if out_path.exists():
            continue
        print(f"  Compiling {name}...")
        result = subprocess.run(
            ["g++", "-O2", str(src_path), "-o", str(out_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr}")
            return False

    print("MUSTER compilation complete.")
    return True


def check_muster_installation() -> bool:
    required_programs = [
        "MusicXMLToFmt3x",
        "MusicXMLToHMM",
        "Fmt3xToSpr",
        "ScorePerfmMatcher",
        "ErrorDetection",
        "RealignmentMOHMM",
        "ScoreMatchEvaluation_VoicePlus",
    ]

    missing = []
    for prog in required_programs:
        exe_path = resolve_muster_program_path(prog)
        if not exe_path.exists():
            missing.append(prog)

    if not missing:
        return True

    if IS_WINDOWS:
        print(f"ERROR: Missing MUSTER programs: {missing}")
        print("Please compile MUSTER first inside the MUSTER folder.")
        sys.exit(1)

    print(f"Missing MUSTER programs: {missing}")
    if not compile_muster_linux():
        print("ERROR: Failed to compile MUSTER programs.")
        sys.exit(1)

    still_missing = [
        prog
        for prog in required_programs
        if not resolve_muster_program_path(prog).exists()
    ]
    if still_missing:
        print(f"ERROR: Still missing after compilation: {still_missing}")
        sys.exit(1)

    return True


def load_model(checkpoint_path: str, config_source: str | None = None):
    checkpoint = Path(checkpoint_path)
    if config_source is None:
        config_source = DEFAULT_CONFIG_SOURCE

    print(f"Loading model from {checkpoint_path}...")
    if (checkpoint / "config.json").exists():
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint),
            local_files_only=True,
        )
    else:
        weight_path = checkpoint / "model.safetensors"
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Could not find config.json or model.safetensors in {checkpoint_path}"
            )
        if not Path(config_source).exists():
            raise FileNotFoundError(
                f"Fallback config source not found: {config_source}"
            )
        config = AutoConfig.from_pretrained(config_source, local_files_only=True)
        model = AutoModelForCausalLM.from_config(config)
        state_dict = load_safetensors(str(weight_path))
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys while loading {checkpoint_path}: {unexpected_keys}"
            )
        allowed_missing = {"lm_head.weight"}
        if set(missing_keys) - allowed_missing:
            raise RuntimeError(
                f"Missing keys while loading {checkpoint_path}: {missing_keys}"
            )
        model.tie_weights()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.config.use_cache = True
    model = model.to(device)
    model.eval()
    print(f"  Model loaded on {device}")
    return model, device


def triplets_to_events(triplets) -> list[int]:
    events = []
    for triplet in triplets:
        events.extend(triplet)
    return events


def normalize_triplet_times(triplets):
    if not triplets:
        return []
    triplets = sorted(triplets, key=lambda t: t[0])
    min_time = min(t[0] - TIME_OFFSET for t in triplets)
    return [[t[0] - min_time, t[1], t[2]] for t in triplets]


def triplets_to_musicxml(triplets, xml_path, beat_seconds=0.5):
    """
    Convert score triplets directly to single-part MusicXML using raw ElementTree.
    """
    try:
        from xml.etree.ElementTree import Element, ElementTree, SubElement, indent

        bins_per_second = TIME_RESOLUTION
        bins_per_quarter = max(1, round(bins_per_second * beat_seconds))
        divisions = bins_per_quarter
        bins_per_measure = bins_per_quarter * 4

        notes = []
        for triplet in triplets:
            onset = triplet[0] - TIME_OFFSET
            dur = triplet[1] - DUR_OFFSET
            pitch = triplet[2] - NOTE_OFFSET
            if pitch < 0 or pitch > 127:
                continue
            if dur <= 0:
                dur = 1
            notes.append((onset, dur, pitch))

        if not notes:
            return False

        notes.sort(key=lambda item: item[0])
        total_bins = max(onset + dur for onset, dur, _ in notes)
        num_measures = max(1, (total_bins + bins_per_measure - 1) // bins_per_measure)

        def midi_to_pitch_elements(parent, midi_pitch):
            pitch_el = SubElement(parent, "pitch")
            pc = midi_pitch % 12
            octave = midi_pitch // 12 - 1
            names = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
            alters = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
            SubElement(pitch_el, "step").text = names[pc]
            if alters[pc]:
                SubElement(pitch_el, "alter").text = "1"
            SubElement(pitch_el, "octave").text = str(octave)

        def dur_to_type(dur_bins):
            quarter = divisions
            mapping = [
                (quarter * 8, "breve"),
                (quarter * 4, "whole"),
                (quarter * 2, "half"),
                (quarter, "quarter"),
                (quarter // 2, "eighth"),
                (quarter // 4, "16th"),
                (quarter // 8, "32nd"),
            ]
            best_type, best_dur = "quarter", quarter
            best_dist = abs(dur_bins - quarter)
            for mapped_dur, note_type in mapping:
                if mapped_dur > 0 and abs(dur_bins - mapped_dur) < best_dist:
                    best_dist = abs(dur_bins - mapped_dur)
                    best_type = note_type
                    best_dur = mapped_dur
            return best_type, best_dur

        def emit_forward(parent, duration_bins):
            if duration_bins <= 0:
                return
            forward_el = SubElement(parent, "forward")
            SubElement(forward_el, "duration").text = str(duration_bins)

        root = Element("score-partwise", version="3.0")
        part_list = SubElement(root, "part-list")
        score_part = SubElement(part_list, "score-part", id="P1")
        SubElement(score_part, "part-name").text = "Piano"
        part_el = SubElement(root, "part", id="P1")

        for measure_idx in range(num_measures):
            measure_start = measure_idx * bins_per_measure
            measure_end = measure_start + bins_per_measure
            measure_el = SubElement(part_el, "measure", number=str(measure_idx + 1))

            if measure_idx == 0:
                attrs = SubElement(measure_el, "attributes")
                SubElement(attrs, "divisions").text = str(divisions)
                key_el = SubElement(attrs, "key")
                SubElement(key_el, "fifths").text = "0"
                time_el = SubElement(attrs, "time")
                SubElement(time_el, "beats").text = "4"
                SubElement(time_el, "beat-type").text = "4"
                clef_el = SubElement(attrs, "clef")
                SubElement(clef_el, "sign").text = "G"
                SubElement(clef_el, "line").text = "2"

            measure_notes = [
                (onset - measure_start, dur, pitch)
                for onset, dur, pitch in notes
                if measure_start <= onset < measure_end
            ]

            cursor = 0
            note_pos = 0
            while note_pos < len(measure_notes):
                onset = measure_notes[note_pos][0]
                if onset > cursor:
                    emit_forward(measure_el, onset - cursor)
                    cursor = onset

                chord_notes = []
                while note_pos < len(measure_notes) and measure_notes[note_pos][0] == onset:
                    chord_notes.append(measure_notes[note_pos])
                    note_pos += 1

                group_advance = 0
                for chord_idx, (_, dur, pitch) in enumerate(chord_notes):
                    note_el = SubElement(measure_el, "note")
                    if chord_idx > 0:
                        SubElement(note_el, "chord")
                    midi_to_pitch_elements(note_el, pitch)
                    note_type, clamped_dur = dur_to_type(dur)
                    SubElement(note_el, "duration").text = str(clamped_dur)
                    SubElement(note_el, "type").text = note_type
                    group_advance = max(group_advance, clamped_dur)

                cursor = max(cursor, onset + group_advance)

            if not measure_notes:
                rest_el = SubElement(measure_el, "note")
                SubElement(rest_el, "rest")
                SubElement(rest_el, "duration").text = str(divisions * 4)
                SubElement(rest_el, "type").text = "whole"

        tree = ElementTree(root)
        try:
            indent(tree, space="  ")
        except TypeError:
            pass

        header = (
            b'<?xml version="1.0" encoding="UTF-8"?>\n'
            b'<!DOCTYPE score-partwise PUBLIC '
            b'"-//Recordare//DTD MusicXML 3.0 Partwise//EN" '
            b'"http://www.musicxml.org/dtds/partwise.dtd">\n'
        )
        with open(xml_path, "wb") as handle:
            handle.write(header)
            tree.write(handle, encoding="utf-8", xml_declaration=False)

        return True
    except Exception as exc:
        print(f"    Warning: Could not create MusicXML from triplets: {exc}")
        import traceback

        traceback.print_exc()
        return False


def save_midi(events, filepath):
    try:
        midi_obj = events_to_midi(events)
        midi_obj.save(filepath)
        return True
    except Exception as exc:
        import traceback

        print(f"    Warning: Could not save {filepath}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def run_muster_evaluation(gt_xml_path, pred_xml_path, output_prefix, work_dir):
    try:
        gt_name = "gt"
        pred_name = "est"

        gt_work = work_dir / f"{gt_name}.xml"
        pred_work = work_dir / f"{pred_name}.xml"
        shutil.copy(gt_xml_path, gt_work)
        shutil.copy(pred_xml_path, pred_work)

        if not gt_work.exists():
            print(f"    GT file not copied: {gt_work}")
            return None
        if not pred_work.exists():
            print(f"    EST file not copied: {pred_work}")
            return None

        steps = [
            (("MusicXMLToFmt3x", f"{pred_name}.xml", f"{pred_name}_fmt3x.txt"), "Step 1 failed (MusicXMLToFmt3x EST)"),
            (("Fmt3xToSpr", f"{pred_name}_fmt3x.txt", f"{pred_name}_spr.txt"), "Step 2 failed (Fmt3xToSpr)"),
            (("MusicXMLToHMM", f"{gt_name}.xml", f"{gt_name}_hmm.txt"), "Step 3 failed (MusicXMLToHMM GT)"),
            (("MusicXMLToFmt3x", f"{gt_name}.xml", f"{gt_name}_fmt3x.txt"), "Step 4 failed (MusicXMLToFmt3x GT)"),
            (("ScorePerfmMatcher", f"{gt_name}_hmm.txt", f"{pred_name}_spr.txt", f"{pred_name}_pre_match.txt", "0.01"), "Step 5 failed (ScorePerfmMatcher)"),
            (("ErrorDetection", f"{gt_name}_fmt3x.txt", f"{gt_name}_hmm.txt", f"{pred_name}_pre_match.txt", f"{pred_name}_err_match.txt"), "Step 6 failed (ErrorDetection)"),
            (("RealignmentMOHMM", f"{gt_name}_fmt3x.txt", f"{gt_name}_hmm.txt", f"{pred_name}_err_match.txt", f"{pred_name}_auto_match.txt", "0.3"), "Step 7 failed (RealignmentMOHMM)"),
        ]

        for command, error_label in steps:
            program = get_muster_exe(command[0])
            result = subprocess.run(
                [program, *command[1:]],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"    {error_label}: {result.stderr}")
                return None

        result = subprocess.run(
            [
                get_muster_exe("ScoreMatchEvaluation_VoicePlus"),
                f"{gt_name}_fmt3x.txt",
                f"{pred_name}_fmt3x.txt",
                f"{pred_name}_auto_match.txt",
                f"{pred_name}_err_detail.txt",
                "-1",
            ],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"    Step 8 failed (ScoreMatchEvaluation): {result.stderr}")
            return None

        output_line = result.stdout.strip()
        if output_line and ":" in output_line:
            values = output_line.split(":")[-1].strip().split()
            if len(values) >= 8:
                metrics = {
                    "pitch_error_rate": float(values[0]),
                    "missing_note_rate": float(values[1]),
                    "extra_note_rate": float(values[2]),
                    "onset_time_error_rate": float(values[3]),
                    "offset_time_error_rate": float(values[4]),
                    "mean_error_rate": float(values[5]),
                    "voice_error_rate": float(values[6]),
                    "mean_error_rate_with_voice": float(values[7]),
                }
                if len(values) >= 13:
                    metrics.update(
                        {
                            "voice_precision": float(values[8]),
                            "voice_recall": float(values[9]),
                            "voice_f_measure": float(values[10]),
                            "note_value_scale_error": float(values[11]),
                            "hand_error_rate": float(values[12]),
                        }
                    )
                with open(work_dir / f"{output_prefix}_ER.txt", "w", encoding="utf-8") as handle:
                    handle.write(output_line)
                return metrics

        preview = output_line[:100] if output_line else "(empty)"
        print(f"    Warning: Could not parse MUSTER output: {preview}")
        return None
    except Exception as exc:
        print(f"    Warning: Error processing {pred_xml_path}: {exc}")
        import traceback

        traceback.print_exc()
        return None


def print_muster_summary(checkpoint_name, stats):
    print(f"\n{'=' * 70}")
    print(f"MUSTER Results for {checkpoint_name}")
    print(f"{'=' * 70}")
    print(f"  Sequences evaluated: {stats['num_sequences_evaluated']}")
    print(f"  Sequences failed: {stats['num_sequences_failed']}")
    print()

    metrics_to_show = [
        ("pitch_error_rate", "Pitch Error Rate (PER)"),
        ("missing_note_rate", "Missing Note Rate (MNR)"),
        ("extra_note_rate", "Extra Note Rate (ENR)"),
        ("onset_time_error_rate", "Onset Time Error Rate (OTER)"),
        ("offset_time_error_rate", "Offset Time Error Rate (OFTER)"),
        ("mean_error_rate", "Mean Error Rate (MER)"),
        ("voice_error_rate", "Voice Error Rate (VER)"),
        ("mean_error_rate_with_voice", "Mean with Voice (MER+V)"),
    ]

    print("  MUSTER Metrics (lower is better):")
    print("  " + "-" * 50)
    for key, name in metrics_to_show:
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        if mean_key in stats:
            print(f"  {name:<30} {stats[mean_key]:>8.2f}% (+-{stats[std_key]:.2f})")
    print()
