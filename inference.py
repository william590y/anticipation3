"""
Autoregressive packed-format inference on sampled ASAP test windows.

This mirrors the older combined autoregressive evaluator, but defaults to the
packed ASAP-normalized files on current main:

  - seed with the control/rest prefix
  - autoregressively predict score triplets
  - insert the ground-truth following control triplet after each prediction
  - save performance / ground-truth score / predicted score MIDIs
  - report time / duration / pitch / exact-triplet accuracy
  - optionally compute MUSTER scores on predicted vs ground-truth score windows
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from anticipation.config import EVENT_SIZE
from anticipation.convert import events_to_midi
from anticipation.packed_sequence import (
    ALTERNATING_START as PACKED_ALTERNATING_START,
    extract_packed_components,
    is_real_score_triplet,
    is_score_triplet as packed_is_score_triplet,
    iter_score_slot_positions,
    triplet_values,
)
from anticipation.score_constraints import constrain_score_token_logits
from anticipation.vocab import (
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    TIME_OFFSET,
)
from evaluate_muster import (
    check_muster_installation,
    run_muster_evaluation,
    triplets_to_musicxml,
)


TEST_FILE = "data/test_normalized.txt"
OUTPUT_BASE = "autoregressive_inference_results"
DEFAULT_NUM_EXAMPLES = 25
DEFAULT_RANDOM_SEED = 41
DEFAULT_CONFIG_SOURCE = "checkpoint-2000"
K_PREFIX = 33
ALTERNATING_START = K_PREFIX * 2 * EVENT_SIZE
TARGET_BEAT_INTERVAL = 0.5
MUSTER_METRIC_KEYS = (
    "pitch_error_rate",
    "missing_note_rate",
    "extra_note_rate",
    "onset_time_error_rate",
    "offset_time_error_rate",
    "mean_error_rate",
    "voice_error_rate",
    "mean_error_rate_with_voice",
)

assert ALTERNATING_START == PACKED_ALTERNATING_START


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


def checkpoint_label(checkpoint_path: str) -> str:
    parts = [part for part in Path(checkpoint_path).parts if part not in (".", "")]
    if not parts:
        return "checkpoint"
    return "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def load_model(checkpoint_path: str, config_source: str):
    checkpoint = Path(checkpoint_path)
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


def sample_test_lines(
    test_file: str,
    num_examples: int,
    seed: int,
) -> tuple[list[tuple[int, str]], int]:
    if num_examples <= 0:
        raise ValueError("--num-examples must be positive")

    rng = random.Random(seed)
    sampled: list[tuple[int, str]] = []
    total_lines = 0

    with open(test_file, "r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line:
                continue

            if len(sampled) < num_examples:
                sampled.append((idx, line))
            else:
                replace_idx = rng.randrange(total_lines + 1)
                if replace_idx < num_examples:
                    sampled[replace_idx] = (idx, line)
            total_lines += 1

    sampled.sort(key=lambda pair: pair[0])
    return sampled, total_lines


def parse_sequence(line: str) -> list[int]:
    token_str = line.split("|", 1)[0].strip()
    tokens = [int(token) for token in token_str.split()]
    return [max(0, token) for token in tokens]


def is_score_triplet(tokens: list[int], pos: int) -> bool:
    return packed_is_score_triplet(tokens, pos, ALTERNATING_START)


def extract_components(tokens: list[int], score_start_idx: int):
    return extract_packed_components(tokens, score_start_idx, include_dummy_score=False)


def autoregressive_generate_score(
    model,
    tokens: list[int],
    score_start_idx: int,
    device: str,
    constrain_score_tokens: bool = True,
) -> list[int]:
    context = list(tokens[:score_start_idx])

    with torch.inference_mode():
        primed = model(
            torch.tensor([context], device=device),
            use_cache=True,
        )
        past = primed.past_key_values
        next_logits = primed.logits[0, -1, :]

        def feed_token(token: int):
            nonlocal past, next_logits
            out = model(
                torch.tensor([[token]], device=device),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_logits = out.logits[0, -1, :]

        pos = score_start_idx
        while pos + 5 < len(tokens):
            if is_score_triplet(tokens, pos):
                for slot in range(3):
                    logits = next_logits
                    if constrain_score_tokens:
                        logits = constrain_score_token_logits(logits, slot)
                    predicted = int(logits.argmax().item())
                    context.append(predicted)
                    feed_token(predicted)

                pos += 3

                if pos + 2 < len(tokens):
                    for control_token in tokens[pos : pos + 3]:
                        context.append(control_token)
                        feed_token(control_token)
                    pos += 3
            else:
                token = tokens[pos]
                context.append(token)
                feed_token(token)
                pos += 1

    return context


def triplets_to_events(triplets: Iterable[list[int]]) -> list[int]:
    events = []
    for triplet in triplets:
        events.extend(triplet)
    return events


def normalize_triplet_times(triplets: list[list[int]]) -> list[list[int]]:
    if not triplets:
        return []
    triplets = sorted(triplets, key=lambda triplet: triplet[0])
    min_time = min(triplet[0] - TIME_OFFSET for triplet in triplets)
    return [[triplet[0] - min_time, triplet[1], triplet[2]] for triplet in triplets]


def raw_triplets_to_event_triplets(triplets: list[list[int]]) -> list[list[int]]:
    return [
        [triplet[0] + TIME_OFFSET, triplet[1] + DUR_OFFSET, triplet[2] + NOTE_OFFSET]
        for triplet in triplets
    ]


def save_midi(triplets: list[list[int]], filepath: Path) -> bool:
    try:
        midi = events_to_midi(triplets_to_events(triplets))
        midi.save(str(filepath))
        return True
    except Exception as exc:
        print(f"  Warning: could not save {filepath}: {type(exc).__name__}: {exc}")
        return False


def format_muster_summary(metrics: dict[str, float | int]) -> str:
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


def evaluate_triplets_with_muster(
    gt_triplets: list[list[int]],
    pred_triplets: list[list[int]],
    seq_dir: Path,
    output_prefix: str,
) -> tuple[dict[str, float] | None, str | None]:
    gt_xml = seq_dir / "ground_truth_score.xml"
    pred_xml = seq_dir / "output_score.xml"
    if not triplets_to_musicxml(gt_triplets, str(gt_xml), beat_seconds=TARGET_BEAT_INTERVAL):
        return None, "ground-truth MusicXML conversion failed"
    if not triplets_to_musicxml(pred_triplets, str(pred_xml), beat_seconds=TARGET_BEAT_INTERVAL):
        return None, "predicted-score MusicXML conversion failed"

    work_dir = seq_dir / "muster_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_muster_evaluation(gt_xml, pred_xml, output_prefix, work_dir)
    if not metrics:
        return None, "MUSTER evaluation failed"
    return metrics, None


def compute_triplet_accuracy_from_tokens(
    gt_tokens: list[int],
    pred_tokens: list[int],
    score_start_idx: int,
) -> dict[str, float | int]:
    compared = 0
    time_correct = 0
    dur_correct = 0
    pitch_correct = 0
    overall_correct = 0

    if len(pred_tokens) < len(gt_tokens):
        max_len = len(pred_tokens)
    else:
        max_len = len(gt_tokens)

    for pos in iter_score_slot_positions(max_len, score_start_idx):
        if pos + 2 >= len(gt_tokens) or pos + 2 >= len(pred_tokens):
            break
        if not is_real_score_triplet(gt_tokens, pos, score_start_idx):
            continue

        gt_triplet = list(triplet_values(gt_tokens, pos))
        pred_triplet = list(triplet_values(pred_tokens, pos))
        compared += 1

        if gt_triplet[0] == pred_triplet[0]:
            time_correct += 1
        if gt_triplet[1] == pred_triplet[1]:
            dur_correct += 1
        if gt_triplet[2] == pred_triplet[2]:
            pitch_correct += 1
        if gt_triplet == pred_triplet:
            overall_correct += 1

    if compared <= 0:
        raise ValueError("No comparable score triplets found")

    return {
        "time_correct": time_correct,
        "time_total": compared,
        "dur_correct": dur_correct,
        "dur_total": compared,
        "pitch_correct": pitch_correct,
        "pitch_total": compared,
        "overall_correct": overall_correct,
        "overall_total": compared,
        "num_gt_notes": compared,
        "time_accuracy": 100.0 * time_correct / compared,
        "dur_accuracy": 100.0 * dur_correct / compared,
        "pitch_accuracy": 100.0 * pitch_correct / compared,
        "overall_accuracy": 100.0 * overall_correct / compared,
        "autoregressive_accuracy": pitch_correct / compared,
        "autoregressive_accuracy_pct": 100.0 * pitch_correct / compared,
    }


def evaluate_checkpoint(
    checkpoint_path: str,
    config_source: str,
    sampled_lines: list[tuple[int, str]],
    output_dir: Path,
    constrain_score_tokens: bool,
    compute_muster: bool,
) -> dict[str, float | int]:
    model, device = load_model(checkpoint_path, config_source)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = {
        "time_correct": 0,
        "time_total": 0,
        "dur_correct": 0,
        "dur_total": 0,
        "pitch_correct": 0,
        "pitch_total": 0,
        "overall_correct": 0,
        "overall_total": 0,
        "num_sequences_evaluated": 0,
        "num_sequences_failed": 0,
        "muster_enabled": compute_muster,
        "num_sequences_muster_evaluated": 0,
        "num_sequences_muster_failed": 0,
    }
    muster_aggregate = {key: [] for key in MUSTER_METRIC_KEYS}
    per_sequence = []
    failed_sequences = []

    for original_index, line in tqdm(
        sampled_lines,
        desc=f"Evaluating {checkpoint_label(checkpoint_path)}",
    ):
        try:
            tokens = parse_sequence(line)
            if len(tokens) <= ALTERNATING_START:
                raise ValueError("Sequence is shorter than the alternating section start")

            gt_performance_raw, gt_score = extract_components(tokens, ALTERNATING_START)
            if not gt_score:
                raise ValueError("No ground-truth score triplets found")

            predicted_tokens = autoregressive_generate_score(
                model,
                tokens,
                ALTERNATING_START,
                device,
                constrain_score_tokens=constrain_score_tokens,
            )
            _, pred_score = extract_components(predicted_tokens, ALTERNATING_START)
            if not pred_score:
                raise ValueError("No predicted score triplets found")

            metrics = compute_triplet_accuracy_from_tokens(
                tokens,
                predicted_tokens,
                ALTERNATING_START,
            )
            metrics["num_gt_notes"] = len(gt_score)
            metrics["num_pred_notes"] = len(pred_score)
            metrics["original_index"] = original_index

            seq_dir = output_dir / f"sequence_{original_index:07d}"
            seq_dir.mkdir(parents=True, exist_ok=True)

            perf_triplets = normalize_triplet_times(
                raw_triplets_to_event_triplets(gt_performance_raw)
            )
            gt_triplets = normalize_triplet_times(gt_score)
            pred_triplets = normalize_triplet_times(pred_score)

            save_midi(perf_triplets, seq_dir / "input_performance.mid")
            save_midi(gt_triplets, seq_dir / "ground_truth_score.mid")
            save_midi(pred_triplets, seq_dir / "output_score.mid")

            if compute_muster:
                muster_metrics, muster_error = evaluate_triplets_with_muster(
                    gt_triplets,
                    pred_triplets,
                    seq_dir,
                    f"sequence_{original_index:07d}",
                )
                if muster_metrics:
                    metrics.update(muster_metrics)
                    metrics["muster_status"] = "ok"
                    aggregate["num_sequences_muster_evaluated"] += 1
                    for key, values in muster_aggregate.items():
                        if key in muster_metrics:
                            values.append(float(muster_metrics[key]))
                    tqdm.write(
                        f"[MUSTER] sequence_{original_index:07d}: "
                        f"{format_muster_summary(muster_metrics)}"
                    )
                    with open(seq_dir / "muster_metrics.json", "w", encoding="utf-8") as handle:
                        json.dump(muster_metrics, handle, indent=2)
                else:
                    metrics["muster_status"] = "failed"
                    metrics["muster_error"] = muster_error
                    aggregate["num_sequences_muster_failed"] += 1
                    tqdm.write(
                        f"[MUSTER] sequence_{original_index:07d}: failed ({muster_error})"
                    )
                    with open(seq_dir / "muster_metrics.json", "w", encoding="utf-8") as handle:
                        json.dump(
                            {
                                "status": "failed",
                                "error": muster_error,
                            },
                            handle,
                            indent=2,
                        )

            with open(seq_dir / "stats.json", "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)

            per_sequence.append(metrics)
            for key in (
                "time_correct",
                "time_total",
                "dur_correct",
                "dur_total",
                "pitch_correct",
                "pitch_total",
                "overall_correct",
                "overall_total",
            ):
                aggregate[key] += int(metrics[key])
            aggregate["num_sequences_evaluated"] += 1

        except Exception as exc:
            print(f"  Sequence {original_index}: failed - {exc}")
            aggregate["num_sequences_failed"] += 1
            failed_sequences.append(
                {
                    "original_index": original_index,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )

    if aggregate["time_total"] > 0:
        aggregate["time_accuracy"] = 100.0 * aggregate["time_correct"] / aggregate["time_total"]
        aggregate["dur_accuracy"] = 100.0 * aggregate["dur_correct"] / aggregate["dur_total"]
        aggregate["pitch_accuracy"] = (
            100.0 * aggregate["pitch_correct"] / aggregate["pitch_total"]
        )
        aggregate["overall_accuracy"] = (
            100.0 * aggregate["overall_correct"] / aggregate["overall_total"]
        )
        aggregate["autoregressive_accuracy"] = (
            aggregate["pitch_correct"] / aggregate["pitch_total"]
        )
        aggregate["autoregressive_accuracy_pct"] = aggregate["pitch_accuracy"]
    else:
        aggregate["time_accuracy"] = 0.0
        aggregate["dur_accuracy"] = 0.0
        aggregate["pitch_accuracy"] = 0.0
        aggregate["overall_accuracy"] = 0.0
        aggregate["autoregressive_accuracy"] = 0.0
        aggregate["autoregressive_accuracy_pct"] = 0.0

    if compute_muster:
        for key, values in muster_aggregate.items():
            if values:
                aggregate[f"{key}_mean"] = float(np.mean(values))
                aggregate[f"{key}_std"] = float(np.std(values))
                aggregate[f"{key}_min"] = float(np.min(values))
                aggregate[f"{key}_max"] = float(np.max(values))

    if per_sequence:
        aggregate["per_sequence_pitch_accuracy_mean"] = float(
            np.mean([item["pitch_accuracy"] for item in per_sequence])
        )
        aggregate["per_sequence_pitch_accuracy_std"] = float(
            np.std([item["pitch_accuracy"] for item in per_sequence])
        )
    if failed_sequences:
        aggregate["failed_sequences"] = failed_sequences

    with open(output_dir / "per_sequence_stats.json", "w", encoding="utf-8") as handle:
        json.dump(per_sequence, handle, indent=2)
    with open(output_dir / "failed_sequences.json", "w", encoding="utf-8") as handle:
        json.dump(failed_sequences, handle, indent=2)
    with open(output_dir / "aggregate_stats.json", "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    return aggregate


def write_summary(
    output_dir: Path,
    checkpoint_path: str,
    test_file: str,
    total_available: int,
    sampled_lines: list[tuple[int, str]],
    aggregate: dict[str, float | int],
    constrain_score_tokens: bool,
):
    summary_lines = [
        "Packed ASAP Autoregressive Inference",
        "=" * 60,
        f"Checkpoint: {checkpoint_path}",
        f"Test file: {test_file}",
        f"Total sequences available: {total_available}",
        f"Sampled sequences: {len(sampled_lines)}",
        f"Constrained score decoding: {constrain_score_tokens}",
        f"MUSTER scoring enabled: {aggregate.get('muster_enabled', False)}",
        "",
        "Aggregate metrics:",
        f"  Autoregressive accuracy (pitch): {aggregate['autoregressive_accuracy_pct']:.2f}%",
        f"  Time accuracy: {aggregate['time_accuracy']:.2f}%",
        f"  Duration accuracy: {aggregate['dur_accuracy']:.2f}%",
        f"  Exact triplet accuracy: {aggregate['overall_accuracy']:.2f}%",
        f"  Sequences evaluated: {aggregate['num_sequences_evaluated']}",
        f"  Sequences failed: {aggregate['num_sequences_failed']}",
    ]

    if "per_sequence_pitch_accuracy_mean" in aggregate:
        summary_lines.append(
            f"  Mean per-sequence pitch accuracy: "
            f"{aggregate['per_sequence_pitch_accuracy_mean']:.2f}% "
            f"(+-{aggregate['per_sequence_pitch_accuracy_std']:.2f})"
        )

    if aggregate.get("muster_enabled"):
        summary_lines.extend(
            [
                "",
                "Aggregate MUSTER metrics:",
                f"  Windows MUSTER-evaluated: {aggregate['num_sequences_muster_evaluated']}",
                f"  Windows MUSTER-failed: {aggregate['num_sequences_muster_failed']}",
            ]
        )
        if "mean_error_rate_mean" in aggregate:
            summary_lines.extend(
                [
                    f"  MER: {aggregate['mean_error_rate_mean']:.2f}% "
                    f"(+-{aggregate['mean_error_rate_std']:.2f})",
                    f"  PER: {aggregate['pitch_error_rate_mean']:.2f}% "
                    f"(+-{aggregate['pitch_error_rate_std']:.2f})",
                    f"  MNR: {aggregate['missing_note_rate_mean']:.2f}% "
                    f"(+-{aggregate['missing_note_rate_std']:.2f})",
                    f"  ENR: {aggregate['extra_note_rate_mean']:.2f}% "
                    f"(+-{aggregate['extra_note_rate_std']:.2f})",
                    f"  OTER: {aggregate['onset_time_error_rate_mean']:.2f}% "
                    f"(+-{aggregate['onset_time_error_rate_std']:.2f})",
                    f"  OFTER: {aggregate['offset_time_error_rate_mean']:.2f}% "
                    f"(+-{aggregate['offset_time_error_rate_std']:.2f})",
                ]
            )
            if "voice_error_rate_mean" in aggregate:
                summary_lines.append(
                    f"  VER: {aggregate['voice_error_rate_mean']:.2f}% "
                    f"(+-{aggregate['voice_error_rate_std']:.2f})"
                )
            if "mean_error_rate_with_voice_mean" in aggregate:
                summary_lines.append(
                    f"  MER+V: {aggregate['mean_error_rate_with_voice_mean']:.2f}% "
                    f"(+-{aggregate['mean_error_rate_with_voice_std']:.2f})"
                )

    if aggregate["num_sequences_evaluated"] > 0:
        summary_lines.extend(
            [
                "",
                "Per-sequence outputs (for successful sequences):",
                "  input_performance.mid",
                "  ground_truth_score.mid",
                "  output_score.mid",
                "  stats.json",
            ]
        )
        if aggregate.get("muster_enabled"):
            summary_lines.extend(
                [
                    "  ground_truth_score.xml",
                    "  output_score.xml",
                    "  muster_metrics.json",
                    "  muster_work/",
                ]
            )
    else:
        summary_lines.extend(
            [
                "",
                "Per-sequence outputs:",
                "  No per-sequence outputs were created because all sampled sequences failed.",
                "  See failed_sequences.json for the recorded exceptions.",
            ]
        )

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Autoregressive score inference on sampled packed ASAP test windows"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=guess_default_checkpoint(),
        help="Checkpoint directory to evaluate",
    )
    parser.add_argument(
        "--config-source",
        type=str,
        default=DEFAULT_CONFIG_SOURCE,
        help="Fallback config source for checkpoints that only contain model.safetensors",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=TEST_FILE,
        help="Path to the packed test token file",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
        help="Number of random test windows to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for sampling windows",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=OUTPUT_BASE,
        help="Base directory for outputs",
    )
    parser.add_argument(
        "--no-slot-constraints",
        action="store_true",
        help="Disable score-token type constraints during decoding",
    )
    parser.add_argument(
        "--compute-muster",
        action="store_true",
        help="Also compute MUSTER scores for predicted vs ground-truth score windows",
    )
    args = parser.parse_args()

    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"Test file not found: {args.test_file}")
    if args.compute_muster:
        check_muster_installation()

    sampled_lines, total_available = sample_test_lines(
        args.test_file,
        args.num_examples,
        args.seed,
    )
    if not sampled_lines:
        raise ValueError(f"No sequences found in {args.test_file}")

    output_dir = (
        Path(args.output_base)
        / checkpoint_label(args.checkpoint)
        / f"sample{args.num_examples}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "sampled_indices.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "seed": args.seed,
                "num_samples": len(sampled_lines),
                "total_sequences_available": total_available,
                "indices": [index for index, _ in sampled_lines],
            },
            handle,
            indent=2,
        )

    aggregate = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        config_source=args.config_source,
        sampled_lines=sampled_lines,
        output_dir=output_dir,
        constrain_score_tokens=not args.no_slot_constraints,
        compute_muster=args.compute_muster,
    )

    write_summary(
        output_dir=output_dir,
        checkpoint_path=args.checkpoint,
        test_file=args.test_file,
        total_available=total_available,
        sampled_lines=sampled_lines,
        aggregate=aggregate,
        constrain_score_tokens=not args.no_slot_constraints,
    )

    print("\n" + "=" * 60)
    print("AUTOREGRESSIVE INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sequences evaluated: {aggregate['num_sequences_evaluated']}")
    print(f"Sequences failed: {aggregate['num_sequences_failed']}")
    print(
        f"Autoregressive accuracy (pitch): "
        f"{aggregate['autoregressive_accuracy_pct']:.2f}%"
    )
    print(f"Time accuracy: {aggregate['time_accuracy']:.2f}%")
    print(f"Duration accuracy: {aggregate['dur_accuracy']:.2f}%")
    print(f"Exact triplet accuracy: {aggregate['overall_accuracy']:.2f}%")
    if args.compute_muster:
        print(f"MUSTER windows evaluated: {aggregate['num_sequences_muster_evaluated']}")
        print(f"MUSTER windows failed: {aggregate['num_sequences_muster_failed']}")
        if "mean_error_rate_mean" in aggregate:
            print(f"MUSTER MER: {aggregate['mean_error_rate_mean']:.2f}%")
            print(f"MUSTER PER: {aggregate['pitch_error_rate_mean']:.2f}%")
            print(f"MUSTER MNR: {aggregate['missing_note_rate_mean']:.2f}%")
            print(f"MUSTER ENR: {aggregate['extra_note_rate_mean']:.2f}%")
            print(f"MUSTER OTER: {aggregate['onset_time_error_rate_mean']:.2f}%")
            print(f"MUSTER OFTER: {aggregate['offset_time_error_rate_mean']:.2f}%")
            if "voice_error_rate_mean" in aggregate:
                print(f"MUSTER VER: {aggregate['voice_error_rate_mean']:.2f}%")
            if "mean_error_rate_with_voice_mean" in aggregate:
                print(
                    f"MUSTER MER+V: "
                    f"{aggregate['mean_error_rate_with_voice_mean']:.2f}%"
                )
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
