import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path
import random
import sys

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alignment import align_tokens2, load_annotation_file
from anticipation import ops
from anticipation.config import CONTEXT_SIZE, EVENT_SIZE, MAX_TIME, TIME_RESOLUTION
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


ASAP_PATH = "asap-dataset-master"
ASAP_META_CSV = os.path.join(ASAP_PATH, "metadata.csv")
TARGET_BEAT_INTERVAL = 1.0
PACKED_SEQUENCE_LENGTH = CONTEXT_SIZE - 4
ALTERNATING_START = 33 * 2 * EVENT_SIZE
DEFAULT_WINDOWS_WORKERS = 8 if os.name == "nt" else 32


def load_asap_metadata():
    if not os.path.exists(ASAP_META_CSV):
        raise FileNotFoundError(f"ASAP metadata not found: {ASAP_META_CSV}")

    df = pd.read_csv(ASAP_META_CSV)
    pieces = []
    for _, row in df.iterrows():
        perf_midi = os.path.join(ASAP_PATH, row["midi_performance"])
        score_midi = os.path.join(ASAP_PATH, row["midi_score"])
        perf_beats = os.path.join(ASAP_PATH, row["performance_annotations"])
        score_beats = os.path.join(ASAP_PATH, row["midi_score_annotations"])
        if all(os.path.exists(f) for f in [perf_midi, score_midi, perf_beats, score_beats]):
            pieces.append(
                {
                    "filegroup": ("asap", perf_midi, score_midi, perf_beats, score_beats),
                    "perf_path": row["midi_performance"],
                }
            )
    return pieces


def load_asap_test_perfs(split_file):
    if not os.path.exists(split_file):
        return None

    test_perfs = set()
    in_test = False
    with open(split_file, encoding="utf-8") as f:
        for line in f:
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


def select_asap_test_pieces(split_file, num_pieces, selection, seed):
    piece_infos = load_asap_metadata()
    test_perfs = load_asap_test_perfs(split_file)
    if test_perfs is None:
        raise FileNotFoundError(f"Could not read split file: {split_file}")

    test_pieces = [p for p in piece_infos if p["perf_path"] in test_perfs]
    test_pieces = sorted(test_pieces, key=lambda x: x["perf_path"])

    if selection == "random":
        rng = random.Random(seed)
        rng.shuffle(test_pieces)

    return test_pieces[:num_pieces]


def build_packed_sequences(normalized_matched_tuples, prefix_controls=33):
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

        interleaved_tokens = []

        for i in range(k):
            pt = perf_triplets[i]
            interleaved_tokens.extend(
                [pt[0] + ATIME_OFFSET, pt[1] + ADUR_OFFSET, pt[2] + ANOTE_OFFSET]
            )
            cc_time = max(0, pt[0])
            interleaved_tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])

        for i in range(len(subset)):
            st = score_triplets[i]
            if st[0] is not None:
                interleaved_tokens.extend(st)
            ii = i + k
            if ii < len(subset):
                pt = perf_triplets[ii]
                interleaved_tokens.extend(
                    [pt[0] + ATIME_OFFSET, pt[1] + ADUR_OFFSET, pt[2] + ANOTE_OFFSET]
                )

        if len(interleaved_tokens) < PACKED_SEQUENCE_LENGTH:
            break
        interleaved_tokens = interleaved_tokens[:PACKED_SEQUENCE_LENGTH]

        if ops.max_time(interleaved_tokens, seconds=False) >= MAX_TIME:
            continue

        sequences.append(interleaved_tokens)

    return sequences


def tokenize_asap_piece(filegroup):
    _, perf_midi, score_midi, perf_beats, score_beats = filegroup
    try:
        matched_tuples = align_tokens2(
            perf_midi,
            score_midi,
            perf_beats,
            score_beats,
            skip_Nones=False,
            preserve_unmatched_perf=True,
        )
        if len(matched_tuples) < 20:
            return []

        score_annotations = load_annotation_file(score_beats)
        score_beat_times = [a[0] for a in score_annotations]

        normalized = []
        for match in matched_tuples:
            perf_triplet = match[0]
            score_triplet = match[2]

            if score_triplet[0] is not None:
                orig_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
                orig_dur_sec = (score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION
                pitch = score_triplet[2]

                norm_time_sec = 0.0
                time_scale = 1.0

                if score_beat_times and len(score_beat_times) >= 2:
                    if orig_time_sec < score_beat_times[0]:
                        beat_dur = score_beat_times[1] - score_beat_times[0]
                        progress = ((orig_time_sec - score_beat_times[0]) / beat_dur) if beat_dur > 0 else 0
                        time_scale = TARGET_BEAT_INTERVAL / beat_dur if beat_dur > 0 else 1.0
                        norm_time_sec = progress * TARGET_BEAT_INTERVAL
                    else:
                        found = False
                        for i in range(len(score_beat_times) - 1):
                            if score_beat_times[i] <= orig_time_sec <= score_beat_times[i + 1]:
                                beat_dur = score_beat_times[i + 1] - score_beat_times[i]
                                progress = ((orig_time_sec - score_beat_times[i]) / beat_dur) if beat_dur > 0 else 0
                                time_scale = TARGET_BEAT_INTERVAL / beat_dur if beat_dur > 0 else 1.0
                                norm_time_sec = i * TARGET_BEAT_INTERVAL + progress * TARGET_BEAT_INTERVAL
                                found = True
                                break
                        if not found:
                            last_dur = (score_beat_times[-1] - score_beat_times[-2]) if len(score_beat_times) >= 2 else 1.0
                            progress = ((orig_time_sec - score_beat_times[-1]) / last_dur) if last_dur > 0 else 0
                            time_scale = TARGET_BEAT_INTERVAL / last_dur if last_dur > 0 else 1.0
                            norm_time_sec = (len(score_beat_times) - 1) * TARGET_BEAT_INTERVAL + progress * TARGET_BEAT_INTERVAL

                norm_time_units = max(0, round(norm_time_sec * TIME_RESOLUTION))
                norm_dur_units = max(0, round(orig_dur_sec * time_scale * TIME_RESOLUTION))
                normalized_score = [norm_time_units + TIME_OFFSET, norm_dur_units + DUR_OFFSET, pitch]
            else:
                normalized_score = score_triplet

            normalized.append([perf_triplet, match[1], normalized_score, match[3]])

        return normalized
    except Exception:
        return []


def _tokenize_piece_to_first_window(piece):
    perf_path = piece["perf_path"]
    normalized = tokenize_asap_piece(piece["filegroup"])
    if not normalized:
        return {"perf_path": perf_path, "reason": "no_normalized_tuples"}

    sequences = build_packed_sequences(normalized, prefix_controls=33)
    if not sequences:
        return {"perf_path": perf_path, "reason": "no_packed_windows"}

    seq = sequences[0]
    if len(seq) <= ALTERNATING_START:
        return {"perf_path": perf_path, "reason": "sequence_too_short"}

    return {"perf_path": perf_path, "tokens": seq}


def tokenize_first_window(piece_infos, num_workers):
    windows = []
    failures = []

    if num_workers <= 1:
        iterator = map(_tokenize_piece_to_first_window, piece_infos)
        for result in tqdm(iterator, total=len(piece_infos), desc="Tokenizing pieces", unit="piece"):
            if "tokens" in result:
                windows.append(result)
            else:
                failures.append(result)
        return windows, failures

    with Pool(processes=num_workers) as pool:
        iterator = pool.imap_unordered(_tokenize_piece_to_first_window, piece_infos, chunksize=1)
        for result in tqdm(iterator, total=len(piece_infos), desc="Tokenizing pieces", unit="piece"):
            if "tokens" in result:
                windows.append(result)
            else:
                failures.append(result)

    return windows, failures


def count_score_triplets(tokens):
    total = 0
    pos = ALTERNATING_START
    while pos + 5 < len(tokens):
        if (
            tokens[pos] < CONTROL_OFFSET
            and tokens[pos + 1] < CONTROL_OFFSET
            and tokens[pos + 2] < CONTROL_OFFSET
            and tokens[pos + 2] != REST
        ):
            total += 1
            pos += 6
        else:
            pos += 1
    return total


def evaluate_sequence(
    model,
    device,
    tokens,
    show_triplets=False,
    piece_label="",
    forced=False,
    forced_max_attempts=1000,
):
    import torch

    context = list(tokens[:ALTERNATING_START])
    pos = ALTERNATING_START
    correct = 0
    total = 0
    state = {"past": None, "next_logits": None}
    total_triplet_attempts = 0
    positions_forced = 0

    triplet_total = count_score_triplets(tokens)
    triplet_bar = None
    if show_triplets:
        triplet_bar = tqdm(
            total=triplet_total,
            desc=f"AR {piece_label}"[:60],
            unit="triplet",
            leave=False,
        )

    def prime():
        with torch.no_grad():
            out = model(torch.tensor([context], device=device), use_cache=True)
        state["past"] = out.past_key_values
        state["next_logits"] = out.logits[0, -1, :]

    def greedy_append():
        if state["past"] is None:
            prime()
        tok = state["next_logits"].argmax().item()
        context.append(tok)
        with torch.no_grad():
            out = model(
                torch.tensor([[tok]], device=device),
                past_key_values=state["past"],
                use_cache=True,
            )
        state["past"] = out.past_key_values
        state["next_logits"] = out.logits[0, -1, :]
        return tok

    def sample_append():
        if state["past"] is None:
            prime()
        tok = torch.multinomial(torch.softmax(state["next_logits"], dim=-1), 1).item()
        context.append(tok)
        with torch.no_grad():
            out = model(
                torch.tensor([[tok]], device=device),
                past_key_values=state["past"],
                use_cache=True,
            )
        state["past"] = out.past_key_values
        state["next_logits"] = out.logits[0, -1, :]
        return tok

    def feed_gt(tokens_to_feed):
        if state["past"] is None:
            prime()
        context.extend(tokens_to_feed)
        with torch.no_grad():
            out = model(
                torch.tensor([tokens_to_feed], device=device),
                past_key_values=state["past"],
                use_cache=True,
            )
        state["past"] = out.past_key_values
        state["next_logits"] = out.logits[0, -1, :]

    while pos + 5 < len(tokens):
        if (
            tokens[pos] < CONTROL_OFFSET
            and tokens[pos + 1] < CONTROL_OFFSET
            and tokens[pos + 2] < CONTROL_OFFSET
            and tokens[pos + 2] != REST
        ):
            if forced:
                true_pitch = tokens[pos + 2]
                matched = False
                last_triplet = None
                for _ in range(forced_max_attempts):
                    total_triplet_attempts += 1
                    context_before = list(context)
                    past_before = state["past"]
                    logits_before = state["next_logits"]

                    last_triplet = [sample_append(), sample_append(), sample_append()]
                    if last_triplet[2] == true_pitch:
                        matched = True
                        break

                    context[:] = context_before
                    state["past"] = past_before
                    state["next_logits"] = logits_before

                if last_triplet is None:
                    last_triplet = [greedy_append(), greedy_append(), greedy_append()]
                if not matched:
                    context[-1] = true_pitch
                    state["past"] = None
                    state["next_logits"] = None
                    last_triplet[2] = true_pitch
                    positions_forced += 1
                pred_pitch = last_triplet[2]
            else:
                greedy_append()
                greedy_append()
                pred_pitch = greedy_append()
            true_pitch = tokens[pos + 2]

            if pred_pitch == true_pitch:
                correct += 1
            total += 1

            if triplet_bar is not None:
                triplet_bar.update(1)
                triplet_bar.set_postfix_str(f"{correct}/{total} ({100 * correct / total:.1f}%)")

            pos += 3
            if pos + 2 < len(tokens):
                feed_gt([tokens[pos], tokens[pos + 1], tokens[pos + 2]])
                pos += 3
        else:
            feed_gt([tokens[pos]])
            pos += 1

    if triplet_bar is not None:
        triplet_bar.close()

    return correct, total, total_triplet_attempts, positions_forced


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASAP autoregressive pitch accuracy on freshly tokenized windows."
    )
    parser.add_argument("--checkpoint", default="final", help="Checkpoint directory")
    parser.add_argument("--split-file", default="data/combined_split.txt", help="Path to combined split file")
    parser.add_argument("--num-pieces", type=int, default=20, help="Number of ASAP test pieces to evaluate")
    parser.add_argument("--selection", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random piece selection")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WINDOWS_WORKERS,
        help="Tokenization worker processes; on Windows the safe default is 8",
    )
    parser.add_argument(
        "--forced",
        action="store_true",
        help="Keep regenerating a score triplet until its pitch matches ground truth",
    )
    parser.add_argument(
        "--forced-max-attempts",
        type=int,
        default=1000,
        help="Maximum rejection-sampling attempts per score triplet in forced mode",
    )
    parser.add_argument("--show-triplets", action="store_true")
    parser.add_argument("--output-json", default="test_outputs/asap_ar_pitch_eval.json")
    args = parser.parse_args()

    if os.name == "nt" and args.workers > 8:
        print(
            f"Warning: --workers {args.workers} may exhaust Windows virtual memory. "
            "If you see page table/pagefile errors, retry with --workers 8 or lower."
        )

    selected_pieces = select_asap_test_pieces(
        args.split_file, args.num_pieces, args.selection, args.seed
    )
    print(f"Selected {len(selected_pieces)} ASAP test pieces")

    windows, failures = tokenize_first_window(selected_pieces, num_workers=args.workers)
    print(f"Usable packed windows: {len(windows)}")
    if failures:
        print(f"Skipped during tokenization: {len(failures)}")

    import torch
    from transformers import AutoModelForCausalLM

    print(f"Loading model from {args.checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model on {device}")

    overall_correct = 0
    overall_total = 0
    forced_total_triplet_attempts = 0
    forced_positions_forced = 0
    per_piece = []

    piece_bar = tqdm(windows, desc="Evaluating windows", unit="piece")
    for item in piece_bar:
        correct, total, triplet_attempts, positions_forced = evaluate_sequence(
            model,
            device,
            item["tokens"],
            show_triplets=args.show_triplets,
            piece_label=item["perf_path"],
            forced=args.forced,
            forced_max_attempts=args.forced_max_attempts,
        )
        overall_correct += correct
        overall_total += total
        forced_total_triplet_attempts += triplet_attempts
        forced_positions_forced += positions_forced
        acc = correct / total if total else 0.0
        per_piece.append(
            {
                "perf_path": item["perf_path"],
                "correct": correct,
                "total": total,
                "accuracy": acc,
                "forced_triplet_attempts": triplet_attempts,
                "forced_positions_forced": positions_forced,
            }
        )
        overall_acc = overall_correct / overall_total if overall_total else 0.0
        piece_bar.set_postfix_str(f"{overall_correct}/{overall_total} ({100 * overall_acc:.2f}%)")

    summary = {
        "checkpoint": args.checkpoint,
        "split_file": args.split_file,
        "num_pieces_requested": args.num_pieces,
        "selection": args.selection,
        "seed": args.seed,
        "windowing": "first packed training-style window per piece",
        "workers": args.workers,
        "forced": args.forced,
        "forced_max_attempts": args.forced_max_attempts,
        "forced_total_triplet_attempts": forced_total_triplet_attempts,
        "forced_positions_forced": forced_positions_forced,
        "used_windows": len(windows),
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "overall_accuracy": overall_correct / overall_total if overall_total else 0.0,
        "skipped": failures,
        "per_piece": per_piece,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))

    print("\nSummary")
    print(f"  Accuracy: {100 * summary['overall_accuracy']:.2f}%")
    print(f"  Correct:  {summary['overall_correct']}")
    print(f"  Total:    {summary['overall_total']}")
    print(f"  Windows:  {summary['used_windows']}")
    print(f"  JSON:     {output_path}")


if __name__ == "__main__":
    main()
