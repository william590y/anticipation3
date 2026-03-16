import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anticipation.vocab import CONTROL_OFFSET, REST
from evaluate_muster_asap import (
    ALTERNATING_START,
    _build_sequences,
    load_asap_metadata,
    load_asap_test_perfs,
    tokenize_asap_piece,
)


def select_asap_test_pieces(split_file, num_pieces, selection, seed):
    piece_infos = load_asap_metadata()
    test_perfs = load_asap_test_perfs(split_file)
    if test_perfs is None:
        raise FileNotFoundError(f"Could not read split file: {split_file}")

    test_pieces = [p for p in piece_infos if p["perf_path"] in test_perfs]
    test_pieces = sorted(test_pieces, key=lambda x: x["perf_path"])

    if selection == "random":
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(test_pieces), generator=g).tolist()
        test_pieces = [test_pieces[i] for i in perm]

    return test_pieces[:num_pieces]


NUM_WORKERS = 32


def _tokenize_piece_to_first_window(piece):
    perf_path = piece["perf_path"]
    normalized = tokenize_asap_piece(piece["filegroup"])
    if not normalized:
        return {"perf_path": perf_path, "reason": "no_normalized_tuples"}

    sequences = _build_sequences(normalized, prefix_controls=33)
    if not sequences:
        return {"perf_path": perf_path, "reason": "no_packed_windows"}

    seq = sequences[0]
    if len(seq) <= ALTERNATING_START:
        return {"perf_path": perf_path, "reason": "sequence_too_short"}

    return {"perf_path": perf_path, "tokens": seq}


def tokenize_first_window(piece_infos, num_workers=NUM_WORKERS):
    windows = []
    failures = []

    with Pool(processes=num_workers) as pool:
        iterator = pool.imap(_tokenize_piece_to_first_window, piece_infos)
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


def evaluate_sequence(model, device, tokens, show_triplets=False, piece_label=""):
    context = list(tokens[:ALTERNATING_START])
    pos = ALTERNATING_START
    correct = 0
    total = 0
    state = {"past": None, "next_logits": None}

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

    return correct, total


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASAP autoregressive pitch accuracy on freshly tokenized windows."
    )
    parser.add_argument("--checkpoint", default="final", help="Checkpoint directory")
    parser.add_argument("--split-file", default="data/combined_split.txt", help="Path to combined split file")
    parser.add_argument("--num-pieces", type=int, default=20, help="Number of ASAP test pieces to evaluate")
    parser.add_argument(
        "--selection",
        choices=["first", "random"],
        default="first",
        help="How to choose ASAP test pieces from the split",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random piece selection")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Tokenization worker processes")
    parser.add_argument(
        "--show-triplets",
        action="store_true",
        help="Show a nested tqdm for score-triplet generation within each piece",
    )
    parser.add_argument(
        "--output-json",
        default="test_outputs/asap_ar_pitch_eval.json",
        help="Where to write a JSON summary",
    )
    args = parser.parse_args()

    selected_pieces = select_asap_test_pieces(
        args.split_file, args.num_pieces, args.selection, args.seed
    )
    print(f"Selected {len(selected_pieces)} ASAP test pieces")

    windows, failures = tokenize_first_window(selected_pieces, num_workers=args.workers)
    print(f"Usable packed windows: {len(windows)}")
    if failures:
        print(f"Skipped during tokenization: {len(failures)}")

    print(f"Loading model from {args.checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model on {device}")

    overall_correct = 0
    overall_total = 0
    per_piece = []

    piece_bar = tqdm(windows, desc="Evaluating windows", unit="piece")
    for item in piece_bar:
        correct, total = evaluate_sequence(
            model,
            device,
            item["tokens"],
            show_triplets=args.show_triplets,
            piece_label=item["perf_path"],
        )
        overall_correct += correct
        overall_total += total
        acc = correct / total if total else 0.0
        per_piece.append(
            {
                "perf_path": item["perf_path"],
                "correct": correct,
                "total": total,
                "accuracy": acc,
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
