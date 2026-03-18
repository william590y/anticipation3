import argparse
import json
from pathlib import Path
import random
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anticipation.config import CONTEXT_SIZE, EVENT_SIZE
from anticipation.vocab import CONTROL_OFFSET, REST


PACKED_SEQUENCE_LENGTH = CONTEXT_SIZE - 4
PREFIX_CONTROLS = 33
ALTERNATING_START = PREFIX_CONTROLS * 2 * EVENT_SIZE


class TokenizedDataset:
    def __init__(self, file_path):
        self.file_path = str(file_path)
        self.offsets = []

        with open(self.file_path, "rb") as f:
            offset = 0
            for raw_line in f:
                if raw_line.strip():
                    self.offsets.append(offset)
                offset += len(raw_line)

        if not self.offsets:
            raise ValueError(f"No sequences found in {self.file_path}")

        first = self._read_tokens(0)
        if len(first) != PACKED_SEQUENCE_LENGTH:
            raise ValueError(
                f"Expected {PACKED_SEQUENCE_LENGTH} tokens per sequence, got {len(first)}"
            )

    def __len__(self):
        return len(self.offsets)

    def _read_tokens(self, idx):
        with open(self.file_path, "rb") as f:
            f.seek(self.offsets[idx])
            raw_line = f.readline().decode("utf-8").strip()
        if "|" in raw_line:
            raw_line = raw_line.split("|", 1)[0].strip()
        return [max(0, int(tok)) for tok in raw_line.split()]

    def __getitem__(self, idx):
        return torch.tensor(self._read_tokens(idx), dtype=torch.long)


def evaluate_sequence(model, device, seq, forced=False, forced_max_attempts=1000):
    context = seq[:ALTERNATING_START].tolist()
    pos = ALTERNATING_START
    correct = 0
    total = 0
    total_triplet_attempts = 0
    positions_forced = 0

    def decode_triplet(ctx, sample=False):
        local_ctx = list(ctx)
        triplet = []
        for _ in range(3):
            input_tensor = torch.tensor([local_ctx], device=device)
            with torch.no_grad():
                logits = model(input_tensor).logits[0, -1, :]
            if sample:
                tok = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            else:
                tok = logits.argmax().item()
            local_ctx.append(tok)
            triplet.append(tok)
        return triplet

    while pos + 5 < len(seq):
        if (
            seq[pos] < CONTROL_OFFSET
            and seq[pos + 1] < CONTROL_OFFSET
            and seq[pos + 2] < CONTROL_OFFSET
            and seq[pos + 2] != REST
        ):
            if forced:
                true_pitch = seq[pos + 2].item()
                matched = False
                last_triplet = None
                for _ in range(forced_max_attempts):
                    total_triplet_attempts += 1
                    last_triplet = decode_triplet(context, sample=True)
                    if last_triplet[2] == true_pitch:
                        matched = True
                        break
                if last_triplet is None:
                    last_triplet = decode_triplet(context, sample=False)
                if not matched:
                    last_triplet[2] = true_pitch
                    positions_forced += 1
                pred_time, pred_dur, pred_pitch = last_triplet
                context.extend(last_triplet)
            else:
                pred_time, pred_dur, pred_pitch = decode_triplet(context, sample=False)
                context.extend([pred_time, pred_dur, pred_pitch])

            if pred_pitch == seq[pos + 2].item():
                correct += 1
            total += 1

            pos += 3
            if pos + 2 < len(seq):
                context.extend([seq[pos].item(), seq[pos + 1].item(), seq[pos + 2].item()])
                pos += 3
        else:
            context.append(seq[pos].item())
            pos += 1

    return correct, total, total_triplet_attempts, positions_forced


def main():
    parser = argparse.ArgumentParser(
        description="Compute autoregressive pitch accuracy on the validation token file."
    )
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint directory")
    parser.add_argument("--val-file", default="data/test_combined.txt", help="Validation token file")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of validation sequences to sample; use 0 or a negative value for all",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
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
    parser.add_argument("--output-json", default="", help="Optional path to save JSON summary")
    args = parser.parse_args()

    dataset = TokenizedDataset(args.val_file)
    dataset_size = len(dataset)
    if args.num_samples > 0:
        sample_count = min(args.num_samples, dataset_size)
        rng = random.Random(args.seed)
        indices = rng.sample(range(dataset_size), sample_count)
    else:
        indices = list(range(dataset_size))

    print(f"Loading validation dataset from {args.val_file}")
    print(f"Dataset sequences: {dataset_size}")
    print(f"Evaluating samples: {len(indices)}")

    print(f"Loading model from {args.checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model on {device}")

    overall_correct = 0
    overall_total = 0
    forced_total_triplet_attempts = 0
    forced_positions_forced = 0
    per_sequence = []

    for seq_idx in tqdm(indices, desc="Autoregressive eval", unit="seq"):
        seq = dataset[seq_idx]
        correct, total, triplet_attempts, positions_forced = evaluate_sequence(
            model,
            device,
            seq,
            forced=args.forced,
            forced_max_attempts=args.forced_max_attempts,
        )
        overall_correct += correct
        overall_total += total
        forced_total_triplet_attempts += triplet_attempts
        forced_positions_forced += positions_forced
        per_sequence.append(
            {
                "dataset_index": seq_idx,
                "correct": correct,
                "total": total,
                "accuracy": (correct / total) if total else 0.0,
                "forced_triplet_attempts": triplet_attempts,
                "forced_positions_forced": positions_forced,
            }
        )

    summary = {
        "checkpoint": args.checkpoint,
        "val_file": args.val_file,
        "dataset_size": dataset_size,
        "num_samples": len(indices),
        "seed": args.seed,
        "forced": args.forced,
        "forced_max_attempts": args.forced_max_attempts,
        "forced_total_triplet_attempts": forced_total_triplet_attempts,
        "forced_positions_forced": forced_positions_forced,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "overall_accuracy": (overall_correct / overall_total) if overall_total else 0.0,
        "per_sequence": per_sequence,
    }

    print("\nSummary")
    print(f"  Accuracy: {100 * summary['overall_accuracy']:.2f}%")
    print(f"  Correct:  {summary['overall_correct']}")
    print(f"  Total:    {summary['overall_total']}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"  JSON:     {output_path}")


if __name__ == "__main__":
    main()
