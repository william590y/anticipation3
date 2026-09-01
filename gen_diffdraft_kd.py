"""Teacher greedy rollouts for the diffusion drafter's knowledge-distillation set.

A speculative-decoding drafter is not trying to be right about the *music*; it is
trying to guess what the **target model's constrained argmax** will be. Those are
very different targets here: the target's own AR exact-match accuracy against the
ground truth is ~16% on onsets, so a drafter trained on ground truth would agree
with the target only where the target happens to be right. Sequence-level KD
(Kim & Rush 2016), i.e. training the drafter to reconstruct the teacher's own
greedy rollout, is the standard fix and is what every production draft model does.

The rollout protocol here is exactly `onpolicy_rollout.rollout_score_slots` at
temperature 0 -- the same decode `inference.batched_autoregressive_generate_score`
and `train.evaluate_model` use -- so the cached tokens are, position for position,
what the greedy production decode emits. Output lines are the *rolled* packed
window (controls and prefix untouched, score triplets replaced), which keeps them
in the ordinary `data/*.txt` format so anything that reads packed windows can read
them.

Windows are drawn at evenly spaced byte offsets rather than as the first N lines:
consecutive lines of a token file are overlapping sliding windows of the same
performance, so the first N lines are N near-duplicates (same reasoning as
`bench/bench_common.load_bench_windows`). Sharding is by strided offset index, so
shard i of n is itself spread over the whole file and any subset of the shards is
still a representative sample -- a job that dies half way through still leaves
usable data.

Usage:
    python gen_diffdraft_kd.py --token-file data/train_paper.txt \
        --num-windows 24000 --shard 0 --num-shards 4 --out data/diffdraft_kd_train
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from evaluate_muster import load_model
from onpolicy_rollout import rollout_score_slots

PACKED_LENGTH = 1020


def window_byte_offsets(path: Path, count: int) -> list[int]:
    """`count` evenly spaced byte offsets, each snapped forward to a line start."""
    size = path.stat().st_size
    stride = max(size // max(count, 1), 1)
    return [stride * i for i in range(count)]


def read_window_at(handle, offset: int, expect_length: int) -> list[int] | None:
    handle.seek(offset)
    if offset > 0:
        handle.readline()  # discard the partial line we landed inside
    line = handle.readline()
    if not line:
        return None
    text = line.decode("utf-8").split("|")[0]
    try:
        tokens = [int(tok) for tok in text.split()]
    except ValueError:
        return None
    if expect_length and len(tokens) != expect_length:
        return None
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="run_paper_split_v2/checkpoint-2500")
    parser.add_argument("--token-file", default="data/train_paper.txt")
    parser.add_argument("--num-windows", type=int, default=24000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--out", default="data/diffdraft_kd_train")
    parser.add_argument("--max-minutes", type=float, default=0.0,
                        help="stop early and keep what was written (0 = no limit)")
    args = parser.parse_args()

    token_path = Path(args.token_file)
    offsets = window_byte_offsets(token_path, args.num_windows)
    offsets = offsets[args.shard :: args.num_shards]

    model, device = load_model(args.checkpoint)
    model.config.use_cache = True

    out_path = Path(f"{args.out}.shard{args.shard}of{args.num_shards}.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    started = time.perf_counter()
    with token_path.open("rb") as handle, out_path.open("w", encoding="utf-8") as out:
        batch: list[list[int]] = []
        for index, offset in enumerate(offsets + [None]):
            if offset is not None:
                tokens = read_window_at(handle, offset, PACKED_LENGTH)
                if tokens is None:
                    skipped += 1
                    continue
                batch.append(tokens)
                if len(batch) < args.batch_size:
                    continue
            if not batch:
                continue

            input_ids = torch.tensor(batch, dtype=torch.long, device=device)
            rollout = rollout_score_slots(
                model,
                input_ids,
                temperature=0.0,
                constrain=True,
                collect_logprobs=False,
                collect_gt_ce=False,
            )
            for row in rollout["rolled"].tolist():
                out.write(" ".join(str(t) for t in row) + " | \n")
            written += len(batch)
            batch = []

            elapsed = time.perf_counter() - started
            print(
                f"[shard {args.shard}] {written} windows  {written / elapsed:.2f} win/s  "
                f"{elapsed / 60:.1f} min",
                flush=True,
            )
            out.flush()
            if args.max_minutes and elapsed > args.max_minutes * 60:
                print(f"[shard {args.shard}] hit --max-minutes, stopping early", flush=True)
                break

    print(
        f"[shard {args.shard}] done: {written} windows -> {out_path} "
        f"({skipped} offsets skipped) in {(time.perf_counter() - started) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
