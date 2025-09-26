import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from anticipation.convert import midi_to_events
from anticipation.vocab import *
from anticipation.config import *
from alignment import align_tokens2


def to_pairs(triples: List[int]) -> List[Tuple[int,int,int]]:
    return list(zip(triples[0::3], triples[1::3], triples[2::3]))


def plot_piano_roll(ax, tokens: List[int], title: str, is_control: bool = False, color: str = "C0"):
    # Build a simple scatter-like piano roll: y=pitch, x=time
    xs, ys = [], []
    for (time, dur, note) in to_pairs(tokens):
        if note == SEPARATOR or note == REST:
            continue
        if is_control:
            time = time - ATIME_OFFSET
            note = note - ANOTE_OFFSET
        else:
            time = time - TIME_OFFSET
            note = note - NOTE_OFFSET
        pitch = note % 128
        xs.append(time)
        ys.append(pitch)
    ax.scatter(xs, ys, s=4, alpha=0.6, color=color)
    ax.set_title(title)
    ax.set_xlabel("time (ticks)")
    ax.set_ylabel("pitch")


def main():
    parser = argparse.ArgumentParser(description="Visualize score vs performance and interleaving order")
    parser.add_argument("--asap-root", default="./asap-dataset-master")
    parser.add_argument("--row", type=int, default=0, help="Row index in metadata.csv to visualize")
    parser.add_argument("--skip-nones", action="store_true")
    parser.add_argument("--out", default="./data/interleave_vis.png")
    parser.add_argument("--method", choices=["t3", "t4"], default="t4", help="Interleaving style: t3 (time prefix) or t4 (fixed-count control+pad prefix)")
    parser.add_argument("--prefix-controls", type=int, default=33, help="For t4: number of initial control tokens to prefix (each followed by a pad)")
    args = parser.parse_args()

    meta = os.path.join(args.asap_root, "metadata.csv")
    df = pd.read_csv(meta)
    row = df.iloc[args.row]
    perf = os.path.join(args.asap_root, row["midi_performance"])
    score = os.path.join(args.asap_root, row["midi_score"])
    perf_ann = os.path.join(args.asap_root, row["performance_annotations"])
    score_ann = os.path.join(args.asap_root, row["midi_score_annotations"])

    matched = align_tokens2(perf, score, perf_ann, score_ann, skip_Nones=args.skip_nones)

    # Separate perf and score streams (as tokens)
    perf_tokens = []
    score_tokens = []
    order_indices = []  # sequence index of appearance

    for i, m in enumerate(matched):
        cc = m[0]
        sc = m[2]
        perf_tokens.extend(cc)
        if sc[0] is not None:
            score_tokens.extend(sc)

    # Build the interleaved order in the requested style
    interleaved = []
    if args.method == "t3":
        prefix = []
        for m in matched:
            if m[0][0] - CONTROL_OFFSET <= DELTA * TIME_RESOLUTION:
                prefix.extend(m[0])
        prefix_len = int(len(prefix) / 3)
        for i, m in enumerate(matched):
            sc = m[2]
            if sc[0] is not None:
                interleaved.extend(sc)
                order_indices.append((len(interleaved) // 3 - 1, "score"))
            ii = i + prefix_len
            if ii < len(matched):
                interleaved.extend(matched[ii][0])
                order_indices.append((len(interleaved) // 3 - 1, "control"))
    else:  # t4 style
        k = min(args.prefix_controls, len(matched))
        # fixed-length control+pad prefix
        for m in matched[:k]:
            cc = m[0]
            interleaved.extend(cc)
            order_indices.append((len(interleaved) // 3 - 1, "control"))
            # add REST pad at same time
            cc_time = cc[0] - CONTROL_OFFSET
            interleaved.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])
            order_indices.append((len(interleaved) // 3 - 1, "pad"))
        # main alternation
        for i, m in enumerate(matched):
            sc = m[2]
            if sc[0] is not None:
                interleaved.extend(sc)
                order_indices.append((len(interleaved) // 3 - 1, "score"))
            ii = i + k
            if ii < len(matched):
                interleaved.extend(matched[ii][0])
                order_indices.append((len(interleaved) // 3 - 1, "control"))

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    plot_piano_roll(axes[0], score_tokens, "Score (events)", is_control=False, color="C0")
    plot_piano_roll(axes[1], perf_tokens, "Performance (controls)", is_control=True, color="C1")

    # Interleaving order: plot index vs type (0=score,1=control,2=pad)
    idxs = list(range(len(order_indices)))
    mapping = {"score": 0, "control": 1, "pad": 2}
    labels = [mapping[t] for _, t in order_indices]
    axes[2].plot(idxs, labels, marker="o", linestyle="-")
    axes[2].set_yticks([0, 1, 2], labels=["score", "control", "pad"])  # type: ignore
    axes[2].set_xlabel("Interleaving step")
    axes[2].set_title(f"Interleaving order ({args.method})")

    plt.tight_layout()
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
