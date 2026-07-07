"""
Histogram of score-note durations, either from the tokenized dataset or from a
model's autoregressive rollout, with gridlines marking standard note values.

Assumes 4/4 time and the tokenizer's fixed beat grid (TARGET_BEAT_INTERVAL in
anticipation/asap_aligned_stream.py): a quarter note = 0.5 s. All standard note
values (and their common triplet/dotted forms) fall out as multiples of that.

Duration tokens are already quantized to 10 ms bins (TIME_RESOLUTION), so counts
are tallied per exact token value -- no re-binning or precision loss.

Usage:
  # Ground-truth score durations in the tokenized corpus (fast, uses mawk/awk).
  python analysis/duration_histogram.py dataset \
      --files data/train_normalized.txt data/test_normalized.txt \
      --output analysis/duration_hist_dataset

  # Model's predicted score durations, sampled from autoregressive rollouts.
  python analysis/duration_histogram.py model \
      --checkpoint run_nodummy/final --test-file data/test_normalized.txt \
      --num-examples 300 --output analysis/duration_hist_model
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from anticipation.config import MAX_DUR, TIME_RESOLUTION  # noqa: E402
from anticipation.packed_sequence import (  # noqa: E402
    ALTERNATING_START,
    is_real_score_triplet,
    iter_score_slot_positions,
    triplet_values,
)
from anticipation.vocab import DUR_OFFSET  # noqa: E402

# Body score triplets sit at (0-indexed) ALTERNATING_START + 6*i; duration is
# the triplet's 2nd token (+1). AWK fields are 1-indexed, so the field number
# is pos + 2. Kept in sync with ALTERNATING_START via the assert below.
assert ALTERNATING_START == 192, "AWK field offsets below assume ALTERNATING_START=192"
_AWK_DUR_FIRST_FIELD = ALTERNATING_START + 2   # 194
_AWK_DUR_STEP = 6

# The tokenizer pins whatever ASAP calls "the beat" to 0.5s (see
# asap_aligned_stream.TARGET_BEAT_INTERVAL). Every ASAP time signature reduces
# to exactly one of two duration ladders depending on whether that beat is a
# *plain* note (simple meter) or a *dotted* note (compound meter) -- the
# specific signature (4/4 vs 3/4 vs 2/2 vs ...) doesn't matter, only which
# family it belongs to, since within a family all signatures share identical
# durations and differ only in which note NAME sits at a given duration.
#
# Measured against ASAP's own annotations (asap_annotations.json, classified
# by comparing each signature's notated beat to its actual number_of_beats):
#   simple:   4/4, 2/4, 3/4, 2/2, 3/8, 5/8, 3/2, 4/8, 5/4, 4/2, 1/2, 1/4
#             (~82% of the corpus by beat-time)
#   compound: 6/8, 9/8, 12/8, 6/4, 12/16, 6/16, 24/16
#             (~18% of the corpus by beat-time; beat = dotted note, e.g. 6/8's
#             beat is a dotted quarter = 3 eighth notes)
BEAT_SECONDS = 0.5

# Simple-meter ladder: beat = a plain note, so halving/doubling from the beat
# never leaves powers of 2.
SIMPLE_NOTE_VALUES = {
    "32nd": BEAT_SECONDS / 8,
    "16th": BEAT_SECONDS / 4,
    "dotted 16th": BEAT_SECONDS * 3 / 8,
    "8th": BEAT_SECONDS / 2,
    "dotted 8th": BEAT_SECONDS * 3 / 4,
    "quarter": BEAT_SECONDS,
    "dotted quarter": BEAT_SECONDS * 3 / 2,
    "half": BEAT_SECONDS * 2,
    "dotted half": BEAT_SECONDS * 3,
    "whole": BEAT_SECONDS * 4,
}

# Compound-meter ladder: beat = a dotted note (3 base units), so the base
# unit is beat/3, then halved/doubled from there. Values at 0.5/1.0/2.0s
# (the beat itself and its doublings) coincide exactly with the simple
# ladder's quarter/half/whole -- intentionally omitted here to avoid drawing
# duplicate lines; see the plot legend for that overlap.
COMPOUND_NOTE_VALUES = {
    "32nd (compound)": BEAT_SECONDS / 12,
    "16th (compound)": BEAT_SECONDS / 6,
    "8th (compound)": BEAT_SECONDS / 3,
    "quarter (compound)": BEAT_SECONDS * 2 / 3,
}

# Kept for any external callers; dataset/model histograms now draw both
# families (see plot_note_duration_histogram).
NOTE_VALUES = {**SIMPLE_NOTE_VALUES, **COMPOUND_NOTE_VALUES}


def extract_dataset_histogram(files: list[str]) -> np.ndarray:
    """Tally score-note duration tokens across packed sequence files via AWK.

    Every body score triplet holds a real note post-refactor (unmatched
    performance notes are dropped at tokenization time, so there are no
    dummy REST placeholders left in the body) -- no REST filtering needed.
    """
    awk_bin = shutil.which("mawk") or shutil.which("awk")
    if awk_bin is None:
        raise RuntimeError("Neither mawk nor awk found on PATH.")

    script = (
        f"{{ for (i={_AWK_DUR_FIRST_FIELD}; i<=NF; i+={_AWK_DUR_STEP}) {{"
        f" v = $i - {DUR_OFFSET};"
        f" if (v >= 0 && v < {MAX_DUR}) count[v]++ }} }}"
        f" END {{ for (v = 0; v < {MAX_DUR}; v++) if (count[v] > 0) print v, count[v] }}"
    )

    result = subprocess.run(
        [awk_bin, script, *files],
        capture_output=True, text=True, check=True,
    )

    counts = np.zeros(MAX_DUR, dtype=np.int64)
    for line in result.stdout.splitlines():
        v_str, c_str = line.split()
        counts[int(v_str)] = int(c_str)
    return counts


def extract_model_histogram(
    checkpoint: str,
    config_source: str | None,
    test_file: str,
    num_examples: int,
    seed: int,
) -> np.ndarray:
    """Tally predicted score-note durations from full autoregressive rollouts."""
    from evaluate_muster import load_model
    from inference import (
        autoregressive_generate_score,
        parse_sequence,
        sample_test_lines,
    )

    model, device = load_model(checkpoint, config_source)
    sampled, total = sample_test_lines(test_file, num_examples, seed)
    print(f"Sampled {len(sampled)} of {total} validation windows (seed={seed})")

    counts = np.zeros(MAX_DUR, dtype=np.int64)
    for i, (original_index, line) in enumerate(sampled):
        tokens = parse_sequence(line)
        if len(tokens) <= ALTERNATING_START:
            continue
        pred_tokens, _ = autoregressive_generate_score(
            model,
            tokens,
            ALTERNATING_START,
            device,
            constrain_score_tokens=True,
            ground_truth_score_tokens_to_feed=0,
        )
        for pos in iter_score_slot_positions(len(pred_tokens), ALTERNATING_START):
            if pos + 2 >= len(pred_tokens) or not is_real_score_triplet(
                pred_tokens, pos, ALTERNATING_START
            ):
                continue
            _, dur_tok, _ = triplet_values(pred_tokens, pos)
            v = dur_tok - DUR_OFFSET
            if 0 <= v < MAX_DUR:
                counts[v] += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(sampled)} windows decoded (running total notes: {counts.sum()})")

    return counts


# (family label, color, note-value dict, list of time signatures it covers)
NOTE_FAMILIES = [
    (
        "simple meter",
        "#3fbf6e",
        SIMPLE_NOTE_VALUES,
        "4/4, 2/4, 3/4, 2/2, 3/8, 5/8, 3/2, 4/8, 5/4, 4/2 (~82% of corpus)",
    ),
    (
        "compound meter",
        "#e0763d",
        COMPOUND_NOTE_VALUES,
        "6/8, 9/8, 12/8, 6/4, 12/16, 6/16, 24/16 (~18% of corpus)",
    ),
]
# Values shared by both families (the compound beat and its doublings), drawn
# once in a neutral color rather than duplicated per family.
SHARED_NOTE_VALUES = {
    "quarter / compound beat (dotted quarter or dotted eighth)": BEAT_SECONDS,
    "half / 2 compound beats": BEAT_SECONDS * 2,
    "whole / 4 compound beats": BEAT_SECONDS * 4,
}


def plot_note_duration_histogram(
    counts: np.ndarray,
    title: str,
    output_path: Path,
    max_seconds: float = 2.5,
    families: list[tuple] = NOTE_FAMILIES,
    shared_note_values: dict[str, float] = SHARED_NOTE_VALUES,
    log_scale: bool = True,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total = int(counts.sum())
    seconds = np.arange(len(counts)) / TIME_RESOLUTION
    tail_fraction = counts[seconds > max_seconds].sum() / max(total, 1)

    fig, ax = plt.subplots(figsize=(15, 7))
    in_range = seconds <= max_seconds
    ax.bar(seconds[in_range], counts[in_range], width=1.0 / TIME_RESOLUTION,
           color="#4c8bf5", align="edge", zorder=2)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlim(0, max_seconds)
    ax.set_xlabel("Note duration (seconds)")
    ax.set_ylabel("Count (log scale)" if log_scale else "Count")
    ax.set_title(f"{title}\n(n={total:,} score notes; {100*tail_fraction:.2f}% beyond {max_seconds}s not shown)")

    ytop = None
    shared_color = "#8a8f98"
    for name, sec in shared_note_values.items():
        if sec > max_seconds:
            continue
        ax.axvline(sec, color=shared_color, linestyle="-", linewidth=1.3, alpha=0.85, zorder=3)
        ytop = ax.get_ylim()[1] if ytop is None else ytop
        ax.text(sec, ytop, name, rotation=90, va="top", ha="right",
                fontsize=7.5, color=shared_color)

    legend_handles = [
        plt.Line2D([0], [0], color=shared_color, lw=1.3, label="shared by both families (see labels)")
    ]
    for family_name, color, values, sig_list in families:
        for name, sec in values.items():
            if sec > max_seconds:
                continue
            ax.axvline(sec, color=color, linestyle="--", linewidth=1.1, alpha=0.85, zorder=3)
            ax.text(sec, ytop, name, rotation=90, va="top", ha="right", fontsize=7.5, color=color)
        legend_handles.append(
            plt.Line2D([0], [0], color=color, lw=1.3, linestyle="--",
                       label=f"{family_name}: {sig_list}")
        )

    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, which="both", axis="y", alpha=0.15)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def save_counts(counts: np.ndarray, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_base.with_suffix(".npy"), counts)
    with open(output_base.with_suffix(".txt"), "w", encoding="utf-8") as handle:
        for v, c in enumerate(counts):
            if c > 0:
                handle.write(f"{v} {c}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_data = sub.add_parser("dataset", help="Histogram of ground-truth score durations in the tokenized corpus.")
    p_data.add_argument("--files", nargs="+", required=True, help="Packed sequence file(s), e.g. data/train_normalized.txt data/test_normalized.txt")
    p_data.add_argument("--output", type=Path, required=True, help="Output path prefix (writes <output>.png/.npy/.txt)")
    p_data.add_argument("--max-seconds", type=float, default=2.5)
    p_data.add_argument("--title", default="Score note duration distribution (tokenized dataset)")
    p_data.add_argument("--linear", action="store_true", help="Plot counts on a linear (not log) y-axis.")

    p_model = sub.add_parser("model", help="Histogram of a model's predicted score durations from autoregressive rollouts.")
    p_model.add_argument("--checkpoint", required=True)
    p_model.add_argument("--config-source", default=None)
    p_model.add_argument("--test-file", default="data/test_normalized.txt")
    p_model.add_argument("--num-examples", type=int, default=300)
    p_model.add_argument("--seed", type=int, default=17)
    p_model.add_argument("--output", type=Path, required=True)
    p_model.add_argument("--max-seconds", type=float, default=2.5)
    p_model.add_argument("--title", default=None)
    p_model.add_argument("--linear", action="store_true", help="Plot counts on a linear (not log) y-axis.")

    args = parser.parse_args()

    if args.mode == "dataset":
        counts = extract_dataset_histogram(args.files)
        title = args.title
    else:
        counts = extract_model_histogram(
            args.checkpoint, args.config_source, args.test_file, args.num_examples, args.seed
        )
        title = args.title or f"Score note duration distribution (model rollout: {args.checkpoint})"

    save_counts(counts, args.output)
    plot_note_duration_histogram(counts, title, args.output.with_suffix(".png"),
                                  max_seconds=args.max_seconds, log_scale=not args.linear)


if __name__ == "__main__":
    main()
