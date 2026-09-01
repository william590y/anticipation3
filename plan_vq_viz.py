#!/usr/bin/env python
"""Piano rolls for the stage-1 VQ: performance, ground-truth score, reconstruction.

One figure per validation, logged to wandb as ``piano_rolls/*``. Each row is one
fixed validation window (the same windows every time, so the panels are watchable
as a time series across a run); the columns are:

    performance      the conditioning control stream (original tempo)
    score (GT)       the target the codes + performance have to reproduce
    reconstruction   the decoder's argmax, drawn over the ground truth in grey

The performance and the score live on **different time axes** -- performance
times keep the original tempo while score onsets are normalized to a 0.5 s beat
grid -- so the panels are not meant to line up horizontally. What matters is
whether the reconstruction's rhythm and contour match the middle panel.

A reconstructed note is drawn green when it matches the ground-truth note in
that slot exactly (onset, duration and pitch) and red when it does not, so the
error structure is legible without reading the accuracy numbers.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PERFORMANCE_COLOR = "#4C7BA6"
SCORE_COLOR = "#2F3E4F"
CORRECT_COLOR = "#2E7D5B"
WRONG_COLOR = "#B3453C"
GHOST_COLOR = "#C9CDD3"


def _draw(ax, notes, color, alpha=0.95, height=0.9, zorder=2):
    for onset, duration, pitch in notes:
        ax.add_patch(
            patches.Rectangle(
                (onset, pitch - height / 2.0),
                max(int(duration), 1),
                height,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=zorder,
            )
        )


def _frame(ax, notes, title, xlabel):
    """Scale the axes to ``notes`` and return the (x, y) window in view.

    The reconstruction panel is always framed on the *ground truth*, never on
    its own predictions: an untrained decoder emits onsets an order of magnitude
    out of range, and letting those set the limits squashes the real music into
    a few pixels exactly when you most want to look at it.
    """
    onsets, pitches = [], []
    for onset, duration, pitch in notes:
        onsets.extend([onset, onset + max(int(duration), 1)])
        pitches.append(pitch)
    xlim = ylim = None
    if onsets:
        span = max(onsets) - min(onsets)
        xlim = (min(onsets) - 0.02 * span - 1, max(onsets) + 0.02 * span + 1)
        ax.set_xlim(*xlim)
    if pitches:
        ylim = (min(pitches) - 3, max(pitches) + 3)
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("MIDI pitch", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.15, linewidth=0.5)
    return xlim, ylim


def _count_offscreen(notes, xlim, ylim):
    if xlim is None or ylim is None:
        return 0
    return sum(
        1
        for onset, _, pitch in notes
        if not (xlim[0] <= onset <= xlim[1] and ylim[0] <= pitch <= ylim[1])
    )


def piano_roll_figure(
    perf,
    score,
    score_valid,
    predicted,
    step=None,
    max_examples=4,
    *,
    suptitle="Stage-1 VQ reconstruction",
    pred_heading="reconstruction from the plan",
    window_ids=None,
):
    """Build the (examples x 3) piano-roll figure.

    All note arguments are ``(B, N, 3)`` tensors (any device) — the same
    layout ``PlanVQ.reconstruct`` and ``split_packed_batch`` return. Only the
    first ``max_examples`` rows are drawn.

    ``suptitle`` / ``pred_heading`` / ``window_ids`` let other trainers reuse
    the same visual language (LTLM AR decode, etc.) without forking the
    drawing code. Defaults keep the stage-1 VQ figure byte-identical.
    """
    perf = perf.detach().cpu().tolist()
    score = score.detach().cpu().tolist()
    valid = score_valid.detach().cpu().tolist()
    predicted = predicted.detach().cpu().tolist()

    rows = min(max_examples, len(perf))
    fig, axes = plt.subplots(rows, 3, figsize=(16, 2.9 * rows), squeeze=False)

    for row in range(rows):
        keep = [i for i, ok in enumerate(valid[row]) if ok]
        true_notes = [score[row][i] for i in keep]
        pred_notes = [predicted[row][i] for i in keep]
        exact = [
            tuple(predicted[row][i]) == tuple(score[row][i])
            for i in keep
        ]

        window_tag = ""
        if window_ids is not None and row < len(window_ids):
            window_tag = f"window {window_ids[row]}  "

        _draw(axes[row][0], perf[row], PERFORMANCE_COLOR)
        _frame(
            axes[row][0],
            perf[row],
            f"{window_tag}performance ({len(perf[row])} notes)",
            "performance time (10 ms bins)",
        )

        _draw(axes[row][1], true_notes, SCORE_COLOR)
        _frame(
            axes[row][1],
            true_notes,
            f"score, ground truth ({len(true_notes)} notes)",
            "score time (10 ms bins, 0.5 s beat grid)",
        )

        # Ground truth underneath in grey so displacement is visible.
        _draw(axes[row][2], true_notes, GHOST_COLOR, alpha=1.0, height=0.9, zorder=1)
        _draw(
            axes[row][2],
            [note for note, ok in zip(pred_notes, exact) if ok],
            CORRECT_COLOR,
            height=0.6,
            zorder=3,
        )
        _draw(
            axes[row][2],
            [note for note, ok in zip(pred_notes, exact) if not ok],
            WRONG_COLOR,
            height=0.6,
            zorder=3,
        )
        matched = sum(exact)
        xlim, ylim = _frame(
            axes[row][2],
            true_notes,
            f"{pred_heading} -- {matched}/{len(keep)} notes exact "
            "(grey = ground truth)",
            "score time (10 ms bins, 0.5 s beat grid)",
        )
        offscreen = _count_offscreen(pred_notes, xlim, ylim)
        if offscreen:
            axes[row][2].text(
                0.99, 0.02, f"{offscreen} predicted notes off-frame",
                transform=axes[row][2].transAxes, ha="right", va="bottom",
                fontsize=7, color=WRONG_COLOR,
            )

    title = suptitle
    if step is not None:
        title += f" -- step {step}"
    fig.suptitle(title, fontsize=12, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig
