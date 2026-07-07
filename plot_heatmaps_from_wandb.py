#!/usr/bin/env python
"""Render the train.py autoregressive-error heatmaps from a finished W&B run.

The native ``wandb/heatmap/v0`` charts logged by ``train.py`` don't always render
nicely in the W&B UI, so this pulls the underlying tables (each validation re-logs
the full accumulated matrix, so the final table per metric holds everything) and
draws them with matplotlib.

Usage:
    python plot_heatmaps_from_wandb.py \
        --run wjl86-cornell-university/anticipation-asap/l2ryqjx8
"""

import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

# key -> (human title, colorbar label, colormap)
METRICS = {
    "pitch_error_freq": ("Autoregressive pitch error frequency", "error frequency", "magma"),
    "onset_mae": ("Autoregressive onset MAE", "MAE (10 ms bins)", "viridis"),
    "duration_mae": ("Autoregressive duration MAE", "MAE (10 ms bins)", "viridis"),
}

_FILE_RE = re.compile(r"heatmaps/(?P<metric>[a-z_]+)_table_(?P<step>\d+)_[0-9a-f]+\.table\.json$")


def find_final_tables(run):
    """Return {metric: wandb File} picking the max-step (then max-size) table file."""
    best = {}
    for f in run.files():
        m = _FILE_RE.search(f.name)
        if not m:
            continue
        metric = m.group("metric")
        if metric not in METRICS:
            continue
        key = (int(m.group("step")), f.size)
        if metric not in best or key > best[metric][0]:
            best[metric] = (key, f)
    return {metric: f for metric, (key, f) in best.items()}


def table_to_matrix(table_path):
    """Load a wandb table JSON (columns x_axis,y_axis,value) into (matrix, steps, slots)."""
    with open(table_path) as handle:
        payload = json.load(handle)
    cols = payload["columns"]
    xi, yi, vi = cols.index("x_axis"), cols.index("y_axis"), cols.index("value")
    rows = payload["data"]

    steps = sorted({int(r[yi]) for r in rows})
    slots = sorted({int(r[xi]) for r in rows})
    step_pos = {s: i for i, s in enumerate(steps)}
    slot_pos = {s: i for i, s in enumerate(slots)}

    matrix = np.full((len(steps), len(slots)), np.nan)
    for r in rows:
        matrix[step_pos[int(r[yi])], slot_pos[int(r[xi])]] = float(r[vi])
    return matrix, np.array(steps), np.array(slots)


def render_metric(ax, matrix, steps, slots, title, cbar_label, cmap_name):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("lightgrey")  # empty score slots
    im = ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        origin="upper",
        cmap=cmap,
        extent=[slots.min() - 0.5, slots.max() + 0.5, steps.max(), steps.min()],
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("score-slot index (k-th predicted note)")
    ax.set_ylabel("training step")
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label(cbar_label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default="wjl86-cornell-university/anticipation-asap/l2ryqjx8",
        help="W&B run path entity/project/run_id.",
    )
    parser.add_argument("--download_dir", default="scratch_heatmaps")
    parser.add_argument("--out_prefix", default="heatmaps_run_genai")
    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(args.run)
    print(f"Run: {run.name} ({run.id})  state={run.state}  step={run.summary.get('_step')}")

    tables = find_final_tables(run)
    missing = [m for m in METRICS if m not in tables]
    if missing:
        print(f"WARNING: no heatmap tables found for: {missing}")

    os.makedirs(args.download_dir, exist_ok=True)

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(13, 4.2 * len(METRICS)))
    if len(METRICS) == 1:
        axes = [axes]

    for ax, (metric, (title, cbar_label, cmap_name)) in zip(axes, METRICS.items()):
        if metric not in tables:
            ax.set_visible(False)
            continue
        f = tables[metric]
        f.download(root=args.download_dir, replace=True)
        matrix, steps, slots = table_to_matrix(os.path.join(args.download_dir, f.name))
        print(f"{metric}: {matrix.shape[0]} validations x {matrix.shape[1]} slots "
              f"(steps {steps.min()}..{steps.max()})")
        render_metric(ax, matrix, steps, slots, title, cbar_label, cmap_name)

        # also save an individual figure per metric
        single = plt.figure(figsize=(13, 4.6))
        render_metric(single.gca(), matrix, steps, slots, title, cbar_label, cmap_name)
        single.suptitle(f"{run.name} ({run.id})", y=1.02, fontsize=9)
        single_path = f"{args.out_prefix}_{metric}.png"
        single.savefig(single_path, dpi=150, bbox_inches="tight")
        plt.close(single)
        print(f"  wrote {single_path}")

    fig.suptitle(f"Autoregressive error heatmaps — {run.name} ({run.id})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    combined_path = f"{args.out_prefix}.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote combined figure: {combined_path}")


if __name__ == "__main__":
    main()
