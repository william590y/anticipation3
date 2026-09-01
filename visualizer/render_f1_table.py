#!/usr/bin/env python
"""Render the mean note-level F1 table (PNG + standalone LaTeX) from rl_f1_table.json.

The earlier table was hand-written; this one is generated, so the figure and the
numbers cannot drift apart. Row order, shading, and the caption follow
mean_f1_by_checkpoint.png, with the two RL post-training rows inserted between
the supervised checkpoints and the reference papers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

VARIANTS = ("onset_pitch", "onset_pitch_dur", "onset_pitch_tol1")

INK = "#1B1F24"
MUTED = "#4C5661"
RULE = "#386A8C"
HEADER_BG = "#EAF2F8"
BAND_BG = "#F7F9FB"

# (key, model column, checkpoint column, muted parenthetical, shaded?, rule above?)
LAYOUT = [
    ("base_loss", "Ours (FT)", "best val loss", None, False, False),
    ("base_pitch", "", "best AR pitch", None, True, False),
    ("lora_loss", "Ours (LoRA)", "best val loss", None, False, False),
    ("lora_pitch", "", "best AR pitch", None, True, False),
    ("grpo", "Ours (GRPO)", "best val reward", None, False, True),
    ("ppo", "Ours (PPO)", "best val reward", None, False, False),
    ("paper1", "Paper 1", "Zeng+", "(joint-apt-epr)", False, True),
    ("paper2", "Paper 2", "Beyer & Dai", "(MIDI2ScoreTF)", False, False),
]

CKPT_PAREN = {
    "base_loss": "(ckpt 2500)",
    "base_pitch": "(ckpt 7500)",
    "lora_loss": "(ckpt 10000)",
    "lora_pitch": "(ckpt 15000)",
}


def paren_for(key, row):
    if key in CKPT_PAREN:
        return CKPT_PAREN[key]
    step = row.get("best_step")
    return f"(step {step})" if step is not None else ""


def text_right(ax, artist):
    """Right edge of a drawn text, in axes coordinates."""
    renderer = ax.figure.canvas.get_renderer()
    bbox = artist.get_window_extent(renderer=renderer)
    return ax.transAxes.inverted().transform((bbox.x1, bbox.y0))[0]


def render_png(rows, out_path, title, subtitle):
    n = len(rows)
    fig_h = 1.9 + 0.42 * n
    fig, ax = plt.subplots(figsize=(8.7, fig_h), dpi=200)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    x_model, x_ckpt = 0.015, 0.215
    x_num = (0.665, 0.825, 0.968)

    top = 0.845
    row_h = 0.088
    header_h = 0.082

    ax.text(0.5, 0.975, title, ha="center", va="center",
            fontsize=15.5, fontweight="bold", color=INK)
    ax.text(0.5, 0.905, subtitle, ha="center", va="center",
            fontsize=9.5, color=MUTED)

    # Header band
    ax.add_patch(Rectangle((0, top - header_h), 1, header_h,
                           facecolor=HEADER_BG, edgecolor="none",
                           transform=ax.transAxes, zorder=0))
    ax.plot([0, 1], [top, top], color=RULE, lw=2.0, transform=ax.transAxes, zorder=3)
    ax.plot([0, 1], [top - header_h] * 2, color=RULE, lw=2.0,
            transform=ax.transAxes, zorder=3)
    hy = top - header_h / 2
    ax.text(x_model, hy, "Model", ha="left", va="center",
            fontsize=11, fontweight="bold", color=INK)
    ax.text(x_ckpt, hy, "Checkpoint", ha="left", va="center",
            fontsize=11, fontweight="bold", color=INK)
    for x, label in zip(x_num, ("Onset + pitch", "+ duration", "±1 bin")):
        ax.text(x, hy, label, ha="right", va="center",
                fontsize=11, fontweight="bold", color=INK)

    best = {
        crit: max(row["macro_f1"][crit] for row in rows.values() if row)
        for crit in VARIANTS
    }

    y = top - header_h
    for key, model, ckpt, extra, shaded, rule_above in LAYOUT:
        row = rows.get(key)
        if row is None:
            continue
        y_bottom = y - row_h
        if shaded:
            ax.add_patch(Rectangle((0, y_bottom), 1, row_h, facecolor=BAND_BG,
                                   edgecolor="none", transform=ax.transAxes, zorder=0))
        if rule_above:
            ax.plot([0, 1], [y, y], color=RULE, lw=0.6, alpha=0.45,
                    transform=ax.transAxes, zorder=2)
        cy = y - row_h / 2
        if model:
            ax.text(x_model, cy, model, ha="left", va="center", fontsize=11, color=INK)
        paren = extra if extra is not None else paren_for(key, row)
        label = ax.text(x_ckpt, cy, ckpt, ha="left", va="center",
                        fontsize=10.5, color=MUTED)
        if paren:
            # Place the muted parenthetical against the measured right edge of
            # the label; a fixed offset overlaps as soon as a label changes width.
            ax.text(text_right(ax, label) + 0.014, cy, paren, ha="left",
                    va="center", fontsize=9.5, color=MUTED)
        for x, crit in zip(x_num, VARIANTS):
            value = row["macro_f1"][crit]
            is_best = abs(value - best[crit]) < 1e-12
            ax.text(x, cy, f"{100 * value:.1f}%", ha="right", va="center",
                    fontsize=11, color=INK,
                    fontweight="bold" if is_best else "normal")
        y = y_bottom

    ax.plot([0, 1], [y, y], color=RULE, lw=2.0, transform=ax.transAxes, zorder=3)

    caption = (
        "Macro-average over pieces. “±1 bin” permits an onset error of at "
        "most one 10 ms score-grid bin while requiring an exact pitch match.\n"
        "Shaded rows = checkpoint chosen by AR pitch accuracy.   GRPO and PPO are "
        "post-trained from Ours (FT) best val loss, selected by validation reward\n"
        "(exact-match onset + duration + pitch).   Bold = best in column."
    )
    ax.text(0.5, y - 0.052, caption, ha="center", va="top",
            fontsize=8.2, color=MUTED, linespacing=1.5)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.22, facecolor="white")
    plt.close(fig)


def latex_escape(text):
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def render_tex(rows, out_path, title, subtitle):
    best = {
        crit: max(row["macro_f1"][crit] for row in rows.values() if row)
        for crit in VARIANTS
    }
    body = []
    for key, model, ckpt, extra, shaded, rule_above in LAYOUT:
        row = rows.get(key)
        if row is None:
            continue
        if rule_above:
            body.append(r"    \midrule")
        if shaded:
            body.append(r"    \rowcolor{bandgrey}")
        paren = extra if extra is not None else paren_for(key, row)
        ckpt_cell = latex_escape(ckpt)
        if paren:
            ckpt_cell += r" \textcolor{muted}{" + latex_escape(paren) + "}"
        cells = []
        for crit in VARIANTS:
            value = row["macro_f1"][crit]
            text = f"{100 * value:.1f}\\%"
            if abs(value - best[crit]) < 1e-12:
                text = r"\textbf{" + text + "}"
            cells.append(text)
        body.append(
            f"    {latex_escape(model):<12} & {ckpt_cell} & "
            + " & ".join(cells)
            + r" \\"
        )

    document = r"""\documentclass[border=14pt]{standalone}

\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{array}
\usepackage[table]{xcolor}

\definecolor{headerblue}{HTML}{EAF2F8}
\definecolor{ruleblue}{HTML}{386A8C}
\definecolor{bandgrey}{HTML}{F7F9FB}
\definecolor{muted}{HTML}{4C5661}

\begin{document}
\begin{minipage}{15.4cm}
  \centering
  {\Large\bfseries TITLE\par}
  \vspace{3pt}
  {\small\color{muted}SUBTITLE\par}
  \vspace{10pt}

  \renewcommand{\arraystretch}{1.35}
  \setlength{\tabcolsep}{12pt}
  \begin{tabular}{>{\raggedright\arraybackslash}p{2.5cm}
                  >{\raggedright\arraybackslash}p{4.5cm} r r r}
    \arrayrulecolor{ruleblue}
    \toprule
    \rowcolor{headerblue}
    \textbf{Model} &
    \textbf{Checkpoint} &
    \textbf{Onset + pitch} &
    \textbf{+ duration} &
    \textbf{$\boldsymbol{\pm 1}$ bin} \\
    \midrule
BODY
    \bottomrule
  \end{tabular}

  \vspace{8pt}
  {\footnotesize\color{muted}
    Macro-average over pieces. ``$\pm 1$ bin'' permits an onset error of at most
    one 10\,ms score-grid bin while requiring an exact pitch match.
    Shaded rows are the checkpoint chosen by autoregressive pitch accuracy.
    GRPO and PPO are post-trained from Ours (FT) best val loss and selected by
    validation reward (exact-match onset + duration + pitch).\par}
\end{minipage}
\end{document}
"""
    document = (
        document.replace("TITLE", latex_escape(title))
        .replace("SUBTITLE", latex_escape(subtitle))
        .replace("BODY", "\n".join(body))
    )
    Path(out_path).write_text(document, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="visualizer/rl_f1_table.json")
    ap.add_argument("--png", default="visualizer/mean_f1_with_rl.png")
    ap.add_argument("--tex", default="visualizer/mean_f1_with_rl.tex")
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = {row["key"]: row for row in payload["rows"]}
    n_pieces = max(row["n_pieces"] for row in payload["rows"])
    title = f"Mean Note-Level F1 Across {n_pieces} Pieces"
    subtitle = "Unfiltered performance input; no ground-truth seeding"

    render_png(rows, args.png, title, subtitle)
    render_tex(rows, args.tex, title, subtitle)
    missing = [key for key, *_ in LAYOUT if key not in rows]
    if missing:
        print(f"warning: no data for {missing}")
    print(f"wrote {args.png} and {args.tex}")


if __name__ == "__main__":
    main()
