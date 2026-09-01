#!/usr/bin/env python
"""Regenerate the communications-credit sample figure from the on-disk experiment logs.

Every plotted value is read from a file in this repository; nothing is typed in
by hand.  See SOURCES.md (written alongside the figure) for the value-by-value
provenance table, and ACCESSIBILITY.md for the audit this script measures.

Outputs, all into this script's own directory:
    figure_reward_vs_f1.png        300 dpi raster
    figure_reward_vs_f1.pdf        vector
    figure_reward_vs_f1_gray.png   measured grayscale rendering (accessibility test)
    accessibility_audit.json       measured contrast / CVD / grayscale numbers
    plotted_values.json            every plotted series, verbatim

Run:  python figures/comms_credit/make_figure.py      (from the repo root or anywhere)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / cluster-safe backend, as train.py does
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# --------------------------------------------------------------------------
# Paths.  REPO is resolved from this file so the script runs from anywhere.
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

GRPO_CSV = REPO / "run_grpo_acc_reward" / "ar_val_loss.csv"
PPO_CSV = REPO / "run_ppo_corrected_20260814_020654_2364547" / "ar_val_loss.csv"
PPO_BEST = REPO / "run_ppo_corrected_20260814_020654_2364547" / "best_val_reward.json"
PPOF1_CSV = REPO / "run_ppo_f1_triplet_20260814_135022" / "val_f1.csv"
PPOF1_BEST = REPO / "run_ppo_f1_triplet_20260814_135022" / "best_val_f1.json"
F1_TABLE = REPO / "visualizer" / "rl_f1_table.json"
DIAGNOSIS = REPO / "visualizer" / "reward_vs_f1_diagnosis.json"

# --------------------------------------------------------------------------
# Palette.  Colours are the Okabe-Ito colour-vision-deficiency-safe set plus
# the repository's own ink/muted greys (visualizer/render_f1_table.py).
# Contrast and CVD separation are *measured* below, not assumed.
# --------------------------------------------------------------------------
INK = "#1B1F24"        # repo ink; body text, axes, outlines
MUTED = "#4C5661"      # repo muted grey; reference lines, secondary text
BG = "#FFFFFF"
GRID = "#CFD4D9"

# Matching criterion -> colour.  Ordinal: strictest criterion is darkest.
CRIT_STYLE = {
    "onset_pitch_dur": dict(color=INK, hatch="", ls="-", marker="o",
                            label="onset + pitch + duration (strictest)",
                            short="onset + pitch + duration"),
    "onset_pitch": dict(color="#0072B2", hatch="///", ls="--", marker="s",
                        label="onset + pitch",
                        short="onset + pitch"),
    "onset_pitch_tol1": dict(color="#D55E00", hatch="...", ls="-.", marker="^",
                             label="onset within ±1 bin + pitch",
                             short="onset ±1 bin + pitch"),
}
CRIT_ORDER = ["onset_pitch_dur", "onset_pitch", "onset_pitch_tol1"]

# Post-training arm -> colour (panel a only; models are named on the x axis in b).
ARM_STYLE = {
    "grpo": dict(color=INK, ls="-", marker="o", label="GRPO"),
    "ppo": dict(color="#AA4499", ls="--", marker="s", label="PPO (token-level)"),
}

MODEL_LABELS = {
    "base_loss": "Supervised\ninitialisation",
    "grpo": "GRPO\n(ckpt-250)",
    "ppo": "PPO\n(step 775)",
}


# --------------------------------------------------------------------------
# WCAG 2.x measurement helpers (Cornell adopts WCAG AA; see ACCESSIBILITY.md).
# --------------------------------------------------------------------------
def _srgb_to_linear(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_colour: str) -> float:
    """WCAG 2.x relative luminance L of an sRGB colour."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (0.2126 * _srgb_to_linear(r)
            + 0.7152 * _srgb_to_linear(g)
            + 0.0722 * _srgb_to_linear(b))


def contrast_ratio(a: str, b: str) -> float:
    """WCAG 2.x contrast ratio (L1 + 0.05) / (L2 + 0.05)."""
    l1, l2 = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def to_gray_hex(hex_colour: str) -> str:
    """The neutral grey with the same WCAG relative luminance."""
    lum = relative_luminance(hex_colour)
    v = 12.92 * lum if lum <= 0.0031308 else 1.055 * lum ** (1 / 2.4) - 0.055
    q = max(0, min(255, round(v * 255)))
    return f"#{q:02X}{q:02X}{q:02X}"


def simulate_cvd(hex_colour: str, cvd_type: str, severity: int = 100):
    """Simulated colour under a colour-vision deficiency, or None if unavailable."""
    try:
        from colorspacious import cspace_convert
    except ImportError:
        return None
    h = hex_colour.lstrip("#")
    rgb = np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
    space = {"name": "sRGB1+CVD", "cvd_type": cvd_type, "severity": severity}
    out = np.clip(cspace_convert(rgb, space, "sRGB1"), 0, 1)
    return "#{:02X}{:02X}{:02X}".format(*(int(round(v * 255)) for v in out))


# --------------------------------------------------------------------------
# Data loading.  One function per source file; each returns raw file values.
# --------------------------------------------------------------------------
def read_csv_cols(path: Path, cols):
    rows = list(csv.DictReader(path.open()))
    return {c: [float(r[c]) for r in rows] for c in cols}, len(rows)


def load_all():
    data = {}

    grpo, n_grpo = read_csv_cols(GRPO_CSV, ["step", "REWARD"])
    data["grpo_curve"] = dict(step=grpo["step"], reward=grpo["REWARD"],
                              n_rows=n_grpo, source=str(GRPO_CSV))

    ppo, n_ppo = read_csv_cols(PPO_CSV, ["step", "REWARD"])
    data["ppo_curve"] = dict(step=ppo["step"], reward=ppo["REWARD"],
                             n_rows=n_ppo, source=str(PPO_CSV))

    f1cols = ["step", "REWARD", "f1_onset_pitch", "f1_onset_pitch_dur"]
    pf1, n_pf1 = read_csv_cols(PPOF1_CSV, f1cols)
    data["ppo_f1_curve"] = dict(
        step=pf1["step"],
        onset_pitch_tol1=pf1["REWARD"],          # this run's reward, per train_ppo_f1.py
        onset_pitch=pf1["f1_onset_pitch"],
        onset_pitch_dur=pf1["f1_onset_pitch_dur"],
        n_rows=n_pf1, source=str(PPOF1_CSV))

    table = json.loads(F1_TABLE.read_text())
    rows = {r["key"]: r for r in table["rows"]}
    data["bars"] = {
        k: dict(macro_f1=rows[k]["macro_f1"],
                checkpoint=rows[k]["checkpoint"],
                n_windows=rows[k]["n_windows"],
                n_pieces=rows[k]["n_pieces"],
                variant=rows[k]["variant"])
        for k in ("base_loss", "grpo", "ppo")
    }
    data["bars_source"] = str(F1_TABLE)

    data["ppo_best"] = json.loads(PPO_BEST.read_text())
    data["ppo_f1_best"] = json.loads(PPOF1_BEST.read_text())

    diag = json.loads(DIAGNOSIS.read_text())
    data["diagnosis"] = {
        slice_: {name: block["f1"] for name, block in diag[slice_].items()}
        for slice_ in ("raw", "filtered", "filtered_val", "filtered_test")
    }
    data["diagnosis_source"] = str(DIAGNOSIS)
    return data


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.9,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
    })


def running_median(y, window=9):
    """Centred running median of a 1-D list; a declared, labelled smoothing of
    the raw validation points, which are also drawn."""
    y = np.asarray(y, dtype=float)
    half = window // 2
    out = np.empty_like(y)
    for i in range(len(y)):
        lo, hi = max(0, i - half), min(len(y), i + half + 1)
        out[i] = np.median(y[lo:hi])
    return out


def build(data):
    style()
    fig = plt.figure(figsize=(6.8, 6.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.97],
                          hspace=0.40, wspace=0.30,
                          left=0.105, right=0.985, top=0.945, bottom=0.085)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # ---- (a) the reward that GRPO and PPO were trained on -----------------
    init_reward = data["grpo_curve"]["reward"][0]
    assert data["ppo_curve"]["reward"][0] == init_reward, "arms must share the init"

    for key, series in (("grpo", data["grpo_curve"]), ("ppo", data["ppo_curve"])):
        st = ARM_STYLE[key]
        ax_a.plot(series["step"], series["reward"], color=st["color"],
                  ls="-", lw=0.6, alpha=0.30, zorder=2)
        med = running_median(series["reward"], 9)
        every = max(1, len(series["step"]) // 10)
        ax_a.plot(series["step"], med, color=st["color"], ls=st["ls"], lw=1.5,
                  marker=st["marker"], markersize=3.6, markevery=every,
                  markerfacecolor=BG, markeredgewidth=1.0,
                  label=st["label"], zorder=4)

    ax_a.axhline(init_reward, color=MUTED, ls=":", lw=1.4, zorder=3)

    grpo_sel_step = 250
    gi = data["grpo_curve"]["step"].index(float(grpo_sel_step))
    ppo_sel_step = int(data["ppo_best"]["step"])
    pi = data["ppo_curve"]["step"].index(float(ppo_sel_step))
    for x, y, col in ((grpo_sel_step, data["grpo_curve"]["reward"][gi], INK),
                      (ppo_sel_step, data["ppo_curve"]["reward"][pi],
                       ARM_STYLE["ppo"]["color"])):
        ax_a.plot([x], [y], marker="*", markersize=11, color=col,
                  markerfacecolor=col, markeredgecolor=BG,
                  markeredgewidth=0.8, zorder=6, linestyle="none")

    ax_a.set_xlabel("Post-training optimiser step (count)")
    ax_a.set_ylabel("Validation reward\n(onset + duration + pitch accuracy, 0–3)")
    ax_a.set_xlim(-140, 5150)
    ax_a.set_ylim(1.20, 1.72)
    ax_a.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    ax_a.set_yticks([1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    ax_a.grid(True, axis="y", zorder=0)
    ax_a.set_axisbelow(True)
    handles, labels = ax_a.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="*", markersize=9, linestyle="none",
                          color=INK, label="checkpoint scored in (b)"))
    handles.append(Line2D([], [], color=MUTED, lw=1.4, ls=":",
                          label=f"supervised init. ({init_reward:.3f})"))
    ax_a.legend(handles=handles, loc="lower right", frameon=True,
                framealpha=1.0, edgecolor=GRID, borderpad=0.35,
                handlelength=2.2, fontsize=7.2, labelspacing=0.32)
    ax_a.text(0.025, 0.968, "pale line: every validation;\nbold: 9-point running median",
              transform=ax_a.transAxes, fontsize=7.0, color=MUTED,
              ha="left", va="top", linespacing=1.4)
    ax_a.set_title("(a) Reward climbs during RL", loc="left",
                   fontsize=8.8, fontweight="bold", pad=5)

    # ---- (b) note-level F1 of those same checkpoints ----------------------
    models = ["base_loss", "grpo", "ppo"]
    xs = np.arange(len(models), dtype=float)
    offsets = (-0.28, 0.0, 0.28)
    width = 0.25
    for crit, off in zip(CRIT_ORDER, offsets):
        st = CRIT_STYLE[crit]
        vals = [100.0 * data["bars"][m]["macro_f1"][crit] for m in models]
        ax_b.bar(xs + off, vals, width=width,
                 color=st["color"], edgecolor=INK, linewidth=0.8,
                 hatch=st["hatch"], label=st["short"], zorder=3)
        for x, v in zip(xs + off, vals):
            ax_b.text(x, v + 0.5, f"{v:.1f}", ha="center", va="bottom",
                      fontsize=7.4, color=INK, zorder=4)

    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=7.8)
    ax_b.set_ylabel("Macro note-level F1 (%)")
    ax_b.set_ylim(0, 35)
    ax_b.set_yticks([0, 5, 10, 15, 20, 25])
    ax_b.grid(True, axis="y", zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor=GRID,
                fontsize=7.2, handlelength=1.6, borderpad=0.35,
                labelspacing=0.35, title="note-matching criterion",
                title_fontsize=7.2)
    ax_b.set_title("(b) Note-level F1 does not", loc="left",
                   fontsize=8.8, fontweight="bold", pad=5)

    # ---- (c) PPO-F1: when the reward IS the F1 ---------------------------
    cur = data["ppo_f1_curve"]
    for crit in CRIT_ORDER:
        st = CRIT_STYLE[crit]
        vals = [100.0 * v for v in cur[crit]]
        every = max(1, len(cur["step"]) // 14)
        lbl = st["short"] + (" — this run's reward"
                             if crit == "onset_pitch_tol1" else "")
        ax_c.plot(cur["step"], vals, color=st["color"], ls="-", lw=0.6,
                  alpha=0.30, zorder=2)
        ax_c.plot(cur["step"], running_median(vals, 9), color=st["color"],
                  ls=st["ls"], lw=1.5, marker=st["marker"], markersize=3.6,
                  markevery=every, markerfacecolor=BG, markeredgewidth=1.0,
                  label=lbl, zorder=4)
        ax_c.axhline(vals[0], color=st["color"], ls=":", lw=0.9,
                     alpha=0.65, zorder=3)

    best_step = int(data["ppo_f1_best"]["step"])
    best_val = 100.0 * data["ppo_f1_best"]["val_f1"]
    ax_c.plot([best_step], [best_val], marker="*", markersize=11,
              color=CRIT_STYLE["onset_pitch_tol1"]["color"],
              markeredgecolor=BG, markeredgewidth=0.8, linestyle="none",
              zorder=6, label=f"best checkpoint (step {best_step}, {best_val:.1f}%)")

    ax_c.set_xlabel("Post-training optimiser step (count)")
    ax_c.set_ylabel("Validation note-level F1 (%)")
    ax_c.set_xlim(-140, 5560)
    ax_c.set_ylim(15, 57)
    ax_c.set_yticks([15, 20, 25, 30, 35, 40, 45, 50])
    ax_c.grid(True, axis="y", zorder=0)
    ax_c.set_axisbelow(True)
    ax_c.legend(loc="upper left", ncol=2, frameon=True, framealpha=1.0,
                edgecolor=GRID, fontsize=7.2, handlelength=2.2,
                borderpad=0.35, columnspacing=1.4, labelspacing=0.35)
    ax_c.set_title("(c) When the reward is note-level F1, note-level F1 climbs",
                   loc="left", fontsize=8.8, fontweight="bold", pad=5)
    ax_c.text(0.02, 0.025,
              "pale lines: every validation;  bold: 9-point running median;"
              "  dotted: value at step 0",
              transform=ax_c.transAxes, fontsize=7.0, color=MUTED,
              ha="left", va="bottom")

    return fig


# --------------------------------------------------------------------------
# Accessibility audit: measure, do not assert.
# --------------------------------------------------------------------------
def audit(png_path: Path, gray_path: Path):
    report = {"standard": "WCAG 2.2 Level AA (adopted by Cornell University "
                          "Policy 5.12 / it.cornell.edu/accessibility)"}

    inks = {"body ink (INK)": INK, "secondary text (MUTED)": MUTED,
            "gridlines (non-essential)": GRID}
    report["text_contrast_vs_background"] = {
        name: {"colour": c, "contrast_ratio": round(contrast_ratio(c, BG), 2),
               "wcag_1_4_3_min_normal_text": 4.5,
               "passes": contrast_ratio(c, BG) >= 4.5}
        for name, c in inks.items() if "grid" not in name
    }
    report["gridline_contrast_vs_background"] = {
        "colour": GRID, "contrast_ratio": round(contrast_ratio(GRID, BG), 2),
        "note": ("gridlines are not required to read the figure (every bar is "
                 "labelled with its value and every axis is ticked), so SC "
                 "1.4.11 does not apply to them"),
    }

    series = {}
    for crit in CRIT_ORDER:
        series[CRIT_STYLE[crit]["label"]] = CRIT_STYLE[crit]["color"]
    for k, st in ARM_STYLE.items():
        series[st["label"]] = st["color"]
    report["data_mark_contrast_vs_background"] = {
        name: {"colour": c, "contrast_ratio": round(contrast_ratio(c, BG), 2),
               "wcag_1_4_11_min": 3.0, "passes": contrast_ratio(c, BG) >= 3.0}
        for name, c in series.items()
    }

    # Pairwise separation only matters between marks that share a panel.
    panels = {
        "panel_a": {ARM_STYLE[k]["label"]: ARM_STYLE[k]["color"] for k in ARM_STYLE},
        "panels_b_and_c": {CRIT_STYLE[c]["label"]: CRIT_STYLE[c]["color"]
                           for c in CRIT_ORDER},
    }
    report["pairwise_series_contrast_within_panel"] = {}
    for pname, pal in panels.items():
        nm = list(pal)
        report["pairwise_series_contrast_within_panel"][pname] = {
            f"{nm[i]} vs {nm[j]}": round(contrast_ratio(pal[nm[i]], pal[nm[j]]), 2)
            for i in range(len(nm)) for j in range(i + 1, len(nm))
        }
    names = list(series)

    report["grayscale_luminance"] = {
        name: {"colour": c,
               "relative_luminance": round(relative_luminance(c), 4),
               "equivalent_grey": to_gray_hex(c)}
        for name, c in series.items()
    }

    cvd = {}
    for cvd_type in ("deuteranomaly", "protanomaly", "tritanomaly"):
        sim = {name: simulate_cvd(c, cvd_type, 100) for name, c in series.items()}
        if any(v is None for v in sim.values()):
            cvd[cvd_type] = "colorspacious not installed - simulation not run"
            continue
        cvd[cvd_type] = {
            "simulated_colours": sim,
            "min_contrast_vs_background": round(
                min(contrast_ratio(v, BG) for v in sim.values()), 2),
            "pairwise_within_panel": {
                pname: {
                    f"{a} vs {b}": round(contrast_ratio(sim[a], sim[b]), 2)
                    for i, a in enumerate(list(pal))
                    for b in list(pal)[i + 1:]
                }
                for pname, pal in panels.items()
            },
        }
    report["cvd_simulation_severity_100"] = cvd

    report["redundant_encodings"] = {
        "panel_a": "line style (solid / dashed) + marker shape (circle / square)",
        "panel_b": ("hatch (none / diagonal / dotted) + fixed bar order + a printed "
                    "numeric value above every bar + white gaps so no two fills touch"),
        "panel_c": ("line style (solid / dashed / dash-dot) + marker shape "
                    "(circle / square / triangle)"),
    }

    # measured grayscale rendering of the finished raster
    from PIL import Image
    img = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.float64) / 255.0
    lin = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    lum = lin @ np.array([0.2126, 0.7152, 0.0722])
    enc = np.where(lum <= 0.0031308, 12.92 * lum, 1.055 * lum ** (1 / 2.4) - 0.055)
    Image.fromarray((np.clip(enc, 0, 1) * 255).round().astype(np.uint8),
                    mode="L").save(gray_path)
    report["grayscale_render"] = {
        "path": str(gray_path),
        "distinct_grey_levels_in_render": int(np.unique(
            (np.clip(enc, 0, 1) * 255).round().astype(np.uint8)).size),
    }

    report["font_sizes_pt"] = {
        "axis labels": 9.0,
        "panel titles (bold)": 8.8,
        "tick labels": 8.5,
        "panel (b) category tick labels": 7.8,
        "bar value labels": 7.4,
        "legends": 7.2,
        "in-axes explanatory notes (smallest text in the figure)": 7.0,
    }
    report["smallest_text_pt"] = 7.0
    report["figure_size_in"] = [6.8, 6.1]
    report["dpi"] = 300
    return report


def main():
    data = load_all()
    fig = build(data)

    png = HERE / "figure_reward_vs_f1.png"
    pdf = HERE / "figure_reward_vs_f1.pdf"
    gray = HERE / "figure_reward_vs_f1_gray.png"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    report = audit(png, gray)
    (HERE / "accessibility_audit.json").write_text(json.dumps(report, indent=2) + "\n")

    plotted = {
        "panel_a": {"grpo": data["grpo_curve"], "ppo": data["ppo_curve"],
                    "ppo_best_val_reward": data["ppo_best"]},
        "panel_b": {"bars": data["bars"], "source": data["bars_source"]},
        "panel_c": {"curve": data["ppo_f1_curve"], "best": data["ppo_f1_best"]},
        "cross_check": {"reward_vs_f1_diagnosis": data["diagnosis"],
                        "source": data["diagnosis_source"]},
    }
    (HERE / "plotted_values.json").write_text(json.dumps(plotted, indent=1) + "\n")

    print(f"wrote {png}")
    print(f"wrote {pdf}")
    print(f"wrote {gray}")
    print("\n-- measured contrast (WCAG 2.x) --")
    for k, v in report["text_contrast_vs_background"].items():
        print(f"  text  {k:<28} {v['colour']}  {v['contrast_ratio']:>6.2f}:1  "
              f"{'PASS' if v['passes'] else 'FAIL'} (>=4.5)")
    for k, v in report["data_mark_contrast_vs_background"].items():
        print(f"  mark  {k:<40} {v['colour']}  {v['contrast_ratio']:>6.2f}:1  "
              f"{'PASS' if v['passes'] else 'FAIL'} (>=3.0)")
    print("\n-- pairwise series contrast, within a panel --")
    for pname, pal in report["pairwise_series_contrast_within_panel"].items():
        print(f"  [{pname}]")
        for k, v in pal.items():
            print(f"    {k:<62} {v:>6.2f}:1")
    print("\n-- grayscale equivalents --")
    for k, v in report["grayscale_luminance"].items():
        print(f"  {k:<44} {v['colour']} -> {v['equivalent_grey']} (L={v['relative_luminance']:.4f})")
    print("\n-- CVD simulation (severity 100) --")
    for t, v in report["cvd_simulation_severity_100"].items():
        if isinstance(v, str):
            print(f"  {t}: {v}")
        else:
            print(f"  {t}: min contrast vs background {v['min_contrast_vs_background']:.2f}:1")
            for pname, pal in v["pairwise_within_panel"].items():
                print(f"    [{pname}]")
                for k2, v2 in pal.items():
                    print(f"      {k2:<60} {v2:>6.2f}:1")


if __name__ == "__main__":
    main()
