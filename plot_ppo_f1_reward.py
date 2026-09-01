#!/usr/bin/env python
"""Smoothed train/REWARD (and validation) for the triplet-action PPO-F1 run.

train/REWARD is a rollout's F1 on AUGMENTED training windows sampled at T=0.7, so
it is far noisier than the clean greedy validation number; a raw trace mostly
shows which windows happened to be drawn. This plots an EMA over it and reports
whether the trend is real (OLS slope with a t-test, plus a first-vs-last quarter
comparison that assumes nothing about linearity).
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEP_RE = re.compile(
    r"^step (\d+) \| train REWARD ([\d.]+)% \(total ([\d.]+)%\).*?"
    r"KL\(ref\) ([-\d.]+).*?EV\(lambda\) ([-\d.]+)"
)

INK, ACCENT, MUTED = "#1B1F24", "#386A8C", "#8892A0"
GOOD = "#2E7D5B"


def parse_log(path):
    steps, reward, kl, ev = [], [], [], []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = STEP_RE.match(line)
        if m:
            steps.append(int(m.group(1)))
            reward.append(float(m.group(2)))
            kl.append(float(m.group(4)))
            ev.append(float(m.group(5)))
    return steps, reward, kl, ev


def parse_val(path):
    if not Path(path).is_file():
        return [], [], [], []
    steps, r, op, opd = [], [], [], []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            steps.append(int(row["step"]))
            r.append(100 * float(row["REWARD"]))
            op.append(100 * float(row["f1_onset_pitch"]))
            opd.append(100 * float(row["f1_onset_pitch_dur"]))
    return steps, r, op, opd


def ema(values, alpha=0.12):
    out, running = [], values[0]
    for v in values:
        running = alpha * v + (1 - alpha) * running
        out.append(running)
    return out


def ols_trend(x, y):
    """Slope per 100 steps, with a two-sided t-test against zero slope."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = my - slope * mx
    residuals = [yi - (intercept + slope * xi) for xi, yi in zip(x, y)]
    dof = n - 2
    s2 = sum(r * r for r in residuals) / dof
    se = math.sqrt(s2 / sxx)
    t = slope / se if se else float("inf")
    return slope * 100, t, dof


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="visualizer/ppo_f1_reward_curve.png")
    ap.add_argument("--alpha", type=float, default=0.12)
    args = ap.parse_args()

    steps, reward, kl, ev = parse_log(args.log)
    if not steps:
        raise SystemExit(f"no step lines parsed from {args.log}")
    vsteps, vreward, vop, vopd = parse_val(args.val)
    smooth = ema(reward, args.alpha)

    slope, t, dof = ols_trend(steps, reward)
    half = len(reward) // 4
    first_q = sum(reward[:half]) / half
    last_q = sum(reward[-half:]) / half

    print(f"{len(steps)} points, steps {steps[0]}-{steps[-1]}")
    print(f"train/REWARD  OLS slope {slope:+.4f} %-points per 100 steps "
          f"(t={t:+.2f}, dof={dof})")
    print(f"              first quarter {first_q:.2f}%  ->  last quarter {last_q:.2f}%  "
          f"({last_q - first_q:+.2f})")
    verdict = ("improving" if t > 2 else "declining" if t < -2 else "FLAT (no significant trend)")
    print(f"              verdict: {verdict}")
    if vsteps:
        print(f"val/REWARD    {vreward[0]:.2f}% -> {vreward[-1]:.2f}% "
              f"({vreward[-1] - vreward[0]:+.2f}), best {max(vreward):.2f}% "
              f"at step {vsteps[vreward.index(max(vreward))]}")

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), dpi=180, sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1.5, 1.0]})

    ax = axes[0]
    ax.plot(steps, reward, color=MUTED, lw=0.9, alpha=0.55, label="raw (per 10 steps)")
    ax.plot(steps, smooth, color=ACCENT, lw=2.4, label=f"EMA (alpha={args.alpha})")
    fit = [(sum(reward) / len(reward)) + (slope / 100) * (s - sum(steps) / len(steps))
           for s in steps]
    ax.plot(steps, fit, color=INK, lw=1.2, ls="--",
            label=f"OLS {slope:+.3f} pts/100 steps (t={t:+.1f})")
    ax.set_ylabel("train/REWARD  (rollout F1, %)")
    ax.set_title("Triplet-action PPO-F1: is train reward improving?",
                 fontsize=13, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.15)

    ax = axes[1]
    if vsteps:
        ax.plot(vsteps, vreward, color=GOOD, lw=2.0, marker="o", ms=3.2,
                label="val REWARD (col 3, greedy)")
        ax.plot(vsteps, vop, color=ACCENT, lw=1.3, alpha=0.8, label="val onset+pitch")
        ax.plot(vsteps, vopd, color=MUTED, lw=1.3, alpha=0.8, label="val +duration")
        ax.axhline(vreward[0], color=GOOD, ls=":", lw=1.0, alpha=0.7)
        ax.annotate(f"baseline {vreward[0]:.2f}%", (vsteps[0], vreward[0]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color=GOOD)
    ax.set_ylabel("validation F1 (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.15)

    ax = axes[2]
    ax.plot(steps, ev, color="#8A5A2B", lw=1.6, label="critic EV")
    ax.plot(steps, kl, color="#7B3F8C", lw=1.6, label="KL to reference")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("critic / drift")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
