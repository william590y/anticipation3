#!/usr/bin/env python
"""Pick a run's highest-AR-pitch-accuracy checkpoint that was actually SAVED.

Parses a training log for autoregressive pitch accuracy readings and pairs each
with the training step it was reported at, then returns the best step for which
`<output_dir>/checkpoint-<step>` exists on disk. (Validations run more often than
checkpoints are saved, so the best validation is not always a saved checkpoint.)

Each accuracy line is attributed to the step from the nearest preceding tqdm
progress counter rather than to a running validation counter -- validation and
accuracy lines do not appear 1:1 in these logs.
"""
import argparse
import re
from pathlib import Path

STEP_RE = re.compile(r"(\d+)/(\d+) \[")
ACC_RE = re.compile(r"Autoregressive .*?Pitch(?: Accuracy)?[: ]+([0-9.]+)", re.I)
ALT_ACC_RE = re.compile(r"ar_pitch_accuracy[\"']?[:= ]+([0-9.]+)", re.I)


def parse(log_path):
    """Return {step: accuracy} from the log."""
    text = Path(log_path).read_text(errors="ignore").replace("\r", "\n")
    lines = text.split("\n")

    # The log interleaves several tqdm bars (training, teacher-forced validation,
    # autoregressive validation), and only the training bar's numerator is the
    # training step. Pick it out by its total: it counts to max_steps, which is
    # far larger than any per-validation bar's length.
    totals = [int(m.group(2)) for line in lines for m in STEP_RE.finditer(line)]
    if not totals:
        return {}
    train_total = max(totals)

    readings = {}
    last_step = None
    for line in lines:
        for m in STEP_RE.finditer(line):
            if int(m.group(2)) == train_total:
                last_step = int(m.group(1))
        m = ACC_RE.search(line) or ALT_ACC_RE.search(line)
        if m and last_step is not None:
            val = float(m.group(1))
            if val <= 1.0:
                val *= 100.0
            readings[last_step] = val
    return readings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--quiet", action="store_true",
                    help="Print only the chosen checkpoint path.")
    args = ap.parse_args()

    readings = parse(args.log)
    out = Path(args.output_dir)
    saved = {int(p.name.split("-")[1]): p for p in out.glob("checkpoint-*")
             if p.name.split("-")[-1].isdigit()}
    if not saved:
        raise SystemExit(f"no checkpoints found in {out}")

    # train.py validates at every --save_steps as well as every --eval_steps, so
    # each saved checkpoint normally has a reading at its own step. Prefer those
    # exact pairings; only if none line up fall back to attributing each reading
    # to the nearest saved checkpoint at or before it, which is a rougher proxy
    # (it credits a checkpoint with accuracy the model reached after it).
    TOL = 50
    scored = {}
    for step, acc in readings.items():
        exact = [s for s in saved if abs(s - step) <= TOL]
        if exact:
            scored.setdefault(min(exact, key=lambda s: abs(s - step)), []).append(acc)
    exact_match = bool(scored)
    if not exact_match:
        for step, acc in readings.items():
            eligible = [s for s in saved if s <= step + TOL]
            if eligible:
                scored.setdefault(max(eligible), []).append(acc)
    best = max(scored, key=lambda s: max(scored[s])) if scored else max(saved)

    if not args.quiet:
        if scored and not exact_match:
            print("  (no reading at a checkpoint step; using nearest-preceding "
                  "attribution)")
        for s in sorted(scored):
            print(f"  checkpoint-{s}: AR pitch {max(scored[s]):.2f}%"
                  + ("   <-- best" if s == best else ""))
        if not scored:
            print(f"  (no AR readings parsed; falling back to latest checkpoint-{best})")
    print(saved[best])


if __name__ == "__main__":
    main()
