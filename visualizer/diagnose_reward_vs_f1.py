#!/usr/bin/env python
"""Why did validation reward rise while note-level F1 did not?

Two candidate explanations, both decidable from data.js alone:

  H1  metric shape.  The reward is `onset_acc + duration_acc + pitch_acc`, three
      marginals scored independently at each slot (onpolicy_rollout.per_rollout_accuracy).
      F1 keys a note on the *conjunction* (onset, pitch) and matches one-to-one.
      Credit earned on one field at a slot whose other field is wrong moves the
      reward and cannot move F1.

  H2  distribution shift.  Training validated on 96 windows of val_paper.txt with
      the model's normal *filtered* input; the F1 table uses the 24 visualizer
      windows (12 val + 12 test) on the *unfiltered* stream.

So for every model/variant this recomputes the reward's own definition on the
same windows F1 uses, splits it val/test, and reports the joint accuracies that
bridge the two metrics.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

MODELS = [
    ("base (ckpt 2500)", "rollouts_valloss"),
    ("GRPO (step 250)", "rollouts_grpo"),
    ("PPO (step 775)", "rollouts_ppo"),
]
STREAMS = ("filtered", "raw")


def load_payload(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text[text.find("{") : text.rfind("}") + 1])


def slot_stats(pred, gt):
    """Positional (slot-indexed) exact-match counts -- the reward's own criterion."""
    n = min(len(pred), len(gt))
    out = defaultdict(int)
    out["slots"] = n
    out["len_mismatch"] = int(len(pred) != len(gt))
    gt_onsets = {note["t"] for note in gt}
    gt_keys = {(note["t"], note["p"]) for note in gt}
    onset_errors = []
    wrong_onset = 0
    for a, b in zip(pred[:n], gt[:n]):
        if a["t"] != b["t"]:
            wrong_onset += 1
            onset_errors.append(abs(a["t"] - b["t"]))
            # Does a wrong onset at least land somewhere the window really has a
            # note?  That is the only way an off-slot F1 match can happen.
            on_grid = a["t"] in gt_onsets
            out["wrong_onset_on_grid"] += on_grid
            # ...and having landed there, is the pitch the one that window
            # actually has at that time?  That is a whole correct note emitted at
            # the wrong slot -- invisible to the slot-indexed reward, credited in
            # full by order-free F1.
            if on_grid:
                out["drifted_but_coherent"] += (a["t"], a["p"]) in gt_keys
    out["wrong_onset"] = wrong_onset
    out["onset_abs_err_sum"] = sum(onset_errors)
    for a, b in zip(pred[:n], gt[:n]):
        on = a["t"] == b["t"]
        du = a["d"] == b["d"]
        pi = a["p"] == b["p"]
        out["onset"] += on
        out["dur"] += du
        out["pitch"] += pi
        out["onset_and_pitch"] += on and pi
        out["all_three"] += on and du and pi
        # Where does marginal credit land relative to the joint?
        out["onset_ok_pitch_bad"] += on and not pi
        out["pitch_ok_onset_bad"] += pi and not on
        out["dur_ok_onset_bad"] += du and not on
        out["onset_within_1"] += abs(a["t"] - b["t"]) <= 1
        out["tol1_and_pitch"] += (abs(a["t"] - b["t"]) <= 1) and pi
    return out


def aggregate(payload, group, stream, window_filter=None):
    """Macro-average over windows (each window equally weighted, as F1 is)."""
    examples = payload["examples"]
    order = [
        eid for eid in (payload.get("example_order") or examples)
        if window_filter is None or window_filter(eid)
    ]
    rates = defaultdict(list)
    f1s = defaultdict(list)
    n_used = 0
    for eid in order:
        block = (examples[eid].get(group) or {}).get(stream)
        if not isinstance(block, dict):
            continue
        pred = [n for n in block.get("pred_score") or [] if n.get("p") is not None]
        gt = [n for n in examples[eid].get("gt_score") or [] if n.get("p") is not None]
        if not pred or not gt:
            continue
        stats = slot_stats(pred, gt)
        n_used += 1
        for key, value in stats.items():
            if key in ("slots", "len_mismatch"):
                rates[key].append(value)
            else:
                rates[key].append(value / stats["slots"])
        # How much distinct material the prediction offers the one-to-one matcher.
        rates["distinct_keys"].append(len({(n["t"], n["p"]) for n in pred}))
        rates["distinct_onsets"].append(len({n["t"] for n in pred}))
        rates["gt_distinct_keys"].append(len({(n["t"], n["p"]) for n in gt}))
        rates["gt_distinct_onsets"].append(len({n["t"] for n in gt}))
        for crit, scores in (block.get("f1") or {}).items():
            f1s[crit].append(scores["f1"])
    if not n_used:
        return None
    mean = {k: sum(v) / len(v) for k, v in rates.items()}
    mean["n_windows"] = n_used
    mean["f1"] = {k: sum(v) / len(v) for k, v in f1s.items()}
    return mean


def show(title, rows):
    print(f"\n{title}")
    print(
        f"  {'model':<18} {'reward':>7} {'onset':>7} {'dur':>7} {'pitch':>7} "
        f"{'on&pi':>7} {'all3':>7} | {'F1 o+p':>7} {'F1+dur':>7} {'F1 ±1':>7}"
    )
    for label, m in rows:
        if m is None:
            print(f"  {label:<18} (absent)")
            continue
        reward = m["onset"] + m["dur"] + m["pitch"]
        f1 = m["f1"]
        print(
            f"  {label:<18} {reward:>7.3f} {100*m['onset']:>6.1f}% {100*m['dur']:>6.1f}% "
            f"{100*m['pitch']:>6.1f}% {100*m['onset_and_pitch']:>6.1f}% "
            f"{100*m['all_three']:>6.1f}% | "
            f"{100*f1.get('onset_pitch', 0):>6.1f}% "
            f"{100*f1.get('onset_pitch_dur', 0):>6.1f}% "
            f"{100*f1.get('onset_pitch_tol1', 0):>6.1f}%"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--output", default="visualizer/reward_vs_f1_diagnosis.json")
    args = ap.parse_args()

    payload = load_payload(args.data)
    record = {}

    for stream in STREAMS:
        rows = [(label, aggregate(payload, group, stream)) for label, group in MODELS]
        show(f"=== {stream} stream, all 24 windows ===", rows)
        record[stream] = {label: m for label, m in rows}

    for split in ("val", "test"):
        rows = [
            (label, aggregate(payload, group, "filtered",
                              lambda eid, s=split: eid.startswith(s)))
            for label, group in MODELS
        ]
        show(f"=== filtered stream, {split} windows only ===", rows)
        record[f"filtered_{split}"] = {label: m for label, m in rows}

    print("\n=== set-matching slack: order-free F1 matches vs slot-aligned ones ===")
    print(f"  {'model':<18} {'stream':<9} {'slot-aligned':>13} {'F1 matches':>11} "
          f"{'off-slot':>9} {'distinct (t,p)':>15} {'distinct onsets':>16}")
    for stream in STREAMS:
        for label, group in MODELS:
            m = aggregate(payload, group, stream)
            if m is None:
                continue
            # n_pred == n_gt == 138 for our rollouts, so F1 == tp/138 exactly.
            slot = 138 * m["onset_and_pitch"]
            f1_tp = 138 * m["f1"].get("onset_pitch", 0.0)
            print(
                f"  {label:<18} {stream:<9} {slot:>13.1f} {f1_tp:>11.1f} "
                f"{f1_tp - slot:>9.1f} {m['distinct_keys']:>15.1f} "
                f"{m['distinct_onsets']:>16.1f}"
            )

    print("\n=== behaviour of the wrong onsets (the only source of off-slot matches) ===")
    print(f"  {'model':<18} {'stream':<9} {'wrong onsets':>13} {'land on a real gt onset':>24} "
          f"{'mean |onset err| bins':>22} {'...and pitch fits it':>22}")
    for stream in STREAMS:
        for label, group in MODELS:
            m = aggregate(payload, group, stream)
            if m is None:
                continue
            wrong = m["wrong_onset"]
            on_grid_n = m["wrong_onset_on_grid"]
            on_grid = on_grid_n / wrong if wrong else 0.0
            mean_err = m["onset_abs_err_sum"] / wrong if wrong else 0.0
            coherent = m["drifted_but_coherent"] / on_grid_n if on_grid_n else 0.0
            print(
                f"  {label:<18} {stream:<9} {138 * wrong:>13.1f} "
                f"{100 * on_grid:>23.1f}% {mean_err:>22.1f} "
                f"{100 * coherent:>21.1f}%"
            )

    print("\n=== where the marginal credit lands (filtered, 24 windows) ===")
    print(f"  {'model':<18} {'onset ok / pitch wrong':>24} {'pitch ok / onset wrong':>24} "
          f"{'dur ok / onset wrong':>22}")
    for label, group in MODELS:
        m = aggregate(payload, group, "filtered")
        if m is None:
            continue
        print(
            f"  {label:<18} {100*m['onset_ok_pitch_bad']:>23.1f}% "
            f"{100*m['pitch_ok_onset_bad']:>23.1f}% {100*m['dur_ok_onset_bad']:>21.1f}%"
        )

    Path(args.output).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
