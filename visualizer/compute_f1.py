#!/usr/bin/env python
"""Note-level F1 for predicted vs ground-truth score notes (replaces MUSTER).

CPU-only post-process over a visualizer data.js: for every window and every
rollout variant present (ours base / ours LoRA / ours masked / masked 40k /
paper 1 / paper 2, plus any extra ``rollouts*`` key) it scores the predicted
score notes against that window's `gt_score`
and writes the result back as `<rollout>.f1`.

Three matching criteria are reported side by side (all requested):

  onset_pitch      pitch AND onset must match exactly (on the window's bin grid)
  onset_pitch_dur  pitch AND onset AND duration must all match exactly
  onset_pitch_tol1 pitch must match, onset may differ by up to +-1 bin

Matching is one-to-one: a ground-truth note can be consumed by at most one
predicted note, so duplicated predictions cannot inflate the score. For the
tolerant variant that means a greedy nearest-onset assignment over candidates
sharing a pitch (exact variants are order-independent multiset intersections).

Notes are compared on the window-local grid the tokenizer already established
(`min_score_time_units`), so both sides share one fixed origin and no
re-anchoring is applied -- same reasoning as compute_muster.py.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

VARIANTS = ("onset_pitch", "onset_pitch_dur", "onset_pitch_tol1")


def _prf(tp, n_pred, n_gt):
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": n_pred - tp, "fn": n_gt - tp,
            "n_pred": n_pred, "n_gt": n_gt}


def _match_exact(pred, gt, keyfn):
    """One-to-one multiset intersection on an exact key."""
    gt_counts = Counter(keyfn(n) for n in gt)
    tp = 0
    for n in pred:
        k = keyfn(n)
        if gt_counts.get(k, 0) > 0:
            gt_counts[k] -= 1
            tp += 1
    return tp


def _match_tolerant(pred, gt, tol):
    """Greedy one-to-one match: same pitch, |onset difference| <= tol.
    Predictions are taken in onset order and each takes the closest unused GT
    note of that pitch, so a prediction cannot consume a GT note that an exactly
    matching prediction still needs."""
    by_pitch = defaultdict(list)
    for i, n in enumerate(gt):
        by_pitch[n["p"]].append([n["t"], False])  # [onset, used]
    tp = 0
    for n in sorted(pred, key=lambda x: x["t"]):
        cands = by_pitch.get(n["p"])
        if not cands:
            continue
        best, best_d = None, None
        for entry in cands:
            if entry[1]:
                continue
            d = abs(entry[0] - n["t"])
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = entry, d
        if best is not None:
            best[1] = True
            tp += 1
    return tp


def score_notes(pred, gt):
    """All three F1 variants for one window/model. `pred`/`gt` are lists of
    {"t": onset_bins, "d": duration_bins, "p": midi_pitch}."""
    n_pred, n_gt = len(pred), len(gt)
    return {
        "onset_pitch": _prf(
            _match_exact(pred, gt, lambda n: (n["t"], n["p"])), n_pred, n_gt),
        "onset_pitch_dur": _prf(
            _match_exact(pred, gt, lambda n: (n["t"], n["d"], n["p"])), n_pred, n_gt),
        "onset_pitch_tol1": _prf(
            _match_tolerant(pred, gt, tol=1), n_pred, n_gt),
    }


def load_payload(path, expected_prefix="window.VISUALIZER_DATA = "):
    txt = Path(path).read_text(encoding="utf-8")
    if not txt.startswith(expected_prefix):
        # tolerate other global names (e.g. re-keyed datasets)
        if not txt.lstrip().startswith("window."):
            raise ValueError(f"{path} does not look like a visualizer data file")
    return json.loads(txt[txt.index("{"): txt.rindex("}") + 1]), txt[: txt.index("{")]


KNOWN_ROLLOUT_GROUPS = (
    "rollouts",
    "rollouts_lora",
    "rollouts_masked",
    "rollouts_masked_40k",
    "rollouts_masked_40k_final",
    "rollouts_paper1",
    "rollouts_paper2",
)


def iter_rollout_groups(example):
    """Yield (group_name, variant_name, rollout_dict) for every rollout present.

    Known groups are scored in a stable order; any extra ``rollouts*`` key is
    included too so a new vis model does not need a compute_f1.py edit.
    """
    names = [g for g in KNOWN_ROLLOUT_GROUPS if g in example]
    names.extend(
        sorted(
            g for g in example
            if isinstance(g, str) and g.startswith("rollouts") and g not in KNOWN_ROLLOUT_GROUPS
        )
    )
    for group in names:
        block = example.get(group)
        if not isinstance(block, dict):
            continue
        for variant, roll in block.items():
            if isinstance(roll, dict) and "pred_score" in roll:
                yield group, variant, roll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print scores without rewriting the file.")
    args = ap.parse_args()

    payload, prefix = load_payload(args.data)
    examples = payload["examples"]

    # scores[(group, variant)][window_key] = {criterion: f1}
    scores = defaultdict(dict)
    n_scored = 0
    for key, ex in examples.items():
        gt = ex.get("gt_score") or []
        for group, variant, roll in iter_rollout_groups(ex):
            res = score_notes(roll["pred_score"], gt)
            roll["f1"] = res
            n_scored += 1
            scores[(group, variant)][key] = {v: res[v]["f1"] for v in VARIANTS}

    print(f"Scored {n_scored} rollouts across {len(examples)} windows.\n")

    def table(title, rows, keys):
        """rows: list of ((group, variant), per_window dict). keys: windows to average."""
        print(title)
        print(f"  {'model / variant':30s} {'n':>4s} "
              + "  ".join(f"{v:>17s}" for v in VARIANTS))
        for (group, variant), per_window in rows:
            use = [k for k in keys if k in per_window]
            if not use:
                continue
            line = f"  {group + ' / ' + variant:30s} {len(use):4d} "
            for v in VARIANTS:
                mean = sum(per_window[k][v] for k in use) / len(use)
                line += f"  {mean * 100:16.2f}%"
            print(line)
        print()

    # Every rollout on whatever windows it has -- diagnostic, NOT a comparison:
    # the groups cover different window sets and our groups carry extra variants.
    table("All rollouts (per-model window coverage differs -- not comparable):",
          sorted(scores.items()), list(examples))

    # The like-for-like comparison. Restrict to windows where EVERY model produced
    # output, and to the one variant that means the same thing for all of them:
    # `filtered` is a plain rollout with no ground-truth seeding. Pooling our
    # `*_seeded` variants (first score note force-fed from the GT) or averaging
    # over a larger window set would flatter whichever side had the easier subset.
    comparable = [(gv, pw) for gv, pw in scores.items() if gv[1] == "filtered"]
    if len(comparable) > 1:
        common = set(examples)
        for _, per_window in comparable:
            common &= set(per_window)
        missing = len(examples) - len(common)
        table(f"Like-for-like: variant='filtered', {len(common)} windows all models "
              f"scored ({missing} dropped):",
              sorted(comparable), sorted(common))

    if not args.dry_run:
        out = Path(args.data)
        with out.open("w", encoding="utf-8") as fh:
            fh.write(prefix)
            json.dump(payload, fh)
            fh.write(";\n")
        print(f"\nWrote F1 scores into {out}")


if __name__ == "__main__":
    main()
