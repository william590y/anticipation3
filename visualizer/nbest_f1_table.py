#!/usr/bin/env python
"""Generate the N-best selection F1 table -- the standing deliverable.

Until now this table existed only as prose in CLAUDE.md. `visualizer/rl_f1_table.py`
is NOT it: that one is the older RL table, macro-averages by piece, and its ROWS
still name a `rollouts_valloss` group that no longer exists in data.js.

Reads `visualizer/data_slim.js` (the split payload; same f1 values as data.js,
21MB instead of 494MB) and the trainers' own SLURM logs. Emits two tables that
are deliberately NOT merged, because their numbers are not comparable:

  A. The 24 visualizer windows -- 12 val + 12 test, one distinct piece each,
     scored by `visualizer/compute_f1.score_notes` against the FULL gt_score.
     This is the number every result is finally judged on.
  B. Trainer holdouts -- each run's own `sel_f1`, on ITS own holdout, at ITS own
     N. Four independent breaks make these non-comparable across rows (see the
     printed warnings); they are here because four methods have no row in A.

Everything the table cannot honestly claim is printed as a caveat rather than
silently dropped. Run: python visualizer/nbest_f1_table.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CRITERIA = ("onset_pitch", "onset_pitch_dur", "onset_pitch_tol1")

# Groups that are NOT selection-from-a-pool. Kept out of table A's main block.
BEAM = re.compile(r"^rollouts_rerank(_ab\d+)?$")
PLAIN = {"rollouts", "rollouts_lora", "rollouts_paper1", "rollouts_paper2",
         "rollouts_masked", "rollouts_masked_40k", "rollouts_masked_40k_final"}

LABEL = {
    "rollouts_oracle_fp32": "[fp32] pool oracle (of the POOL DIAGNOSTIC F1)",
    "rollouts_rerank_pairwise32_fp32": "[fp32] pairwise32",
    "rollouts_rerank_pw32feat_fp32": "[fp32] pairwise32feat",
    "rollouts_rerank_pairwise_fp32": "[fp32] pairwise (9-cand training)",
    "rollouts_fitted_fp32": "[fp32] fitted alpha/beta/gamma",
    "rollouts_duel32_fp32": "[fp32] duel32 knockout",
    "rollouts_genrm_fp32": "[fp32] GenRM",
    "rollouts_mbr_fp32": "[fp32] MBR consensus",
    "rollouts_rerank_listwise32_fp32": "[fp32] listwise32",
    "rollouts_sample_t1_fp32": "[fp32] T=1 sample (control)",
    "rollouts_rerank_pw32feat": "pairwise32feat (per-token generative feats)",
    "rollouts_duel32": "duel32 knockout (PairJudge RM)",
    "rollouts_genrm": "GenRM (generative verifier, argmax log p(YES))",
    "rollouts_rerank_sample_oracle": "pool oracle (of the POOL DIAGNOSTIC F1)",
    "rollouts_rerank_pairwise32": "pairwise32 (reranker trained on 33-cand pools)",
    "rollouts_rerank_pairwise": "pairwise (reranker trained on 9-cand pools)",
    "rollouts_rerank_sample": "fitted alpha/beta/gamma objective",
    "rollouts_mbr": "MBR consensus (unweighted)",
    "rollouts_rerank_listwise32": "listwise32",
    "rollouts_sample_t1": "T=1 sample (no selection -- the honest control)",
    "rollouts": "greedy 'ours'",
    "rollouts_lora": "greedy, LoRA r512",
    # paper1 == external/paper1_joint_apt_epr == Zeng+ (wei-zeng98/joint-apt-epr)
    # paper2 == external/paper2_midi2score == Beyer & Dai (TimFelixBeyer/
    #   MIDI2ScoreTransformer) -- the one needing his music21 fork.
    # These were swapped here, so the table reported each paper's score under
    # the other's name. Verified against run_paper_models.py's own repo map.
    "rollouts_paper1": "Zeng+ (ICLR 2026) [joint-apt-epr]",
    "rollouts_paper2": "Beyer & Dai (ISMIR 2024) [MIDI2ScoreTransformer]",
    "rollouts_masked": "greedy, score-only loss",
    "rollouts_masked_40k": "greedy, score-only loss, 40k",
    "rollouts_masked_40k_final": "greedy, score-only loss, 40k final",
}


def load_payload() -> dict:
    p = os.path.join(HERE, "data_slim.js")
    if not os.path.exists(p):
        p = os.path.join(HERE, "data.js")
    s = open(p, encoding="utf-8").read()
    return json.loads(s[s.index("=") + 1:].strip().rstrip(";")), os.path.basename(p)


def sign_flip_p(deltas: np.ndarray, n_perm: int = 200_000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation test on the MEAN delta.

    The right test here: windows are paired (same window, two methods) and n=24
    is far too small to lean on a t-test's normality assumption.
    """
    d = deltas[~np.isnan(deltas)]
    if d.size == 0:
        return float("nan")
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.count_nonzero(null >= obs - 1e-15) + 1) / (n_perm + 1))


def main() -> None:
    data, src = load_payload()
    ex = data["examples"]
    order = data.get("example_order") or sorted(ex)
    groups: dict[str, set] = {}
    for k in order:
        for g, v in ex[k].items():
            if g.startswith("rollouts") and isinstance(v, dict):
                for variant, blob in v.items():
                    if isinstance(blob, dict) and "f1" in blob:
                        groups.setdefault(f"{g}|{variant}", set()).add(k)

    # --- recover the TRUE baseline: candidate 0 of the pool, ckpt-2500 greedy.
    # There is no ckpt-2500 greedy row in data.js. But any group that SELECTED
    # index 0 for a window has scored exactly that candidate under the table's
    # own metric, so its f1 IS the baseline for that window.
    # ONLY the pool-selection groups. The beam rows (rollouts_rerank,
    # rollouts_rerank_ab*) also carry a `rerank_meta`, but their index 0 is the
    # argmax of a BEAM rescoring, not a pool candidate -- and it is index 0 in
    # 24/24 windows for all five ab* rows. Including them pulled this baseline
    # to 4.02% (exactly rollouts_rerank_ab10's own score) instead of ~25.8%,
    # and made all 24 windows "disagree". Seeded variants are excluded too:
    # they are force-fed the true first note and are not the same task.
    def is_pool_group(g: str) -> bool:
        return (g.startswith("rollouts") and not BEAM.match(g) and g not in PLAIN)

    # TWO baselines, split by decode dtype. The fp32 rows come from a DIFFERENT
    # pool -- fp32 decoding changes the sampled candidates, so their candidate 0
    # is not the bf16 candidate 0. One shared baseline mixed them and made 13 of
    # 24 windows disagree, measuring every fp32 delta against the wrong greedy.
    def dtype_of(g: str) -> str:
        return "fp32" if g.endswith("_fp32") else "bf16"

    base = {"bf16": {}, "fp32": {}}
    base_src = {"bf16": {}, "fp32": {}}
    for k in order:
        for g, v in ex[k].items():
            if not is_pool_group(g):
                continue
            dt = dtype_of(g)
            for variant, blob in (v.items() if isinstance(v, dict) else []):
                if not isinstance(blob, dict) or variant.endswith("_seeded"):
                    continue
                m = blob.get("rerank_meta")
                if isinstance(m, dict) and m.get("selected") == 0 and "f1" in blob:
                    base[dt].setdefault(k, blob["f1"])
                    base_src[dt].setdefault(k, []).append(f"{g}|{variant}")

    # Any window with two such groups must agree, or the baseline is not well defined.
    disagree = {"bf16": [], "fp32": []}
    for dt in ("bf16", "fp32"):
        for k, srcs in base_src[dt].items():
            vals = set()
            for src_gv in srcs:
                g, variant = src_gv.split("|")
                vals.add(round(ex[k][g][variant]["f1"]["onset_pitch_tol1"]["f1"], 12))
            if len(vals) > 1:
                disagree[dt].append((k, sorted(vals)))

    def series(gv: str, crit: str) -> np.ndarray:
        g, variant = gv.split("|")
        return np.array([ex[k].get(g, {}).get(variant, {}).get("f1", {})
                         .get(crit, {}).get("f1", np.nan) for k in order], float)

    base_series_of = {dt: np.array(
        [base[dt].get(k, {}).get("onset_pitch_tol1", {}).get("f1", np.nan)
         for k in order], float) for dt in ("bf16", "fp32")}

    print("=" * 100)
    print("TABLE A -- the 24 visualizer windows (12 val + 12 test, one distinct piece each)")
    print(f"source: visualizer/{src}   metric: compute_f1.score_notes vs the FULL gt_score")
    print("=" * 100)
    for dt in ("bf16", "fp32"):
        bs = base_series_of[dt]
        n_b = int(np.count_nonzero(~np.isnan(bs)))
        if n_b == 0:
            print(f"\nBASELINE [{dt}]: not recoverable")
            continue
        print(f"\nBASELINE [{dt}] (pool candidate 0 = ckpt-2500 greedy), recovered "
              f"on {n_b}/{len(order)} windows: tol1 = {np.nanmean(bs)*100:.2f}%")
        if disagree[dt]:
            print(f"  !! {len(disagree[dt])} window(s) disagree: {disagree[dt][:3]}")
        else:
            print("  cross-checked: every window with >1 recovering group agrees exactly")
        miss = [k for k in order if k not in base[dt]]
        if miss:
            print(f"  not recoverable: {miss}")

    sel = sorted(g for g in groups
                 if is_pool_group(g.split("|")[0])
                 and not g.split("|")[1].endswith("_seeded"))
    print(f"\n{'method':50} {'n':>3} {'op':>7} {'opd':>7} {'tol1':>7} "
          f"{'d.tol1':>8} {'p':>7}")
    print("-" * 100)
    rows = []
    for gv in sel:
        g, variant = gv.split("|")
        vals = {c: series(gv, c) for c in CRITERIA}
        n = int(np.count_nonzero(~np.isnan(vals["onset_pitch_tol1"])))
        paired = vals["onset_pitch_tol1"] - base_series_of[dtype_of(g)]
        d = np.nanmean(paired) * 100
        p = sign_flip_p(paired)
        rows.append((np.nanmean(vals["onset_pitch_tol1"]), gv, n, vals, d, p))
    for mean_tol1, gv, n, vals, d, p in sorted(rows, key=lambda r: -r[0]):
        g, variant = gv.split("|")
        lab = LABEL.get(g, g)
        print(f"{lab[:48]:50} {n:>3} "
              f"{np.nanmean(vals['onset_pitch'])*100:>7.2f} "
              f"{np.nanmean(vals['onset_pitch_dur'])*100:>7.2f} "
              f"{mean_tol1*100:>7.2f} {d:>+8.2f} {p:>7.3f}")
    for dt in ("bf16", "fp32"):
        if not base[dt]:
            continue
        print(f"{f'BASELINE [{dt}] pool greedy (candidate 0)':50} {len(base[dt]):>3} "
              f"{np.nanmean([base[dt][k]['onset_pitch']['f1'] for k in base[dt]])*100:>7.2f} "
              f"{np.nanmean([base[dt][k]['onset_pitch_dur']['f1'] for k in base[dt]])*100:>7.2f} "
              f"{np.nanmean(base_series_of[dt])*100:>7.2f} {'--':>8} {'--':>7}")

    print("\nNON-POOL ROWS (different models and/or different conditioning -- NOT a "
          "'gain over greedy' comparison):")
    print(f"{'row':46} {'variant':>16} {'n':>3} {'tol1':>7}  checkpoint")
    print("-" * 100)
    for gv in sorted(groups):
        g, variant = gv.split("|")
        if not (BEAM.match(g) or g in PLAIN):
            continue
        v = series(gv, "onset_pitch_tol1")
        # rollouts_lora is decoded through the LoRA adapter, NOT through
        # `checkpoint` -- falling back to the top-level key mislabels it as the
        # full-FT model it is meant to be compared against.
        if g == "rollouts_lora":
            ck = data.get("lora_checkpoint", "?")
        else:
            ck = data.get(f"checkpoint_{g}") or data.get("checkpoint", "?")
        note = "  [GT-SEEDED: slot 0 force-fed]" if variant.endswith("_seeded") else ""
        print(f"{LABEL.get(g, g)[:44]:46} {variant:>16} "
              f"{int(np.count_nonzero(~np.isnan(v))):>3} {np.nanmean(v)*100:>7.2f}  {ck}{note}")

    print("\nCHECKPOINT PROVENANCE for the selection rows:")
    cks = {data.get(f"checkpoint_{gv.split('|')[0]}", "?") for gv in sel}
    for c in sorted(cks):
        print(f"  {c}")

    # ---------------- Table B: trainer holdouts, from the trainers' own logs.
    print("\n" + "=" * 100)
    print("TABLE B -- trainer holdouts (each run's OWN holdout, at its OWN N)")
    print("=" * 100)
    pat = re.compile(r"eval @ (\d+): (\{.*\})")
    runs = {
        "pairwise_0821": "logs/pw_train_361262.out",
        "pairwise32_0821": "logs/pw32_train_402657.out",
        "pairwise32feat_0822": "logs/pwfeat_train_461862.out",
        "listwise32_0821": "logs/lw32_train_402658.out",
        "duel32_0822": "logs/duel_train_474782.out",
        "listt5_0822": "logs/listt5_train_461930.out",
        "genrm_0822": "logs/genrm_train_461931.out",
        # --- fp32 arms: identical launch args, dtype the only difference ---
        "pairwise_FP32": "logs/fp32sel_549000_0.out",
        "pairwise32_FP32": "logs/fp32sel_549000_1.out",
        "listwise32_FP32": "logs/fp32sel_549000_2.out",
        "genrm_FP32": "logs/fp32sel_549000_3.out",
        "duel32_FP32": "logs/fp32feat_592239_1.out",
        "pairwise32feat_FP32": "logs/fp32feat_592239_0.out",
    }
    print(f"{'run':24} {'N':>5} {'greedy':>8} {'oracle':>8} {'best':>8} "
          f"{'@step':>7} {'final':>8} {'%headroom(final)':>17}")
    print("-" * 100)
    for name, rel in runs.items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print(f"{name:24} LOG NOT FOUND: {rel}")
            continue
        evals = []
        for line in open(p, errors="ignore"):
            m = pat.search(line)
            if m:
                try:
                    evals.append((int(m.group(1)), eval(m.group(2), {"__builtins__": {}})))
                except Exception:
                    pass
        if not evals:
            print(f"{name:24} no eval lines parsed")
            continue
        last = evals[-1][1]
        bstep, best = max(evals, key=lambda e: e[1].get("sel_f1", -1))
        g, o = last.get("greedy_f1", float("nan")), last.get("oracle_f1", float("nan"))
        head = (last["sel_f1"] - g) / (o - g) * 100 if o > g else float("nan")
        print(f"{name:24} {last.get('n_windows','?'):>5} {g:>8.5f} {o:>8.5f} "
              f"{best['sel_f1']:>8.5f} {bstep:>7} {last['sel_f1']:>8.5f} {head:>16.1f}%")

    print("""
READ BEFORE QUOTING ANY OF THIS
  Table A
    * n=24. Only the oracle separates from the baseline; every other p is >0.1.
      The table can RANK methods; it cannot certify any non-oracle gap.
    * "pairwise32"/"listwise32" name the reranker's TRAINING pool depth (33).
      Both SELECT from the same 9-candidate pool -- rerank_sample_viz.py's
      --n-sampled defaults to 8 and the sbatch never overrides it.
    * The oracle row maximises the POOL DIAGNOSTIC F1 (matched against a
      raw-aligned GT subset, as few as 94/138 notes on val-07), not the table
      metric. It is not the pool's true ceiling -- regenerate before calling it one.
    * The greedy 'ours' row is checkpoint-7500; every selection row is
      checkpoint-2500. They are different models -- that is why the baseline
      above is recovered from candidate 0 instead.
    * Selection rows are variant=raw; both paper rows are variant=filtered.
      Conditioning alone is worth ~2.35 pt on the same checkpoint.
  Table B
    * NOT comparable across rows: (i) holdouts are a PREFIX of the token file,
      not a sample, and the head is easier -- greedy is 0.36643 at N=120 vs
      0.28069 at N=1796; (ii) 9-cand shards are unaugmented, 33-cand are not;
      (iii) best.pt is max-over-evals on the same holdout it is reported from;
      (iv) optimisation budget varies ~4x.
    * The only defensible cross-row protocol: one shard family (tokfeat32_*),
      the full 1796-window holdout, and delta-over-greedy or %-of-headroom.
    * pairwise32feat and duel32 are token_features=2 and are SILENTLY MIS-SCORED
      by rerank_sample_viz.py, which passes no feats. They cannot get a Table A
      row until that plumbing exists.""")


if __name__ == "__main__":
    main()
