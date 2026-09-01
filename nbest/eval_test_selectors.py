#!/usr/bin/env python
"""Does the 24-window pairwise32 / pairwise32feat result survive the TEST SET?

The 24-window table can rank methods but certifies nothing: only the pool
oracle separated from greedy (p<1e-4); pairwise32's +5.32 pt had p=0.311. This
re-runs the same selectors over ~1,180 disjoint windows of the paper-split test
file and reports the comparison at TWO levels, because they answer different
questions and only one of them is honest about the sample size:

  window level -- n ~ 1,180. OVERSTATES significance. Windows within a piece
                  share a performance, a performer and a musical style, so they
                  are not independent draws no matter how far apart they sit.
  PIECE level  -- n = 14. The paper-split test set contains 59 performances but
                  only FOURTEEN distinct musical works. That is the real unit of
                  replication, and no amount of extra windows raises it. Each
                  piece contributes ONE number (the mean over its windows), and
                  the paired test runs over those 14.

Scored with `visualizer/compute_f1.score_notes` against the FULL ground truth --
the table's own metric -- not the shard's `cand_f1` (which is
`f1_reward.final_f1`, emission-ordered). The two agree to <=0.004, but the
table is the deliverable, so the deliverable's function is the one used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from f1_reward import score_triplet_to_note                      # noqa: E402
from nbest.reranker import build_reranker_from_ckpt, substitute_candidates  # noqa: E402
from onpolicy_rollout import score_token_positions               # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def notes_from_flat(flat) -> list:
    toks = [int(t) for t in flat]
    out = []
    for k in range(len(toks) // 3):
        n = score_triplet_to_note(toks[3 * k], toks[3 * k + 1], toks[3 * k + 2])
        if n is not None:
            out.append({"t": int(n[0]), "d": int(n[1]), "p": int(n[2])})
    return out


def sign_flip_p(d: np.ndarray, n_perm: int = 200_000, seed: int = 0) -> float:
    d = d[~np.isnan(d)]
    if d.size == 0:
        return float("nan")
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice((-1.0, 1.0), size=(n_perm, d.size)) * d).mean(1))
    return float((np.count_nonzero(null >= obs - 1e-15) + 1) / (n_perm + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", default="nbest_data/test9feat_stride150.pt")
    ap.add_argument("--token-file", default="data/test_paper.txt")
    ap.add_argument("--out", default="nbest_data/test_set_selector_eval.json")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--arms", default="pairwise32,pairwise32feat")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="smoke-test on the first N windows only")
    ap.add_argument("--no-pieces", action="store_true",
                    help="skip piece attribution (window-level stats only)")
    a = ap.parse_args()

    cf1 = _load("compute_f1", "visualizer/compute_f1.py")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(a.shard, map_location="cpu", weights_only=False)
    has_feats = "cand_tok_logp_ft" in d
    print(f"{a.shard}: {d['window_tokens'].shape[0]} windows, "
          f"{d['cand_tokens'].shape[0]} candidates, feats={has_feats}", flush=True)

    CKPT = {"pairwise32": "run_nbest_reranker/pairwise32_0821/final.pt",
            "pairwise32feat": "run_nbest_reranker/pairwise32feat_0822/final.pt"}
    arms = [x for x in a.arms.split(",") if x]
    models = {}
    for name in arms:
        m = build_reranker_from_ckpt(str(ROOT / CKPT[name]), dev)
        tf = getattr(m.cfg, "token_features", 0)
        if tf and not has_feats:
            raise SystemExit(
                f"{name} declares token_features={tf} but the shard has none. "
                "Scoring it would silently skip its feature pathway.")
        models[name] = m
        print(f"  {name}: token_features={tf}", flush=True)

    row_of = {int(l): i for i, l in enumerate(d["window_line_idx"].tolist())}
    by_win = defaultdict(list)
    for ci, l in enumerate(d["cand_line_idx"].tolist()):
        by_win[row_of[int(l)]].append(ci)

    flat_pos = score_token_positions(d["window_tokens"].shape[1], device=dev)
    per_win = []
    with torch.inference_mode():
        wins = sorted(by_win)
        if a.max_windows:
            wins = wins[:a.max_windows]
        for wi in wins:
            cids = sorted(by_win[wi])            # greedy is always candidate 0
            win = d["window_tokens"][wi:wi + 1].to(dev).long()
            cands = d["cand_tokens"][cids].to(dev).long()
            gt = notes_from_flat(d["window_tokens"][wi][flat_pos.cpu()])
            f1s = [cf1.score_notes(notes_from_flat(cands[i].cpu()), gt)
                   ["onset_pitch_tol1"]["f1"] for i in range(len(cids))]
            rec = {"window": wi, "line": int(d["window_line_idx"][wi]),
                   "greedy": f1s[0], "oracle": max(f1s), "n_cand": len(cids)}
            feats = (torch.stack([d["cand_tok_logp_ft"][cids],
                                  d["cand_tok_logp_base"][cids]], -1).to(dev)
                     if has_feats else None)
            for name, m in models.items():
                tf = getattr(m.cfg, "token_features", 0)
                q = []
                for s in range(0, len(cids), a.batch):
                    sl = slice(s, min(s + a.batch, len(cids)))
                    toks = substitute_candidates(win.expand(sl.stop - sl.start, -1),
                                                 cands[sl], flat_pos)
                    q.append(m(toks, feats[sl] if tf else None).float().cpu())
                rec[name] = f1s[int(torch.cat(q).argmax())]
            per_win.append(rec)
            if len(per_win) % 100 == 0:
                print(f"  {len(per_win)}/{len(wins)} windows", flush=True)

    # ---- piece attribution: the honest clustering unit -----------------------
    pieces_of = {}
    if not a.no_pieces:
        try:
            pv = _load("precompute_visualizer", "visualizer/precompute_visualizer.py")
            pieces = pv._load_cache_pieces()
            wanted = {r["line"] for r in per_win}
            lines = {}
            with open(ROOT / a.token_file, encoding="utf-8") as fh:
                for i, raw in enumerate(fh):
                    if i in wanted:
                        lines[i] = raw.strip()
                        if len(lines) == len(wanted):
                            break
            # The shard stores LINE INDICES, not the token file it came from.
            # Point this at the wrong file and every index still resolves --
            # to a different window -- and locate_window then attributes it
            # confidently and wrongly. Verified in a smoke test: a val shard
            # read against train_paper.txt produced clean-looking attributions
            # for windows it had never seen. So check the tokens match.
            probe = per_win[0]
            ptoks = [int(t) for t in lines[probe["line"]].split("|")[0].split()]
            shard_win = d["window_tokens"][probe["window"]].tolist()
            if ptoks[:len(shard_win)] != shard_win:
                raise SystemExit(
                    f"--token-file {a.token_file} does NOT match this shard: "
                    f"line {probe['line']} differs from the stored window "
                    "tokens. Piece attribution against the wrong file silently "
                    "produces plausible, wrong clusters.")
            print(f"  token-file matches shard (checked line {probe['line']})",
                  flush=True)

            unlocated = 0
            for r in per_win:
                toks = [int(t) for t in lines[r["line"]].split("|")[0].split()]
                # extract_window_controls is the function select_paper_windows.py
                # itself uses; the piece_id is a PERFORMANCE path, so strip the
                # final component to cluster by the musical WORK -- several
                # performers play the same sonata and those are not independent.
                ctl = pv.extract_window_controls(toks)
                pc, _ = pv.locate_window(pieces, ctl)
                if pc is None:
                    unlocated += 1
                    continue
                pid = pc["piece_id"].split("asap-dataset-master/")[-1]
                pieces_of[r["window"]] = pid.rsplit("/", 1)[0]
            if unlocated:
                print(f"  WARNING: {unlocated}/{len(per_win)} windows could not "
                      "be attributed to a piece; they are excluded from the "
                      "piece-level test", flush=True)
        except Exception as exc:
            # Loud, not silent: without piece attribution the only numbers left
            # are the window-level ones, and those OVERSTATE significance --
            # which is the whole thing this run exists to avoid.
            import traceback
            traceback.print_exc()
            print(f"  !! PIECE ATTRIBUTION FAILED ({type(exc).__name__}: {exc}). "
                  "Window-level p-values below are NOT trustworthy on their own.",
                  flush=True)

    base = np.array([r["greedy"] for r in per_win])
    print("\n" + "=" * 78)
    print(f"TEST SET -- {len(per_win)} disjoint windows"
          + (f", {len(set(pieces_of.values()))} distinct pieces" if pieces_of else ""))
    print("=" * 78)
    print(f"\nWINDOW LEVEL (n={len(per_win)}) -- OVERSTATES significance; windows "
          "within a piece are correlated")
    print(f"  {'arm':16} {'mean tol1':>10} {'d vs greedy':>12} {'p':>8}")
    for name in ["oracle"] + arms:
        v = np.array([r[name] for r in per_win])
        print(f"  {name:16} {v.mean()*100:>9.2f}% {(v-base).mean()*100:>+11.2f} "
              f"{sign_flip_p(v - base):>8.4f}")
    print(f"  {'greedy (baseline)':16} {base.mean()*100:>9.2f}%")

    if pieces_of:
        groups = defaultdict(list)
        for r in per_win:
            if r["window"] in pieces_of:
                groups[pieces_of[r["window"]]].append(r)
        print(f"\nPIECE LEVEL (n={len(groups)}) -- one mean per musical work. "
              "THIS is the number to believe.")
        print(f"  {'arm':16} {'mean tol1':>10} {'d vs greedy':>12} {'p':>8} "
              f"{'wins':>6}")
        pb = np.array([np.mean([r["greedy"] for r in g]) for g in groups.values()])
        for name in ["oracle"] + arms:
            pv_ = np.array([np.mean([r[name] for r in g]) for g in groups.values()])
            dd = pv_ - pb
            print(f"  {name:16} {pv_.mean()*100:>9.2f}% {dd.mean()*100:>+11.2f} "
                  f"{sign_flip_p(dd):>8.4f} {int((dd>0).sum()):>3}/{len(dd)}")
        print(f"  {'greedy (baseline)':16} {pb.mean()*100:>9.2f}%")
    else:
        print("\nPIECE LEVEL: unavailable -- window-level p-values above are "
              "NOT trustworthy on their own.")

    json.dump({"windows": per_win, "pieces": pieces_of,
               "shard": a.shard, "arms": arms},
              open(ROOT / a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
