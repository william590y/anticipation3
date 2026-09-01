#!/usr/bin/env python
"""Sample-then-rerank decode for the visualizer windows.

Runs the discriminative reranker in its NATIVE regime (the setting it was
fitted and validated in): build C(x) = 8 sampled (T=1.0, constrained) + 1
greedy candidate scores per window from the FT model -- exactly
``nbest.generate_nbest`` -- then select

    argmax_{y in C(x)}  alpha * z(logp_FT) + beta * z(logp_base) + gamma * z(q_phi)

with the fitted weights/stats. This is selection over sampled candidates, NOT
beam-search maximisation of the objective (``rerank_viz_rollout.py``), which
collapses into max-FT-likelihood because the features have no spread across
beam survivors.

Windows are the viz examples' packed contexts (``tokens_from_controls`` on the
requested variant's controls); candidate features follow generate_nbest:
logp_ft from the rollout's own constrained logprobs (greedy re-scored at
T=1.0), logp_base via ``logp_base_batch`` (so_c convention), q_phi from the
trained reranker. Per-candidate diagnostic F1 (onset_pitch_tol1 semantics) is
scored against ``gt_notes_for_variant``.

Writes shards in the ``rerank_viz_rollout`` schema; merge with
  python visualizer/merge_rerank_rollouts.py --group rollouts_rerank_sample \
      --shards visualizer/rerank_sample_shards/shard_*.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from anticipation.config import CONTEXT_SIZE  # noqa: E402
from eval_base_score_ppl import load_base_model as load_base_amt  # noqa: E402
from eval_base_score_ppl import slot_logit_masks  # noqa: E402
from evaluate_muster import load_model  # noqa: E402
from f1_reward import final_f1  # noqa: E402
from nbest.generate_nbest import flat_notes, logp_base_batch  # noqa: E402
from onpolicy_rollout import (rollout_score_slots, score_token_logprob,  # noqa: E402
                              score_token_positions)
from precompute_beams import note_from_tokens  # noqa: E402
from precompute_visualizer import tokens_from_controls  # noqa: E402
from compute_f1 import score_notes  # noqa: E402
from compute_sequence_ppl import (control_notes_for_variant,  # noqa: E402
                                  gt_notes_for_variant, load_payload)
from rerank_viz_rollout import load_weights, normalise  # noqa: E402


def _feats_for(model, cand_feats, i):
    """The (1, 414, F) feature row for candidate `i`, or None if unused.

    Hard-fails instead of returning None when the model declares
    token_features > 0 and the features are missing: that combination used to
    score silently, with the model's feature pathway skipped, and produced
    plausible-looking rows for a differently-behaved model.
    """
    if not getattr(getattr(model, "cfg", None), "token_features", 0):
        return None
    if cand_feats is None:
        raise SystemExit(
            f"{type(model).__name__} declares token_features="
            f"{model.cfg.token_features} but no per-token features were "
            "computed. Scoring it would silently skip its feature pathway "
            "(measured: 0.001x-73x score change, 83% of candidates moving "
            ">3.2x). Refusing.")
    return cand_feats[i:i + 1]


def notes_as_tuples(notes):
    """Slot-aligned note dicts/None -> f1_reward note tuples/None."""
    return [None if n is None else (int(n["t"]), int(n["d"]), int(n["p"]))
            for n in notes]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--weights", default="nbest_data/decode_weights_unfiltered.json")
    ap.add_argument("--checkpoint", default="run_paper_split_v2/checkpoint-2500")
    ap.add_argument("--variant", default="raw", choices=["filtered", "raw"])
    ap.add_argument("--n-sampled", type=int, default=8)
    ap.add_argument("--score-chunk", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--example-keys", default=None,
                    help="comma-separated; default = all in example_order")
    ap.add_argument("--fp32", action="store_true",
                    help="decode the pools in fp32 (no bf16 autocast), so the "
                         "table's rows share numerics with its greedy row")
    ap.add_argument("--fast-decode", action="store_true",
                    help="bit-identical fp32 fast path (~1.75x)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--output-oracle", default=None,
                    help="also write a shard selecting each window's best-F1 "
                         "candidate (ties keep greedy) -- the pool oracle")
    ap.add_argument("--output-sample1", default=None,
                    help="also write a shard taking each window's FIRST "
                         "sampled candidate -- one fair draw at T=1.0")
    ap.add_argument("--output-mbr", default=None,
                    help="also write a shard selecting by unweighted candidate "
                         "consensus (MBR): argmax_i mean_{j!=i} F1(y_i, y_j)")
    ap.add_argument("--qsel-reranker", default=None,
                    help="checkpoint of a second reranker; with --output-qsel, "
                         "write a shard selecting argmax of ITS score alone")
    ap.add_argument("--output-qsel", default=None)
    ap.add_argument("--duel-ckpt", default=None,
                    help="DuelComparator checkpoint; with --output-tournament, "
                         "select by knockout tournament (arXiv 2501.13007)")
    ap.add_argument("--output-tournament", default=None)
    ap.add_argument("--tournament-seed", type=int, default=0,
                    help="the bracket is randomised; fix it for reproducibility")
    ap.add_argument("--listt5-ckpt", default=None,
                    help="ListwiseFiD checkpoint; with --output-listt5, select "
                         "by m-ary tournament sort (arXiv 2402.15838)")
    ap.add_argument("--output-listt5", default=None)
    ap.add_argument("--genrm-ckpt", default=None,
                    help="generative-verifier checkpoint; with --output-genrm, "
                         "select argmax p(YES) (arXiv 2408.15240)")
    ap.add_argument("--output-genrm", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = load_weights(args.weights)
    alpha, beta, gamma = weights["alpha"], weights["beta"], weights["gamma"]
    stats = weights.get("feature_stats")

    reranker = None
    reranker_ckpt = (weights.get("reranker_ckpt")
                     or weights.get("reranker_checkpoint"))
    if gamma != 0.0 and reranker_ckpt:
        from nbest.reranker import build_reranker_from_ckpt
        reranker = build_reranker_from_ckpt(reranker_ckpt, device)

    payload, _ = load_payload(args.data)
    keys = (args.example_keys.split(",") if args.example_keys
            else list(payload["example_order"]))

    ft, _ = load_model(args.checkpoint)
    ft = ft.to(device).eval()
    # fp32 must reach the base model's WEIGHTS, not just the autocast around
    # the FT decode: load_base_model casts to bf16 by default, which is how
    # nbest_data/fp32_*.pt ended up fp32 on the FT channel and bf16 on the
    # logp_base channel despite the name.
    base = load_base_amt(device,
                         dtype=torch.float32 if args.fp32 else torch.bfloat16)
    masks = slot_logit_masks(device)
    # DTYPE. These pools are what EVERY selection row in the F1 table is
    # chosen from, while the table's greedy "ours" row comes from
    # precompute_visualizer.py, which has no autocast at all (fp32). Running
    # this under bf16 therefore made the table mix numerics -- and bf16
    # agrees with fp32 on only 63.3% of greedy score tokens here
    # (visualizer/dtype_ab_viz.py). --fp32 removes that confound; pair it
    # with --fast-decode (anticipation/fast_decode.py), a bit-identical fp32
    # path, so full precision costs no wall clock.
    import contextlib
    if args.fp32:
        autocast = lambda: contextlib.nullcontext()   # noqa: E731
    else:
        autocast = lambda: torch.autocast(  # noqa: E731
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available())
    # cuda_graph is greedy-only (a graph-private RNG cannot reproduce a
    # seeded eager multinomial), so sampled rollouts get buckets only.
    fast_s = {"buckets": 8} if args.fast_decode else None
    fast_g = ({"buckets": 8, "cuda_graph": True} if args.fast_decode else None)

    meta_dtype = {"decode_dtype": "fp32" if args.fp32 else "bf16_autocast",
                  "fast_decode": bool(args.fast_decode)}
    shard = {"checkpoint": args.checkpoint, "weights": weights,
             "n_sampled": args.n_sampled, "variant": args.variant,
             **meta_dtype, "examples": {}}
    shard_oracle = {"checkpoint": args.checkpoint, "weights": weights,
                    "n_sampled": args.n_sampled, "variant": args.variant,
                    **meta_dtype, "examples": {}}
    shard_sample1 = {"checkpoint": args.checkpoint, "weights": weights,
                     "n_sampled": args.n_sampled, "variant": args.variant,
                     **meta_dtype, "examples": {}}
    shard_mbr = {"checkpoint": args.checkpoint, "weights": weights,
                 "n_sampled": args.n_sampled, "variant": args.variant,
                 **meta_dtype, "examples": {}}
    reranker2 = None
    if args.qsel_reranker and args.output_qsel:
        from nbest.reranker import build_reranker_from_ckpt as _brc
        reranker2 = _brc(args.qsel_reranker, device)
    shard_qsel = {"checkpoint": args.checkpoint,
                  "weights": {"qsel_reranker": args.qsel_reranker},
                  "n_sampled": args.n_sampled, "variant": args.variant,
                  **meta_dtype, "examples": {}}
    duel = None
    if args.duel_ckpt and args.output_tournament:
        from nbest.duel import DuelComparator, DuelConfig
        dck = torch.load(args.duel_ckpt, map_location="cpu",
                         weights_only=False)
        duel = DuelComparator(DuelConfig(**dck["model_cfg"]))
        duel.load_state_dict(dck["model"])
        duel = duel.to(device).eval()
    shard_tour = {"checkpoint": args.checkpoint,
                  "weights": {"duel_ckpt": args.duel_ckpt},
                  "n_sampled": args.n_sampled, "variant": args.variant,
                  **meta_dtype, "examples": {}}
    # Do any loaded selector actually consume per-token features? Both
    # `Reranker.forward` and `DuelComparator.forward` default `feats=None` and
    # SILENTLY SKIP the index_add that injects them -- no exception, just a
    # differently-scored model. Measured on pairwise32feat_0822/final.pt:
    # dropping feats moves the score by 0.001x-73x, with 83% of candidates
    # moving >3.2x, i.e. arbitrary re-ranking rather than a uniform shift.
    # So compute them whenever a checkpoint declares token_features > 0, and
    # refuse to score such a checkpoint without them (below).
    need_feats = any(
        m is not None and getattr(getattr(m, "cfg", None), "token_features", 0)
        for m in (reranker, reranker2, duel))
    if need_feats:
        from nbest.add_token_features import base_token_logp, ft_token_logp

    listt5 = None
    if args.listt5_ckpt and args.output_listt5:
        from nbest.listt5 import ListwiseFiD, ListwiseFiDConfig
        lck = torch.load(args.listt5_ckpt, map_location="cpu",
                         weights_only=False)
        listt5 = ListwiseFiD(ListwiseFiDConfig(**lck["model_cfg"]))
        listt5.load_state_dict(lck["model"])
        listt5 = listt5.to(device).eval()
    shard_listt5 = {"checkpoint": args.checkpoint,
                    "weights": {"listt5_ckpt": args.listt5_ckpt},
                    "n_sampled": args.n_sampled, "variant": args.variant,
                    **meta_dtype, "examples": {}}
    genrm = None
    if args.genrm_ckpt and args.output_genrm:
        from nbest.train_genrm import NEW_VOCAB
        gck = torch.load(args.genrm_ckpt, map_location="cpu",
                         weights_only=False)
        genrm, _ = load_model(args.checkpoint)
        genrm.resize_token_embeddings(NEW_VOCAB, mean_resizing=False)
        genrm.load_state_dict(gck["model"])
        genrm = genrm.to(device).eval()
    shard_genrm = {"checkpoint": args.checkpoint,
                   "weights": {"genrm_ckpt": args.genrm_ckpt},
                   "n_sampled": args.n_sampled, "variant": args.variant,
                   **meta_dtype, "examples": {}}
    # SHARD-INVARIANT SEEDING. This used to key on the loop index within THIS
    # process's key list, while rerank_sample_viz.sbatch shards the windows
    # with order[TASK_ID::4] -- so the same window drew a different seed, and
    # therefore a different candidate pool, depending only on how the run was
    # sharded (23 of 24 windows changed). Every merged row happens to come
    # from single-process 24-key runs so the existing table is self-consistent,
    # but a sharded rerun would have silently produced non-comparable rows.
    # Keying on the window's position in the payload's own example_order makes
    # the pool a function of the window, not of the job layout.
    order = list(payload["example_order"])
    t0 = time.time()
    for ki, key in enumerate(keys):
        gi = order.index(key) if key in order else ki
        torch.manual_seed(args.seed + 1000 * gi + sum(map(ord, key)))
        ex = payload["examples"][key]
        controls = control_notes_for_variant(ex, args.variant)
        tokens = tokens_from_controls(controls, CONTEXT_SIZE - 4)
        window = torch.tensor([tokens], dtype=torch.long, device=device)
        positions = score_token_positions(len(tokens), device=device)
        n_slots = positions.shape[0] // 3

        rep = window.repeat_interleave(args.n_sampled, dim=0)
        out = rollout_score_slots(
            ft, rep, temperature=1.0, constrain=True, collect_logprobs=True,
            collect_gt_ce=False, autocast_ctx=autocast, fast=fast_s)
        sampled_flat = out["rolled"][:, positions]
        # tokens_from_controls fills score slots with REST placeholders, so the
        # rollout's REST-based `valid` mask is all-False here (unlike
        # generate_nbest's GT-bearing windows, where it is all-True). Every
        # body slot is a generated score token; sum them all.
        sampled_lp = out["logprob"].sum(dim=1)

        greedy = rollout_score_slots(
            ft, window, temperature=0.0, constrain=True, collect_logprobs=False,
            collect_gt_ce=False, autocast_ctx=autocast, fast=fast_g)
        greedy_flat = greedy["rolled"][:, positions]
        with autocast():
            logits = ft(greedy["rolled"]).logits
        lp = score_token_logprob(logits, greedy["rolled"], positions,
                                 temperature=1.0, constrain=True)
        greedy_lp = lp.sum(dim=1)
        del logits

        cand_flat = torch.cat([greedy_flat, sampled_flat])   # greedy first
        cand_lp_ft = torch.cat([greedy_lp, sampled_lp]).float()
        cand_lp_base = logp_base_batch(base, masks, cand_flat,
                                       args.score_chunk).float()

        # The PER-TOKEN version of the two scalars just computed. Both are
        # already produced upstream and thrown away by `.sum(dim=1)`
        # (sampled_lp above, greedy_lp above, and logp_base_batch's own
        # `-nll_c.sum(dim=1)`), so this costs one extra teacher-forced FT pass
        # per candidate and nothing else.
        #
        # Computed by add_token_features.py's OWN functions, deliberately: the
        # features these checkpoints were trained on came from that module, and
        # re-deriving them here would be a second implementation to keep in
        # sync. Stacked [ft, base] to match how the trainers stack them
        # (train_reranker_pairwise.py:82-83, train_duel.py:85-86) -- reversing
        # the channels would be silently wrong, not an error.
        #
        # The FT channel is a TEACHER-FORCED rescore for EVERY candidate,
        # including sampled ones. `sampled_lp` above instead comes from the
        # rollout's own logprobs; that is a different quantity (the two
        # disagree by ~1.27 nats summed, which is the known `ft`-side gap in
        # add_token_features' consistency print) and is not what was trained on.
        cand_feats = None
        if need_feats:
            wins = window.repeat(cand_flat.shape[0], 1)
            f_ft = ft_token_logp(ft, wins, cand_flat, positions,
                                 args.score_chunk)
            f_base = base_token_logp(base, masks, cand_flat, args.score_chunk)
            cand_feats = torch.stack([f_ft, f_base], dim=-1).to(device)
            del wins, f_ft, f_base

        # SCORE CANDIDATES WITH THE TABLE'S METRIC, AGAINST THE FULL GT.
        # This used to use gt_notes_for_variant, a raw-ALIGNED subset holding
        # only 2998 of the 3312 GT notes across the 24 windows (as few as
        # 94/138 on val-07), scored by f1_reward.final_f1 in emission order --
        # while the table scores every row with compute_f1.score_notes against
        # the full gt_score, onset-sorted. Selected predictions differed by up
        # to 4.87 points between the two. Mostly that only mislabeled the
        # diagnostics, but the ORACLE row selects argmax over these numbers,
        # so it could crown a candidate that is not the pool's best under the
        # metric the table reports. (Note the TRAINING shards never had this
        # problem: they score against the window's own full GT, and relabeling
        # them with the table metric moves labels by at most 0.004.)
        gt_full = [{"t": int(n["t"]), "d": int(n["d"]), "p": int(n["p"])}
                   for n in (ex.get("gt_score") or [])]
        gt_tuples = [t for t in notes_as_tuples(
            gt_notes_for_variant(ex, args.variant, n_slots)) if t is not None]
        cand_cpu = cand_flat.cpu()

        rows = []
        for i in range(cand_cpu.shape[0]):
            q_phi = 0.0
            if reranker is not None:
                from nbest.reranker import substitute_candidates
                q_phi = float(reranker(
                    substitute_candidates(window, cand_flat[i:i + 1],
                                          positions),
                    _feats_for(reranker, cand_feats, i)).item())
            pred_notes = [{"t": int(t), "d": int(d), "p": int(p)}
                          for t, d, p in
                          (x for x in flat_notes(cand_cpu[i]) if x is not None)]
            f1 = score_notes(pred_notes, gt_full)["onset_pitch_tol1"]["f1"]
            f1_emit = final_f1(flat_notes(cand_cpu[i]), gt_tuples)
            objective = (alpha * normalise(float(cand_lp_ft[i]), stats, "logp_ft")
                         + beta * normalise(float(cand_lp_base[i]), stats, "logp_base")
                         + gamma * normalise(q_phi, stats, "q_phi"))
            toks = cand_cpu[i].tolist()
            pred = [note_from_tokens(toks[3 * k], toks[3 * k + 1], toks[3 * k + 2])
                    for k in range(n_slots)]
            row = {"f1_emission_order": f1_emit,
                   "logp_ft": float(cand_lp_ft[i]),
                   "logp_base": float(cand_lp_base[i]),
                   "q_phi": q_phi, "f1": f1, "objective": objective,
                   "kind": "greedy" if i == 0 else "sampled",
                   "pred": pred}
            if reranker2 is not None:
                from nbest.reranker import substitute_candidates
                with torch.inference_mode():
                    row["q_pw"] = float(reranker2(
                        substitute_candidates(window, cand_flat[i:i + 1],
                                              positions),
                        _feats_for(reranker2, cand_feats, i)).item())
            rows.append(row)
        # Candidate-consensus (MBR): score each candidate by its mean F1
        # against the other candidates (no GT involved). F1 is directional
        # (pred vs reference), so use y_i as pred and y_j's real notes as
        # reference for each ordered pair.
        k_pool = len(rows)
        preds_t = [[None if n is None else (int(n["t"]), int(n["d"]), int(n["p"]))
                    for n in r["pred"]] for r in rows]
        refs_t = [[t for t in p if t is not None] for p in preds_t]
        pair = [[final_f1(preds_t[i], refs_t[j]) if i != j else 0.0
                 for j in range(k_pool)] for i in range(k_pool)]
        for i, r in enumerate(rows):
            r["s_mbr"] = sum(pair[i]) / (k_pool - 1)
        mbr = max(range(k_pool), key=lambda i: rows[i]["s_mbr"])
        shard_mbr["examples"][key] = {
            "pred_score": rows[mbr]["pred"],
            "rerank_meta": {
                "selected": mbr, "selection": "mbr_unweighted",
                "candidates": [{k: v for k, v in r.items() if k != "pred"}
                               for r in rows],
            },
        }
        if duel is not None:
            import random as _random
            from nbest.duel import knockout
            from nbest.reranker import substitute_candidates as _sub
            calls = [0]

            def _match(i, j, _c=calls):
                _c[0] += 1
                a_tok = _sub(window, cand_flat[i:i + 1], positions)
                b_tok = _sub(window, cand_flat[j:j + 1], positions)
                fa = _feats_for(duel, cand_feats, i)
                fb = _feats_for(duel, cand_feats, j)
                with autocast():
                    # order-symmetrised: the features must follow their own
                    # candidate when the pair is swapped, not the slot.
                    ab = duel(a_tok, cand_flat[j:j + 1], fa, fb).float()
                    ba = duel(b_tok, cand_flat[i:i + 1], fb, fa).float()
                return float((ab - ba).item()) > 0    # order-symmetrised

            champ, log = knockout(
                k_pool, _match,
                _random.Random(args.tournament_seed + ki))
            shard_tour["examples"][key] = {
                "pred_score": rows[champ]["pred"],
                "rerank_meta": {
                    "selected": champ, "selection": "knockout_tournament",
                    "matches": calls[0], "bracket": log,
                    "candidates": [{k: v for k, v in r.items() if k != "pred"}
                                   for r in rows],
                },
            }
        if listt5 is not None:
            from nbest.listt5 import tournament_sort
            from nbest.reranker import substitute_candidates as _sub2
            lpasses = [0]

            def _pick(group, _lp=lpasses):
                _lp[0] += 1
                m = listt5.cfg.m
                g = list(group)
                toks = torch.stack([
                    _sub2(window, cand_flat[i:i + 1], positions)[0]
                    for i in g])
                if len(g) < m:      # pad by repeating slot 0, as ListT5 does
                    toks = torch.cat(
                        [toks, toks[:1].expand(m - len(g), -1)], dim=0)
                with autocast():
                    perm = listt5.generate(toks.unsqueeze(0)[:, :len(g)])
                return g[int(perm[0, -1])]      # best is emitted LAST

            champ_l, n_pass = tournament_sort(k_pool, listt5.cfg.m, _pick)
            shard_listt5["examples"][key] = {
                "pred_score": rows[champ_l]["pred"],
                "rerank_meta": {
                    "selected": champ_l, "selection": "listt5_tournament_sort",
                    "passes": n_pass,
                    "candidates": [{k: v for k, v in r.items() if k != "pred"}
                                   for r in rows],
                },
            }
        if genrm is not None:
            from nbest.train_genrm import ASK, NO, YES
            from nbest.reranker import substitute_candidates as _sub3
            p_yes = []
            for i in range(k_pool):
                seq = torch.cat([
                    _sub3(window, cand_flat[i:i + 1], positions),
                    torch.full((1, 1), ASK, dtype=torch.long, device=device)],
                    dim=1)
                with autocast():
                    lg = genrm(seq, use_cache=False).logits[:, -1].float()
                p_yes.append(float(torch.log_softmax(lg, -1)[0, YES]))
            gi = max(range(k_pool), key=lambda i: p_yes[i])
            for i, r in enumerate(rows):
                r["logp_yes"] = p_yes[i]
            shard_genrm["examples"][key] = {
                "pred_score": rows[gi]["pred"],
                "rerank_meta": {
                    "selected": gi, "selection": "genrm_p_yes",
                    "candidates": [{k: v for k, v in r.items() if k != "pred"}
                                   for r in rows],
                },
            }
        if reranker2 is not None:
            qi = max(range(k_pool), key=lambda i: rows[i]["q_pw"])
            shard_qsel["examples"][key] = {
                "pred_score": rows[qi]["pred"],
                "rerank_meta": {
                    "selected": qi, "selection": "pairwise_q",
                    "candidates": [{k: v for k, v in r.items() if k != "pred"}
                                   for r in rows],
                },
            }
        # Likelihood-weighted consensus diagnostics at a few temperatures.
        wdiag = []
        for temp in (5.0, 20.0, 100.0):
            import math
            mx = max(r["logp_ft"] for r in rows)
            w = [math.exp((r["logp_ft"] - mx) / temp) for r in rows]
            sw = [sum(w[j] * pair[i][j] for j in range(k_pool) if j != i)
                  / max(sum(w[j] for j in range(k_pool) if j != i), 1e-12)
                  for i in range(k_pool)]
            wi = max(range(k_pool), key=lambda i: sw[i])
            wdiag.append((temp, wi, rows[wi]["f1"]))
        shard_sample1["examples"][key] = {
            "pred_score": rows[1]["pred"],
            "rerank_meta": {
                "selected": 1, "selection": "sample_t1",
                "candidates": [{k: v for k, v in r.items() if k != "pred"}
                               for r in rows],
            },
        }
        oracle = max(range(len(rows)),
                     key=lambda i: (rows[i]["f1"], 1 if i == 0 else 0))
        shard_oracle["examples"][key] = {
            "pred_score": rows[oracle]["pred"],
            "rerank_meta": {
                "selected": oracle, "selection": "oracle_f1",
                "candidates": [{k: v for k, v in r.items() if k != "pred"}
                               for r in rows],
            },
        }
        best = max(range(len(rows)), key=lambda i: rows[i]["objective"])
        shard["examples"][key] = {
            "pred_score": rows[best]["pred"],
            "rerank_meta": {
                "selected": best,
                "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
                "candidates": [{k: v for k, v in r.items() if k != "pred"}
                               for r in rows],
            },
        }
        print(f"{key}: selected={best} ({rows[best]['kind']}) "
              f"f1_sel={rows[best]['f1']:.3f} "
              f"f1_greedy={rows[0]['f1']:.3f} "
              f"f1_mbr={rows[mbr]['f1']:.3f} "
              f"f1_oracle={max(r['f1'] for r in rows):.3f} "
              f"wmbr={[(t, i, round(f, 3)) for t, i, f in wdiag]} "
              f"[{ki + 1}/{len(keys)}  {time.time() - t0:.0f}s]", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(shard), encoding="utf-8")
    print(f"wrote {out_path}")
    if args.output_oracle:
        op = Path(args.output_oracle)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(shard_oracle), encoding="utf-8")
        print(f"wrote {op}")
    if args.output_sample1:
        sp = Path(args.output_sample1)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(shard_sample1), encoding="utf-8")
        print(f"wrote {sp}")
    if args.output_mbr:
        mp = Path(args.output_mbr)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(shard_mbr), encoding="utf-8")
        print(f"wrote {mp}")
    if reranker2 is not None:
        qp = Path(args.output_qsel)
        qp.parent.mkdir(parents=True, exist_ok=True)
        qp.write_text(json.dumps(shard_qsel), encoding="utf-8")
        print(f"wrote {qp}")
    if duel is not None:
        tp = Path(args.output_tournament)
        tp.parent.mkdir(parents=True, exist_ok=True)
        tp.write_text(json.dumps(shard_tour), encoding="utf-8")
        print(f"wrote {tp}")
    for obj, path, shard in ((listt5, args.output_listt5, shard_listt5),
                             (genrm, args.output_genrm, shard_genrm)):
        if obj is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(shard), encoding="utf-8")
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
