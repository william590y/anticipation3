#!/usr/bin/env python
"""Discriminative-rerank decode for the visualizer windows.

Implements the reranking decode rule

    max_{y in C(x)}  alpha * logp_FT(y|x)  +  beta * logp_base(y)  +  gamma * q_phi(x, y)

with beam search: C(x) is built by a constrained triplet beam search (fork of
``precompute_beams.beam_search_triplets``) whose PRUNING score is already the
combined incremental term ``alpha * lp_FT + beta * lp_base`` --

  * lp_FT: constrained log-prob of the score triplet in the INTERLEAVED packed
    context under the fine-tuned model (teacher-forced GT controls), the same
    quantity ``precompute_beams`` accumulates;
  * lp_base: constrained log-prob of the same tokens in the ISOLATED score
    stream (``AUTOREGRESS`` + flattened triplets) under the untuned base AMT
    (``eval_base_score_ppl``'s ``so_c`` convention), accumulated with a second,
    score-only KV cache per beam.

The K complete beams are then rescored with the exact fitted objective
(sequence-level features normalised per ``nbest_data/decode_weights.json``,
plus ``gamma * q_phi`` from the trained reranker) and the argmax is emitted.

Writes a JSON shard: per example key -> {pred_score, rerank_meta{candidates,
weights, components}}; ``merge_rerank_rollouts.py`` folds it into ``data.js``
as ``rollouts_rerank.filtered``.

Smoke (no reranker yet, gamma=0):
  python visualizer/rerank_viz_rollout.py --example-keys val-01 \
      --weights none --output visualizer/rerank_shards/smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from anticipation.config import CONTEXT_SIZE, MAX_PITCH  # noqa: E402
from anticipation.packed_sequence import (  # noqa: E402
    ALTERNATING_START, iter_score_slot_positions)
from anticipation.score_constraints import constrain_score_token_logits  # noqa: E402
from anticipation.vocab import (  # noqa: E402
    AUTOREGRESS, DUR_OFFSET, NOTE_OFFSET, REST, TIME_OFFSET, VOCAB_SIZE)
from evaluate_muster import load_model  # noqa: E402
from precompute_beams import (  # noqa: E402
    _stack_pasts, _unstack_past, clone_past, note_from_tokens)
from precompute_visualizer import to_legacy_past, tokens_from_controls  # noqa: E402
from compute_sequence_ppl import control_notes_for_variant, load_payload  # noqa: E402

DEFAULT_FT = "run_paper_split_v2/checkpoint-7500"
BASE_MODEL = "stanford-crfm/music-medium-800k"
DEFAULT_WEIGHTS = "nbest_data/decode_weights.json"


def load_base_model(device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    if model.config.vocab_size != VOCAB_SIZE:
        raise SystemExit("base model vocab mismatch")
    return model.to(device).eval()


def _lp(logits_row, slot, tok):
    constrained = constrain_score_token_logits(logits_row.float(), slot)
    return float(F.log_softmax(constrained, dim=-1)[int(tok)].item())


def _top(logits_row, slot, k):
    constrained = constrain_score_token_logits(logits_row.float(), slot)
    log_probs = F.log_softmax(constrained, dim=-1)
    finite = torch.isfinite(log_probs)
    if not bool(finite.any()):
        return []
    values, indices = torch.topk(log_probs, min(int(k), int(finite.sum())))
    return [(int(t), float(v)) for t, v in zip(indices.tolist(), values.tolist())
            if v > float("-inf")]


class _Cache:
    """One model's KV state + last-position logits for one beam."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def feed_one(self, past, tok):
        out = self.model(torch.tensor([[tok]], device=self.device),
                         past_key_values=past, use_cache=True)
        return to_legacy_past(out.past_key_values), out.logits[0, -1, :]

    def feed_seq_all(self, pasts, seq):
        ids = torch.tensor([seq] * len(pasts), device=self.device)
        if len(pasts) == 1:
            out = self.model(ids, past_key_values=pasts[0], use_cache=True)
            return [to_legacy_past(out.past_key_values)], out.logits[:, -1, :]
        out = self.model(ids, past_key_values=_stack_pasts(pasts), use_cache=True)
        return _unstack_past(to_legacy_past(out.past_key_values)), out.logits[:, -1, :]


@torch.inference_mode()
def rerank_beam_search(ft_model, base_model, device, tokens, num_beams: int,
                       alpha: float, beta: float):
    """K-beam constrained triplet search pruned on alpha*lp_FT + beta*lp_base.

    Returns a list of K candidates, each a dict with slot-aligned ``pred``
    (note dicts / None), ``tokens`` (per-slot triplets), ``logp_ft``,
    ``logp_base``.
    """
    use_base = base_model is not None
    ft = _Cache(ft_model, device)
    base = _Cache(base_model, device) if use_base else None

    prime = ft_model(torch.tensor([tokens[:ALTERNATING_START]], device=device),
                     use_cache=True)
    beam0 = {
        "ft_past": to_legacy_past(prime.past_key_values),
        "ft_logits": prime.logits[0, -1, :],
        "base_past": None, "base_logits": None,
        "logp_ft": 0.0, "logp_base": 0.0,
        "pred": [], "tokens": [],
    }
    if use_base:
        base_prime = base_model(torch.tensor([[AUTOREGRESS]], device=device),
                                use_cache=True)
        beam0["base_past"] = to_legacy_past(base_prime.past_key_values)
        beam0["base_logits"] = base_prime.logits[0, -1, :]
    beams = [beam0]

    for s, pos in enumerate(iter_score_slot_positions(len(tokens),
                                                      ALTERNATING_START)):
        if pos + 5 >= len(tokens):
            for b in beams:
                b["pred"].append(None)
                b["tokens"].append(None)
            continue
        candidates = []  # (combined, lp_ft3, lp_base3, onset, dur, pitch, parent)
        for pi, b in enumerate(beams):
            for onset_tok, lp_o in _top(b["ft_logits"], 0, num_beams):
                lp_bo = _lp(b["base_logits"], 0, onset_tok) if use_base else 0.0
                ft_po, ft_ld = ft.feed_one(b["ft_past"], onset_tok)
                if use_base:
                    ba_po, ba_ld = base.feed_one(b["base_past"], onset_tok)
                for dur_tok, lp_d in _top(ft_ld, 1, num_beams):
                    lp_bd = _lp(ba_ld, 1, dur_tok) if use_base else 0.0
                    ft_pod, ft_lp_row = ft.feed_one(ft_po, dur_tok)
                    if use_base:
                        ba_pod, ba_lp_row = base.feed_one(ba_po, dur_tok)
                    for pitch_tok, lp_p in _top(ft_lp_row, 2, num_beams):
                        lp_bp = (_lp(ba_lp_row, 2, pitch_tok)
                                 if use_base else 0.0)
                        ft3 = lp_o + lp_d + lp_p
                        base3 = lp_bo + lp_bd + lp_bp
                        combined = (alpha * (b["logp_ft"] + ft3)
                                    + beta * (b["logp_base"] + base3))
                        candidates.append((combined, ft3, base3, onset_tok,
                                           dur_tok, pitch_tok, pi))
        if not candidates:
            for b in beams:
                b["pred"].append(None)
                b["tokens"].append(None)
        else:
            candidates.sort(key=lambda x: x[0], reverse=True)
            survivors = candidates[:num_beams]
            fanout = {}
            for *_, pi in survivors:
                fanout[pi] = fanout.get(pi, 0) + 1
            new_beams = []
            for _, ft3, base3, onset_tok, dur_tok, pitch_tok, pi in survivors:
                parent = beams[pi]
                ft_past = (clone_past(parent["ft_past"]) if fanout[pi] > 1
                           else parent["ft_past"])
                base_past = base_logits = None
                if use_base:
                    base_past = (clone_past(parent["base_past"])
                                 if fanout[pi] > 1 else parent["base_past"])
                for tok in (onset_tok, dur_tok, pitch_tok):
                    ft_past, ft_logits = ft.feed_one(ft_past, tok)
                    if use_base:
                        base_past, base_logits = base.feed_one(base_past, tok)
                new_beams.append({
                    "ft_past": ft_past, "ft_logits": ft_logits,
                    "base_past": base_past, "base_logits": base_logits,
                    "logp_ft": parent["logp_ft"] + ft3,
                    "logp_base": parent["logp_base"] + base3,
                    "pred": parent["pred"]
                    + [note_from_tokens(onset_tok, dur_tok, pitch_tok)],
                    "tokens": parent["tokens"]
                    + [(onset_tok, dur_tok, pitch_tok)],
                })
            beams = new_beams

        ctrl = [int(tokens[pos + 3 + k]) for k in range(3)]
        pasts, logits = ft.feed_seq_all([b["ft_past"] for b in beams], ctrl)
        for b, p, i in zip(beams, pasts, range(len(beams))):
            b["ft_past"] = p
            b["ft_logits"] = logits[i]
        # The base (score-only) stream skips controls entirely.

    return [{"pred": b["pred"], "tokens": b["tokens"],
             "logp_ft": b["logp_ft"], "logp_base": b["logp_base"]}
            for b in beams]


def load_weights(path: str):
    if path == "none":
        return {"alpha": 1.0, "beta": 1.0, "gamma": 0.0,
                "feature_stats": None, "reranker_ckpt": None}
    with open(path) as f:
        return json.load(f)


def normalise(value: float, stats: dict | None, name: str) -> float:
    if not stats or name not in stats:
        return value
    s = stats[name]
    return (value - s["mean"]) / max(s.get("std", 1.0), 1e-8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--checkpoint", default=DEFAULT_FT)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="decode_weights.json from nbest/fit_weights.py; "
                         "'none' -> alpha=beta=1, gamma=0")
    ap.add_argument("--alpha", type=float, default=None,
                    help="override alpha (with --beta: gamma=0 baseline mode)")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--reranker-ckpt", default=None,
                    help="override the reranker checkpoint in --weights")
    ap.add_argument("--num-beams", type=int, default=8)
    ap.add_argument("--example-keys", default=None,
                    help="comma-separated; default = all in example_order")
    ap.add_argument("--variant", default="filtered", choices=["filtered", "raw"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = load_weights(args.weights)
    if args.alpha is not None or args.beta is not None:
        weights = {"alpha": args.alpha if args.alpha is not None else 1.0,
                   "beta": args.beta if args.beta is not None else 0.0,
                   "gamma": 0.0, "feature_stats": None, "reranker_ckpt": None}
    alpha, beta, gamma = weights["alpha"], weights["beta"], weights["gamma"]
    stats = weights.get("feature_stats")

    reranker = None
    reranker_ckpt = (args.reranker_ckpt or weights.get("reranker_ckpt")
                     or weights.get("reranker_checkpoint"))
    if gamma != 0.0 and reranker_ckpt:
        from nbest.reranker import build_reranker_from_ckpt
        reranker = build_reranker_from_ckpt(reranker_ckpt, device)

    payload, _ = load_payload(args.data)
    order = payload["example_order"]
    keys = (args.example_keys.split(",") if args.example_keys else order)

    ft_model, _ = load_model(args.checkpoint)
    ft_model = ft_model.to(device).eval()
    base_model = load_base_model(device) if beta != 0.0 else None

    shard = {"checkpoint": args.checkpoint, "weights": weights,
             "num_beams": args.num_beams, "variant": args.variant,
             "examples": {}}
    for key in keys:
        ex = payload["examples"][key]
        control_notes = control_notes_for_variant(ex, args.variant)
        tokens = tokens_from_controls(control_notes, CONTEXT_SIZE - 4)
        t0 = time.time()
        cands = rerank_beam_search(ft_model, base_model, device, tokens,
                                   args.num_beams, alpha, beta)
        flat_pos = torch.tensor(
            [p + j for p in iter_score_slot_positions(len(tokens))
             if p + 5 < len(tokens) for j in range(3)], device=device)
        window_t = torch.tensor([tokens], dtype=torch.long, device=device)
        rows = []
        for c in cands:
            q_phi = 0.0
            if reranker is not None:
                from nbest.reranker import substitute_candidates
                cand_flat = [t for trip in c["tokens"] if trip is not None
                             for t in trip]
                cand_t = torch.tensor([cand_flat], dtype=torch.long,
                                      device=device)
                with torch.inference_mode():
                    q_phi = float(reranker(substitute_candidates(
                        window_t, cand_t, flat_pos[:cand_t.shape[1]])).item())
            objective = (alpha * normalise(c["logp_ft"], stats, "logp_ft")
                         + beta * normalise(c["logp_base"], stats, "logp_base")
                         + gamma * normalise(q_phi, stats, "q_phi"))
            rows.append({"logp_ft": c["logp_ft"], "logp_base": c["logp_base"],
                         "q_phi": q_phi, "objective": objective,
                         "pred": c["pred"]})
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
        print(f"{key}: {len(cands)} beams  best={best}  "
              f"logp_ft={rows[best]['logp_ft']:.1f}  "
              f"logp_base={rows[best]['logp_base']:.1f}  "
              f"q_phi={rows[best]['q_phi']:.3f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(shard, f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
