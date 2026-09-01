#!/usr/bin/env python
"""Add PER-TOKEN log-prob features to existing N-best shards.

The shards store only the SUMMED `cand_logp_ft` / `cand_logp_base`. Both were
computed from per-position tensors that were then reduced away, so recovering
the sequences needs no re-sampling -- one teacher-forced forward per candidate
under each model, which is cheap next to the 414-step decode that produced the
candidates in the first place.

Per candidate this writes two (414,) float16 rows, matching the scalars'
conventions exactly so the new features are a strict refinement:

  cand_tok_logp_ft   -- constrained score-token log-prob under the FT model
                        with the candidate substituted into the FULL
                        interleaved window (onpolicy_rollout.score_token_logprob,
                        temperature 1.0, constrain=True). Summing this row
                        reproduces `cand_logp_ft` for greedy candidates; for
                        sampled ones the stored scalar came from the rollout's
                        own logprobs, which is the same quantity computed
                        during decode.
  cand_tok_logp_base -- MINUS the constrained score-only NLL under the untuned
                        base AMT (eval_base_score_ppl `so_c`: AUTOREGRESS-primed
                        flat 414-token layout). Summing reproduces
                        `cand_logp_base` exactly.

Usage (one shard):
  python -m nbest.add_token_features --shard nbest_data/unf32_train_shard00.pt \
      --output nbest_data/tokfeat32_train_shard00.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import contextlib

import torch

# --fp32 makes this pass genuinely full precision on BOTH channels. Three
# separate things had to change for that, and missing any one of them leaves a
# shard that is labelled fp32 but is not:
#   1. this autocast, which governs the FT channel;
#   2. load_base_model's WEIGHT cast (it hardcoded bf16 and took no dtype), which
#      governs the base channel -- this is why nbest_data/fp32_*.pt are fp32 on
#      the FT channel and bf16 on the base channel despite the name;
#   3. the float16 STORAGE below, which would re-quantise whatever precision
#      the first two produced.
_FP32 = False
_STORE = torch.float16


def _AC():
    if _FP32:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16,
                          enabled=torch.cuda.is_available())

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from anticipation.vocab import AUTOREGRESS
from eval_base_score_ppl import (load_base_model, nll_at_positions,
                                 slot_logit_masks)
from evaluate_muster import load_model
from nbest.reranker import substitute_candidates
from onpolicy_rollout import score_token_logprob, score_token_positions


@torch.inference_mode()
def ft_token_logp(ft, windows, cands, flat_pos, chunk):
    """(B,1020) windows + (B,414) candidates -> (B,414) constrained log-prob."""
    out = []
    for s in range(0, windows.shape[0], chunk):
        toks = substitute_candidates(windows[s:s + chunk],
                                     cands[s:s + chunk], flat_pos)
        with _AC():
            logits = ft(toks, use_cache=False).logits
        out.append(score_token_logprob(logits, toks, flat_pos,
                                       temperature=1.0, constrain=True)
                   .to(_STORE).cpu())
        del logits, toks
    return torch.cat(out)


@torch.inference_mode()
def base_token_logp(base, masks, cands, chunk):
    """(B,414) score tokens -> (B,414) so_c per-token log-prob."""
    device = cands.device
    n = cands.shape[1]
    prime = torch.full((cands.shape[0], 1), AUTOREGRESS, dtype=torch.long,
                       device=device)
    seqs = torch.cat([prime, cands.to(torch.long)], dim=1)
    target_pos = torch.arange(1, n + 1, device=device)
    _, nll_c = nll_at_positions(base, seqs, target_pos - 1, target_pos,
                                (target_pos - 1) % 3, masks, chunk)
    return (-nll_c).to(_STORE).cpu()


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="default: the FT checkpoint recorded in the shard")
    ap.add_argument("--batch", type=int, default=24,
                    help="candidates per FT forward")
    ap.add_argument("--score-chunk", type=int, default=48)
    ap.add_argument("--report-every", type=int, default=20)
    ap.add_argument("--fp32", action="store_true",
                    help="full precision on BOTH channels (no bf16 autocast, "
                         "fp32 base weights, fp32 feature storage), and "
                         "rewrite cand_logp_base from the fp32 per-token sum")
    a = ap.parse_args()
    global _FP32, _STORE
    _FP32 = bool(a.fp32)
    _STORE = torch.float32 if _FP32 else torch.float16
    print(f"feature dtype: {'fp32' if _FP32 else 'bf16 autocast / fp16 store'}",
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(a.shard, map_location="cpu", weights_only=False)
    ckpt = a.checkpoint or d["checkpoint"]
    print(f"{a.shard}: {d['cand_tokens'].shape[0]} candidates, FT {ckpt}",
          flush=True)

    ft, _ = load_model(ckpt)
    ft = ft.to(device).eval()
    base = load_base_model(device,
                           dtype=torch.float32 if a.fp32 else torch.bfloat16)
    masks = slot_logit_masks(device)
    flat_pos = score_token_positions(d["window_tokens"].shape[1], device=device)

    # Candidate i belongs to window row_of_line[cand_line_idx[i]].
    row_of_line = {int(l): i for i, l in enumerate(d["window_line_idx"].tolist())}
    cand_row = torch.tensor([row_of_line[int(l)]
                             for l in d["cand_line_idx"].tolist()],
                            dtype=torch.long)

    n = d["cand_tokens"].shape[0]
    ft_rows, base_rows = [], []
    t0 = time.time()
    for s in range(0, n, a.batch):
        idx = slice(s, min(s + a.batch, n))
        cands = d["cand_tokens"][idx].to(device)
        wins = d["window_tokens"][cand_row[idx]].to(device).long()
        ft_rows.append(ft_token_logp(ft, wins, cands, flat_pos, a.batch))
        base_rows.append(base_token_logp(base, masks, cands, a.score_chunk))
        if (s // a.batch) % a.report_every == 0:
            done = min(s + a.batch, n)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{n} candidates  {rate:.0f}/s  "
                  f"eta={(n - done) / max(rate, 1e-6) / 60:.1f}min", flush=True)

    d["cand_tok_logp_ft"] = torch.cat(ft_rows)
    d["cand_tok_logp_base"] = torch.cat(base_rows)
    d["feature_dtype"] = "fp32" if _FP32 else "bf16_autocast_fp16_store"

    if _FP32:
        # The stored scalar was computed with a bf16 BASE MODEL even in shards
        # named fp32 (load_base_model cast the weights and --fp32 only swapped
        # the FT autocast). `logp_base` is a pure teacher-forced function of the
        # candidate tokens, so the fp32 value is recoverable here by summation
        # -- no re-decode needed. Keep the old value for audit rather than
        # overwriting silently: fit_weights.py's alpha/beta and feature_stats
        # are fitted to whichever scale is in the shard.
        old_base = d["cand_logp_base"].float()
        new_base = d["cand_tok_logp_base"].float().sum(1)
        d["cand_logp_base_bf16_base_model"] = old_base
        d["cand_logp_base"] = new_base
        shift = (new_base - old_base)
        print(f"  base scalar rewritten in fp32: mean shift {shift.mean():+.4f}"
              f"  max |shift| {shift.abs().max():.4f}", flush=True)

    # Consistency check against the stored scalars (the base side must match
    # to float error; the FT side matches for greedy candidates, while sampled
    # ones were scored from the rollout's own logprobs during decode).
    for name, per_tok, scalar in (
            ("base", d["cand_tok_logp_base"], d["cand_logp_base"]),
            ("ft", d["cand_tok_logp_ft"], d["cand_logp_ft"])):
        got = per_tok.float().sum(1)
        err = (got - scalar).abs()
        print(f"  {name}: |sum(per-token) - stored| mean {err.mean():.3f} "
              f"max {err.max():.3f}", flush=True)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
