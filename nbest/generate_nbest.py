#!/usr/bin/env python
"""N-best candidate generation for discriminative reranking.

For a stride-sample of packed windows, generates C(x) = 8 sampled (T=1.0,
constrained) + 1 greedy candidate score per window from the fine-tuned model,
and records per candidate:

  * ``logp_ft``   -- summed constrained interleaved score-token log-prob under
                     the FT model. For sampled candidates this is the rollout's
                     own ``logprob`` at T=1.0 (`onpolicy_rollout.rollout_score_slots`,
                     the train_grpo.py repeat_interleave idiom); the greedy
                     candidate is re-scored teacher-forced via
                     `onpolicy_rollout.score_token_logprob` at temperature=1.0.
  * ``logp_base`` -- MINUS the summed constrained score-only NLL under the
                     untuned base AMT, exactly `eval_base_score_ppl`'s ``so_c``
                     convention (AUTOREGRESS-primed flat 414-token layout,
                     reusing its `slot_logit_masks` + `nll_at_positions`).
  * ``f1``        -- `f1_reward.final_f1` (onset_pitch_tol1 semantics) against
                     the window's own GT body score triplets.
  * the candidate's 414 score tokens (int16 -- score vocab < 27513 fits) and
    the source line index, so the reranker can rebuild (window, candidate).

FT model of record: ``run_paper_split_v2/checkpoint-7500`` -- the visualizer's
FT model, deliberately chosen over the RL-init checkpoint-2500 so reranking is
consistent with the viz rollouts.

Usage (one shard):
  python -m nbest.generate_nbest --token-file data/train_paper.txt \
      --stride 40 --shard-index 0 --num-shards 2 \
      --output nbest_data/train_shard00.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from anticipation.config import MAX_DUR, MAX_PITCH, MAX_TIME
from anticipation.vocab import (ADUR_OFFSET, ANOTE_OFFSET, ATIME_OFFSET,
                                AUTOREGRESS, NOTE_OFFSET, REST)
from eval_base_score_ppl import load_base_model, nll_at_positions, slot_logit_masks
from packed_dataset import iter_sequence_triplets
from evaluate_muster import load_model
from f1_reward import final_f1, score_triplet_to_note
from onpolicy_rollout import (rollout_score_slots, score_token_logprob,
                              score_token_positions)

DEFAULT_CKPT = "run_paper_split_v2/checkpoint-7500"


def read_shard_windows(path, stride, shard_index, num_shards, max_windows):
    line_indices, windows = [], []
    selected = 0
    with open(path) as f:
        for line_idx, line in enumerate(f):
            if line_idx % stride != 0:
                continue
            keep = selected % num_shards == shard_index
            selected += 1
            if not keep:
                continue
            tokens = [int(t) for t in line.split("|", 1)[0].split()]
            if len(tokens) != 1020:
                continue
            line_indices.append(line_idx)
            windows.append(torch.tensor(tokens, dtype=torch.long))
            if max_windows is not None and len(windows) >= max_windows:
                break
    return line_indices, windows


def _transpose_tok(tok, note_base, shift):
    """train.py's octave-folded transposition of one pitch token."""
    raw = tok - note_base
    instr = raw // MAX_PITCH
    pitch = raw % MAX_PITCH + shift
    while pitch > MAX_PITCH - 1:
        pitch -= 12
    while pitch < 0:
        pitch += 12
    return note_base + instr * MAX_PITCH + max(0, min(MAX_PITCH - 1, pitch))


def augment_window(tokens, rng, transpose_range=12, tempo_range=0.2):
    """train.py-style augmentation of one packed window.

    Transposition shifts BOTH streams' pitches consistently (octave-folded,
    as `train.py._augment_sequence`); tempo scaling rescales only the
    PERFORMANCE (control) times and durations -- score onsets live on the
    fixed beat grid and stay untouched. One (shift, factor) pair per window.
    """
    shift = rng.randint(-transpose_range, transpose_range)
    factor = 1.0 + (rng.random() * 2.0 - 1.0) * tempo_range
    out = list(tokens)
    for pos, tok0, tok1, tok2, is_control in iter_sequence_triplets(out):
        if is_control:
            t = int(round((tok0 - ATIME_OFFSET) * factor))
            out[pos] = ATIME_OFFSET + max(0, min(MAX_TIME - 1, t))
            d = int(round((tok1 - ADUR_OFFSET) * factor))
            out[pos + 1] = ADUR_OFFSET + max(0, min(MAX_DUR - 1, d))
            if shift:
                out[pos + 2] = _transpose_tok(tok2, ANOTE_OFFSET, shift)
        elif shift and tok2 != REST:
            out[pos + 2] = _transpose_tok(tok2, NOTE_OFFSET, shift)
    return out


def flat_notes(flat_tokens):
    """(414,) flat score tokens -> list of 138 (onset, dur, pitch) | None."""
    toks = flat_tokens.tolist()
    return [score_triplet_to_note(toks[3 * k], toks[3 * k + 1], toks[3 * k + 2])
            for k in range(len(toks) // 3)]


@torch.inference_mode()
def logp_base_batch(base_model, masks, cand_flat, chunk):
    """(M, 414) score tokens -> (M,) summed constrained log-prob (so_c)."""
    device = cand_flat.device
    m = cand_flat.shape[0]
    prime = torch.full((m, 1), AUTOREGRESS, dtype=torch.long, device=device)
    seqs = torch.cat([prime, cand_flat], dim=1)
    n = cand_flat.shape[1]
    target_pos = torch.arange(1, n + 1, device=device)
    pred_pos = target_pos - 1
    slot_ids = (target_pos - 1) % 3
    _, nll_c = nll_at_positions(base_model, seqs, pred_pos, target_pos,
                                slot_ids, masks, chunk)
    return -nll_c.sum(dim=1)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--token-file", default="data/train_paper.txt")
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--n-sampled", type=int, default=8)
    ap.add_argument("--window-batch", type=int, default=24)
    ap.add_argument("--score-chunk", type=int, default=16)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true",
                    help="apply train.py-style augmentation (transpose both "
                         "streams, tempo-scale the performance) per window, "
                         "seeded by line index")
    ap.add_argument("--aug-transpose", type=int, default=12)
    ap.add_argument("--aug-tempo", type=float, default=0.2)
    ap.add_argument("--save-every", type=int, default=50,
                    help="write a resumable .partial every N window batches "
                         "(0 disables)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <output>.partial if present")
    ap.add_argument("--fp32", action="store_true",
                    help="decode in fp32 (no bf16 autocast) -- see the dtype "
                         "note at the autocast site")
    ap.add_argument("--fast-decode", action="store_true",
                    help="use anticipation.fast_decode (bit-identical fp32, "
                         "~1.75x at these batch sizes)")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed + 100 * a.shard_index)
    ft, _ = load_model(a.checkpoint)   # returns (model, device)
    ft = ft.to(device).eval()
    base = load_base_model(
        device, dtype=torch.float32 if a.fp32 else torch.bfloat16)
    masks = slot_logit_masks(device)

    line_indices, windows = read_shard_windows(
        a.token_file, a.stride, a.shard_index, a.num_shards, a.max_windows)
    if a.augment:
        import random
        windows = [
            torch.tensor(
                augment_window(w.tolist(),
                               random.Random(a.seed * 1_000_003 + int(li)),
                               a.aug_transpose, a.aug_tempo),
                dtype=torch.long)
            for li, w in zip(line_indices, windows)
        ]
    print(f"{a.token_file}: shard {a.shard_index}/{a.num_shards} stride "
          f"{a.stride} -> {len(windows)} windows"
          f"{' (augmented)' if a.augment else ''}", flush=True)
    positions = score_token_positions(1020, device=device)

    # DTYPE MATTERS AND IS NOW EXPLICIT. bf16 autocast was the original
    # default here; it disagrees with fp32 on ~37% of greedy score tokens
    # (visualizer/dtype_ab_viz.py, 24 viz windows: 63.3% token agreement),
    # which meant candidate pools and the visualizer's own greedy row were
    # decoded under different numerics. --fp32 runs the decode in full
    # precision; pair it with --fast-decode, which is a bit-identical fp32
    # reimplementation (anticipation/fast_decode.py) that claws back the
    # speed (1.75x at these batch sizes) instead of buying it with precision.
    import contextlib
    if a.fp32:
        autocast = lambda: contextlib.nullcontext()   # noqa: E731
    else:
        autocast = lambda: torch.autocast(  # noqa: E731
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available())
    # cuda_graph is greedy-only (a graph-private RNG cannot reproduce eager
    # multinomial), so sampled rollouts get buckets only.
    fast_s = {"buckets": 8} if a.fast_decode else None
    fast_g = ({"buckets": 8, "cuda_graph": True} if a.fast_decode else None)

    w_line, w_tokens = [], []
    c_line, c_tokens, c_logp_ft, c_logp_base, c_f1, c_kind = [], [], [], [], [], []
    def _pack():
        return {
        "token_file": a.token_file, "stride": a.stride,
        "shard_index": a.shard_index, "num_shards": a.num_shards,
        "checkpoint": a.checkpoint, "n_sampled": a.n_sampled,
        "window_line_idx": torch.tensor(w_line, dtype=torch.long),
        "window_tokens": torch.cat(w_tokens) if w_tokens else torch.empty(0),
        "cand_line_idx": torch.tensor(c_line, dtype=torch.long),
        "cand_tokens": torch.cat(c_tokens) if c_tokens else torch.empty(0),
        "cand_logp_ft": torch.cat(c_logp_ft) if c_logp_ft else torch.empty(0),
        "cand_logp_base": torch.cat(c_logp_base) if c_logp_base else torch.empty(0),
        "cand_f1": torch.tensor(c_f1, dtype=torch.float32),
        "cand_kind": torch.tensor(c_kind, dtype=torch.int8),
        }


    resume_from = 0
    part_path = Path(str(a.output) + ".partial")
    if a.resume and part_path.exists():
        pd = torch.load(part_path, map_location="cpu", weights_only=False)
        resume_from = int(pd.get("resume_done", 0))
        w_line = pd["window_line_idx"].tolist()
        w_tokens = [pd["window_tokens"]]
        c_line = pd["cand_line_idx"].tolist()
        c_tokens = [pd["cand_tokens"]]
        c_logp_ft = [pd["cand_logp_ft"]]
        c_logp_base = [pd["cand_logp_base"]]
        c_f1 = pd["cand_f1"].tolist()
        c_kind = pd["cand_kind"].tolist()
        print(f"resuming from {part_path}: {resume_from} windows already done",
              flush=True)
    t0 = time.time()
    for start in range(resume_from, len(windows), a.window_batch):
        rows = windows[start:start + a.window_batch]
        idxs = line_indices[start:start + a.window_batch]
        batch = torch.stack(rows).to(device)
        b = batch.shape[0]

        rep = batch.repeat_interleave(a.n_sampled, dim=0)
        out = rollout_score_slots(
            ft, rep, temperature=1.0, constrain=True, collect_logprobs=True,
            collect_gt_ce=False, autocast_ctx=autocast, fast=fast_s)
        sampled_flat = out["rolled"][:, positions]            # (b*N, 414)
        sampled_lp = (out["logprob"] * out["valid"]).sum(dim=1)

        greedy = rollout_score_slots(
            ft, batch, temperature=0.0, constrain=True, collect_logprobs=False,
            collect_gt_ce=False, autocast_ctx=autocast, fast=fast_g)
        greedy_flat = greedy["rolled"][:, positions]          # (b, 414)
        greedy_lp = torch.empty(b, device=device)
        for lo in range(0, b, a.score_chunk):
            hi = min(lo + a.score_chunk, b)
            with autocast():
                logits = ft(greedy["rolled"][lo:hi]).logits
            lp = score_token_logprob(logits, greedy["rolled"][lo:hi],
                                     positions, temperature=1.0, constrain=True)
            greedy_lp[lo:hi] = (lp * greedy["valid"][lo:hi]).sum(dim=1)
            del logits

        cand_flat = torch.cat([greedy_flat, sampled_flat])    # greedy first
        cand_line = idxs + [i for i in idxs for _ in range(a.n_sampled)]
        cand_lp_ft = torch.cat([greedy_lp, sampled_lp])
        cand_kind = [0] * b + [1] * (b * a.n_sampled)
        cand_lp_base = logp_base_batch(base, masks, cand_flat, a.score_chunk)

        gt_flat = batch[:, positions].cpu()
        gt_notes = {idx: flat_notes(gt_flat[i]) for i, idx in enumerate(idxs)}
        cand_flat_cpu = cand_flat.cpu()
        for row in range(cand_flat_cpu.shape[0]):
            c_f1.append(final_f1(flat_notes(cand_flat_cpu[row]),
                                 gt_notes[cand_line[row]]))
        c_line.extend(cand_line)
        c_tokens.append(cand_flat_cpu.to(torch.int16))
        c_logp_ft.append(cand_lp_ft.float().cpu())
        c_logp_base.append(cand_lp_base.float().cpu())
        c_kind.extend(cand_kind)
        w_line.extend(idxs)
        w_tokens.append(batch.cpu().to(torch.int32))

        done = start + b
        # Checkpoint mid-shard: these jobs run for hours on preemptible
        # nodes, and this file used to be written only at completion (a
        # cancellation at 7h lost an entire shard once). The partial is
        # written to a temp path and renamed, so a kill mid-write cannot
        # leave a half-file that a resume would trust.
        if a.save_every and (start // a.window_batch) % a.save_every == 0 \
                and start > 0:
            part = Path(str(a.output) + ".partial")
            part.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(part) + ".tmp")
            torch.save({**_pack(), "resume_done": done}, tmp)
            tmp.replace(part)
            print(f"  [checkpoint] {done} windows -> {part}", flush=True)
        rate = done / max(time.time() - t0, 1e-6)
        if (start // a.window_batch) % 10 == 0:
            print(f"  {done}/{len(windows)} windows  {rate:.2f} win/s  "
                  f"eta={(len(windows)-done)/max(rate,1e-6)/3600:.2f}h",
                  flush=True)

    result = _pack()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    Path(str(out) + ".partial").unlink(missing_ok=True)
    n = len(c_line)
    print(f"wrote {out}: {len(w_line)} windows, {n} candidates  "
          f"mean_f1={result['cand_f1'].mean():.4f}  "
          f"greedy_mean_f1={result['cand_f1'][result['cand_kind'] == 0].mean():.4f}",
          flush=True)


if __name__ == "__main__":
    main()
