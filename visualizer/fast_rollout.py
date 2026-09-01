"""Faster visualizer AR rollouts: SDPA + TF32 + torch.compile + batched decode.

Token-identical to ``precompute_visualizer.rollout_with_candidates`` for greedy
constrained AR (same ``constrain_score_token_logits``, same top-k candidate
tree). Batching is across the four filtered/raw × plain/GT-seed variants of a
window (and optionally multiple windows that share the packed length).

TensorRT is attempted via ``torch.compile(..., backend="tensorrt")`` and
skipped if the package is missing. Growing KV-cache decode is compiled with
inductor ``mode=default`` (not ``reduce-overhead``): CUDA graphs recapture on
every cache length and would be slower than eager.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from anticipation.config import CONTEXT_SIZE, MAX_PITCH
from anticipation.score_constraints import constrain_score_token_logits
from anticipation.packed_sequence import ALTERNATING_START, iter_score_slot_positions
from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, REST, TIME_OFFSET

from precompute_visualizer import (  # noqa: E402
    build_branches_from_slots,
    compact_entropy,
    compact_perplexity,
    encode_score_note,
    mark_greedy_candidates,
    to_legacy_past,
    tokens_from_controls,
)


def enable_fast_kernels():
    """TF32 + SDPA flash/mem-efficient. No-op on CPU."""
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


def _local_inductor_cache():
    """Keep inductor/triton caches on local /tmp, not NFS home."""
    job = os.environ.get("SLURM_JOB_ID", "local")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu").replace(",", "_")
    root = f"/tmp/vis_compile_{os.environ.get('USER', 'user')}_{job}_{gpu}"
    os.makedirs(root, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(root, "inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(root, "triton"))
    os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)
    return root


def load_and_compile_model(
    checkpoint,
    device,
    *,
    compile_model=True,
    compile_mode="default",
    try_tensorrt=True,
    attn_implementation="sdpa",
):
    """Load a full-FT causal LM for greedy vis decode; optionally compile it."""
    enable_fast_kernels()
    cache_root = _local_inductor_cache()
    print(f"compile cache: {cache_root}")

    kwargs = dict(local_files_only=True, use_cache=True)
    model = None
    if attn_implementation and attn_implementation != "eager":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint),
                attn_implementation=attn_implementation,
                **kwargs,
            )
            print(f"loaded {checkpoint} attn={attn_implementation}")
        except (TypeError, ValueError) as exc:
            print(f"attn_implementation={attn_implementation!r} rejected ({exc}); eager attn")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), **kwargs)
        print(f"loaded {checkpoint} default attn")

    model.config.use_cache = True
    model.to(device)
    model.eval()

    compile_backend = "eager"
    if compile_model and hasattr(torch, "compile"):
        try:
            torch._dynamo.config.optimize_ddp = False
            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass
        if try_tensorrt:
            try:
                model = torch.compile(model, backend="tensorrt", dynamic=True)
                compile_backend = "tensorrt"
                print("torch.compile backend=tensorrt enabled")
                return model, compile_backend
            except Exception as exc:
                print(
                    f"TensorRT compile unavailable ({type(exc).__name__}: {exc}); "
                    "falling back to inductor"
                )
        try:
            # Growing KV cache: reduce-overhead CUDA graphs recapture every
            # length and are slower than eager. default still fuses kernels.
            model = torch.compile(model, mode=compile_mode, dynamic=True)
            compile_backend = f"inductor:{compile_mode}"
            print(f"torch.compile enabled ({compile_backend}, dynamic=True)")
        except Exception as exc:
            print(f"torch.compile failed, running eager: {exc}")
            compile_backend = "eager"
    return model, compile_backend


def warmup_compile(model, device, prefix_tokens, batch_size=4):
    """Prime inductor on the packed prefix + one decode step."""
    prefix = torch.tensor(
        [list(prefix_tokens[:ALTERNATING_START]) for _ in range(batch_size)],
        device=device,
        dtype=torch.long,
    )
    with torch.inference_mode():
        out = model(prefix, use_cache=True)
        past = to_legacy_past(out.past_key_values)
        nxt = torch.zeros(batch_size, dtype=torch.long, device=device)
        nxt.fill_(int(prefix_tokens[0]) if prefix_tokens else TIME_OFFSET)
        model(nxt.unsqueeze(1), past_key_values=past, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"compile warmup done (B={batch_size}, prefix={ALTERNATING_START})")


def _batched_feed(model, token_col, past):
    out = model(token_col.unsqueeze(1), past_key_values=past, use_cache=True)
    return to_legacy_past(out.past_key_values), out.logits[:, -1, :]


def _batched_top(logits, slot, k):
    constrained = constrain_score_token_logits(logits.float(), slot)
    probs = F.softmax(constrained, dim=-1)
    values, indices = torch.topk(probs, k, dim=-1)
    return indices, values


def _batched_logprob(logits, slot, token_col):
    constrained = constrain_score_token_logits(logits.float(), slot)
    log_probs = F.log_softmax(constrained, dim=-1)
    return log_probs.gather(-1, token_col.unsqueeze(-1)).squeeze(-1)


def _batched_entropy(logits, slot):
    constrained = constrain_score_token_logits(logits.float(), slot)
    log_probs = F.log_softmax(constrained, dim=-1)
    probs = log_probs.exp()
    term = torch.where(probs > 0, probs * log_probs, torch.zeros_like(log_probs))
    return -term.sum(dim=-1)


@torch.inference_mode()
def batched_rollout_with_candidates(
    model,
    device,
    token_seqs,
    seed_notes,
    topk_onset,
    topk_dur,
    topk_pitch,
    max_candidates,
):
    """Greedy constrained AR + candidate tree for ``len(token_seqs)`` packed windows.

    Every sequence must share the same packed length / slot layout. ``seed_notes[i]``
    is a ``{t,d,p}`` dict to force-feed at slot 0, or ``None``.
    """
    if not token_seqs:
        return []
    batch = len(token_seqs)
    if len(seed_notes) != batch:
        raise ValueError("seed_notes must match token_seqs")
    length = len(token_seqs[0])
    if any(len(seq) != length for seq in token_seqs):
        raise ValueError("batched rollout requires identical packed lengths")

    prefix = torch.tensor(
        [seq[:ALTERNATING_START] for seq in token_seqs],
        device=device,
        dtype=torch.long,
    )
    tokens_t = torch.tensor(token_seqs, device=device, dtype=torch.long)
    prime = model(prefix, use_cache=True)
    past = to_legacy_past(prime.past_key_values)
    next_logits = prime.logits[:, -1, :]

    slot_positions = list(iter_score_slot_positions(length, ALTERNATING_START))
    pred_by_slot = [[] for _ in range(batch)]
    candidates_by_slot = [[] for _ in range(batch)]
    perplexity_by_slot = [[] for _ in range(batch)]
    entropy_by_slot = [[] for _ in range(batch)]

    for s, pos in enumerate(slot_positions):
        if pos + 5 >= length:
            for i in range(batch):
                pred_by_slot[i].append(None)
                candidates_by_slot[i].append([])
                perplexity_by_slot[i].append(None)
                entropy_by_slot[i].append(None)
            continue

        slot_past = past
        onset_idx, onset_pr = _batched_top(next_logits, 0, topk_onset)
        greedy_pitch_logits = None
        greedy_dur_logits = None
        greedy_onset = onset_idx[:, 0]
        greedy_dur = None
        cand_rows = [[] for _ in range(batch)]

        for oi in range(topk_onset):
            past_o, dur_logits = _batched_feed(model, onset_idx[:, oi], slot_past)
            dur_idx, dur_pr = _batched_top(dur_logits, 1, topk_dur)
            for di in range(topk_dur):
                past_od, pitch_logits = _batched_feed(model, dur_idx[:, di], past_o)
                if oi == 0 and di == 0:
                    greedy_pitch_logits = pitch_logits
                    greedy_dur_logits = dur_logits
                    greedy_dur = dur_idx[:, 0]
                pitch_idx, pitch_pr = _batched_top(pitch_logits, 2, topk_pitch)
                onset_tok = onset_idx[:, oi].tolist()
                dur_tok = dur_idx[:, di].tolist()
                pitch_tok = pitch_idx.tolist()
                o_p = onset_pr[:, oi].tolist()
                d_p = dur_pr[:, di].tolist()
                p_p = pitch_pr.tolist()
                for i in range(batch):
                    ot, dt = onset_tok[i], dur_tok[i]
                    op, dp = o_p[i], d_p[i]
                    if op <= 0 or dp <= 0:
                        continue
                    for kk in range(topk_pitch):
                        pp = p_p[i][kk]
                        if pp <= 0:
                            continue
                        pt = pitch_tok[i][kk]
                        cand_rows[i].append({
                            "t": ot - TIME_OFFSET,
                            "d": dt - DUR_OFFSET,
                            "p": (pt - NOTE_OFFSET) % MAX_PITCH if pt != REST else None,
                            "rest": pt == REST,
                            "prob": op * dp * pp,
                            "probs": [op, dp, pp],
                        })

        if greedy_pitch_logits is None or greedy_dur is None:
            for i in range(batch):
                pred_by_slot[i].append(None)
                candidates_by_slot[i].append([])
                perplexity_by_slot[i].append(None)
                entropy_by_slot[i].append(None)
            continue

        pitch_idx1, _ = _batched_top(greedy_pitch_logits, 2, 1)
        greedy_pitch = pitch_idx1[:, 0]
        ppl_onset = torch.exp(-_batched_logprob(next_logits, 0, greedy_onset))
        ppl_dur = torch.exp(-_batched_logprob(greedy_dur_logits, 1, greedy_dur))
        ppl_pitch = torch.exp(-_batched_logprob(greedy_pitch_logits, 2, greedy_pitch))
        ent_onset = _batched_entropy(next_logits, 0)
        ent_dur = _batched_entropy(greedy_dur_logits, 1)
        ent_pitch = _batched_entropy(greedy_pitch_logits, 2)

        go = greedy_onset.tolist()
        gd = greedy_dur.tolist()
        gp = greedy_pitch.tolist()
        ppl_o = ppl_onset.tolist()
        ppl_d = ppl_dur.tolist()
        ppl_p = ppl_pitch.tolist()
        e_o = ent_onset.tolist()
        e_d = ent_dur.tolist()
        e_p = ent_pitch.tolist()

        advance_onset = greedy_onset.clone()
        advance_dur = greedy_dur.clone()
        advance_pitch = greedy_pitch.clone()
        greedy_notes = []
        for i in range(batch):
            g_onset_tok, g_dur_tok, g_pitch_tok = go[i], gd[i], gp[i]
            greedy_note = {
                "t": g_onset_tok - TIME_OFFSET,
                "d": g_dur_tok - DUR_OFFSET,
                "p": (g_pitch_tok - NOTE_OFFSET) % MAX_PITCH if g_pitch_tok != REST else None,
            }
            greedy_notes.append(greedy_note)
            mark_greedy_candidates(cand_rows[i], greedy_note)
            cand_rows[i].sort(key=lambda c: -c["prob"])
            candidates_by_slot[i].append(cand_rows[i][:max_candidates])
            perplexity_by_slot[i].append({
                "time": float(ppl_o[i]),
                "dur": float(ppl_d[i]),
                "pitch": float(ppl_p[i]),
            })
            entropy_by_slot[i].append({
                "time": float(e_o[i]),
                "dur": float(e_d[i]),
                "pitch": float(e_p[i]),
            })
            if s == 0 and seed_notes[i] is not None:
                ot, dt, pt = encode_score_note(seed_notes[i])
                advance_onset[i] = ot
                advance_dur[i] = dt
                advance_pitch[i] = pt
                pred_by_slot[i].append({
                    "t": ot - TIME_OFFSET,
                    "d": dt - DUR_OFFSET,
                    "p": (pt - NOTE_OFFSET) % MAX_PITCH,
                    "seeded": True,
                })
            else:
                pred_by_slot[i].append(
                    None if greedy_note["p"] is None else greedy_note
                )

        past, _ = _batched_feed(model, advance_onset, slot_past)
        past, _ = _batched_feed(model, advance_dur, past)
        past, next_logits = _batched_feed(model, advance_pitch, past)
        control_pos = pos + 3
        for k in range(3):
            past, next_logits = _batched_feed(model, tokens_t[:, control_pos + k], past)

    return list(zip(pred_by_slot, candidates_by_slot, perplexity_by_slot, entropy_by_slot))


def _pack_rollout(pred, cands, perplexity, entropy, key_for_slot, slot_meta):
    branches = build_branches_from_slots(cands, key_for_slot=key_for_slot, slot_meta=slot_meta)
    ent = compact_entropy(entropy)
    return {
        "pred_score": pred,
        "branches": branches,
        "perplexity": compact_perplexity(perplexity),
        "entropy": ent["entropy"],
        "log_entropy": ent["log_entropy"],
    }


@torch.inference_mode()
def compute_rollout_sets_batched(model, device, windows, args):
    """``windows`` is a list of dicts with tokens/raw_notes/gt_by_slot.

    All four variants of every window are decoded in one batched pass when
    they share packed length (they do: CONTEXT_SIZE-4).
    """
    jobs_tokens = []
    jobs_seeds = []
    jobs_meta = []
    target_len = CONTEXT_SIZE - 4

    def _add(tokens, seed, eid, variant, key_for_slot, slot_meta):
        jobs_tokens.append(tokens)
        jobs_seeds.append(seed)
        jobs_meta.append((eid, variant, key_for_slot, slot_meta))

    for win in windows:
        eid = win["eid"]
        tokens = win["tokens"]
        raw_notes = win.get("raw_notes")
        gt_by_slot = win["gt_by_slot"]
        seed_note = gt_by_slot[0] if gt_by_slot else None
        filtered_key = lambda s: s
        filtered_meta = lambda s: {"gt_slot": s, "filtered_index": s}
        _add(tokens, None, eid, "filtered", filtered_key, filtered_meta)
        if seed_note is not None:
            _add(tokens, seed_note, eid, "filtered_seeded", filtered_key, filtered_meta)
        if raw_notes:
            raw_tokens = tokens_from_controls(raw_notes, target_len)
            if len(raw_tokens) != len(tokens):
                raw_tokens = (raw_tokens + [0] * len(tokens))[:len(tokens)]

            def raw_key_for_slot(s, notes=raw_notes):
                return s if s < len(notes) else None

            def raw_slot_meta(s, notes=raw_notes):
                return {
                    "gt_slot": notes[s].get("j") if s < len(notes) else None,
                    "raw_index": s if s < len(notes) else None,
                }

            _add(raw_tokens, None, eid, "raw", raw_key_for_slot, raw_slot_meta)
            if seed_note is not None:
                _add(raw_tokens, seed_note, eid, "raw_seeded", raw_key_for_slot, raw_slot_meta)

    results = batched_rollout_with_candidates(
        model, device, jobs_tokens, jobs_seeds,
        args.topk_onset, args.topk_dur, args.topk_pitch, args.max_candidates,
    )
    by_eid = {win["eid"]: {
        "filtered": None, "filtered_seeded": None, "raw": None, "raw_seeded": None,
    } for win in windows}
    for (eid, variant, key_for_slot, slot_meta), (pred, cands, ppl, ent) in zip(
        jobs_meta, results
    ):
        by_eid[eid][variant] = _pack_rollout(pred, cands, ppl, ent, key_for_slot, slot_meta)
    return by_eid


def default_roll_args(**overrides):
    ns = SimpleNamespace(
        topk_onset=5,
        topk_dur=4,
        topk_pitch=8,
        max_candidates=40,
        slot_progress=False,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns
