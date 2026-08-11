#!/usr/bin/env python
"""Precompute constrained beam-search rollouts for the visualizer (beams 1..10).

Reads an existing format-4 ``data.js``, rebuilds packed control streams from the
stored notes, and for each of our rollouts (base FT + LoRA × filtered/raw ×
plain/seeded) writes::

    rollouts*.<variant>.beams = {
        "1": {"pred_score": [...]},
        ...
        "10": {"pred_score": [...]},
    }

Beam width 1 is constrained greedy (matches the existing AR path). Wider beams
keep the top-``n`` partial hypotheses after every score onset/duration/pitch
token, then teacher-force the ground-truth control triplet (same schedule as
``inference.autoregressive_generate_score`` / ``precompute_visualizer``).

Designed to shard: ``--example-keys val-01,val-02`` writes a compact JSON shard
that ``merge_beam_shards.py`` folds back into ``data.js``.

Example:
  python visualizer/precompute_beams.py \\
      --data visualizer/data.js \\
      --example-keys val-01,val-02 \\
      --output visualizer/beam_shards/shard_0.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from anticipation.config import CONTEXT_SIZE, MAX_PITCH  # noqa: E402
from anticipation.packed_sequence import (  # noqa: E402
    ALTERNATING_START,
    iter_score_slot_positions,
)
from anticipation.score_constraints import constrain_score_token_logits  # noqa: E402
from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, REST, TIME_OFFSET  # noqa: E402
from evaluate_muster import load_model  # noqa: E402
from precompute_visualizer import (  # noqa: E402
    encode_score_note,
    load_lora_model,
    to_legacy_past,
    tokens_from_controls,
)
from compute_sequence_ppl import (  # noqa: E402
    control_notes_for_variant,
    load_payload,
)

ALL_VARIANTS = ("filtered", "filtered_seeded", "raw", "raw_seeded")


def clone_past(past):
    past = to_legacy_past(past)
    if past is None:
        return None
    return tuple(tuple(t.clone() for t in layer) for layer in past)


def note_from_tokens(onset_tok, dur_tok, pitch_tok, seeded=False):
    if pitch_tok == REST:
        return None
    note = {
        "t": int(onset_tok) - TIME_OFFSET,
        "d": int(dur_tok) - DUR_OFFSET,
        "p": (int(pitch_tok) - NOTE_OFFSET) % MAX_PITCH,
    }
    if seeded:
        note["seeded"] = True
    return note


def _stack_pasts(pasts):
    """Stack per-beam legacy pasts along the batch dimension."""
    n_layers = len(pasts[0])
    stacked = []
    for layer in range(n_layers):
        keys = torch.cat([p[layer][0] for p in pasts], dim=0)
        vals = torch.cat([p[layer][1] for p in pasts], dim=0)
        stacked.append((keys, vals))
    return tuple(stacked)


def _unstack_past(batched_past):
    batch = batched_past[0][0].shape[0]
    return [
        tuple(
            (layer[0][i : i + 1].contiguous(), layer[1][i : i + 1].contiguous())
            for layer in batched_past
        )
        for i in range(batch)
    ]


@torch.inference_mode()
def beam_search_score(model, device, tokens, num_beams: int, seed_note=None):
    """Constrained triplet beam search with teacher-forced controls.

    Beams advance with batched KV-cache forwards (batch = current beam count)
    and control triplets are fed as a length-3 sequence, so width-10 is much
    closer to width-1 cost than a sequential per-beam loop.

    Returns ``pred_by_slot`` aligned with body score slots (None where skipped).
    """
    if num_beams < 1:
        raise ValueError("num_beams must be >= 1")

    def feed_batch(token_ids, pasts):
        """Feed one token per beam. ``token_ids`` length = len(pasts)."""
        ids = torch.tensor(token_ids, device=device, dtype=torch.long).view(-1, 1)
        if len(pasts) == 1:
            out = model(ids, past_key_values=pasts[0], use_cache=True)
            new_past = to_legacy_past(out.past_key_values)
            return [new_past], out.logits[:, -1, :]
        stacked = _stack_pasts(pasts)
        out = model(ids, past_key_values=stacked, use_cache=True)
        new_past = to_legacy_past(out.past_key_values)
        return _unstack_past(new_past), out.logits[:, -1, :]

    def feed_seq_batch(seq_tokens, pasts):
        """Feed the same token sequence to every beam in one forward."""
        B = len(pasts)
        ids = torch.tensor([seq_tokens] * B, device=device, dtype=torch.long)
        if B == 1:
            out = model(ids, past_key_values=pasts[0], use_cache=True)
            new_past = to_legacy_past(out.past_key_values)
            return [new_past], out.logits[:, -1, :]
        stacked = _stack_pasts(pasts)
        out = model(ids, past_key_values=stacked, use_cache=True)
        new_past = to_legacy_past(out.past_key_values)
        return _unstack_past(new_past), out.logits[:, -1, :]

    def expand_slot(scores, pasts, logits, extras, slot):
        """Expand/prune beams for one score-token slot.

        ``extras`` is a list of tuples carried alongside each beam.
        Returns scores, pasts, logits, list of (chosen_token, parent_extra).
        """
        constrained = constrain_score_token_logits(logits.float(), slot)
        log_probs = F.log_softmax(constrained, dim=-1)
        k_eff = min(num_beams, log_probs.shape[-1])
        top_lp, top_idx = torch.topk(log_probs, k_eff, dim=-1)

        score_t = torch.tensor(scores, device=device, dtype=top_lp.dtype).unsqueeze(1)
        combined = (score_t + top_lp).view(-1)
        take = min(num_beams, combined.numel())
        best_flat_scores, best_flat_idx = torch.topk(combined, take)

        cand_scores = []
        cand_meta = []
        for fs, fi in zip(best_flat_scores.tolist(), best_flat_idx.tolist()):
            parent = int(fi) // k_eff
            j = int(fi) % k_eff
            tok = int(top_idx[parent, j].item())
            cand_scores.append(float(fs))
            cand_meta.append((parent, tok, extras[parent]))

        fanout = {}
        for parent, _, _ in cand_meta:
            fanout[parent] = fanout.get(parent, 0) + 1
        feed_pasts = []
        feed_tokens = []
        new_extras = []
        for parent, tok, extra in cand_meta:
            past = pasts[parent]
            feed_pasts.append(clone_past(past) if fanout[parent] > 1 else past)
            feed_tokens.append(tok)
            new_extras.append((tok, extra))

        new_pasts, new_logits = feed_batch(feed_tokens, feed_pasts)
        return cand_scores, new_pasts, new_logits, new_extras

    prime = model(
        torch.tensor([tokens[:ALTERNATING_START]], device=device),
        use_cache=True,
    )
    scores = [0.0]
    pasts = [to_legacy_past(prime.past_key_values)]
    logits = prime.logits[0, -1, :].unsqueeze(0)
    # extras[i] = (pred_notes, pending_onset, pending_dur)
    extras = [([], None, None)]

    for s, pos in enumerate(iter_score_slot_positions(len(tokens), ALTERNATING_START)):
        if pos + 5 >= len(tokens):
            extras = [(preds + [None], None, None) for preds, _, _ in extras]
            continue

        force_seed = s == 0 and seed_note is not None and seed_note.get("p") is not None

        if force_seed:
            onset_tok, dur_tok, pitch_tok = encode_score_note(seed_note)
            new_scores = None
            cur_logits = logits
            cur_pasts = pasts
            for slot, tok in enumerate((onset_tok, dur_tok, pitch_tok)):
                base = scores if slot == 0 else new_scores
                step_scores = []
                for i, sc in enumerate(base):
                    constrained = constrain_score_token_logits(cur_logits[i].float(), slot)
                    step_scores.append(
                        sc + float(F.log_softmax(constrained, dim=-1)[int(tok)].item())
                    )
                new_scores = step_scores
                cur_pasts, cur_logits = feed_batch([tok] * len(cur_pasts), cur_pasts)
            scores, pasts, logits = new_scores, cur_pasts, cur_logits
            note = note_from_tokens(onset_tok, dur_tok, pitch_tok, seeded=True)
            extras = [(preds + [note], None, None) for preds, _, _ in extras]
        else:
            scores, pasts, logits, packed = expand_slot(
                scores, pasts, logits,
                [(preds, None, None) for preds, _, _ in extras],
                slot=0,
            )
            mid = [(preds, onset_tok, None) for onset_tok, (preds, _, _) in packed]

            scores, pasts, logits, packed = expand_slot(
                scores, pasts, logits, mid, slot=1,
            )
            mid = [
                (preds, onset_tok, dur_tok)
                for dur_tok, (preds, onset_tok, _) in packed
            ]

            scores, pasts, logits, packed = expand_slot(
                scores, pasts, logits, mid, slot=2,
            )
            extras = [
                (preds + [note_from_tokens(onset_tok, dur_tok, pitch_tok)], None, None)
                for pitch_tok, (preds, onset_tok, dur_tok) in packed
            ]

        control_pos = pos + 3
        ctrl = [int(tokens[control_pos + k]) for k in range(3)]
        pasts, logits = feed_seq_batch(ctrl, pasts)

    best = max(range(len(scores)), key=lambda i: scores[i])
    return extras[best][0]


@torch.inference_mode()
def beam_search_triplets(
    model,
    device,
    tokens,
    num_beams: int,
    seed_note=None,
    gt_pitches_by_slot=None,
    pitch_force=False,
):
    """Beam search that expands/prunes only after a full score triplet.

    At each body slot, every live beam expands to a grid of
    top-``num_beams`` onsets × top-``num_beams`` durations × top-``num_beams``
    pitches (or a single forced pitch when ``pitch_force`` and a GT pitch exist
    for that slot). Candidates are scored by the sum of constrained log-probs
    and the global top-``num_beams`` triplets survive.

    ``gt_pitches_by_slot`` is a list aligned with score slots (None where unknown).
    When ``pitch_force`` is set but the slot has no GT pitch (e.g. unmatched raw
    control), that slot falls back to unconstrained triplet expansion.
    """
    if num_beams < 1:
        raise ValueError("num_beams must be >= 1")
    gt_pitches_by_slot = gt_pitches_by_slot or []

    def feed_batch(token_ids, pasts):
        ids = torch.tensor(token_ids, device=device, dtype=torch.long).view(-1, 1)
        if len(pasts) == 1:
            out = model(ids, past_key_values=pasts[0], use_cache=True)
            new_past = to_legacy_past(out.past_key_values)
            return [new_past], out.logits[:, -1, :]
        stacked = _stack_pasts(pasts)
        out = model(ids, past_key_values=stacked, use_cache=True)
        new_past = to_legacy_past(out.past_key_values)
        return _unstack_past(new_past), out.logits[:, -1, :]

    def feed_seq_batch(seq_tokens, pasts):
        B = len(pasts)
        ids = torch.tensor([seq_tokens] * B, device=device, dtype=torch.long)
        if B == 1:
            out = model(ids, past_key_values=pasts[0], use_cache=True)
            new_past = to_legacy_past(out.past_key_values)
            return [new_past], out.logits[:, -1, :]
        stacked = _stack_pasts(pasts)
        out = model(ids, past_key_values=stacked, use_cache=True)
        new_past = to_legacy_past(out.past_key_values)
        return _unstack_past(new_past), out.logits[:, -1, :]

    def top_tokens(logits_row, slot, k):
        constrained = constrain_score_token_logits(logits_row.float(), slot)
        log_probs = F.log_softmax(constrained, dim=-1)
        finite = torch.isfinite(log_probs)
        if not bool(finite.any()):
            return []
        k_eff = min(int(k), int(finite.sum().item()))
        values, indices = torch.topk(log_probs, k_eff)
        return [
            (int(tok), float(lp))
            for tok, lp in zip(indices.tolist(), values.tolist())
            if lp > float("-inf")
        ]

    def token_lp(logits_row, slot, tok):
        constrained = constrain_score_token_logits(logits_row.float(), slot)
        return float(F.log_softmax(constrained, dim=-1)[int(tok)].item())

    prime = model(
        torch.tensor([tokens[:ALTERNATING_START]], device=device),
        use_cache=True,
    )
    scores = [0.0]
    pasts = [to_legacy_past(prime.past_key_values)]
    logits = prime.logits[0, -1, :]  # (V,) for single beam; we'll keep a list
    logits_list = [logits]
    preds_list = [[]]

    for s, pos in enumerate(iter_score_slot_positions(len(tokens), ALTERNATING_START)):
        if pos + 5 >= len(tokens):
            preds_list = [preds + [None] for preds in preds_list]
            continue

        force_seed = s == 0 and seed_note is not None and seed_note.get("p") is not None
        forced_pitch = None
        if pitch_force and s < len(gt_pitches_by_slot):
            forced_pitch = gt_pitches_by_slot[s]

        if force_seed:
            onset_tok, dur_tok, pitch_tok = encode_score_note(seed_note)
            new_scores, new_pasts, new_logits, new_preds = [], [], [], []
            for i, (sc, past, logit, preds) in enumerate(
                zip(scores, pasts, logits_list, preds_list)
            ):
                sc = sc + token_lp(logit, 0, onset_tok)
                past, logit = feed_batch([onset_tok], [past])
                past, logit = past[0], logit[0]
                sc = sc + token_lp(logit, 1, dur_tok)
                past, logit = feed_batch([dur_tok], [past])
                past, logit = past[0], logit[0]
                sc = sc + token_lp(logit, 2, pitch_tok)
                past, logit = feed_batch([pitch_tok], [past])
                past, logit = past[0], logit[0]
                note = note_from_tokens(onset_tok, dur_tok, pitch_tok, seeded=True)
                new_scores.append(sc)
                new_pasts.append(past)
                new_logits.append(logit)
                new_preds.append(preds + [note])
            scores, pasts, logits_list, preds_list = (
                new_scores, new_pasts, new_logits, new_preds
            )
        else:
            # Collect full-triplet candidates from every parent, then prune.
            # Paths are expanded sequentially; we batch where parents share a step.
            candidates = []  # (score, onset, dur, pitch, parent_idx)
            for pi, (sc, logit) in enumerate(zip(scores, logits_list)):
                onsets = top_tokens(logit, 0, num_beams)
                if not onsets:
                    continue
                for onset_tok, lp_o in onsets:
                    # Feed onset for this single parent path
                    past_o, logit_d = feed_batch([onset_tok], [pasts[pi]])
                    past_o, logit_d = past_o[0], logit_d[0]
                    durs = top_tokens(logit_d, 1, num_beams)
                    if not durs:
                        continue
                    for dur_tok, lp_d in durs:
                        past_od, logit_p = feed_batch([dur_tok], [past_o])
                        past_od, logit_p = past_od[0], logit_p[0]
                        if forced_pitch is not None:
                            pitch_tok = NOTE_OFFSET + (int(forced_pitch) % MAX_PITCH)
                            lp_p = token_lp(logit_p, 2, pitch_tok)
                            # Skip illegal / -inf pitches
                            if lp_p > float("-inf"):
                                candidates.append(
                                    (sc + lp_o + lp_d + lp_p, onset_tok, dur_tok, pitch_tok, pi)
                                )
                        else:
                            for pitch_tok, lp_p in top_tokens(logit_p, 2, num_beams):
                                candidates.append(
                                    (sc + lp_o + lp_d + lp_p, onset_tok, dur_tok, pitch_tok, pi)
                                )

            if not candidates:
                preds_list = [preds + [None] for preds in preds_list]
            else:
                candidates.sort(key=lambda x: x[0], reverse=True)
                survivors = candidates[:num_beams]
                # Advance each survivor: re-feed onset/dur/pitch from parent past.
                # (Re-forward is simpler/safer than caching every intermediate past.)
                fanout = {}
                for *_, pi in survivors:
                    fanout[pi] = fanout.get(pi, 0) + 1
                new_scores, new_pasts, new_logits, new_preds = [], [], [], []
                for sc, onset_tok, dur_tok, pitch_tok, pi in survivors:
                    past = clone_past(pasts[pi]) if fanout[pi] > 1 else pasts[pi]
                    past, _ = feed_batch([onset_tok], [past])
                    past = past[0]
                    past, _ = feed_batch([dur_tok], [past])
                    past = past[0]
                    past, logit = feed_batch([pitch_tok], [past])
                    past, logit = past[0], logit[0]
                    note = note_from_tokens(onset_tok, dur_tok, pitch_tok)
                    new_scores.append(sc)
                    new_pasts.append(past)
                    new_logits.append(logit)
                    new_preds.append(preds_list[pi] + [note])
                scores, pasts, logits_list, preds_list = (
                    new_scores, new_pasts, new_logits, new_preds
                )

        control_pos = pos + 3
        ctrl = [int(tokens[control_pos + k]) for k in range(3)]
        pasts, logits_batch = feed_seq_batch(ctrl, pasts)
        logits_list = [logits_batch[i] for i in range(len(pasts))]

    best = max(range(len(scores)), key=lambda i: scores[i])
    return preds_list[best]


def gt_pitches_for_variant(ex, variant, n_slots):
    """Per-slot GT pitch for pitch-forced decoding, or None if unmatched/missing."""
    gt = ex.get("gt_score") or []
    out = []
    if variant.startswith("raw"):
        raw = ex.get("raw_notes") or []
        for s in range(n_slots):
            if s >= len(raw):
                out.append(None)
                continue
            j = raw[s].get("j")
            if j is None or j >= len(gt) or not gt[j]:
                out.append(None)
            else:
                out.append(gt[j].get("p"))
        return out
    for s in range(n_slots):
        if s < len(gt) and gt[s]:
            out.append(gt[s].get("p"))
        else:
            out.append(None)
    return out


def build_control_tokens(ex, variant):
    controls = control_notes_for_variant(ex, variant)
    if not controls:
        return None
    return tokens_from_controls(controls, CONTEXT_SIZE - 4)


def seed_for_variant(ex, variant, n_slots=None):
    if not variant.endswith("_seeded"):
        return None
    # Match precompute_visualizer: always seed filtered slot-0 GT (even for raw_seeded).
    gt0 = (ex.get("gt_score") or [None])[0]
    return gt0 if gt0 is not None and gt0.get("p") is not None else None


def notes_equal(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return a.get("t") == b.get("t") and a.get("d") == b.get("d") and a.get("p") == b.get("p")


def compute_beams_for_rollout(
    model,
    device,
    tokens,
    seed_note,
    beam_widths,
    desc=None,
    reuse_greedy=None,
    mode="token",
    gt_pitches=None,
):
    """Decode one rollout under ``mode`` ∈ {token, triplet, triplet_pitch_forced}.

    If ``reuse_greedy`` is the stored greedy ``pred_score``, beam width 1 for
    token mode is copied instead of re-decoded.
    """
    out = {}
    iterator = beam_widths
    if desc:
        iterator = tqdm(beam_widths, desc=desc, leave=False, file=sys.stdout)
    pitch_force = mode == "triplet_pitch_forced"
    for bw in iterator:
        if mode == "token" and int(bw) == 1 and reuse_greedy is not None:
            out["1"] = {"pred_score": reuse_greedy}
            continue
        if mode == "token":
            pred = beam_search_score(
                model, device, tokens, num_beams=int(bw), seed_note=seed_note,
            )
        else:
            pred = beam_search_triplets(
                model, device, tokens, num_beams=int(bw), seed_note=seed_note,
                gt_pitches_by_slot=gt_pitches,
                pitch_force=pitch_force,
            )
        out[str(int(bw))] = {"pred_score": pred}
    return out


def process_example(ex, models, device, beam_widths, variants, modes):
    """Return shard fragment for one window.

    Token-only mode emits the flat schema used by the visualizer UI / merge::

        {group: {variant: {"1": {pred_score}, ..., "10": {...}}}}

    When additional modes are requested, nests them under field names::

        {group: {variant: {beams: {...}, beams_triplet: {...}, ...}}}
    """
    mode_to_key = {
        "token": "beams",
        "triplet": "beams_triplet",
        "triplet_pitch_forced": "beams_triplet_pitch_forced",
    }
    flat_token = list(modes) == ["token"]
    result = {}
    for group_name, model in models:
        block = ex.get(group_name)
        if not isinstance(block, dict):
            continue
        group_out = {}
        for variant in variants:
            roll = block.get(variant)
            if not isinstance(roll, dict) or "pred_score" not in roll:
                continue
            tokens = build_control_tokens(ex, variant)
            if tokens is None:
                continue
            n_slots = len(roll["pred_score"])
            seed = seed_for_variant(ex, variant, n_slots)
            gt_pitches = gt_pitches_for_variant(ex, variant, n_slots)
            variant_out = {}
            for mode in modes:
                field = mode_to_key[mode]
                widths = list(beam_widths)
                if mode == "token":
                    existing = set((roll.get("beams") or {}).keys())
                    widths = [w for w in widths if str(int(w)) not in existing]
                    if not widths:
                        continue
                decoded = compute_beams_for_rollout(
                    model, device, tokens, seed, widths,
                    desc=f"{group_name}/{variant}/{mode}",
                    reuse_greedy=None,  # always decode beam=1 for parity checks
                    mode=mode,
                    gt_pitches=gt_pitches,
                )
                if flat_token and mode == "token":
                    variant_out = decoded
                else:
                    variant_out[field] = decoded
            if variant_out:
                group_out[variant] = variant_out
        if group_out:
            result[group_name] = group_out
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--output", required=True, help="Shard JSON path.")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--lora-checkpoint", default=None)
    ap.add_argument(
        "--example-keys", default=None,
        help="Comma-separated example_order keys (default: all).",
    )
    ap.add_argument("--beam-min", type=int, default=1)
    ap.add_argument("--beam-max", type=int, default=10)
    ap.add_argument(
        "--beam-list", default=None,
        help="Comma-separated beam widths (overrides --beam-min/--beam-max).",
    )
    ap.add_argument("--skip-lora", action="store_true")
    ap.add_argument(
        "--variants", default=",".join(ALL_VARIANTS),
        help="Comma-separated subset of variants to compute.",
    )
    ap.add_argument(
        "--modes",
        default="token,triplet,triplet_pitch_forced",
        help="Comma-separated decode modes: token, triplet, triplet_pitch_forced.",
    )
    args = ap.parse_args()

    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    allowed = {"token", "triplet", "triplet_pitch_forced"}
    bad = [m for m in modes if m not in allowed]
    if bad:
        raise SystemExit(f"unknown modes {bad}; expected subset of {sorted(allowed)}")

    t0 = time.perf_counter()
    payload, _prefix = load_payload(args.data)
    order = payload.get("example_order") or list(payload["examples"])
    keys = (
        [k.strip() for k in args.example_keys.split(",") if k.strip()]
        if args.example_keys else list(order)
    )
    missing = [k for k in keys if k not in payload["examples"]]
    if missing:
        raise SystemExit(f"unknown example keys: {missing}")

    ckpt = args.checkpoint or payload.get("checkpoint")
    lora_ckpt = args.lora_checkpoint or payload.get("lora_checkpoint")
    if not ckpt:
        raise SystemExit("no checkpoint in data.js; pass --checkpoint")

    print(f"Loading base model {ckpt}...", flush=True)
    model, device = load_model(ckpt, config_source=None)
    model.eval()
    print(f"  ready on {device} ({time.perf_counter() - t0:.1f}s)", flush=True)

    models = [("rollouts", model)]
    if not args.skip_lora and lora_ckpt:
        needs = any(payload["examples"][k].get("rollouts_lora") for k in keys)
        if needs:
            print(f"Loading LoRA {lora_ckpt}...", flush=True)
            lora_model = load_lora_model(lora_ckpt)
            lora_model.eval()
            models.append(("rollouts_lora", lora_model))
            print(f"  LoRA ready ({time.perf_counter() - t0:.1f}s)", flush=True)

    if args.beam_list:
        beam_widths = [int(x) for x in args.beam_list.split(",") if x.strip()]
    else:
        beam_widths = list(range(args.beam_min, args.beam_max + 1))
    shard = {
        "checkpoint": ckpt,
        "lora_checkpoint": lora_ckpt,
        "beam_widths": beam_widths,
        "variants": list(variants),
        "modes": list(modes),
        "examples": {},
    }

    for key in tqdm(keys, desc="windows", file=sys.stdout):
        ex = payload["examples"][key]
        t_win = time.perf_counter()
        print(f"\n=== {key} ===", flush=True)
        beams = process_example(ex, models, device, beam_widths, variants, modes)
        shard["examples"][key] = beams
        print(f"  done in {time.perf_counter() - t_win:.1f}s", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(shard, fh)
    print(f"Wrote {out} ({time.perf_counter() - t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
