#!/usr/bin/env python
"""Add sequence-level perplexity + per-slot entropy to visualizer data.js.

For each of our rollouts (base + LoRA), builds the packed score/control stream
and scores it with a single teacher-forced forward pass:

  sequence_perplexity
      generated    — PPL of the rollout's ``pred_score`` tokens
      ground_truth — PPL of the aligned GT score tokens
      (exp(mean NLL) over score onset/duration/pitch; controls not counted)

  entropy / log_entropy
      Shannon entropy (nats) of the constrained predictive distribution at each
      score-token position along the *generated* walk, plus ``log(H + eps)`` for
      heatmap coloring. Parallel arrays aligned with ``pred_score`` slots.

Also scores every beam / triplet decode entry under
``beams`` / ``beams_triplet`` / ``beams_triplet_pitch_forced``, writing
``perplexity`` (per-slot exp(NLL)), ``sequence_perplexity``, and entropy onto
that entry so the visualizer can swap metrics with ``pred_score``.

Much cheaper than re-running ``precompute_visualizer.py`` (no candidate expansion,
one forward per sequence).

Example:
  python visualizer/compute_sequence_ppl.py --data visualizer/data.js
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from anticipation.config import MAX_DUR, MAX_PITCH, MAX_TIME  # noqa: E402
from anticipation.packed_sequence import (  # noqa: E402
    ALTERNATING_START,
    PREFIX_CONTROLS,
    dummy_rest_triplet,
    iter_score_slot_positions,
)
from anticipation.score_constraints import constrain_score_token_logits  # noqa: E402
from anticipation.vocab import (  # noqa: E402
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    TIME_OFFSET,
)
from evaluate_muster import load_model  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "visualizer"))
from precompute_visualizer import LORA_BASE_MODEL, load_lora_model  # noqa: E402

LOG_ENTROPY_EPS = 1e-12


def load_payload(path):
    text = Path(path).read_text(encoding="utf-8")
    prefix = "window.VISUALIZER_DATA = "
    if not text.startswith(prefix):
        raise ValueError(f"unexpected data.js format (missing '{prefix}' prefix): {path}")
    body = text[len(prefix):].rstrip()
    if body.endswith(";"):
        body = body[:-1]
    return json.loads(body), prefix


def _clamp(v, lo_hi):
    return max(0, min(int(v), lo_hi))


def encode_control_triplet(raw):
    t, d, p = raw
    return [
        ATIME_OFFSET + _clamp(t, MAX_TIME),
        ADUR_OFFSET + _clamp(d, MAX_DUR),
        ANOTE_OFFSET + (int(p) % MAX_PITCH),
    ]


def encode_score_note(note):
    return (
        TIME_OFFSET + _clamp(note["t"], MAX_TIME),
        DUR_OFFSET + _clamp(note["d"], MAX_DUR),
        NOTE_OFFSET + (int(note["p"]) % MAX_PITCH),
    )


def build_packed_tokens(control_notes, notes_by_slot):
    """Packed alternating sequence: score slots from ``notes_by_slot``, controls
    from ``control_notes``. Missing score notes become dummy rests (not scored)."""
    k = PREFIX_CONTROLS
    controls = [(int(n["t"]), int(n["d"]), int(n["p"])) for n in control_notes]
    n_slots = len(notes_by_slot)
    num_slots = min(n_slots, max(0, len(controls) - k))
    packed = []
    scored_positions = []  # (slot_index, onset_pos, dur_pos, pitch_pos) for real notes

    for i in range(k):
        packed.extend(encode_control_triplet(controls[i]))
        packed.extend(dummy_rest_triplet(0))

    for s in range(num_slots):
        note = notes_by_slot[s]
        onset_pos = len(packed)
        if note is not None and note.get("p") is not None:
            packed.extend(encode_score_note(note))
            scored_positions.append((s, onset_pos, onset_pos + 1, onset_pos + 2))
        else:
            packed.extend(dummy_rest_triplet(0))
        packed.extend(encode_control_triplet(controls[s + k]))

    return packed, scored_positions, num_slots


def gt_notes_for_variant(ex, variant, n_slots):
    gt = ex.get("gt_score") or []
    if variant.startswith("raw"):
        raw = ex.get("raw_notes") or []
        out = []
        for s in range(n_slots):
            if s >= len(raw):
                out.append(None)
                continue
            j = raw[s].get("j")
            out.append(gt[j] if j is not None and j < len(gt) and gt[j] else None)
        return out
    return [gt[s] if s < len(gt) and gt[s] else None for s in range(n_slots)]


def control_notes_for_variant(ex, variant):
    if variant.startswith("raw") and ex.get("raw_notes"):
        return ex["raw_notes"]
    return ex.get("perf_notes") or []


def _token_nll_and_entropy(logits_row, slot, token):
    """NLL of ``token`` and Shannon entropy (nats) of the constrained distribution."""
    constrained = constrain_score_token_logits(logits_row.float(), slot)
    log_probs = F.log_softmax(constrained, dim=-1)
    probs = log_probs.exp()
    term = probs * log_probs
    term = torch.where(probs > 0, term, torch.zeros_like(term))
    entropy = float((-term.sum()).item())
    nll = -float(log_probs[int(token)].item())
    return nll, entropy


@torch.inference_mode()
def score_packed_sequence(
    model,
    device,
    tokens,
    scored_positions,
    n_slots,
    with_entropy=False,
    with_perplexity=False,
):
    """One forward pass over ``tokens``; score listed score-token positions.

    HuggingFace causal LM: ``logits[t]`` predicts ``tokens[t+1]``, so the
    distribution for a token at position ``pos`` is ``logits[pos - 1]``.

    When ``with_perplexity`` is set, also returns parallel ``perplexity`` arrays
    (``exp(NLL)`` per onset/duration/pitch slot) aligned with score slots.
    """
    if len(tokens) < 2 or not scored_positions:
        empty = {
            "ppl": None, "mean_nll": None, "n_tokens": 0, "n_notes": 0,
        }
        if with_entropy:
            empty["entropy"] = {
                "time": [None] * n_slots,
                "dur": [None] * n_slots,
                "pitch": [None] * n_slots,
            }
            empty["log_entropy"] = {
                "time": [None] * n_slots,
                "dur": [None] * n_slots,
                "pitch": [None] * n_slots,
                "triplet": [None] * n_slots,
            }
        if with_perplexity:
            empty["perplexity"] = {
                "time": [None] * n_slots,
                "dur": [None] * n_slots,
                "pitch": [None] * n_slots,
            }
        return empty

    input_ids = torch.tensor([tokens], device=device)
    logits = model(input_ids, use_cache=False).logits[0]  # (T, V)

    nlls = []
    ent_time = [None] * n_slots
    ent_dur = [None] * n_slots
    ent_pitch = [None] * n_slots
    log_time = [None] * n_slots
    log_dur = [None] * n_slots
    log_pitch = [None] * n_slots
    log_trip = [None] * n_slots
    ppl_time = [None] * n_slots
    ppl_dur = [None] * n_slots
    ppl_pitch = [None] * n_slots

    for s, onset_pos, dur_pos, pitch_pos in scored_positions:
        if onset_pos == 0:
            continue
        nll_t, h_t = _token_nll_and_entropy(logits[onset_pos - 1], 0, tokens[onset_pos])
        nll_d, h_d = _token_nll_and_entropy(logits[dur_pos - 1], 1, tokens[dur_pos])
        nll_p, h_p = _token_nll_and_entropy(logits[pitch_pos - 1], 2, tokens[pitch_pos])
        nlls.extend([nll_t, nll_d, nll_p])
        if with_entropy:
            ent_time[s] = h_t
            ent_dur[s] = h_d
            ent_pitch[s] = h_p
            log_time[s] = math.log(h_t + LOG_ENTROPY_EPS)
            log_dur[s] = math.log(h_d + LOG_ENTROPY_EPS)
            log_pitch[s] = math.log(h_p + LOG_ENTROPY_EPS)
            log_trip[s] = math.log(h_t + h_d + h_p + LOG_ENTROPY_EPS)
        if with_perplexity:
            ppl_time[s] = math.exp(nll_t)
            ppl_dur[s] = math.exp(nll_d)
            ppl_pitch[s] = math.exp(nll_p)

    if not nlls:
        result = {"ppl": None, "mean_nll": None, "n_tokens": 0, "n_notes": 0}
    else:
        mean_nll = sum(nlls) / len(nlls)
        result = {
            "ppl": math.exp(mean_nll),
            "mean_nll": mean_nll,
            "n_tokens": len(nlls),
            "n_notes": len(scored_positions),
        }
    if with_entropy:
        result["entropy"] = {"time": ent_time, "dur": ent_dur, "pitch": ent_pitch}
        result["log_entropy"] = {
            "time": log_time, "dur": log_dur, "pitch": log_pitch, "triplet": log_trip,
        }
    if with_perplexity:
        result["perplexity"] = {"time": ppl_time, "dur": ppl_dur, "pitch": ppl_pitch}
    return result


def notes_from_pred(pred):
    n_slots = len(pred or [])
    return [
        (pred[s] if s < len(pred) and pred[s] and pred[s].get("p") is not None else None)
        for s in range(n_slots)
    ]


def apply_gen_metrics(target, gen, gt_res, *, write_entropy=True, write_perplexity=False):
    """Write sequence / slot metrics from a scored generated walk onto ``target``."""
    pair = summarize_pair(gen, gt_res)
    target["sequence_perplexity"] = pair
    if write_entropy and gen.get("entropy") is not None:
        target["entropy"] = gen["entropy"]
        target["log_entropy"] = gen["log_entropy"]
    if write_perplexity and gen.get("perplexity") is not None:
        target["perplexity"] = gen["perplexity"]
    return pair


def summarize_pair(gen, gt):
    if not gen or gen.get("ppl") is None or not gt or gt.get("ppl") is None:
        return None
    return {
        "generated": gen["ppl"],
        "ground_truth": gt["ppl"],
        "gt_higher": gt["ppl"] > gen["ppl"],
        "ratio_gt_over_gen": (gt["ppl"] / gen["ppl"]) if gen["ppl"] > 0 else None,
        "n_tokens_generated": gen["n_tokens"],
        "n_tokens_ground_truth": gt["n_tokens"],
        "n_notes_generated": gen["n_notes"],
        "n_notes_ground_truth": gt["n_notes"],
    }


ROLL_METRIC_KEYS = ("sequence_perplexity", "entropy", "log_entropy")
BEAM_METRIC_KEYS = ("sequence_perplexity", "entropy", "log_entropy", "perplexity")
BEAM_FIELDS = ("beams", "beams_triplet", "beams_triplet_pitch_forced")


def extract_metrics_patch(ex):
    """Keep only score metrics (not pred_score) for merging shard outputs."""
    patch = {}
    for group_name in ("rollouts", "rollouts_lora"):
        block = ex.get(group_name)
        if not isinstance(block, dict):
            continue
        group_patch = {}
        for variant, roll in block.items():
            if not isinstance(roll, dict):
                continue
            roll_patch = {
                k: roll[k] for k in ROLL_METRIC_KEYS if k in roll and roll[k] is not None
            }
            for field in BEAM_FIELDS:
                beams = roll.get(field)
                if not isinstance(beams, dict):
                    continue
                beams_patch = {}
                for width, entry in beams.items():
                    if not isinstance(entry, dict):
                        continue
                    entry_patch = {
                        k: entry[k]
                        for k in BEAM_METRIC_KEYS
                        if k in entry and entry[k] is not None
                    }
                    if entry_patch:
                        beams_patch[str(width)] = entry_patch
                if beams_patch:
                    roll_patch[field] = beams_patch
            if roll_patch:
                group_patch[variant] = roll_patch
        if group_patch:
            patch[group_name] = group_patch
    return patch


def merge_metrics_into_roll(roll, patch):
    for k in ROLL_METRIC_KEYS:
        if k in patch:
            roll[k] = patch[k]
    for field in BEAM_FIELDS:
        if field not in patch:
            continue
        beams = roll.setdefault(field, {})
        for width, entry_patch in patch[field].items():
            entry = beams.setdefault(str(width), {})
            entry.update(entry_patch)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--checkpoint", default=None,
                    help="Override base checkpoint (default: data.js metadata).")
    ap.add_argument("--lora-checkpoint", default=None,
                    help="Override LoRA checkpoint (default: data.js metadata).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--shard-index", type=int, default=None,
        help="0-based shard index; with --num-shards, score only this slice of windows.",
    )
    ap.add_argument(
        "--num-shards", type=int, default=None,
        help="Total shard count (requires --shard-index).",
    )
    ap.add_argument(
        "--output", default=None,
        help="Write path. For shards use a .json patch; default rewrites --data.",
    )
    args = ap.parse_args()

    if (args.shard_index is None) ^ (args.num_shards is None):
        raise SystemExit("pass both --shard-index and --num-shards, or neither")
    if args.num_shards is not None and (
        args.num_shards < 1 or args.shard_index < 0 or args.shard_index >= args.num_shards
    ):
        raise SystemExit("invalid --shard-index / --num-shards")

    payload, prefix = load_payload(args.data)
    examples = payload["examples"]
    ckpt = args.checkpoint or payload.get("checkpoint")
    lora_ckpt = args.lora_checkpoint or payload.get("lora_checkpoint")
    if not ckpt:
        raise SystemExit("no checkpoint in data.js metadata; pass --checkpoint")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Loading base model {ckpt} on {device}...")
    model, loaded_device = load_model(ckpt, config_source=None)
    if args.device is None:
        device = loaded_device if isinstance(loaded_device, torch.device) else torch.device(loaded_device)
    model.to(device)
    model.eval()

    lora_model = None
    needs_lora = any(ex.get("rollouts_lora") for ex in examples.values())
    if needs_lora:
        if not lora_ckpt:
            print("WARNING: rollouts_lora present but no lora_checkpoint; skipping LoRA")
        else:
            print(f"Loading LoRA model {lora_ckpt} (base {LORA_BASE_MODEL})...")
            lora_model = load_lora_model(lora_ckpt)
            lora_model.to(device)
            lora_model.eval()

    groups = [("rollouts", model)]
    if lora_model is not None:
        groups.append(("rollouts_lora", lora_model))

    n_scored = 0
    n_beams = 0
    gt_higher = 0
    gt_lower = 0
    ratios = []

    order = list(payload.get("example_order") or list(examples))
    if args.num_shards is not None:
        order = order[args.shard_index :: args.num_shards]
        print(
            f"Shard {args.shard_index}/{args.num_shards}: "
            f"{len(order)} window(s) {order}"
        )

    for key in tqdm(order, desc="windows"):
        ex = examples[key]
        for group_name, mdl in groups:
            block = ex.get(group_name)
            if not isinstance(block, dict):
                continue
            for variant, roll in block.items():
                if not isinstance(roll, dict) or "pred_score" not in roll:
                    continue
                controls = control_notes_for_variant(ex, variant)
                pred = roll.get("pred_score") or []
                n_slots = len(pred)
                gen_notes = notes_from_pred(pred)
                gt_notes = gt_notes_for_variant(ex, variant, n_slots)

                gen_tokens, gen_pos, _ = build_packed_tokens(controls, gen_notes)
                gt_tokens, gt_pos, _ = build_packed_tokens(controls, gt_notes)

                # Greedy / top-level rollout: keep AR ``perplexity`` from
                # precompute_visualizer; only refresh sequence PPL + entropy.
                gen = score_packed_sequence(
                    mdl, device, gen_tokens, gen_pos, n_slots, with_entropy=True,
                )
                gt_res = score_packed_sequence(
                    mdl, device, gt_tokens, gt_pos, n_slots, with_entropy=False,
                )
                pair = apply_gen_metrics(roll, gen, gt_res, write_entropy=True)
                if pair:
                    n_scored += 1
                    if pair["gt_higher"]:
                        gt_higher += 1
                    else:
                        gt_lower += 1
                    if pair["ratio_gt_over_gen"] is not None:
                        ratios.append(pair["ratio_gt_over_gen"])

                # Beam (and triplet) entries: score the notes shown for that width
                # so the viz can swap perplexity with pred_score.
                for field in BEAM_FIELDS:
                    beams = roll.get(field)
                    if not isinstance(beams, dict):
                        continue
                    for width, entry in beams.items():
                        if not isinstance(entry, dict):
                            continue
                        beam_pred = entry.get("pred_score")
                        if not isinstance(beam_pred, list):
                            continue
                        beam_notes = notes_from_pred(beam_pred)
                        beam_n = len(beam_pred)
                        beam_tokens, beam_pos, _ = build_packed_tokens(
                            controls, beam_notes,
                        )
                        # GT walk is slot-aligned to the rollout; reuse when lengths match.
                        if beam_n == n_slots:
                            beam_gt = gt_res
                        else:
                            gt_notes_b = gt_notes_for_variant(ex, variant, beam_n)
                            gt_tok_b, gt_pos_b, _ = build_packed_tokens(
                                controls, gt_notes_b,
                            )
                            beam_gt = score_packed_sequence(
                                mdl, device, gt_tok_b, gt_pos_b, beam_n,
                            )
                        beam_gen = score_packed_sequence(
                            mdl, device, beam_tokens, beam_pos, beam_n,
                            with_entropy=True,
                            with_perplexity=True,
                        )
                        apply_gen_metrics(
                            entry, beam_gen, beam_gt,
                            write_entropy=True,
                            write_perplexity=True,
                        )
                        n_beams += 1

    print(f"\nScored {n_scored} rollout sequence pairs.")
    print(f"Scored {n_beams} beam/triplet decode entries (with per-slot perplexity).")
    if ratios:
        ratios_sorted = sorted(ratios)
        mean_r = sum(ratios) / len(ratios)
        med_r = ratios_sorted[len(ratios) // 2]
        print(f"GT PPL > gen PPL in {gt_higher}/{n_scored} "
              f"({100 * gt_higher / n_scored:.1f}%)")
        print(f"GT PPL < gen PPL in {gt_lower}/{n_scored} "
              f"({100 * gt_lower / n_scored:.1f}%)")
        print(f"ratio GT/gen: mean={mean_r:.3f} median={med_r:.3f}")

    if args.dry_run:
        print("dry-run: not writing")
        return

    out = Path(args.output) if args.output else Path(args.data)
    if out.suffix == ".json" or args.num_shards is not None:
        # Shard / patch mode: only metrics for scored windows (safe parallel merge).
        patch = {"examples": {}}
        for key in order:
            metrics = extract_metrics_patch(examples[key])
            if metrics:
                patch["examples"][key] = metrics
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(patch, fh)
        print(f"Wrote metrics patch for {len(patch['examples'])} window(s) -> {out}")
    else:
        with out.open("w", encoding="utf-8") as fh:
            fh.write(prefix)
            json.dump(payload, fh)
            fh.write(";\n")
        print(f"Wrote sequence_perplexity + entropy (+ beam perplexity) into {out}")


if __name__ == "__main__":
    main()
