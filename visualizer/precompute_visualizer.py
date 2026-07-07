"""
Precompute piano-roll + logit data for the HTML visualizer (format 3).

For each sampled validation window this stores:
  - the filtered performance (control) notes exactly as the model sees them
  - the UNFILTERED raw performance stream (recovered from the aligned-stream
    cache by locating the window's source piece): every performance note the
    performer actually played in the window's span, including notes the aligner
    could not match to any score note (performer mistakes / extra notes)
  - the ground-truth score notes
  - TWO autoregressive greedy rollouts with per-slot candidate capture:
      * ``rollouts.filtered`` — conditioning on the filtered packed controls
      * ``rollouts.raw`` — conditioning on the literal raw performance stream
        (mistakes included); each raw perf note r maps to score slot r
  - candidates are captured DURING each rollout (model's own prior score
    mistakes), not from a teacher-forced ground-truth walk
  - per-slot score-token perplexity (exp(-log p) for greedy onset/duration/pitch)

Candidates are a top-A onsets x top-B durations x top-C pitches expansion of
the constrained logits at the slot's rollout state (probabilities from softmax
over constrained logits; joint = product), trimmed to the most probable
``--max-candidates``. The greedy rollout path is flagged with ``greedy: true``
(note: the greedy stepwise path is not always the argmax joint candidate).

Output is a single ``data.js`` (``window.VISUALIZER_DATA = {...}``) so
``visualizer.html`` works from file:// with no server.

Run:
  python visualizer/precompute_visualizer.py \
      --checkpoint run_nodummy_v2/checkpoint-15000 \
      --test-file data/test_normalized.txt \
      --num-examples 8 --output visualizer/data.js
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from anticipation.config import MAX_PITCH, MAX_TIME, MAX_DUR, TIME_RESOLUTION  # noqa: E402
from anticipation.packed_sequence import (  # noqa: E402
    ALTERNATING_START,
    PREFIX_CONTROLS,
    control_triplet_to_raw,
    dummy_rest_triplet,
    is_control_triplet,
    iter_score_slot_positions,
    triplet_values,
)
from anticipation.score_constraints import constrain_score_token_logits  # noqa: E402
from anticipation.vocab import (  # noqa: E402
    ATIME_OFFSET,
    ADUR_OFFSET,
    ANOTE_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    REST,
    TIME_OFFSET,
)
from evaluate_muster import load_model  # noqa: E402
from inference import parse_sequence, sample_test_lines  # noqa: E402

STREAM_CACHE_DIR = REPO_ROOT / "data" / "asap_aligned_stream_cache"


def to_legacy_past(past):
    """Return past_key_values as legacy tuples so a snapshot can be re-fed to
    multiple branch continuations without in-place Cache mutation."""
    if past is None or isinstance(past, tuple):
        return past
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()
    return past


def score_triplet_to_note(triplet) -> dict | None:
    t0, t1, t2 = int(triplet[0]), int(triplet[1]), int(triplet[2])
    if t2 == REST:
        return None
    return {
        "t": t0 - TIME_OFFSET,
        "d": t1 - DUR_OFFSET,
        "p": (t2 - NOTE_OFFSET) % MAX_PITCH,
    }


# ---------------------------------------------------------------------------
# Unfiltered stream recovery
# ---------------------------------------------------------------------------

def _clamp(value, max_value):
    return max(0, min(max_value - 1, int(value)))


def _load_cache_pieces():
    """Load every cached aligned stream as (piece_path, items) with raw control
    tuples. Returns list of dicts with:
      raw:      [(t, d, p)] for ALL performance notes (unfiltered)
      matched:  [bool] per raw note (True if aligned to a score note)
      filtered_to_raw: raw index of each filtered (matched) note
    """
    pieces = []
    for path in sorted(glob.glob(str(STREAM_CACHE_DIR / "*.json"))):
        try:
            payload = json.load(open(path))
        except Exception:
            continue
        piece_id = payload.get("fingerprint", {}).get("perf_midi", {}).get("path", path)
        raw, matched = [], []
        for item in payload.get("items", []):
            c = item["control"]
            raw.append((c[0] - ATIME_OFFSET, c[1] - ADUR_OFFSET, c[2] - ANOTE_OFFSET))
            matched.append(item.get("score") is not None)
        filtered_to_raw = [i for i, m in enumerate(matched) if m]
        # pitch bytes of the FILTERED stream, for fast subsequence search
        pitch_bytes = bytes((raw[i][2] % 256) for i in filtered_to_raw)
        pieces.append({
            "piece_id": piece_id,
            "raw": raw,
            "matched": matched,
            "filtered_to_raw": filtered_to_raw,
            "pitch_bytes": pitch_bytes,
        })
    return pieces


def _window_matches(piece, start, window_controls):
    """Exact check: does the piece's filtered stream at `start` reproduce the
    window's encoded control triplets (same re-anchoring + clamping as the
    tokenizer)?"""
    ftr = piece["filtered_to_raw"]
    if start + len(window_controls) > len(ftr):
        return False
    anchor = piece["raw"][ftr[start]][0]
    for i, (wt, wd, wp) in enumerate(window_controls):
        t, d, p = piece["raw"][ftr[start + i]]
        if p != wp:
            return False
        if _clamp(t - anchor, MAX_TIME) != wt:
            return False
        if _clamp(d, MAX_DUR) != wd:
            return False
    return True


def locate_window(pieces, window_controls):
    """Find (piece, start_index_in_filtered_stream) for a test window.

    window_controls: [(t, d, p)] as decoded from the packed sequence (window-
    anchored). Uses a fast pitch-bytes prefilter, then exact verification.
    """
    probe = bytes((p % 256) for (_, _, p) in window_controls[:16])
    for piece in pieces:
        pos = piece["pitch_bytes"].find(probe)
        while pos != -1:
            if _window_matches(piece, pos, window_controls):
                return piece, pos
            pos = piece["pitch_bytes"].find(probe, pos + 1)
    return None, None


def build_raw_window(piece, start, num_notes):
    """Unfiltered display stream: all raw performance notes from the first to
    the last matched note of the window, times re-anchored to the window.

    Each note: {t, d, p, j} where j = the note's filtered index within the
    window (0..num_notes-1) for matched notes, or null for unmatched notes
    (performer mistakes / extra notes the aligner dropped).
    """
    ftr = piece["filtered_to_raw"]
    lo = ftr[start]
    hi = ftr[start + num_notes - 1]
    anchor = piece["raw"][lo][0]
    raw_to_filtered = {
        ftr[fidx]: fidx - start
        for fidx in range(start, start + num_notes)
    }
    raw_notes = []
    for r, ri in enumerate(range(lo, hi + 1)):
        t, d, p = piece["raw"][ri]
        note = {"t": int(t - anchor), "d": int(d), "p": int(p % MAX_PITCH), "r": r}
        note["j"] = raw_to_filtered.get(ri)
        raw_notes.append(note)
    return raw_notes


# ---------------------------------------------------------------------------
# Autoregressive rollout with per-slot candidate capture
# ---------------------------------------------------------------------------

def encode_control_triplet(raw):
    t, d, p = raw
    return [
        ATIME_OFFSET + _clamp(t, MAX_TIME),
        ADUR_OFFSET + _clamp(d, MAX_DUR),
        ANOTE_OFFSET + (int(p) % MAX_PITCH),
    ]


def tokens_from_controls(control_notes, target_len):
    """Build a packed-format context from a performance/control stream only.

    Score slots are placeholders because ``rollout_with_candidates`` ignores
    ground-truth score tokens; it generates each score triplet, then feeds the
    following control triplet. This lets us compare filtered vs raw controls
    without requiring score alignments for raw unmatched notes.
    """
    k = PREFIX_CONTROLS
    max_slots = max(0, (target_len - ALTERNATING_START) // 6)
    controls = [
        (int(n["t"]), int(n["d"]), int(n["p"]))
        if isinstance(n, dict) else (int(n[0]), int(n[1]), int(n[2]))
        for n in control_notes
    ]
    num_slots = min(max_slots, max(0, len(controls) - k))

    packed = []
    for i in range(k):
        packed.extend(encode_control_triplet(controls[i]))
        packed.extend(dummy_rest_triplet(0))
    for s in range(num_slots):
        packed.extend(dummy_rest_triplet(0))
        packed.extend(encode_control_triplet(controls[s + k]))
    return packed[:target_len]


def raw_perf_index_to_slot(r: int) -> int:
    """Map a raw performance note index to its score-slot index."""
    return r


def build_branches_from_slots(
    candidates_by_slot,
    key_for_slot,
    slot_meta=None,
) -> dict[str, dict]:
    branches = {}
    for s, cands in enumerate(candidates_by_slot):
        if not cands:
            continue
        key = key_for_slot(s)
        if key is None:
            continue
        entry = {"slot": s, "candidates": cands}
        if slot_meta:
            entry.update(slot_meta(s))
        branches[str(key)] = entry
    return branches


def mark_greedy_candidates(candidates, greedy_note):
    greedy_marked = False
    for c in candidates:
        is_greedy = (
            not greedy_marked
            and c["t"] == greedy_note["t"]
            and c["d"] == greedy_note["d"]
            and (c["p"] if not c["rest"] else None) == greedy_note["p"]
        )
        c["greedy"] = is_greedy
        if is_greedy:
            greedy_marked = True


@torch.inference_mode()
def rollout_with_candidates(
    model,
    device,
    tokens,
    topk_onset,
    topk_dur,
    topk_pitch,
    max_candidates,
    slot_progress=False,
):
    """Greedy AR rollout (teacher-forced ground-truth controls) that also
    captures, at every body score slot, the candidate expansion from the
    ROLLOUT state -- i.e. conditioned on the model's own generated score notes
    so far, not the ground truth.

    Returns (pred_by_slot, candidates_by_slot, perplexity_by_slot).
    """

    def token_logprob(logits, slot, token):
        constrained = constrain_score_token_logits(logits.float(), slot)
        log_probs = F.log_softmax(constrained, dim=-1)
        return float(log_probs[int(token)].item())

    def feed(token, past):
        out = model(
            torch.tensor([[int(token)]], device=device),
            past_key_values=past,
            use_cache=True,
        )
        return to_legacy_past(out.past_key_values), out.logits[0, -1, :]

    def top_tokens(logits, slot, k):
        constrained = constrain_score_token_logits(logits.float(), slot)
        probs = F.softmax(constrained, dim=-1)
        values, indices = torch.topk(probs, k)
        return [
            (int(tok), float(pr))
            for tok, pr in zip(indices.tolist(), values.tolist())
            if pr > 0
        ]

    prime = model(
        torch.tensor([tokens[:ALTERNATING_START]], device=device),
        use_cache=True,
    )
    past = to_legacy_past(prime.past_key_values)
    next_logits = prime.logits[0, -1, :]

    slot_positions = list(iter_score_slot_positions(len(tokens), ALTERNATING_START))
    pred_by_slot = []
    candidates_by_slot = []
    perplexity_by_slot = []

    slot_iter = enumerate(slot_positions)
    if slot_progress:
        slot_iter = enumerate(
            tqdm(
                slot_positions,
                desc="score slots",
                unit="slot",
                leave=False,
                mininterval=2.0,
                file=sys.stdout,
            )
        )

    for s, pos in slot_iter:
        if pos + 5 >= len(tokens):
            pred_by_slot.append(None)
            candidates_by_slot.append([])
            perplexity_by_slot.append(None)
            continue

        # --- expand candidates from the current ROLLOUT state ---
        candidates = []
        greedy_state = None  # (past_after_onset_dur, pitch_logits, dur_logits)
        onsets = top_tokens(next_logits, 0, topk_onset)
        for oi, (onset_tok, onset_p) in enumerate(onsets):
            past_o, dur_logits = feed(onset_tok, past)
            durs = top_tokens(dur_logits, 1, topk_dur)
            for di, (dur_tok, dur_p) in enumerate(durs):
                past_od, pitch_logits = feed(dur_tok, past_o)
                if oi == 0 and di == 0:
                    greedy_state = (past_od, pitch_logits, dur_logits)
                for pitch_tok, pitch_p in top_tokens(pitch_logits, 2, topk_pitch):
                    candidates.append({
                        "t": onset_tok - TIME_OFFSET,
                        "d": dur_tok - DUR_OFFSET,
                        "p": (pitch_tok - NOTE_OFFSET) % MAX_PITCH if pitch_tok != REST else None,
                        "rest": pitch_tok == REST,
                        "prob": onset_p * dur_p * pitch_p,
                        "probs": [onset_p, dur_p, pitch_p],
                    })

        # --- greedy continuation: top onset -> top dur -> top pitch ---
        if not candidates or greedy_state is None:
            pred_by_slot.append(None)
            candidates_by_slot.append([])
            perplexity_by_slot.append(None)
            continue

        g_onset_tok, _ = onsets[0]
        past_od, pitch_logits, dur_logits = greedy_state
        g_pitches = top_tokens(pitch_logits, 2, 1)
        g_pitch_tok, _ = g_pitches[0]
        # dur token of the greedy path is the rank-0 dur under the rank-0 onset
        g_dur_tok = candidates[0]["d"] + DUR_OFFSET
        # (candidates[0] is rank-0 onset x rank-0 dur x rank-0 pitch by
        # construction of the nested loops before sorting)
        greedy_note = {
            "t": g_onset_tok - TIME_OFFSET,
            "d": g_dur_tok - DUR_OFFSET,
            "p": (g_pitch_tok - NOTE_OFFSET) % MAX_PITCH if g_pitch_tok != REST else None,
        }
        pred_by_slot.append(None if greedy_note["p"] is None else greedy_note)

        mark_greedy_candidates(candidates, greedy_note)
        candidates.sort(key=lambda c: -c["prob"])
        candidates_by_slot.append(candidates[:max_candidates])
        perplexity_by_slot.append({
            "time": math.exp(-token_logprob(next_logits, 0, g_onset_tok)),
            "dur": math.exp(-token_logprob(dur_logits, 1, g_dur_tok)),
            "pitch": math.exp(-token_logprob(pitch_logits, 2, g_pitch_tok)),
        })

        # --- advance the rollout: feed greedy pitch, then GT control triplet ---
        past, next_logits = feed(g_pitch_tok, past_od)
        control_pos = pos + 3
        for k in range(3):
            past, next_logits = feed(tokens[control_pos + k], past)

    return pred_by_slot, candidates_by_slot, perplexity_by_slot


def compact_perplexity(perplexity_by_slot):
    """Parallel arrays aligned with score slots (null where rollout skipped)."""
    return {
        "time": [None if p is None else p["time"] for p in perplexity_by_slot],
        "dur": [None if p is None else p["dur"] for p in perplexity_by_slot],
        "pitch": [None if p is None else p["pitch"] for p in perplexity_by_slot],
    }


@torch.inference_mode()
def compute_example(model, device, tokens, pieces, args):
    k = PREFIX_CONTROLS

    # filtered performance notes in window order (= model's control stream)
    control_positions = list(range(0, ALTERNATING_START, 6))
    for pos in iter_score_slot_positions(len(tokens), ALTERNATING_START):
        cpos = pos + 3
        if cpos + 2 < len(tokens) and is_control_triplet(tokens, cpos):
            control_positions.append(cpos)
    window_controls = [
        tuple(control_triplet_to_raw(triplet_values(tokens, pos)))
        for pos in control_positions
    ]
    perf_notes = [{"t": t, "d": d, "p": p % MAX_PITCH} for (t, d, p) in window_controls]

    # unfiltered stream from the source piece
    piece, start = locate_window(pieces, window_controls)
    if piece is None:
        print("    WARNING: could not locate source piece; raw stream unavailable")
        piece_id, raw_notes = None, None
    else:
        piece_id = piece["piece_id"].split("asap-dataset-master/")[-1]
        raw_notes = build_raw_window(piece, start, len(window_controls))
        n_extra = sum(1 for n in raw_notes if n["j"] is None)
        print(f"    source: {piece_id} @ filtered idx {start} "
              f"({len(raw_notes)} raw notes, {n_extra} unmatched)")

    gt_by_slot = [
        score_triplet_to_note(triplet_values(tokens, pos))
        for pos in iter_score_slot_positions(len(tokens), ALTERNATING_START)
    ]

    filtered_pred_by_slot, filtered_candidates_by_slot, filtered_perplexity = (
        rollout_with_candidates(
            model, device, tokens,
            args.topk_onset, args.topk_dur, args.topk_pitch, args.max_candidates,
            slot_progress=args.slot_progress,
        )
    )

    filtered_branches = build_branches_from_slots(
        filtered_candidates_by_slot,
        key_for_slot=lambda s: s,
        slot_meta=lambda s: {"gt_slot": s, "filtered_index": s},
    )
    filtered_rollout = {
        "pred_score": filtered_pred_by_slot,
        "branches": filtered_branches,
        "perplexity": compact_perplexity(filtered_perplexity),
    }

    raw_rollout = None
    if raw_notes:
        raw_tokens = tokens_from_controls(raw_notes, len(tokens))
        raw_pred_by_slot, raw_candidates_by_slot, raw_perplexity = rollout_with_candidates(
            model, device, raw_tokens,
            args.topk_onset, args.topk_dur, args.topk_pitch, args.max_candidates,
            slot_progress=args.slot_progress,
        )

        def raw_key_for_slot(s):
            return s if s < len(raw_notes) else None

        raw_branches = build_branches_from_slots(
            raw_candidates_by_slot,
            key_for_slot=raw_key_for_slot,
            slot_meta=lambda s: {
                "gt_slot": raw_notes[s].get("j") if s < len(raw_notes) else None,
                "raw_index": s if s < len(raw_notes) else None,
            },
        )
        raw_rollout = {
            "pred_score": raw_pred_by_slot,
            "branches": raw_branches,
            "perplexity": compact_perplexity(raw_perplexity),
        }

    return {
        "piece": piece_id,
        "prefix_controls": k,
        "perf_notes": perf_notes,
        "raw_notes": raw_notes,
        "gt_score": gt_by_slot,
        # Legacy aliases for old HTML; the UI now uses ``rollouts``.
        "pred_score": filtered_pred_by_slot,
        "branches": filtered_branches,
        "rollouts": {
            "filtered": filtered_rollout,
            "raw": raw_rollout,
        },
        "time_resolution": TIME_RESOLUTION,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-source", default=None)
    parser.add_argument("--test-file", default="data/test_normalized.txt")
    parser.add_argument("--num-examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="visualizer/data.js")
    parser.add_argument("--topk-onset", type=int, default=5)
    parser.add_argument("--topk-dur", type=int, default=4)
    parser.add_argument("--topk-pitch", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument(
        "--slot-progress",
        action="store_true",
        help="show per-slot tqdm bars during each rollout (verbose SLURM logs)",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()
    print("Loading aligned-stream cache for raw-stream recovery...", flush=True)
    pieces = _load_cache_pieces()
    print(f"  {len(pieces)} cached pieces", flush=True)

    model, device = load_model(args.checkpoint, args.config_source)
    print(f"Model ready on {device} ({time.perf_counter() - t_start:.1f}s since start)", flush=True)

    sampled, total = sample_test_lines(args.test_file, args.num_examples, args.seed)
    print(f"Sampled {len(sampled)} of {total} validation windows (seed={args.seed})", flush=True)

    examples = {}
    for original_index, line in tqdm(
        sampled,
        desc="validation windows",
        unit="win",
        mininterval=5.0,
        file=sys.stdout,
    ):
        tokens = parse_sequence(line)
        if len(tokens) <= ALTERNATING_START:
            continue
        t_win = time.perf_counter()
        tqdm.write(f"  window {original_index}: dual AR rollout + perplexity...")
        example = compute_example(model, device, tokens, pieces, args)
        n_real = sum(1 for n in example["gt_score"] if n is not None)
        raw_rollout = example["rollouts"].get("raw")
        raw_slots = len(raw_rollout["branches"]) if raw_rollout else 0
        tqdm.write(
            f"    {len(example['perf_notes'])} filtered perf notes, {n_real} GT score notes, "
            f"{len(example['branches'])} filtered slots, {raw_slots} raw slots "
            f"({time.perf_counter() - t_win:.1f}s)"
        )
        examples[str(original_index)] = example

    payload = {
        "format": 3,
        "checkpoint": args.checkpoint,
        "test_file": args.test_file,
        "seed": args.seed,
        "logits_conditioning": "autoregressive_rollout",
        "examples": examples,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("window.VISUALIZER_DATA = ")
        json.dump(payload, handle)
        handle.write(";\n")
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"Wrote {output_path} and {output_path.with_suffix('.json')}", flush=True)
    print(f"Total elapsed: {time.perf_counter() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
