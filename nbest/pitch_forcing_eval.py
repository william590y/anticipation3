#!/usr/bin/env python
"""Curiosity (2): does PITCH FORCING improve test-set F1?

Greedy windowed decode, except at each score slot's PITCH token: if the argmax
pitch differs from the GT pitch, substitute the GT pitch token and continue
decoding with the corrected context. Under the model's time->dur->pitch
factorization this IS "walk down the ranked triplet list until the pitch is
correct": time/dur are already fixed by the chain, and every pitch appears in
the ranked pitch list, so the first correct-pitch triplet is (t, d, gt_pitch).
(The stricter joint ranking would also reconsider t/d -- not what's tested.)

This is an ORACLE DIAGNOSTIC (it reads GT pitch), not a method. It answers:
how much of our F1 deficit is pitch identity vs onset placement, and does a
corrected pitch context improve SUBSEQUENT onsets?

Scores the forced rollout with the table's own metric on the same stride-150
test windows, against the stored plain-greedy baseline (candidate 0).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anticipation.score_constraints import constrain_score_token_logits  # noqa: E402
from anticipation.vocab import REST                                      # noqa: E402
from f1_reward import score_triplet_to_note                              # noqa: E402
from onpolicy_rollout import (ALTERNATING_START,                        # noqa: E402
                              body_score_slot_starts, score_token_positions)
from evaluate_muster import load_model                                   # noqa: E402


def _cf1():
    spec = importlib.util.spec_from_file_location(
        "compute_f1", ROOT / "visualizer/compute_f1.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@torch.inference_mode()
def rollout_pitch_forced(model, input_ids):
    """rollout_score_slots (greedy, constrained), plus the pitch intervention.
    Returns (rolled, n_forced_per_row, n_slots_per_row)."""
    device = input_ids.device
    length = input_ids.shape[1]
    starts = body_score_slot_starts(length)
    rolled = input_ids.clone()
    forced = torch.zeros(input_ids.shape[0], dtype=torch.long, device=device)
    nvalid = torch.zeros_like(forced)
    primed = model(input_ids[:, :ALTERNATING_START], use_cache=True)
    past, next_logits = primed.past_key_values, primed.logits[:, -1, :]
    for start in starts:
        slot_valid = input_ids[:, start + 2] != REST
        nvalid += slot_valid.long()
        sampled = []
        for role in range(3):
            logits = constrain_score_token_logits(next_logits.float(), role)
            token = logits.argmax(dim=-1)
            if role == 2:
                gt_pitch = input_ids[:, start + 2]
                need = slot_valid & (token != gt_pitch)
                forced += need.long()
                token = torch.where(need, gt_pitch, token)
            sampled.append(token)
            if role < 2:
                step = model(token.unsqueeze(1), past_key_values=past,
                             use_cache=True)
                past, next_logits = step.past_key_values, step.logits[:, -1, :]
        chunk = torch.cat([sampled[2].unsqueeze(1),
                           input_ids[:, start + 3: start + 6]], dim=1)
        step = model(chunk, past_key_values=past, use_cache=True)
        past, next_logits = step.past_key_values, step.logits[:, -1, :]
        for role in range(3):
            rolled[:, start + role] = sampled[role]
    return rolled, forced, nvalid


def notes_of(flat):
    toks = [int(t) for t in flat]
    out = []
    for k in range(len(toks) // 3):
        n = score_triplet_to_note(toks[3 * k], toks[3 * k + 1], toks[3 * k + 2])
        if n is not None:
            out.append({"t": int(n[0]), "d": int(n[1]), "p": int(n[2])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", default="nbest_data/test9_stride150.pt")
    ap.add_argument("--checkpoint", default="run_paper_split_v2/checkpoint-2500")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--payload", default=None,
                    help="piece map from this payload's `piece` fields "
                         "(use for val; default = test selector-eval json)")
    ap.add_argument("--out", default="nbest_data/pitch_forcing_eval.json")
    ap.add_argument("--save-tokens", default=None,
                    help="save the forced rollouts' score tokens (W,414) int16 "
                         "for slot-level analysis")
    a = ap.parse_args()

    cf1 = _cf1()
    d = torch.load(ROOT / a.shard, map_location="cpu", weights_only=False)
    W = d["window_tokens"].long()
    flat = score_token_positions(W.shape[1])
    row = {int(l): i for i, l in enumerate(d["window_line_idx"].tolist())}
    first_cand = {}
    for ci, l in enumerate(d["cand_line_idx"].tolist()):
        first_cand.setdefault(row[int(l)], ci)
    # Piece attribution: the test shard has the selector-eval map; other
    # shards (val) carry pieces in their payload. Fall back to window-only
    # stats if neither exists rather than mislabeling.
    pieces = {}
    if a.payload:
        t = open(ROOT / a.payload, encoding="utf-8").read()
        pj = json.loads(t[t.index("{"): t.rindex("}") + 1])["examples"]
        kp = (sorted(pj)[0]).split("-")[0]
        for k, e in pj.items():
            if e.get("piece"):
                pieces[int(k.split("-")[1])] = e["piece"]
    else:
        sel = json.load(open(ROOT / "nbest_data/test_set_selector_eval.json"))
        pieces = sel["pieces"]

    model, device = load_model(str(ROOT / a.checkpoint))
    model.eval()
    recs, all_tokens = [], []
    for s in range(0, W.shape[0], a.batch):
        ids = W[s: s + a.batch].to(device)
        rolled, forced, nvalid = rollout_pitch_forced(model, ids)
        rolled = rolled.cpu()
        if a.save_tokens:
            all_tokens.append(rolled[:, flat.cpu()].to(torch.int16))
        for j in range(ids.shape[0]):
            wi = s + j
            gt = notes_of(W[wi][flat])
            fr = cf1.score_notes(notes_of(rolled[j][flat]), gt)
            base = cf1.score_notes(
                notes_of(d["cand_tokens"][first_cand[wi]].long()), gt)
            w = pieces.get(str(wi), pieces.get(wi))
            recs.append({
                "window": wi,
                "work": w.rsplit("/", 1)[0] if w else None,
                "forced_frac": float(forced[j]) / max(1, int(nvalid[j])),
                **{f"forced_{k}": fr[k]["f1"] for k in fr},
                **{f"greedy_{k}": base[k]["f1"] for k in base}})
        print(f"  {min(s + a.batch, W.shape[0])}/{W.shape[0]} windows "
              f"(mean forced {np.mean([r['forced_frac'] for r in recs]):.1%})",
              flush=True)

    print("\n=== PITCH FORCING vs plain greedy (test set) ===")
    byw = defaultdict(list)
    for r in recs:
        if r["work"]:
            byw[r["work"]].append(r)
    for k in ("onset_pitch", "onset_pitch_dur", "onset_pitch_tol1"):
        g = 100 * np.mean([r[f"greedy_{k}"] for r in recs])
        f = 100 * np.mean([r[f"forced_{k}"] for r in recs])
        pg = 100 * np.mean([np.mean([r[f"greedy_{k}"] for r in v])
                            for v in byw.values()])
        pf = 100 * np.mean([np.mean([r[f"forced_{k}"] for r in v])
                            for v in byw.values()])
        print(f"  {k:18} window: {g:6.2f} -> {f:6.2f}  ({f-g:+.2f})   "
              f"piece: {pg:6.2f} -> {pf:6.2f}  ({pf-pg:+.2f})")
    print(f"  pitch tokens forced: "
          f"{np.mean([r['forced_frac'] for r in recs]):.1%} of slots")
    json.dump({"records": recs}, open(ROOT / a.out, "w"))
    if a.save_tokens:
        torch.save({"forced_tokens": torch.cat(all_tokens)}, ROOT / a.save_tokens)
        print(f"wrote {a.save_tokens}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
