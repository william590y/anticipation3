"""Localise where a greedy speculative rollout stops agreeing with the baseline.

Two independent checks, because they fail in different places:

1. *Internal consistency.*  At temperature 0 the accept/reject rule collapses to
   "keep the draft token iff it is the target's argmax, else resample from the
   residual, which is a point mass on the target's argmax".  So every finalised
   token must equal the argmax of the constrained target distribution the
   verifier just used.  If this fails, the rule is wrong.
2. *Conditioning.*  If (1) holds but the output still differs from
   `rollout_score_slots`, then the target distribution itself was conditioned on
   the wrong prefix.  Re-deriving the argmax at the first divergent position from
   a single teacher-forced forward over the speculative rollout settles which
   side is wrong.
"""

from __future__ import annotations

import argparse

import torch

from anticipation.score_constraints import constrain_score_token_logits
from evaluate_muster import load_model
from nbest.draft_ngram import NgramProposer, load_tables
from nbest.speculative import (
    ModelProposer,
    geometry,
    load_draft,
    speculative_rollout_score_slots,
)
from onpolicy_rollout import rollout_score_slots, score_token_positions
from train_draft import PackedLineDataset, stack_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_checkpoint", default="run_paper_split_v2/checkpoint-2500")
    parser.add_argument("--draft", default="")
    parser.add_argument("--ngram_tables", default="debug/ngram_smoke.pt")
    parser.add_argument("--eval_file", default="data/val_paper.txt")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--slots", type=int, default=2)
    args = parser.parse_args()

    target, device = load_model(args.target_checkpoint)
    target.eval()
    data = PackedLineDataset([args.eval_file], 1, args.windows, 0)
    ids = stack_batch(data, range(len(data)), 1020)[: args.windows].to(device)
    geom = geometry(ids.shape[1])
    positions = score_token_positions(ids.shape[1], device=device)

    if args.draft:
        proposer_factory = lambda: ModelProposer(load_draft(args.draft, device), temperature=0.0)
        label = f"model:{args.draft}"
    else:
        tables = load_tables(args.ngram_tables, device)
        proposer_factory = lambda: NgramProposer(tables, temperature=0.0)
        label = "ngram"

    base = rollout_score_slots(
        target, ids, temperature=0.0, collect_logprobs=False, collect_gt_ce=False
    )["rolled"]

    trace, snapshots = [], []
    spec = speculative_rollout_score_slots(
        target, proposer_factory(), ids, slots_per_block=args.slots, temperature=0.0,
        debug_greedy_check=trace, debug_snapshots=snapshots,
    )["rolled"]
    print(f"in-loop greedy rule violations: {len(trace)}  {trace[:6]}")

    differs = (base[:, positions] != spec[:, positions])
    print(f"draft={label}  windows={ids.shape[0]}  "
          f"identical windows {float((~differs.any(dim=1)).float().mean()) * 100:.1f}%  "
          f"identical tokens {float((~differs).float().mean()) * 100:.2f}%")

    # --- check 2: first divergence, re-derived by teacher forcing --------
    with torch.no_grad():
        spec_logits = target(spec, use_cache=False).logits
        base_logits = target(base, use_cache=False).logits
    for row in range(ids.shape[0]):
        index = torch.nonzero(differs[row])
        if index.numel() == 0:
            print(f"  row {row}: identical")
            continue
        i = int(index[0])
        pos = int(positions[i])
        role = pos % 3
        prefix_same = bool(torch.equal(base[row, :pos], spec[row, :pos]))
        col_spec = constrain_score_token_logits(spec_logits[row, pos - 1, :].float(), role)
        col_base = constrain_score_token_logits(base_logits[row, pos - 1, :].float(), role)
        top_spec = col_spec.topk(3)
        # Which block verified this position for this row, and did the prefix it
        # was conditioned on survive to the end?
        blame = None
        for snap in snapshots:
            if pos in snap["positions"] and int(snap["frontier"][row]) == pos:
                out_s = snap["out"]
                changed = (out_s[row, :pos].to(spec.device) != spec[row, :pos]).nonzero().flatten()
                blame = (snap["base"], snap["resume"], snap["end"], snap["positions"],
                         changed[:6].tolist(), int(changed.numel()))
                break
        print(f"      verifying block/prefix-drift: {blame}")
        print(
            f"  row {row}: first divergence at position {pos} (role {role}, "
            f"slot {(pos - geom.gen_start) // 6}), prefixes identical={prefix_same}\n"
            f"      baseline token {int(base[row, pos])}  spec token {int(spec[row, pos])}\n"
            f"      teacher-forced argmax on spec prefix: {int(col_spec.argmax())} "
            f"(top3 {top_spec.indices.tolist()} logits "
            f"{[round(v, 3) for v in top_spec.values.tolist()]})\n"
            f"      teacher-forced argmax on base prefix: {int(col_base.argmax())}"
        )


if __name__ == "__main__":
    main()
