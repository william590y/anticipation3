"""Localise a greedy-exactness failure of the diffusion speculative decoder.

`bench/bench_diffdraft.py --tasks exact` only reports *that* the speculative
output differs from `rollout_score_slots`. This says *where* and *why*, using one
extra teacher-forced forward per batch:

  self-consistency  Feed the speculative output back through the target with
                    every token teacher-forced. At each body score position p,
                    is the target's constrained argmax given committed[<p] equal
                    to committed[p]? If yes at every p, the emitted sequence *is*
                    a fixed point of greedy decoding and any difference from the
                    baseline is a tie / floating-point artefact, not a logic bug.
                    The first p where it fails is the bug, and it is reported with
                    the round that produced it.

The same check is run on the baseline rollout, because a chunked teacher-forced
forward and a 1-token cached decode do not have identical matmul reduction
orders: if the baseline is not self-consistent under this test either, the test
itself is measuring float noise and its verdict has to be read that way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.bench_common import load_bench_windows  # noqa: E402
from evaluate_muster import load_model  # noqa: E402
from nbest.diffdraft import build_drafter, constrain_by_role, load_drafter  # noqa: E402
from nbest.diffdraft_decode import diffdraft_decode  # noqa: E402
from onpolicy_rollout import rollout_score_slots, score_token_positions  # noqa: E402


@torch.no_grad()
def self_consistency(target, sequence, positions):
    """(batch, n) bool: does the target's constrained argmax reproduce each token?"""
    out = target(sequence, use_cache=False)
    dists = out.logits[:, positions - 1, :].float()
    dists = constrain_by_role(dists, positions % 3)
    return dists.argmax(dim=-1) == sequence[:, positions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="run_paper_split_v2/checkpoint-2500")
    parser.add_argument("--drafter", default="untrained:6")
    parser.add_argument("--token-file", default="data/val_paper.txt")
    parser.add_argument("--windows", type=int, default=224)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--block-slots", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--order", default="confidence")
    parser.add_argument("--drafter-dtype", default="bf16")
    args = parser.parse_args()

    target, device = load_model(args.checkpoint)
    target.config.use_cache = True
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.drafter_dtype]
    if args.drafter.startswith("untrained:"):
        from safetensors.torch import load_file as load_safetensors

        state = load_safetensors(str(Path(args.checkpoint) / "model.safetensors"))
        drafter, _ = build_drafter(n_layer=int(args.drafter.split(":")[1]), target_state=state)
        del state
        drafter = drafter.to(device=device, dtype=dtype).eval()
    else:
        drafter, _ = load_drafter(args.drafter, device=device, dtype=dtype)

    windows = load_bench_windows(args.token_file, count=args.windows)[: args.rows].to(device)
    positions = score_token_positions(windows.shape[1], device=device)

    with torch.no_grad():
        base = rollout_score_slots(target, windows, temperature=0.0, constrain=True,
                                   collect_logprobs=False, collect_gt_ce=False)["rolled"]
        spec, stats = diffdraft_decode(target, drafter, windows,
                                       block_slots=args.block_slots, steps=args.steps,
                                       order=args.order, temperature=0.0)
        oracle, _ = diffdraft_decode(target, drafter, windows, block_slots=args.block_slots,
                                     steps=args.steps, order=args.order, temperature=0.0,
                                     oracle_draft=base)

        base_ok = self_consistency(target, base, positions)
        spec_ok = self_consistency(target, spec, positions)

    diff = base[:, positions] != spec[:, positions]
    odiff = base[:, positions] != oracle[:, positions]
    print(f"stats: {stats}")
    print(f"tokens differing spec vs base : {int(diff.sum())} / {diff.numel()}")
    print(f"tokens differing oracle vs base: {int(odiff.sum())} / {odiff.numel()}")
    print(f"baseline self-consistent      : {int(base_ok.sum())} / {base_ok.numel()}")
    print(f"speculative self-consistent   : {int(spec_ok.sum())} / {spec_ok.numel()}")
    for row in range(windows.shape[0]):
        first_diff = int(diff[row].float().argmax()) if bool(diff[row].any()) else -1
        first_base_bad = int((~base_ok[row]).float().argmax()) if bool((~base_ok[row]).any()) else -1
        first_spec_bad = int((~spec_ok[row]).float().argmax()) if bool((~spec_ok[row]).any()) else -1
        print(f"  row {row}: first spec-vs-base diff at score index {first_diff} "
              f"(abs pos {int(positions[first_diff]) if first_diff >= 0 else -1}); "
              f"first non-self-consistent: base {first_base_bad}, spec {first_spec_bad}")


if __name__ == "__main__":
    main()
