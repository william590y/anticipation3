"""Correctness gate for every decode optimization.

Nothing ships that fails this. The rule, in the order it is applied:

1. **Greedy must be bit-identical.** A candidate's `rolled` tokens must equal
   the baseline `rollout_score_slots(temperature=0)` tokens *exactly*, on a
   fixed window set, for every one of the 414 generated positions. Token
   equality -- not accuracy parity -- because a greedy rollout is chaotic: one
   flipped argmax on an onset token cascades through the rest of the window (the
   same effect that made 3- and 4-GPU validation disagree, see
   `onpolicy_rollout.evaluate_autoregressive`). So a single differing token is a
   real behaviour change, and "the F1 is the same" would hide it.
2. **`gt_ce` must match too** when the caller collects it: the RL arms select
   checkpoints on that number, so a candidate that shifts it in the last ulp is
   still a change to what the pipeline reports. Compared exactly, with the
   observed max absolute deviation printed either way.
3. **Sampled decode**: seeded and compared token-for-token. Where a candidate
   cannot be bit-exact by construction, the reason is printed and the gate falls
   back to distributional parity -- exact-match onset/duration/pitch accuracy and
   note-level F1 (`f1_reward.final_f1`, the criterion behind the results table)
   over `--windows` windows, which must be reported side by side rather than
   asserted equal.

Usage:
    python bench/check_identical.py --windows 32 --variants fast,fast:cuda_graph=true
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bench_common import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_TOKEN_FILE,
    describe_model,
    load_bench_model,
    load_bench_windows,
)

from anticipation.fast_decode import StaticKVDecoder, rollout_score_slots_fast  # noqa: E402
from f1_reward import decode_rollout_notes, final_f1  # noqa: E402
from onpolicy_rollout import (  # noqa: E402
    body_score_slot_starts,
    per_rollout_accuracy,
    rollout_score_slots,
)


def parse_options(spec):
    _, _, raw = spec.partition(":")
    opts = {}
    for item in raw.split(",") if raw else []:
        if not item:
            continue
        key, _, value = item.partition("=")
        if value in ("", "true", "True"):
            opts[key] = True
        elif value in ("false", "False"):
            opts[key] = False
        else:
            # Numeric options (e.g. `buckets=8`) must not arrive as strings:
            # `buckets="8"` would be read as the explicit edge list (8,) and
            # silently disable bucketing instead of requesting 8 buckets.
            try:
                opts[key] = int(value)
            except ValueError:
                opts[key] = value
    return opts


def run_baseline(model, batch, *, temperature, collect_gt_ce, seed):
    torch.manual_seed(seed)
    return rollout_score_slots(
        model,
        batch,
        temperature=temperature,
        constrain=True,
        collect_logprobs=False,
        collect_gt_ce=collect_gt_ce,
    )


def run_candidate(model, batch, spec, *, temperature, collect_gt_ce, seed, decoder=None):
    torch.manual_seed(seed)
    return rollout_score_slots_fast(
        model,
        batch,
        temperature=temperature,
        constrain=True,
        collect_logprobs=False,
        collect_gt_ce=collect_gt_ce,
        decoder=decoder,
        **parse_options(spec),
    )


def token_report(reference, candidate, positions):
    ref = reference["rolled"][:, positions]
    cand = candidate["rolled"][:, positions]
    matches = ref == cand
    total = matches.numel()
    equal = int(matches.sum().item())
    report = {
        "tokens": total,
        "identical_tokens": equal,
        "token_agreement": equal / total,
        "identical_windows": int(matches.all(dim=1).sum().item()),
        "windows": ref.shape[0],
        "bit_identical": equal == total,
    }
    if equal != total:
        # First divergence, by generated-position index -- with a chaotic greedy
        # rollout everything after it is downstream noise, so only this index is
        # diagnostic.
        first = int((~matches).float().argmax().item() % matches.shape[1])
        report["first_divergent_slot_index"] = first
    return report


def accuracy_and_f1(rollout, batch, positions, slot_starts):
    accuracy = per_rollout_accuracy(rollout["rolled"], batch, positions, rollout["valid"])
    rolled = rollout["rolled"].cpu()
    truth = batch.cpu()
    scores = []
    for row in range(rolled.shape[0]):
        pred = decode_rollout_notes(rolled[row].tolist(), slot_starts)
        gold = decode_rollout_notes(truth[row].tolist(), slot_starts)
        scores.append(final_f1(pred, gold))
    return {
        "onset_acc": float(accuracy[:, 0].mean().item()),
        "duration_acc": float(accuracy[:, 1].mean().item()),
        "pitch_acc": float(accuracy[:, 2].mean().item()),
        "f1_onset_pitch_tol1": sum(scores) / len(scores),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--variants", default="fast,fast:cuda_graph=true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gt-ce", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--attn", default=None)
    parser.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16", "float16"])
    parser.add_argument("--parity-only", action="store_true",
                        help="skip the bit-exact comparison; report accuracy/F1 parity only")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.dtype)
    model, device = load_bench_model(args.checkpoint, attn_implementation=args.attn, dtype=dtype)
    # The reference is always the production model: fp32, default attention. A
    # dtype/attn override is a property of the *candidate*, so comparing it to a
    # like-for-like reference would prove nothing.
    reference_model = model
    if dtype is not None or args.attn is not None:
        reference_model, _ = load_bench_model(args.checkpoint)

    windows = load_bench_windows(args.token_file, count=args.windows)
    length = windows.shape[1]
    slot_starts = body_score_slot_starts(length)
    positions = torch.tensor(
        [start + role for start in slot_starts for role in range(3)], device=device
    )

    specs = _split_variant_specs(args.variants)
    summary = {"info": describe_model(model), "windows": int(windows.shape[0]), "variants": {}}

    reference_rows, candidate_rows = {spec: [] for spec in specs}, {spec: [] for spec in specs}
    ref_stats, cand_stats = [], {spec: [] for spec in specs}

    for offset in range(0, windows.shape[0], args.batch):
        batch = windows[offset : offset + args.batch].to(device)
        if batch.shape[0] < args.batch:
            break  # a short final batch would change the rollout's batch shape
        reference = run_baseline(
            reference_model, batch, temperature=args.temperature,
            collect_gt_ce=args.gt_ce, seed=args.seed + offset,
        )
        ref_stats.append(accuracy_and_f1(reference, batch, positions, slot_starts))
        print(f"  batch @{offset}: reference done", flush=True)
        for spec in specs:
            decoder = None
            opts = parse_options(spec)
            if opts.get("static_cache", True) is not False:
                # Forward the decoder-shaped options: a decoder passed in wins,
                # so building it without them would silently drop `buckets=` /
                # `prebuilt_mask=` and grade a variant that never ran.
                decoder = StaticKVDecoder(
                    model,
                    batch.shape[0],
                    max_length=length,
                    prebuilt_mask=opts.get("prebuilt_mask", True),
                    buckets=opts.get("buckets"),
                )
            candidate = run_candidate(
                model, batch, spec, temperature=args.temperature,
                collect_gt_ce=args.gt_ce, seed=args.seed + offset, decoder=decoder,
            )
            cand_stats[spec].append(accuracy_and_f1(candidate, batch, positions, slot_starts))
            print(f"  batch @{offset}: {spec} done", flush=True)
            if not args.parity_only:
                row = token_report(reference, candidate, positions)
                if args.gt_ce:
                    delta = (reference["gt_ce"] - candidate["gt_ce"]).abs().max()
                    row["gt_ce_max_abs_delta"] = float(delta.item())
                    row["gt_ce_bit_identical"] = bool(
                        torch.equal(reference["gt_ce"], candidate["gt_ce"])
                    )
                candidate_rows[spec].append(row)
            del candidate, decoder
        del reference, batch
        torch.cuda.empty_cache()

    print(f"\nreference: {describe_model(reference_model)}")
    print(f"candidate: {describe_model(model)}")
    print(f"temperature={args.temperature} gt_ce={args.gt_ce} windows compared per variant:\n")
    for spec in specs:
        rows = candidate_rows[spec]
        stats = _mean_dicts(cand_stats[spec])
        ref = _mean_dicts(ref_stats)
        entry = {"parity": {"reference": ref, "candidate": stats}}
        if rows:
            total = sum(r["tokens"] for r in rows)
            equal = sum(r["identical_tokens"] for r in rows)
            windows_equal = sum(r["identical_windows"] for r in rows)
            windows_total = sum(r["windows"] for r in rows)
            entry["bit_identical"] = equal == total
            entry["token_agreement"] = equal / total
            entry["identical_windows"] = f"{windows_equal}/{windows_total}"
            if any("gt_ce_max_abs_delta" in r for r in rows):
                entry["gt_ce_max_abs_delta"] = max(
                    r.get("gt_ce_max_abs_delta", 0.0) for r in rows
                )
                entry["gt_ce_bit_identical"] = all(r.get("gt_ce_bit_identical") for r in rows)
            verdict = "PASS (bit-identical)" if entry["bit_identical"] else "FAIL (outputs differ)"
            print(
                f"{spec:<40} {verdict}   tokens {equal}/{total} "
                f"({entry['token_agreement']:.6f})  windows {entry['identical_windows']}"
            )
        else:
            print(f"{spec:<40} parity-only")
        print(
            f"{'':<40} ref  onset {ref['onset_acc']:.4f} dur {ref['duration_acc']:.4f} "
            f"pitch {ref['pitch_acc']:.4f} F1 {ref['f1_onset_pitch_tol1']:.4f}"
        )
        print(
            f"{'':<40} cand onset {stats['onset_acc']:.4f} dur {stats['duration_acc']:.4f} "
            f"pitch {stats['pitch_acc']:.4f} F1 {stats['f1_onset_pitch_tol1']:.4f}"
        )
        summary["variants"][spec] = entry

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.json}")


def _mean_dicts(rows):
    if not rows:
        return {}
    return {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]}


def _split_variant_specs(text):
    specs = []
    for chunk in text.split(","):
        # A continuation of the previous variant's option list is a bare
        # `key=value` with no `name:` in front of it. Testing the *previous*
        # spec for a colon instead (the first version of this) glued every
        # option-carrying variant onto its predecessor.
        if specs and "=" in chunk and ":" not in chunk:
            specs[-1] = specs[-1] + "," + chunk
        else:
            specs.append(chunk)
    return specs


if __name__ == "__main__":
    main()
