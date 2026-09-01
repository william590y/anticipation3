"""Correctness gate + speed table for the masked-diffusion speculative drafter.

Four tasks, all on the same fixed window fixture so the numbers compose:

  exact      Greedy: is `nbest/diffdraft_decode.diffdraft_decode` bit-identical to
             the unmodified `onpolicy_rollout.rollout_score_slots` rollout? Also
             runs an `--oracle-draft` control that drafts with the baseline's own
             tokens, which forces 100% acceptance and so isolates the pure
             floating-point difference between chunked verification and
             single-token decode from any algorithmic difference.
  agreement  Standalone drafter quality: per-role top-1 agreement with the
             teacher's greedy tokens at several (block, K, order) settings, plus
             the mean matching-prefix length, which is what acceptance actually
             depends on.
  speed      Wall clock, target forwards, drafter forwards and accepted tokens per
             target forward, versus a baseline measured on the same GPU in the
             same process, at several batch sizes.
  dist       Sampled path (temperature 1): how far the emitted distribution moves
             relative to baseline sampling, for the exact `ltr` schedule and for
             the approximate `confidence` schedule.

The window fixture is `bench.bench_common.load_bench_windows` -- evenly spaced
byte offsets into a token file, deterministic and shared with the other decode
benchmarks so the baseline column here is comparable to `bench/results/*.json`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anticipation.packed_sequence import ALTERNATING_START  # noqa: E402
from anticipation.vocab import CONTROL_OFFSET, DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET  # noqa: E402
from bench.bench_common import load_bench_windows  # noqa: E402
from evaluate_muster import load_model  # noqa: E402
from nbest.diffdraft import (  # noqa: E402
    MASK_ID,
    N_BODY_SLOTS,
    block_token_end,
    denoise_block,
    load_drafter,
    score_positions,
)
from nbest.diffdraft_decode import diffdraft_decode  # noqa: E402
from onpolicy_rollout import rollout_score_slots, score_token_positions  # noqa: E402

ROLE_NAMES = ("onset", "duration", "pitch")
ROLE_RANGES = ((TIME_OFFSET, DUR_OFFSET), (DUR_OFFSET, NOTE_OFFSET), (NOTE_OFFSET, CONTROL_OFFSET))


def task_micro(target, drafter, windows, args, device, dctx):
    """Isolated per-call costs, so the speed table's verdict survives re-tuning.

    The whole argument for a diffusion drafter rests on one ratio: what a drafter
    denoising step costs relative to a target decode step. Measuring the three
    primitives separately means the analysis can be re-run against a cheaper
    target step (the other agent's CUDA-graph / static-cache work) without
    re-running any of the decode experiments.

      target_step       one cached single-token forward -- the baseline's unit,
                        414 of them per window
      target_block      one cached forward over a whole 6*B-token block -- the
                        verifier's unit, and note it processes exactly the same
                        number of tokens the 3B baseline steps would have
      drafter_block     one denoising forward over the same block, with the
                        prefix already in the drafter's KV cache
    """
    from nbest.diffdraft import MASK_ID, block_token_end, score_positions

    rows = []
    for batch_size in args.speed_batches:
        chunk = windows[:batch_size].to(device)
        if chunk.shape[0] < batch_size:
            continue
        try:
            prime = target(chunk[:, :ALTERNATING_START], use_cache=True)
            past = prime.past_key_values
            one = chunk[:, ALTERNATING_START : ALTERNATING_START + 1]

            def step():
                out = target(one, past_key_values=past, use_cache=True)
                past.crop(ALTERNATING_START)
                return out

            step_t, _ = timed(step, 5)

            for block_slots in sorted({b for b, _, _ in args.speed_configs}):
                end = block_token_end(block_slots)
                block = chunk[:, ALTERNATING_START:end]

                def verify():
                    out = target(block, past_key_values=past, use_cache=True)
                    past.crop(ALTERNATING_START)
                    return out

                verify_t, _ = timed(verify, 5)

                with dctx():
                    cache = drafter.encode_prefix(chunk[:, :ALTERNATING_START], 0)
                local = score_positions(0, block_slots, device=device) - ALTERNATING_START
                masked = block.clone()
                masked[:, local] = MASK_ID

                def draft_step():
                    with dctx():
                        hidden = drafter.forward_block(masked, ALTERNATING_START, cache)
                        return drafter.head(hidden[:, local, :]).float()

                draft_t, _ = timed(draft_step, 5)
                rows.append({
                    "batch": batch_size, "block_slots": block_slots,
                    "target_step_ms": 1000 * step_t,
                    "target_block_ms": 1000 * verify_t,
                    "drafter_block_ms": 1000 * draft_t,
                    "drafter_over_target_step": draft_t / step_t,
                    "block_over_step": verify_t / step_t,
                    "baseline_steps_replaced": 3 * block_slots,
                })
                print(f"  batch={batch_size:>3} B={block_slots:>3}: target step "
                      f"{1000*step_t:6.2f} ms | target block({6*block_slots} tok) "
                      f"{1000*verify_t:7.2f} ms = {verify_t/step_t:5.2f} steps | "
                      f"drafter block {1000*draft_t:7.2f} ms = "
                      f"{draft_t/step_t:5.2f} steps | replaces {3*block_slots} steps",
                      flush=True)
            del past, prime
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  batch={batch_size}: OOM", flush=True)
    return rows


def role_of(positions):
    return positions % 3


def per_role_match(a, b, positions):
    roles = role_of(positions)
    match = a[:, positions] == b[:, positions]
    out = {ROLE_NAMES[r]: float(match[:, roles == r].float().mean()) for r in range(3)}
    out["overall"] = float(match.float().mean())
    return out


def autocast_for(dtype):
    if dtype == torch.float32:
        return contextlib.nullcontext
    return lambda: torch.autocast("cuda", dtype=dtype)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------


def task_exact(target, drafter, windows, args, device, dctx):
    positions = score_token_positions(windows.shape[1], device=device)
    rows = []
    for batch_size in args.exact_batches:
        mismatched_windows = 0
        mismatched_tokens = 0
        total_tokens = 0
        oracle_bad_windows = 0
        oracle_bad_tokens = 0
        seen = 0
        for start in range(0, min(len(windows), args.exact_windows), batch_size):
            chunk = windows[start : start + batch_size].to(device)
            if chunk.shape[0] < batch_size:
                break
            base = rollout_score_slots(
                target, chunk, temperature=0.0, constrain=True,
                collect_logprobs=False, collect_gt_ce=False,
            )["rolled"]
            spec, _ = diffdraft_decode(
                target, drafter, chunk, block_slots=args.block_slots, steps=args.steps,
                order=args.order, temperature=0.0, drafter_autocast=dctx,
            )
            diff = base[:, positions] != spec[:, positions]
            mismatched_tokens += int(diff.sum())
            mismatched_windows += int(diff.any(dim=1).sum())
            total_tokens += diff.numel()

            oracle, _ = diffdraft_decode(
                target, drafter, chunk, block_slots=args.block_slots, steps=args.steps,
                order=args.order, temperature=0.0, oracle_draft=base,
            )
            odiff = base[:, positions] != oracle[:, positions]
            oracle_bad_tokens += int(odiff.sum())
            oracle_bad_windows += int(odiff.any(dim=1).sum())
            seen += chunk.shape[0]
        rows.append({
            "batch": batch_size,
            "windows": seen,
            "score_tokens": total_tokens,
            "mismatched_tokens": mismatched_tokens,
            "mismatched_windows": mismatched_windows,
            "token_mismatch_rate": mismatched_tokens / max(total_tokens, 1),
            "oracle_mismatched_tokens": oracle_bad_tokens,
            "oracle_mismatched_windows": oracle_bad_windows,
        })
        print(f"  exact batch={batch_size}: {seen} windows, "
              f"{mismatched_tokens}/{total_tokens} score tokens differ "
              f"({mismatched_windows} windows); oracle-draft control "
              f"{oracle_bad_tokens} tokens ({oracle_bad_windows} windows)", flush=True)
    return rows


def task_agreement(target, drafter, windows, args, device, dctx):
    """Per-role top-1 agreement between the drafter and the teacher's greedy tokens."""
    rows = []
    configs = []
    for block_slots in args.agree_blocks:
        for steps in args.agree_steps:
            configs.append((block_slots, steps, "confidence"))
        configs.append((block_slots, 3, "role"))
        for steps in args.agree_steps:
            configs.append((block_slots, steps, "ltr"))

    batch_size = args.agree_batch
    teacher_cache = {}
    for block_slots, steps, order in configs:
        acc = {name: [0.0, 0.0] for name in (*ROLE_NAMES, "overall")}
        run_slots = []
        seen = 0
        for start in range(0, min(len(windows), args.agree_windows), batch_size):
            chunk = windows[start : start + batch_size].to(device)
            if chunk.shape[0] < batch_size:
                break
            key = start
            if key not in teacher_cache:
                teacher_cache[key] = rollout_score_slots(
                    target, chunk, temperature=0.0, constrain=True,
                    collect_logprobs=False, collect_gt_ce=False,
                )["rolled"]
            teacher = teacher_cache[key]

            # Draft the first `block_slots` slots of a fresh window, with the true
            # (control) context and no committed score prefix -- the hardest case
            # and the one the decode loop opens with.
            prefix_len = ALTERNATING_START
            end = block_token_end(block_slots)
            local = score_positions(0, block_slots, device=device) - prefix_len
            with dctx():
                cache = drafter.encode_prefix(teacher[:, :prefix_len], 0)
                block = teacher[:, prefix_len:end].clone()
                block[:, local] = MASK_ID
                filled, _ = denoise_block(drafter, block, prefix_len, cache, local,
                                          steps=steps, order=order, temperature=0.0)
            truth = teacher[:, prefix_len:end][:, local]
            match = filled[:, local] == truth
            roles = (local + prefix_len) % 3
            for r in range(3):
                sel = match[:, roles == r]
                acc[ROLE_NAMES[r]][0] += float(sel.sum())
                acc[ROLE_NAMES[r]][1] += sel.numel()
            acc["overall"][0] += float(match.sum())
            acc["overall"][1] += match.numel()
            run_slots.append(float((match.float().cumprod(dim=1).sum(dim=1) / 3).mean()))
            seen += chunk.shape[0]
        row = {
            "block_slots": block_slots, "steps": steps, "order": order, "windows": seen,
            **{k: (v[0] / max(v[1], 1)) for k, v in acc.items()},
            "prefix_run_slots": sum(run_slots) / max(len(run_slots), 1),
        }
        rows.append(row)
        print(f"  B={block_slots:>3} K={steps} {order:<10} top1 {row['overall']*100:5.2f}%  "
              f"onset {row['onset']*100:5.2f}  dur {row['duration']*100:5.2f}  "
              f"pitch {row['pitch']*100:5.2f}  prefix-run {row['prefix_run_slots']:.2f} slots",
              flush=True)
    return rows


def timed(fn, repeats):
    times = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        times.append(time.perf_counter() - t0)
    return min(times), out


def task_speed(target, drafter, windows, args, device, dctx):
    rows = []
    for batch_size in args.speed_batches:
        chunk = windows[:batch_size].to(device)
        if chunk.shape[0] < batch_size:
            print(f"  (only {chunk.shape[0]} windows available, skipping batch {batch_size})")
            continue
        try:
            torch.cuda.reset_peak_memory_stats()
            base_t, _ = timed(
                lambda: rollout_score_slots(
                    target, chunk, temperature=0.0, constrain=True,
                    collect_logprobs=False, collect_gt_ce=False,
                ),
                args.repeats,
            )
            base_mem = torch.cuda.max_memory_allocated() / 2**30
            rows.append({
                "variant": "baseline", "batch": batch_size, "seconds": base_t,
                "windows_per_s": batch_size / base_t,
                "ms_per_window": 1000 * base_t / batch_size,
                "target_forwards_per_window": 415,
                "drafter_forwards_per_window": 0,
                "target_tokens_per_window": ALTERNATING_START + 3 * N_BODY_SLOTS + 3 * N_BODY_SLOTS,
                "accepted_tokens_per_target_forward": 3 * N_BODY_SLOTS / 414,
                "peak_mem_gib": base_mem,
            })
            print(f"  baseline  batch={batch_size:>3}: {base_t:7.3f}s  "
                  f"{batch_size/base_t:6.2f} win/s  {1000*base_t/batch_size:8.1f} ms/window  "
                  f"{base_mem:.1f} GiB", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  baseline batch={batch_size}: OOM", flush=True)
            continue

        for block_slots, steps, order in args.speed_configs:
            try:
                torch.cuda.reset_peak_memory_stats()
                holder = {}

                def run():
                    out, stats = diffdraft_decode(
                        target, drafter, chunk, block_slots=block_slots, steps=steps,
                        order=order, temperature=0.0, drafter_autocast=dctx,
                    )
                    holder["stats"] = stats
                    return out

                spec_t, _ = timed(run, args.repeats)
                stats = holder["stats"]
                mem = torch.cuda.max_memory_allocated() / 2**30
                rows.append({
                    "variant": f"diffdraft:B={block_slots},K={steps},{order}",
                    "batch": batch_size, "seconds": spec_t,
                    "windows_per_s": batch_size / spec_t,
                    "ms_per_window": 1000 * spec_t / batch_size,
                    "speedup_vs_baseline": base_t / spec_t,
                    "peak_mem_gib": mem,
                    **{k: v for k, v in stats.items() if k != "order"},
                })
                print(f"  B={block_slots:>3} K={steps} {order:<10} batch={batch_size:>3}: "
                      f"{spec_t:7.3f}s  {batch_size/spec_t:6.2f} win/s  "
                      f"{base_t/spec_t:5.2f}x  "
                      f"tgt-fwd {stats['target_forwards_per_window']:>4}  "
                      f"drf-fwd {stats['drafter_forwards_per_window']:>4}  "
                      f"acc/fwd {stats['accepted_tokens_per_target_forward']:5.2f}  "
                      f"{mem:.1f} GiB", flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  B={block_slots} K={steps} {order} batch={batch_size}: OOM", flush=True)
    return rows


def tv(a, b):
    return float(0.5 * (a - b).abs().sum())


def task_dist(target, drafter, windows, args, device, dctx):
    """Sampled path: how far does the emitted distribution move from baseline sampling?"""
    positions = score_token_positions(windows.shape[1], device=device)
    roles = role_of(positions)
    n = min(len(windows), args.dist_windows)
    batch_size = args.dist_batch

    # `baseline_b` is a second, independently seeded baseline sample: the TV
    # distance between the two baselines is the noise floor at this sample size,
    # and any deviation the approximate schedule shows has to clear it to mean
    # anything. Seeds are explicit offsets rather than hash(name) -- Python
    # randomises str hashing per process, which would make the run irreproducible.
    variants = {"baseline_a": (None, 0), "baseline_b": (None, 500_000)}
    for i, order in enumerate(args.dist_orders):
        variants[f"diffdraft:{order}"] = (order, 1_000_000 * (i + 1))

    collected = {k: [] for k in variants}
    accuracy = {k: [0.0, 0.0, 0.0, 0.0] for k in variants}
    for start in range(0, n, batch_size):
        chunk = windows[start : start + batch_size].to(device)
        if chunk.shape[0] < batch_size:
            break
        for name, (order, seed_offset) in variants.items():
            generator = torch.Generator(device=device).manual_seed(
                args.seed + seed_offset + start
            )
            torch.manual_seed(args.seed + seed_offset + start)
            if order is None:
                out = rollout_score_slots(
                    target, chunk, temperature=1.0, constrain=True,
                    collect_logprobs=False, collect_gt_ce=False,
                )["rolled"]
            else:
                out, _ = diffdraft_decode(
                    target, drafter, chunk, block_slots=args.block_slots,
                    steps=args.steps, order=order, temperature=1.0,
                    generator=generator, drafter_autocast=dctx,
                )
            collected[name].append(out[:, positions].cpu())
            match = out[:, positions] == chunk[:, positions]
            for r in range(3):
                accuracy[name][r] += float(match[:, roles == r].float().sum())
            accuracy[name][3] += float(match[:, roles == 0].numel())

    hists = {}
    for name, parts in collected.items():
        tokens = torch.cat(parts, dim=0).to(device)
        # Column j of `tokens` is absolute position `positions[j]`, so the role
        # vector computed from `positions` indexes the columns directly.
        hists[name] = []
        for r in range(3):
            lo, hi = ROLE_RANGES[r]
            vals = tokens[:, (roles == r).nonzero().squeeze(1)].reshape(-1) - lo
            h = torch.bincount(vals.clamp(0, hi - lo - 1), minlength=hi - lo).float()
            hists[name].append(h / h.sum().clamp(min=1))

    rows = []
    noise_floor = [tv(hists["baseline_a"][r], hists["baseline_b"][r]) for r in range(3)]
    for name in variants:
        acc = accuracy[name]
        row = {
            "variant": name,
            "windows": int(acc[3] / N_BODY_SLOTS) if acc[3] else 0,
            **{ROLE_NAMES[r]: acc[r] / max(acc[3], 1) for r in range(3)},
            **{f"tv_{ROLE_NAMES[r]}": tv(hists[name][r], hists["baseline_a"][r])
               for r in range(3)},
            "tv_noise_floor": noise_floor,
        }
        rows.append(row)
        print(f"  {name:<22} acc onset {row['onset']*100:5.2f} dur {row['duration']*100:5.2f} "
              f"pitch {row['pitch']*100:5.2f} | TV vs baseline_a "
              f"onset {row['tv_onset']:.4f} dur {row['tv_duration']:.4f} "
              f"pitch {row['tv_pitch']:.4f}", flush=True)
    print(f"  (two independent baseline samples differ by TV = "
          f"{noise_floor[0]:.4f}/{noise_floor[1]:.4f}/{noise_floor[2]:.4f} -- that is the "
          f"noise floor at this sample size)", flush=True)
    return rows


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="run_paper_split_v2/checkpoint-2500")
    parser.add_argument("--drafter", default="run_diffdraft/diffdraft_final.pt")
    parser.add_argument("--token-file", default="data/val_paper.txt")
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--tasks", nargs="+",
                        default=["agreement", "exact", "speed", "dist"])
    parser.add_argument("--drafter-dtype", default="bf16", choices=["bf16", "fp32", "fp16"])
    parser.add_argument("--block-slots", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--order", default="confidence")
    parser.add_argument("--exact-windows", type=int, default=208)
    parser.add_argument("--exact-batches", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--agree-windows", type=int, default=128)
    parser.add_argument("--agree-batch", type=int, default=16)
    parser.add_argument("--agree-blocks", type=int, nargs="+", default=[8, 16, 32, 138])
    parser.add_argument("--agree-steps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--speed-batches", type=int, nargs="+", default=[8, 32, 96])
    parser.add_argument("--speed-config", nargs="+",
                        default=["8,4,confidence", "16,4,confidence", "16,2,confidence",
                                 "32,4,confidence", "16,4,ltr"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--dist-windows", type=int, default=208)
    parser.add_argument("--dist-batch", type=int, default=16)
    parser.add_argument("--dist-orders", nargs="+", default=["ltr", "confidence"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="bench/results/diffdraft.json")
    args = parser.parse_args()
    args.speed_configs = [
        (int(a), int(b), c) for a, b, c in (s.split(",") for s in args.speed_config)
    ]

    target, device = load_model(args.checkpoint)
    target.config.use_cache = True
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.drafter_dtype
    ]
    if args.drafter.startswith("untrained:"):
        # Smoke path: an untrained drafter accepts almost nothing, but the greedy
        # exactness argument does not depend on draft quality at all, so this is
        # the right fixture for gating the verification logic before spending GPU
        # hours on training.
        from safetensors.torch import load_file as load_safetensors

        from nbest.diffdraft import build_drafter

        n_layer = int(args.drafter.split(":", 1)[1])
        state = load_safetensors(str(Path(args.checkpoint) / "model.safetensors"))
        drafter, init_info = build_drafter(n_layer=n_layer, target_state=state)
        del state
        drafter = drafter.to(device=device, dtype=dtype).eval()
        blob = {"config": drafter.config.to_dict(), "step": 0, "init_info": init_info}
        print(f"untrained drafter, init info: {json.dumps(init_info)}", flush=True)
    else:
        drafter, blob = load_drafter(args.drafter, device=device, dtype=dtype)
    dctx = autocast_for(dtype)
    n_trunk = sum(p.numel() for p in drafter.h.parameters())
    print(f"drafter: {blob['config']['n_layer']} layers, {n_trunk/1e6:.1f}M non-embedding, "
          f"{args.drafter_dtype}, trained {blob.get('step')} steps", flush=True)

    windows = load_bench_windows(args.token_file, count=args.windows)
    print(f"fixture: {windows.shape[0]} windows of {windows.shape[1]} tokens", flush=True)

    results = {
        "info": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "target_dtype": str(next(target.parameters()).dtype),
            "drafter_dtype": args.drafter_dtype,
            "drafter_layers": blob["config"]["n_layer"],
            "drafter_step": blob.get("step"),
            "torch": torch.__version__,
            "args": {k: v for k, v in vars(args).items() if k != "speed_configs"},
        }
    }
    with torch.no_grad():
        for task in args.tasks:
            print(f"\n=== {task} ===", flush=True)
            fn = {"exact": task_exact, "agreement": task_agreement,
                  "speed": task_speed, "dist": task_dist, "micro": task_micro}[task]
            results[task] = fn(target, drafter, windows, args, device, dctx)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
