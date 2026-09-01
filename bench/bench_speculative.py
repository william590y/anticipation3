"""Benchmark + exactness harness for speculative decoding of the score decoder.

Run as ``python -m bench.bench_speculative --mode all``.  Everything is measured
in one process on one GPU so the comparison table is internally consistent; the
GPU name is recorded in the output JSON because a launch-bound decode's numbers
mean nothing without it.

Modes
-----
micro     per-forward latency of the target, of each draft, and of hypothetical
          narrow-from-scratch drafts.  This is the measurement that decides the
          draft *architecture*: if latency tracks depth and ignores width, a
          shallow full-width distillation of the target is strictly better than
          a narrow model trained from scratch, because it costs the same
          launches and starts from the target's own weights.
accept    acceptance rate per role, tokens per target forward, drafter forwards
          per window.  Hardware-independent -- these stay valid if the target
          step is later made cheaper by CUDA graphs / TensorRT.
speed     end-to-end windows/sec against the unmodified `rollout_score_slots`.
exact     greedy bit-identity vs the baseline (plus a float-noise control), and
          a distributional check at temperature 1.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats

from anticipation.score_constraints import constrain_score_token_logits
from evaluate_muster import load_model
from nbest.draft_ngram import NgramProposer, fit_ngram_tables, load_tables, save_tables
from nbest.speculative import (
    ModelProposer,
    SpeculativeStats,
    StagedProposer,
    build_shallow_draft,
    load_draft,
    speculative_rollout_score_slots,
)
from onpolicy_rollout import rollout_score_slots, score_token_positions, score_token_roles
from train_draft import PackedLineDataset, stack_batch

TOKEN_TYPE_NAMES = ("onset", "duration", "pitch")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_windows(paths, count, length=1020, seed=0, stride=1):
    data = PackedLineDataset([p for p in paths.split(",") if p], stride, count, seed)
    rows = stack_batch(data, range(len(data)), length)
    return rows[:count]


def timed(fn, iters, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def role_accuracy(rolled, truth, positions, roles, valid):
    """Per-window exact-match accuracy by role -> (batch, 3) numpy."""
    matches = rolled[:, positions] == truth[:, positions]
    out = []
    for role in range(3):
        selector = valid & (roles == role).unsqueeze(0)
        hits = (matches & selector).sum(dim=1).float()
        total = selector.sum(dim=1).clamp(min=1).float()
        out.append((hits / total).cpu().numpy())
    return np.stack(out, axis=1)


# ---------------------------------------------------------------------------
# Configurations under test
# ---------------------------------------------------------------------------


def make_proposer(name, models, tables, temperature, generator, inner_score_tokens, slots, target=None):
    # 0 means "one inner (D1) forward per outer block": the inner level tries to
    # cover the whole outer request in a single D1 verification, which is the
    # configuration where staging can actually remove D1 forwards.
    if inner_score_tokens <= 0:
        inner_score_tokens = 3 * slots
    if name == "self":
        # Draft == target: the *only* thing that can then cause a rejection is a
        # float difference between the draft's 1-token forward (identical in
        # shape to the baseline's) and the target's chunked verification
        # forward.  Its acceptance rate is therefore a direct measurement of the
        # numerical noise floor, and its greedy output is the tightest possible
        # bit-identity test of the machinery itself.
        return ModelProposer(target, temperature=temperature, generator=generator, level="draft")
    if name == "ngram":
        return NgramProposer(tables, temperature=temperature, generator=generator)
    if name.startswith("staged:"):
        _, outer = name.split(":", 1)
        inner = NgramProposer(tables, temperature=temperature, generator=generator, level="d2")
        return StagedProposer(
            models[outer],
            inner,
            temperature=temperature,
            generator=generator,
            inner_score_tokens=inner_score_tokens,
            level="d1",
        )
    return ModelProposer(models[name], temperature=temperature, generator=generator, level="draft")


# ---------------------------------------------------------------------------
# micro
# ---------------------------------------------------------------------------


def mode_micro(args, target, models, device, results, tables=None, sample=None):
    """Per-forward latency vs depth / width / chunk length.

    A decode step is a 1-token forward on top of a ~600-token cache; the
    verification step of speculative decoding is the same thing with a chunk of
    `chunk` tokens.  Both are measured, because the whole cost model rests on
    the chunk being nearly as cheap as the single token.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    prime_len = 600
    rows = {}
    probes = {"target": target}
    probes.update({f"draft:{k}": v for k, v in models.items()})
    for n_layer, n_embd in args.micro_configs:
        config = GPT2Config(
            vocab_size=target.config.vocab_size,
            n_positions=1024,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=max(1, n_embd // 64),
            scale_attn_by_inverse_layer_idx=True,
        )
        probes[f"scratch:{n_layer}x{n_embd}"] = GPT2LMHeadModel(config).to(device).eval()

    for batch in args.batch_sizes:
        ids = torch.randint(0, 40000, (batch, prime_len), device=device)
        for name, model in probes.items():
            with torch.no_grad():
                # Prime through the trunk only: a (batch, 600, 55028) logits
                # tensor is 8 GB at batch 64 and is not part of what we time.
                primed = model.transformer(ids, use_cache=True)
                cache = primed.past_key_values
                base_len = cache.get_seq_length()

                def step(width):
                    def run():
                        cache.crop(base_len)
                        model(
                            torch.randint(0, 40000, (batch, width), device=device),
                            past_key_values=cache,
                            use_cache=True,
                        )

                    return run

                one = timed(step(1), args.micro_iters)
                chunk = timed(step(args.micro_chunk), args.micro_iters)
            rows.setdefault(name, {})[batch] = {
                "ms_1tok": one * 1e3,
                f"ms_{args.micro_chunk}tok": chunk * 1e3,
                "n_layer": model.config.n_layer,
                "n_embd": model.config.n_embd,
            }
            print(
                f"  micro batch={batch:3d} {name:22s} "
                f"1tok {one * 1e3:7.3f} ms   {args.micro_chunk}tok {chunk * 1e3:7.3f} ms",
                flush=True,
            )
        # The table draft has no forward at all: time one full proposal (gather +
        # scatter_add + renormalise + multinomial) so the cost model has a real
        # number for the cheapest possible drafting level.
        if tables is not None and sample is not None:
            from nbest.speculative import geometry

            out = sample[:1].to(device).expand(batch, -1).contiguous()
            geom = geometry(out.shape[1])
            proposer = NgramProposer(tables, temperature=1.0)
            probe_stats = SpeculativeStats(device)
            pos = geom.score_positions[40]
            frontier = torch.full((batch,), pos, dtype=torch.long, device=device)
            ms = timed(
                lambda: proposer.propose(out, [pos], frontier, geom, probe_stats),
                args.micro_iters,
            ) * 1e3
            rows.setdefault("ngram_lookup", {})[batch] = {"ms_1tok": ms, "n_layer": 0,
                                                          "n_embd": 0}
            print(f"  micro batch={batch:3d} {'ngram_lookup':22s} 1tok {ms:7.3f} ms", flush=True)
        del ids
        torch.cuda.empty_cache()
    results["micro"] = rows
    # The ratio the cost model needs.
    ratios = {}
    for batch in args.batch_sizes:
        target_ms = rows["target"][batch]["ms_1tok"]
        ratios[batch] = {
            name: rows[name][batch]["ms_1tok"] / target_ms
            for name in rows
            if name != "target" and batch in rows[name]
        }
    results["cost_ratio_vs_target"] = ratios
    return results


# ---------------------------------------------------------------------------
# accept / speed
# ---------------------------------------------------------------------------


def run_config(name, target, models, tables, windows, batch, slots, temperature, device,
               inner_score_tokens, seed=0, collect_rolled=False):
    """One full pass over `windows` in batches; returns stats and (optionally) rolls."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    stats = SpeculativeStats(device)
    rolled = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(0, windows.shape[0], batch):
        chunk = windows[i : i + batch].to(device)
        if chunk.shape[0] == 0:
            continue
        if name == "baseline":
            out = rollout_score_slots(
                target, chunk, temperature=temperature, constrain=True,
                collect_logprobs=False, collect_gt_ce=False,
            )
            stats.rows += chunk.shape[0]
            # Baseline accounting, by construction: 1 prime + 3 forwards per slot
            # = one target forward per generated score token.
            n_positions = score_token_positions(chunk.shape[1], device=device).numel()
            stats.count_forward("target", chunk.shape[0])
            for _ in range(n_positions):
                stats.count_forward("target", chunk.shape[0])
            for role in range(3):
                stats.finalized[role] += n_positions / 3 * chunk.shape[0]
        else:
            proposer = make_proposer(
                name, models, tables, temperature, generator, inner_score_tokens, slots,
                target=target,
            )
            out = speculative_rollout_score_slots(
                target, proposer, chunk, slots_per_block=slots,
                temperature=temperature, generator=generator, stats=stats,
            )
        if collect_rolled:
            rolled.append(out["rolled"].cpu())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    summary = stats.as_dict()
    summary["seconds"] = elapsed
    summary["windows"] = int(windows.shape[0])
    summary["windows_per_sec"] = windows.shape[0] / elapsed
    summary["batch"] = batch
    summary["slots_per_block"] = slots
    summary["config"] = name
    summary["temperature"] = temperature
    return summary, (torch.cat(rolled) if collect_rolled and rolled else None)


def mode_speed(args, target, models, tables, windows, device, results):
    table = []
    for temperature in args.temperatures:
      for batch in args.batch_sizes:
        # Time whole batches, enough of them that a run lasts ~10 s at every
        # batch size (a window is 414 sequential forwards, so batch 1 is already
        # seconds per window and batch 64 is one pass).
        n_windows = min(args.speed_windows, batch * max(1, -(-args.speed_min_windows // batch)))
        for name in (args.speed_configs or args.configs):
            slot_list = [0] if name == "baseline" else args.slots
            for slots in slot_list:
                summary, _ = run_config(
                    name, target, models, tables, windows[:n_windows], batch,
                    slots, temperature, device, args.inner_score_tokens, seed=args.seed,
                )
                table.append(summary)
                print(
                    f"  speed T={temperature} batch={batch:3d} {name:18s} gamma={slots} "
                    f"{summary['windows_per_sec']:7.3f} win/s "
                    f"tok/T-fwd {summary['tokens_per_target_forward']:.3f} "
                    f"accept {summary.get('acceptance_target', float('nan')):.3f}",
                    flush=True,
                )
                results["speed"] = table
                _dump(args, results)
    return results


def mode_accept(args, target, models, tables, windows, device, results):
    """Acceptance/efficiency at batch 1 (per-sequence, no shared-cache penalty)."""
    table = []
    for temperature in args.temperatures:
      for name in args.configs:
        if name == "baseline":
            continue
        for slots in args.slots:
            summary, _ = run_config(
                name, target, models, tables, windows[: args.accept_windows], 1, slots,
                temperature, device, args.inner_score_tokens, seed=args.seed,
            )
            table.append(summary)
            print(
                f"  accept T={temperature} {name:18s} gamma={slots} "
                f"accept={summary.get('acceptance_target', float('nan')):.3f} "
                f"(on {summary.get('acceptance_target_onset', float('nan')):.3f} "
                f"du {summary.get('acceptance_target_duration', float('nan')):.3f} "
                f"pi {summary.get('acceptance_target_pitch', float('nan')):.3f}) "
                f"tok/T-fwd {summary['tokens_per_target_forward']:.3f}",
                flush=True,
            )
            results["accept"] = table
            _dump(args, results)
    return results


# ---------------------------------------------------------------------------
# exact
# ---------------------------------------------------------------------------


def mode_exact(args, target, models, tables, windows, device, results):
    out = {}
    length = windows.shape[1]
    positions = score_token_positions(length, device=device)
    roles = score_token_roles(positions)

    # ---- float-noise control -------------------------------------------
    # A greedy rollout is chaotic (one flipped argmax cascades), and batched
    # matmuls do not reduce in a fixed order, so the BASELINE disagrees with
    # itself when the batch shape changes.  Measure that first: it is the floor
    # below which no "bit-identical" claim about speculative decoding can go.
    sub = windows[: args.exact_windows].to(device)
    big = rollout_score_slots(
        target, sub, temperature=0.0, collect_logprobs=False, collect_gt_ce=False
    )["rolled"]
    small = []
    for i in range(0, sub.shape[0], args.exact_control_batch):
        small.append(
            rollout_score_slots(
                target, sub[i : i + args.exact_control_batch], temperature=0.0,
                collect_logprobs=False, collect_gt_ce=False,
            )["rolled"]
        )
    small = torch.cat(small)
    identical = (big[:, positions] == small[:, positions]).all(dim=1)
    out["control_baseline_batch_reshape"] = {
        "windows": int(sub.shape[0]),
        "batch_a": int(sub.shape[0]),
        "batch_b": args.exact_control_batch,
        "identical_windows": float(identical.float().mean()),
        "identical_tokens": float(
            (big[:, positions] == small[:, positions]).float().mean()
        ),
    }
    print(f"  exact CONTROL (baseline vs itself, batch reshaped): "
          f"{out['control_baseline_batch_reshape']['identical_windows'] * 100:.1f}% windows, "
          f"{out['control_baseline_batch_reshape']['identical_tokens'] * 100:.2f}% tokens identical",
          flush=True)

    # ---- numerical noise floor -----------------------------------------
    # `big` IS the incremental (1-token-forward) greedy argmax at every score
    # position, by construction.  Re-deriving the same argmaxes from a single
    # teacher-forced forward over the finished sequence changes only the GEMM
    # shapes, so any disagreement is pure floating-point non-associativity --
    # the same effect a speculative chunk forward is subject to.
    with torch.no_grad():
        tf_logits = target(big, use_cache=False).logits
    flips, gaps, flip_gaps = 0, [], []
    total = 0
    for i, pos in enumerate(positions.tolist()):
        column = constrain_score_token_logits(tf_logits[:, pos - 1, :].float(), pos % 3)
        top2 = column.topk(2, dim=-1)
        gap = (top2.values[:, 0] - top2.values[:, 1])
        flip = top2.indices[:, 0] != big[:, pos]
        flips += int(flip.sum())
        total += flip.numel()
        gaps.append(gap.cpu())
        if bool(flip.any()):
            flip_gaps.append(gap[flip].cpu())
    gaps = torch.cat(gaps)
    flip_gaps = torch.cat(flip_gaps) if flip_gaps else torch.zeros(0)
    out["numerical_noise_floor"] = {
        "argmax_flip_rate_incremental_vs_single_forward": flips / max(total, 1),
        "positions": total,
        "median_top2_gap": float(gaps.median()),
        "frac_top2_gap_below_1e-3": float((gaps < 1e-3).float().mean()),
        "median_top2_gap_at_flips": float(flip_gaps.median()) if flip_gaps.numel() else None,
        "max_top2_gap_at_flips": float(flip_gaps.max()) if flip_gaps.numel() else None,
    }
    print(f"  exact NOISE FLOOR: argmax flips between the incremental and the "
          f"single-forward path at {out['numerical_noise_floor']['argmax_flip_rate_incremental_vs_single_forward'] * 100:.3f}% "
          f"of score positions; median top1-top2 logit gap "
          f"{out['numerical_noise_floor']['median_top2_gap']:.4f}, at flips "
          f"{out['numerical_noise_floor']['median_top2_gap_at_flips']}", flush=True)

    # ---- greedy: speculative vs baseline --------------------------------
    greedy = {}
    for name in args.configs:
        if name == "baseline":
            continue
        _, rolled = run_config(
            name, target, models, tables, windows[: args.exact_windows],
            args.exact_windows, args.slots[0], 0.0, device, args.inner_score_tokens,
            seed=args.seed, collect_rolled=True,
        )
        rolled = rolled.to(device)
        same_tokens = (big[:, positions] == rolled[:, positions]).float()
        # Where a window first diverges, how close was the decision?  A greedy
        # rollout is chaotic, so one flipped near-tie rewrites everything after
        # it; the gap AT THE FIRST divergence is what says whether the flip was
        # a real disagreement or a last-bit rounding difference.
        first_gaps = []
        differs = same_tokens < 1
        for row in range(rolled.shape[0]):
            idx = torch.nonzero(differs[row])
            if idx.numel() == 0:
                continue
            pos = int(positions[int(idx[0])])
            column = constrain_score_token_logits(tf_logits[row, pos - 1, :].float(), pos % 3)
            top2 = column.topk(2)
            first_gaps.append(float(top2.values[0] - top2.values[1]))
        greedy[name] = {
            "identical_windows": float(same_tokens.all(dim=1).float().mean()),
            "identical_tokens": float(same_tokens.mean()),
            "n_diverging_windows": len(first_gaps),
            "median_top2_gap_at_first_divergence": float(np.median(first_gaps))
            if first_gaps else None,
            "max_top2_gap_at_first_divergence": float(np.max(first_gaps)) if first_gaps else None,
        }
        print(f"  exact GREEDY {name:18s} "
              f"{greedy[name]['identical_windows'] * 100:.1f}% windows, "
              f"{greedy[name]['identical_tokens'] * 100:.2f}% tokens identical to baseline; "
              f"median top1-top2 gap at first divergence "
              f"{greedy[name]['median_top2_gap_at_first_divergence']}",
              flush=True)
    out["greedy"] = greedy
    results["exact"] = out
    _dump(args, results)

    return results


def mode_dist(args, target, models, tables, windows, device, results):
    """Distributional exactness at temperature 1, with a valid i.i.d. design.

    A rollout is ONE dependent trajectory: its 414 tokens are strongly
    correlated (a wrong tempo at slot 3 shifts every later onset), so pooling
    tokens across positions and treating them as independent draws makes any
    frequency test wildly over-powered -- it rejects two runs of the *same*
    sampler.  The valid unit of replication is a whole rollout.

    So: hold a handful of windows fixed and draw `--dist_replicates` INDEPENDENT
    rollouts of each from the baseline and from each speculative configuration.
    Within one window, the token at a fixed position is then an i.i.d. draw from
    the marginal the two samplers are supposed to share, and each rollout's
    role accuracy is an i.i.d. scalar.  Both are tested; under the null the
    p-values should look uniform, not all-large and not concentrated near 0.
    """
    length = windows.shape[1]
    positions = score_token_positions(length, device=device)
    roles = score_token_roles(positions)
    probe_positions = [int(positions[i]) for i in args.dist_probe_indices if i < len(positions)]

    out = {"windows": args.dist_replicate_windows, "replicates": args.dist_replicates,
           "probe_positions": probe_positions, "per_window": []}
    configs = [c for c in (args.dist_configs or args.configs) if c != "baseline"]

    for w in range(args.dist_replicate_windows):
        window = windows[w : w + 1]
        replicated = window.expand(args.dist_replicates, -1).contiguous()
        truth = window.to(device)

        torch.manual_seed(4242 + w)
        base_rolls = []
        for i in range(0, args.dist_replicates, args.dist_batch):
            chunk = replicated[i : i + args.dist_batch].to(device)
            base_rolls.append(
                rollout_score_slots(
                    target, chunk, temperature=1.0,
                    collect_logprobs=False, collect_gt_ce=False,
                )["rolled"].cpu()
            )
        base_rolls = torch.cat(base_rolls).to(device)
        base_acc = role_accuracy(
            base_rolls, truth.expand_as(base_rolls), positions, roles,
            torch.ones_like(base_rolls[:, positions], dtype=torch.bool),
        )

        # Appended BEFORE the config loop, and re-dumped after every config:
        # this mode keeps getting preempted mid-window on the shared partition
        # (jobs 463991 on zabih-compute-01, 466426 on nikola-compute-18), and a
        # per-window dump threw away every completed config of the window in
        # progress.  Each config costs a full replicate sweep, so that is real
        # GPU time, not bookkeeping.
        entry = {"window": w, "configs": {}}
        out["per_window"].append(entry)
        for name in configs:
            _, rolls = run_config(
                name, target, models, tables, replicated, args.dist_batch, args.slots[0],
                1.0, device, args.inner_score_tokens, seed=args.seed + 100 + w,
                collect_rolled=True,
            )
            rolls = rolls.to(device)
            spec_acc = role_accuracy(
                rolls, truth.expand_as(rolls), positions, roles,
                torch.ones_like(rolls[:, positions], dtype=torch.bool),
            )
            record = {"position_chi2": {}, "accuracy_ks": {}}
            for pos in probe_positions:
                stat, dof, pvalue = _marginal_chi2(base_rolls[:, pos], rolls[:, pos])
                record["position_chi2"][pos] = {"chi2": stat, "dof": dof, "p": pvalue}
            for role in range(3):
                ks = scipy_stats.ks_2samp(base_acc[:, role], spec_acc[:, role])
                record["accuracy_ks"][TOKEN_TYPE_NAMES[role]] = {
                    "baseline_mean": float(base_acc[:, role].mean()),
                    "spec_mean": float(spec_acc[:, role].mean()),
                    "ks_statistic": float(ks.statistic),
                    "p": float(ks.pvalue),
                }
            entry["configs"][name] = record
            ps = [v["p"] for v in record["position_chi2"].values()]
            ps += [v["p"] for v in record["accuracy_ks"].values()]
            print(f"  dist window {w} {name:16s} min p={min(ps):.4f}  "
                  f"acc onset {record['accuracy_ks']['onset']['baseline_mean']:.4f}/"
                  f"{record['accuracy_ks']['onset']['spec_mean']:.4f} "
                  f"dur {record['accuracy_ks']['duration']['baseline_mean']:.4f}/"
                  f"{record['accuracy_ks']['duration']['spec_mean']:.4f} "
                  f"pitch {record['accuracy_ks']['pitch']['baseline_mean']:.4f}/"
                  f"{record['accuracy_ks']['pitch']['spec_mean']:.4f}", flush=True)
            out["pooled"] = _dist_pooled(out, configs)
            results["distributional"] = out
            _dump(args, results)

    for name, stats in _dist_pooled(out, configs).items():
        print(f"  dist POOLED {name:16s} n={stats['n_tests']} min_p={stats['min_p']:.4f} "
              f"mean_p={stats['mean_p']:.3f} frac(p<0.05)={stats['frac_below_0.05']:.3f} "
              f"holm_reject={str(stats['holm_reject']):5s} "
              f"skewed_low_p={stats['skewed_low_p']:.3g} "
              f"(uniformity KS p={stats['uniformity_ks_p']:.4f} -- see _dist_pooled docstring, "
              f"compare against the `self` control before believing it)", flush=True)
    out["pooled"] = _dist_pooled(out, configs)
    results["distributional"] = out
    _dump(args, results)
    return results


def _dist_pooled(out, configs):
    """Pool p-values over whatever windows/configs have completed so far.

    Tolerant of a partial run: a preempted job still leaves a usable summary,
    it is just based on fewer tests (`n_tests` says how many).

    READ `uniformity_ks_p` ONLY NEXT TO THE `self` CONTROL.  The tempting
    summary is "under the null these p-values are Uniform(0,1), so KS them
    against U(0,1)" -- and that summary is WRONG here, which the `self` config
    proves rather than argues.  With draft = target, acceptance is exactly
    1.000, so `self` samples from the target's own constrained distribution by
    construction; it is exact or nothing is.  It fails the uniformity test at
    p=0.0014, the same as every other config.  A meta-test that rejects a
    sampler that cannot be wrong is measuring itself, not the sampler.

    The cause is that neither input test is continuous.  The per-position
    chi-square pools sparse categories, and 7-12 of 36 tests end up with all
    mass in one bin and return p = exactly 1.000; `ks_2samp` on a role accuracy
    over 138 slots is a lattice statistic with heavy ties, and is conservative.
    Both push p-values UP, so the pooled set is skewed high (mean 0.61-0.68) --
    the opposite of a real mismatch, which piles p-values up near 0.  Hence the
    one-sided fields: `skewed_low_p` is the one that would be small if the
    sampler were actually wrong, and it is ~1 for every config measured.
    The honest per-test verdict is `holm_reject` (multiplicity-corrected).
    """
    pooled = {}
    for name in configs:
        ps = []
        for entry in out["per_window"]:
            record = entry["configs"].get(name)
            if record is None:
                continue
            ps += [v["p"] for v in record["position_chi2"].values()]
            ps += [v["p"] for v in record["accuracy_ks"].values()]
        ps = [p for p in ps if p == p]
        if not ps:
            continue
        arr = np.sort(np.array(ps))
        n = len(arr)
        pooled[name] = {
            "n_tests": n,
            "min_p": float(arr[0]),
            "mean_p": float(arr.mean()),
            "frac_below_0.05": float(np.mean(arr < 0.05)),
            "uniformity_ks_p": float(scipy_stats.kstest(arr, "uniform").pvalue),
            # scipy: alternative="greater" <=> empirical CDF above uniform
            # <=> p-values bunched LOW <=> the mismatch signature.
            "skewed_low_p": float(
                scipy_stats.kstest(arr, "uniform", alternative="greater").pvalue),
            "skewed_high_p": float(
                scipy_stats.kstest(arr, "uniform", alternative="less").pvalue),
            "holm_reject": bool(any(arr[i] < 0.05 / (n - i) for i in range(n))),
        }
    return pooled


def _marginal_chi2(a, b, min_expected=5):
    """Two-sample chi-square on the marginal token distribution at one position.

    Both arguments are (n,) tokens drawn independently at the SAME position of
    the SAME window, so this is a legitimate homogeneity test.  Categories with
    too little expected mass are pooled into one "other" bin.
    """
    a = a.cpu().numpy()
    b = b.cpu().numpy()
    values = np.unique(np.concatenate([a, b]))
    index = {int(v): i for i, v in enumerate(values)}
    ca = np.zeros(len(values))
    cb = np.zeros(len(values))
    for value, count in zip(*np.unique(a, return_counts=True)):
        ca[index[int(value)]] = count
    for value, count in zip(*np.unique(b, return_counts=True)):
        cb[index[int(value)]] = count
    total_a, total_b = ca.sum(), cb.sum()
    shared = ca + cb
    keep = (shared * total_a / (total_a + total_b) >= min_expected) & (
        shared * total_b / (total_a + total_b) >= min_expected
    )
    ca = np.concatenate([ca[keep], [ca[~keep].sum()]])
    cb = np.concatenate([cb[keep], [cb[~keep].sum()]])
    shared = ca + cb
    expected_a = shared * total_a / (total_a + total_b)
    expected_b = shared * total_b / (total_a + total_b)
    mask = (expected_a > 0) & (expected_b > 0)
    stat = float(
        ((ca[mask] - expected_a[mask]) ** 2 / expected_a[mask]).sum()
        + ((cb[mask] - expected_b[mask]) ** 2 / expected_b[mask]).sum()
    )
    dof = max(int(mask.sum()) - 1, 1)
    return stat, dof, float(scipy_stats.chi2.sf(stat, dof))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _dump(args, results):
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_checkpoint", default="run_paper_split_v2/checkpoint-2500")
    parser.add_argument("--drafts", default="L4=run_draft_L4/final,L2=run_draft_L2/final")
    parser.add_argument("--configs", default="baseline,ngram,L4,L2,staged:L4,staged:L2")
    parser.add_argument("--speed_configs", default="", help="override --configs for `speed`")
    parser.add_argument("--dist_configs", default="", help="override --configs for `dist`")
    parser.add_argument("--slots", default="1,2,4")
    parser.add_argument("--batch_sizes", default="1,4,16,64")
    parser.add_argument("--inner_score_tokens", type=int, default=0,
                        help="score tokens per inner (D2->D1) block; 0 = whole outer block")
    parser.add_argument("--temperatures", default="0,1",
                        help="decode temperatures to sweep (0 = greedy)")
    parser.add_argument("--eval_files", default="data/val_paper.txt")
    parser.add_argument("--eval_stride", type=int, default=1)
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--speed_windows", type=int, default=64)
    parser.add_argument("--speed_min_windows", type=int, default=4)
    parser.add_argument("--accept_windows", type=int, default=16)
    parser.add_argument("--exact_windows", type=int, default=32)
    parser.add_argument("--exact_control_batch", type=int, default=8)
    parser.add_argument("--dist_replicate_windows", type=int, default=4,
                        help="windows held fixed for the replicated i.i.d. test")
    parser.add_argument("--dist_replicates", type=int, default=192,
                        help="independent rollouts per window per sampler")
    parser.add_argument("--dist_probe_indices", default="0,1,2,3,4,5,30,120,300",
                        help="score-token indices whose marginal is chi-square tested")
    parser.add_argument("--dist_batch", type=int, default=32)
    parser.add_argument("--ngram_windows", type=int, default=2048)
    parser.add_argument("--ngram_batch", type=int, default=32)
    parser.add_argument("--ngram_tables", default="run_draft_ngram/tables.pt")
    parser.add_argument("--ngram_fit_files", default="data/train_paper.txt")
    parser.add_argument("--ngram_top_m", type=int, default=16)
    parser.add_argument("--micro_iters", type=int, default=30)
    parser.add_argument("--micro_chunk", type=int, default=13)
    parser.add_argument("--mode", default="all")
    parser.add_argument("--output", default="bench/results_speculative.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.slots = [int(s) for s in args.slots.split(",")]
    args.batch_sizes = [int(s) for s in args.batch_sizes.split(",")]
    args.configs = [s for s in args.configs.split(",") if s]
    args.temperatures = [float(t) for t in args.temperatures.split(",")]
    args.speed_configs = [c for c in args.speed_configs.split(",") if c]
    args.dist_configs = [c for c in args.dist_configs.split(",") if c]
    args.dist_probe_indices = [int(i) for i in args.dist_probe_indices.split(",")]
    args.micro_configs = [(2, 256), (4, 256), (4, 512), (6, 1024)]

    target, device = load_model(args.target_checkpoint)
    target.eval()
    print(f"torch {torch.__version__}", flush=True)
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"GPU: {gpu}", flush=True)

    models = {}
    for spec in args.drafts.split(","):
        if not spec:
            continue
        name, path = spec.split("=", 1)
        if path.startswith("build:"):
            # Untrained shallow truncation of the target -- the ablation that
            # shows how much of the acceptance rate the distillation buys.
            models[name] = build_shallow_draft(target, int(path.split(":")[1])).to(device).eval()
            print(f"  draft {name}: untrained truncation, {path}", flush=True)
            continue
        if not Path(path).exists():
            print(f"  draft {name}: {path} MISSING -- skipping", flush=True)
            continue
        models[name] = load_draft(path, device)
        print(f"  draft {name}: {path} "
              f"({models[name].config.n_layer} layers, "
              f"{sum(p.numel() for p in models[name].parameters()) / 1e6:.1f}M)", flush=True)
    def _available(names):
        return [
            c for c in names
            if c in ("baseline", "ngram", "self") or c.split(":")[-1] in models
        ]

    args.configs = _available(args.configs)
    args.speed_configs = _available(args.speed_configs)
    args.dist_configs = _available(args.dist_configs)

    results = {
        "gpu": gpu,
        "torch": torch.__version__,
        "target_checkpoint": args.target_checkpoint,
        "args": vars(args),
    }

    tables = None
    if any(c == "ngram" or c.startswith("staged:") for c in args.configs):
        path = Path(args.ngram_tables)
        if path.exists():
            tables = load_tables(path, device)
            print(f"  ngram tables loaded from {path}", flush=True)
        else:
            print(f"  fitting ngram tables on {args.ngram_windows} target rollouts...", flush=True)
            fit_windows = load_windows(
                args.ngram_fit_files, args.ngram_windows, seed=args.seed + 5, stride=997
            )
            pool = []
            t0 = time.time()
            for i in range(0, fit_windows.shape[0], args.ngram_batch):
                chunk = fit_windows[i : i + args.ngram_batch].to(device)
                temperature = 0.0 if (i // args.ngram_batch) % 2 == 0 else 1.0
                pool.append(
                    rollout_score_slots(
                        target, chunk, temperature=temperature,
                        collect_logprobs=False, collect_gt_ce=False,
                    )["rolled"].cpu()
                )
                if (i // args.ngram_batch) % 10 == 0:
                    print(f"    rollout {i}/{fit_windows.shape[0]} ({time.time() - t0:.0f}s)",
                          flush=True)
            tables = fit_ngram_tables(torch.cat(pool), top_m=args.ngram_top_m, device=device)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_tables(tables, path)
            print(f"  ngram tables saved to {path}", flush=True)

    windows = load_windows(args.eval_files, args.windows, seed=args.seed, stride=args.eval_stride)
    print(f"eval windows: {tuple(windows.shape)}", flush=True)

    modes = (
        ["micro", "accept", "speed", "exact", "dist"]
        if args.mode == "all"
        else args.mode.split(",")
    )
    for mode in modes:
        print(f"=== mode {mode} ===", flush=True)
        if mode == "micro":
            mode_micro(args, target, models, device, results, tables, windows)
        elif mode == "accept":
            mode_accept(args, target, models, tables, windows, device, results)
        elif mode == "speed":
            mode_speed(args, target, models, tables, windows, device, results)
        elif mode == "exact":
            mode_exact(args, target, models, tables, windows, device, results)
        elif mode == "dist":
            mode_dist(args, target, models, tables, windows, device, results)
        _dump(args, results)

    _dump(args, results)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
