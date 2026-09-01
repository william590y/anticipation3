"""Benchmark the KV-cached autoregressive score decode.

    python bench/bench_decode.py --variants baseline --batch-sizes 8,32,96,198

Measures wall time of a full rollout over the 138 body score slots of a packed
1020-token window (414 sequential cached forwards + one 192-token prefill) and
reports windows/sec and ms per decode step at each batch size.

Two accounting notes, because they decide what the numbers mean:

* "step" = one cached forward. A slot costs three: onset, duration, then the
  pitch token batched with the three teacher-forced control tokens that follow
  it. 138 slots -> 414 steps, matching `onpolicy_rollout`'s docstring.
* the RL arms call the rollout with `collect_gt_ce=True` (they need the
  autoregressive validation loss) but pure inference -- `inference.py`,
  `nbest/generate_nbest.py`'s greedy pass, `train.py` validation's accuracy
  pass -- does not. The two have materially different cost (gt_ce forces a
  full-vocab cross-entropy at every step), so both are benchmarked separately
  rather than averaged into one "decode" number.

`--probe` skips the rollout and times a single cached forward at a sweep of KV
cache lengths, which is what distinguishes a launch-bound decode (flat in cache
length and in batch) from a KV-bandwidth-bound one (linear in both).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from bench_common import (  # noqa: E402  (path bootstrap lives in bench_common)
    DEFAULT_CHECKPOINT,
    DEFAULT_TOKEN_FILE,
    describe_model,
    load_bench_model,
    load_bench_windows,
)

from onpolicy_rollout import body_score_slot_starts, rollout_score_slots  # noqa: E402


def batch_sizes_first(spec):
    return int(spec.split(",")[0])


def steps_per_window(length):
    """Cached forwards per window: 3 per body score slot (see module docstring)."""
    return 3 * len(body_score_slot_starts(length))


def make_batch(windows, batch_size, device):
    """A (batch_size, length) batch, tiling the fixed window pool if needed."""
    if batch_size <= windows.shape[0]:
        rows = windows[:batch_size]
    else:
        repeats = (batch_size + windows.shape[0] - 1) // windows.shape[0]
        rows = windows.repeat(repeats, 1)[:batch_size]
    return rows.to(device)


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------


def _baseline(collect_gt_ce):
    def run(model, batch, **_):
        return rollout_score_slots(
            model,
            batch,
            temperature=0.0,
            constrain=True,
            collect_logprobs=False,
            collect_gt_ce=collect_gt_ce,
        )

    return run


def _fast(collect_gt_ce, **fast_kwargs):
    # One decoder (hence one StaticCache allocation and one CUDA-graph capture)
    # per batch width, reused across repeats -- exactly how a caller would hold
    # it across a validation loop. Rebuilding it per rollout would charge the
    # ablation for a several-second graph capture that production pays once.
    decoders = {}

    def run(model, batch, decoder=None, **_):
        from anticipation.fast_decode import StaticKVDecoder, rollout_score_slots_fast

        if fast_kwargs.get("static_cache", True):
            key = batch.shape
            if key not in decoders:
                # Evict first: a StaticCache is ~0.2 GiB per window, so holding
                # the batch-8/32/96 decoders alive while building the batch-198
                # one costs 67 GiB and OOMs a 48 GiB card on the last row of the
                # sweep -- which would look like "the fast path does not scale".
                decoders.clear()
                torch.cuda.empty_cache()
                # Decoder-shaped options must reach the constructor: a decoder
                # passed into rollout_score_slots_fast wins over its own kwargs,
                # so building a plain one would silently grade `buckets=` and
                # `prebuilt_mask=` variants as the default configuration.
                decoders[key] = StaticKVDecoder(
                    model,
                    batch.shape[0],
                    max_length=batch.shape[1],
                    prebuilt_mask=fast_kwargs.get("prebuilt_mask", True),
                    buckets=fast_kwargs.get("buckets"),
                )
            decoder = decoders[key]

        return rollout_score_slots_fast(
            model,
            batch,
            temperature=0.0,
            constrain=True,
            collect_logprobs=False,
            collect_gt_ce=collect_gt_ce,
            decoder=decoder,
            **fast_kwargs,
        )

    return run


def build_variant(spec):
    """Parse a variant spec: `name[:opt=val,...]`."""
    name, _, raw_opts = spec.partition(":")
    opts = {}
    for item in raw_opts.split(",") if raw_opts else []:
        if not item:
            continue
        key, _, value = item.partition("=")
        if value in ("", "true", "True"):
            opts[key] = True
        elif value in ("false", "False"):
            opts[key] = False
        else:
            try:
                opts[key] = int(value)
            except ValueError:
                opts[key] = value
    gt_ce = bool(opts.pop("gt_ce", False))
    if name == "baseline":
        return _baseline(gt_ce)
    if name == "fast":
        return _fast(gt_ce, **opts)
    raise ValueError(f"Unknown variant: {spec}")


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def time_variant(model, batch, runner, repeats, warmup, decoder=None):
    for _ in range(warmup):
        runner(model, batch, decoder=decoder)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        runner(model, batch, decoder=decoder)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return times, torch.cuda.max_memory_allocated()


def probe_forward(model, device, batch_sizes, cache_lengths, repeats=20):
    """Time one cached forward at a sweep of (batch, cache length) points.

    Diagnostic only: it answers "is the decode launch-bound or bandwidth-bound
    at our operating point", which decides whether CUDA graphs can help at all.
    """
    rows = []
    vocab = model.config.vocab_size
    warmup = 3
    for batch_size in batch_sizes:
        for cache_len in cache_lengths:
            # Every probe step appends to the same cache, so the run must fit
            # inside n_positions -- overrunning wpe is a device-side assert, not
            # a Python error, and it poisons every later config in the process.
            budget = model.config.n_positions - cache_len - warmup - 1
            steps = max(1, min(repeats, budget))
            if budget < 1:
                continue
            prefix = torch.randint(0, vocab, (batch_size, cache_len), device=device)
            token = torch.randint(0, vocab, (batch_size, 1), device=device)
            try:
                with torch.inference_mode():
                    primed = model(prefix, use_cache=True, logits_to_keep=1)
                    past = primed.past_key_values
                    for _ in range(warmup):
                        model(token, past_key_values=past, use_cache=True)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(steps):
                        model(token, past_key_values=past, use_cache=True)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) / steps
            except torch.cuda.OutOfMemoryError:
                # Expected past a point: a DynamicCache holds both the old and the
                # newly concatenated cache during every step, so the ceiling is
                # ~2x the resident cache. Record it rather than aborting the sweep.
                print(f"{batch_size:>6} {cache_len:>6}      OOM", flush=True)
                rows.append({"batch": batch_size, "cache_len": cache_len, "oom": True})
                del prefix, token
                torch.cuda.empty_cache()
                continue
            # KV bytes the step must read: 2 (K,V) * layers * batch * cache * embd.
            itemsize = next(model.parameters()).dtype.itemsize
            kv_bytes = 2 * model.config.n_layer * batch_size * cache_len * model.config.n_embd * itemsize
            rows.append(
                {
                    "batch": batch_size,
                    "cache_len": cache_len,
                    "ms": elapsed * 1e3,
                    "kv_gib": kv_bytes / 2**30,
                    "implied_gib_s": kv_bytes / 2**30 / elapsed,
                }
            )
            del primed, past
            torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--batch-sizes", default="8,32,96,198")
    parser.add_argument("--variants", default="baseline,baseline:gt_ce=true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--windows", type=int, default=64, help="size of the window pool")
    parser.add_argument("--attn", default=None, help="override attn_implementation")
    parser.add_argument("--dtype", default=None, choices=[None, "float32", "bfloat16", "float16"])
    parser.add_argument("--tf32", action="store_true", help="allow TF32 fp32 matmuls (changes numerics)")
    parser.add_argument("--compile-mode", default=None,
                        help="torch.compile mode for the transformer trunk "
                             "(default/reduce-overhead/max-autotune)")
    parser.add_argument("--sync-debug", action="store_true",
                        help="warn on every host-device sync during one rollout, then exit")
    parser.add_argument("--probe", action="store_true", help="run the forward-cost probe instead")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.dtype)
    model, device = load_bench_model(args.checkpoint, attn_implementation=args.attn, dtype=dtype)
    if args.compile_mode:
        # Compile the trunk, not GPT2LMHeadModel: the head is called on a
        # different number of positions on the prefill than on a decode step, so
        # compiling the wrapper just adds a recompile for no benefit. Both the
        # baseline (via model.__call__) and the fast path go through this module.
        model.transformer = torch.compile(model.transformer, mode=args.compile_mode)
    info = describe_model(model)
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    info["torch"] = torch.__version__
    info["tf32"] = torch.backends.cuda.matmul.allow_tf32
    info["compile_mode"] = args.compile_mode
    print(json.dumps(info, indent=2), flush=True)

    if args.sync_debug:
        # A launch-bound loop pays dearly for any implicit .item()/.cpu(); this
        # turns each one into a printed warning so they can be counted.
        windows = load_bench_windows(args.token_file, count=max(args.windows, 4))
        batch = make_batch(windows, batch_sizes_first(args.batch_sizes), device)
        runner = build_variant(_split_variant_specs(args.variants)[0])
        runner(model, batch)  # warm up outside the debug window
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("warn")
        runner(model, batch)
        torch.cuda.set_sync_debug_mode("default")
        print("sync-debug pass complete", flush=True)
        return

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    if args.probe:
        rows = probe_forward(model, device, batch_sizes, [192, 384, 606, 800, 960])
        print(f"\n{'batch':>6} {'cache':>6} {'ms/step':>9} {'KV GiB':>8} {'GiB/s':>9}")
        for row in rows:
            if row.get("oom"):
                print(f"{row['batch']:>6} {row['cache_len']:>6} {'OOM':>9}")
                continue
            print(
                f"{row['batch']:>6} {row['cache_len']:>6} {row['ms']:>9.2f} "
                f"{row['kv_gib']:>8.2f} {row['implied_gib_s']:>9.1f}"
            )
        if args.json:
            Path(args.json).write_text(json.dumps({"info": info, "probe": rows}, indent=2))
        return

    windows = load_bench_windows(args.token_file, count=args.windows)
    length = windows.shape[1]
    n_steps = steps_per_window(length)
    print(f"\nwindows={windows.shape[0]} length={length} steps/window={n_steps}", flush=True)

    results = []
    specs = _split_variant_specs(args.variants)

    for spec in specs:
        runner = build_variant(spec)
        for batch_size in batch_sizes:
            batch = make_batch(windows, batch_size, device)
            decoder = None
            try:
                times, peak = time_variant(model, batch, runner, args.repeats, args.warmup, decoder)
            except torch.cuda.OutOfMemoryError:
                print(f"{spec:<40} b={batch_size:<4} OOM", flush=True)
                torch.cuda.empty_cache()
                continue
            best = min(times)
            mean = sum(times) / len(times)
            row = {
                "variant": spec,
                "batch": batch_size,
                "best_s": best,
                "mean_s": mean,
                "times_s": times,
                "windows_per_s": batch_size / best,
                "ms_per_window": best * 1e3 / batch_size,
                "ms_per_step": best * 1e3 / n_steps,
                "peak_mem_gib": peak / 2**30,
            }
            results.append(row)
            print(
                f"{spec:<40} b={batch_size:<4} {best:8.3f}s  "
                f"{row['windows_per_s']:8.2f} win/s  {row['ms_per_step']:7.2f} ms/step  "
                f"peak {row['peak_mem_gib']:5.2f} GiB",
                flush=True,
            )
            del batch
            torch.cuda.empty_cache()

    print(f"\n{'variant':<40} {'batch':>5} {'s/rollout':>10} {'win/s':>9} {'ms/step':>8} {'peak GiB':>9}")
    for row in results:
        print(
            f"{row['variant']:<40} {row['batch']:>5} {row['best_s']:>10.3f} "
            f"{row['windows_per_s']:>9.2f} {row['ms_per_step']:>8.2f} {row['peak_mem_gib']:>9.2f}"
        )

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps({"info": info, "steps_per_window": n_steps, "results": results}, indent=2)
        )
        print(f"\nwrote {args.json}")


def _split_variant_specs(text):
    """Split a `--variants` string on commas that separate variants, not options.

    `fast:static_cache=true,cuda_graph=true` is one variant with two options;
    `baseline,fast` is two variants. A comma starts a new variant only when the
    token after it is not `key=value`.
    """
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
