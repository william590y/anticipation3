"""Honest TensorRT evaluation for the score decode.

Installed for this evaluation (torch was NOT touched -- it is still the shared
env's 2.11.0+cu128, and every other job on this cluster depends on that):

    pip install "tensorrt-cu12==10.15.1.29" dllist
    pip install --no-deps "torch-tensorrt==2.11.0"

`tensorrt-cu12` rather than the `tensorrt` meta-package on purpose: the meta
package resolves to the **cu13** build, which would put a CUDA 13 runtime in the
same process as a cu128 torch.

The two things TRT would have to accelerate here are very different problems:

* **prefill** -- one dense (batch, 192) forward per window, no cache. A normal
  static-shape graph; TRT can have it.
* **decode** -- 414 forwards of one to four tokens each against a KV cache that
  grows from 192 to 1020. A TRT engine is *stateless*: any tensor it mutates has
  to be an engine input and output, so the whole KV cache would cross the engine
  boundary every step. At batch 96 that cache is ~19 GiB of fp32; moving it in
  and out 414 times is not a plausible decode path regardless of kernel quality.

So this script measures the prefill compile for real, and *attempts* the decode
compile so the failure mode is a captured error rather than an assumption.
Nothing here is reported unless it ran.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from bench_common import DEFAULT_CHECKPOINT, describe_model, load_bench_model  # noqa: E402

from anticipation.packed_sequence import ALTERNATING_START  # noqa: E402


def versions():
    info = {"torch": torch.__version__, "torch_cuda": torch.version.cuda}
    for name in ("tensorrt", "torch_tensorrt"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 - the point is to report why
            info[name] = f"UNAVAILABLE: {type(exc).__name__}: {exc}"
    return info


def timed(fn, repeats, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)
    return best


class PrefillModule(torch.nn.Module):
    """The cacheless prefill: token ids -> last hidden state.

    Wrapped as a plain module with tensor-in/tensor-out so it is traceable;
    `GPT2Model.forward` itself returns a dataclass carrying a `Cache` object,
    which no exporter accepts.

    The 4-D causal mask is built here and passed in, rather than letting
    transformers build it. `masking_utils.create_causal_mask` composes the mask
    with `torch.vmap`, and exporting through that raises "tensor may have
    escaped from inside a function being vmapped" -- which would make this
    measure a transformers/torch.export incompatibility instead of TensorRT.
    A pre-built 4-D mask hits `_preprocess_mask_arguments`' early exit and is
    used verbatim. This is the *cacheless* prefill, so it is unrelated to the
    KV-cache mask trap in CLAUDE.md, which is about 2-D chunk-shaped masks
    passed while `past_key_values` is live.
    """

    def __init__(self, model):
        super().__init__()
        self.transformer = model.transformer

    def forward(self, input_ids, causal_mask):
        return self.transformer(
            input_ids, attention_mask=causal_mask, use_cache=False
        ).last_hidden_state[:, -1, :]


def causal_mask_for(length, dtype, device):
    """(1, 1, L, L) additive causal mask, the form transformers passes to SDPA."""
    allowed = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    mask = torch.zeros(length, length, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.view(1, 1, length, length)


def evaluate_prefill(model, batch_size, repeats, results):
    import torch_tensorrt

    device = next(model.parameters()).device
    module = PrefillModule(model).eval()
    example = torch.randint(0, model.config.vocab_size, (batch_size, ALTERNATING_START), device=device)
    mask = causal_mask_for(ALTERNATING_START, next(model.parameters()).dtype, device)

    with torch.inference_mode():
        eager_ms = timed(lambda: module(example, mask), repeats) * 1e3
        reference = module(example, mask).clone()

    entry = {"batch": batch_size, "eager_ms": eager_ms}
    try:
        build_start = time.perf_counter()
        compiled = torch_tensorrt.compile(
            module,
            ir="dynamo",
            inputs=[example, mask],
            enabled_precisions={torch.float32},
            # Static shape on purpose: the prefill is always exactly
            # ALTERNATING_START tokens, so a dynamic profile would only cost
            # kernel-selection quality.
            min_block_size=1,
        )
        entry["build_s"] = time.perf_counter() - build_start
        with torch.inference_mode():
            trt_ms = timed(lambda: compiled(example, mask), repeats) * 1e3
            produced = compiled(example, mask)
        entry["trt_ms"] = trt_ms
        entry["speedup"] = eager_ms / trt_ms
        entry["max_abs_delta"] = float((produced.float() - reference.float()).abs().max().item())
        entry["bit_identical"] = bool(torch.equal(produced, reference))
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()[-2000:]
    results["prefill"].append(entry)
    return entry


def evaluate_decode_step(model, batch_size, max_length, repeats, results):
    """Attempt the thing that would actually matter: the cached decode step."""
    from transformers import StaticCache

    device = next(model.parameters()).device
    cache = StaticCache(config=model.config, max_cache_len=max_length)
    prompt = torch.randint(0, model.config.vocab_size, (batch_size, ALTERNATING_START), device=device)
    token = torch.randint(0, model.config.vocab_size, (batch_size, 1), device=device)

    with torch.inference_mode():
        model.transformer(
            prompt,
            past_key_values=cache,
            cache_position=torch.arange(ALTERNATING_START, device=device),
            use_cache=True,
        )
        position = torch.tensor([ALTERNATING_START], device=device)

        def step():
            return model.transformer(
                token, past_key_values=cache, cache_position=position, use_cache=True
            ).last_hidden_state

        eager_ms = timed(step, repeats) * 1e3

    entry = {"batch": batch_size, "eager_ms": eager_ms}
    try:
        # The realistic way TRT would be applied to a HuggingFace decode: the
        # torch.compile backend, which partitions the graph and leaves anything
        # it cannot convert in eager. Anything it refuses shows up as a graph
        # break rather than a hard failure, so this measures what TRT can
        # actually take here.
        compiled = torch.compile(model.transformer, backend="torch_tensorrt", dynamic=False)
        build_start = time.perf_counter()
        with torch.inference_mode():
            compiled(token, past_key_values=cache, cache_position=position, use_cache=True)
        entry["build_s"] = time.perf_counter() - build_start

        def compiled_step():
            return compiled(
                token, past_key_values=cache, cache_position=position, use_cache=True
            ).last_hidden_state

        with torch.inference_mode():
            trt_ms = timed(compiled_step, repeats) * 1e3
        entry["trt_ms"] = trt_ms
        entry["speedup"] = eager_ms / trt_ms
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["traceback"] = traceback.format_exc()[-3000:]
    results["decode_step"].append(entry)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-sizes", default="8,32")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=1020)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    info = versions()
    print(json.dumps(info, indent=2), flush=True)
    results = {"versions": info, "prefill": [], "decode_step": []}

    if "UNAVAILABLE" in str(info.get("torch_tensorrt")):
        print("torch_tensorrt unavailable; nothing to measure.")
        if args.json:
            Path(args.json).write_text(json.dumps(results, indent=2))
        return

    model, device = load_bench_model(args.checkpoint)
    results["model"] = describe_model(model)
    info["gpu"] = torch.cuda.get_device_name(0)

    for batch_size in [int(b) for b in args.batch_sizes.split(",")]:
        print(f"\n=== prefill, batch {batch_size} ===", flush=True)
        entry = evaluate_prefill(model, batch_size, args.repeats, results)
        print(json.dumps({k: v for k, v in entry.items() if k != "traceback"}, indent=2), flush=True)
        if "traceback" in entry:
            print(entry["traceback"], flush=True)
        torch.cuda.empty_cache()

        print(f"\n=== cached decode step, batch {batch_size} ===", flush=True)
        entry = evaluate_decode_step(model, batch_size, args.max_length, args.repeats, results)
        print(json.dumps({k: v for k, v in entry.items() if k != "traceback"}, indent=2), flush=True)
        if "traceback" in entry:
            print(entry["traceback"], flush=True)
        torch.cuda.empty_cache()

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
