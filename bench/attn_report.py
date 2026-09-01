"""Which attention backend the decode actually runs on, per dtype.

Deliverable-3 evidence. Three separate questions get conflated easily:

1. `attn_implementation` -- transformers' dispatch (`eager` / `sdpa` /
   `flash_attention_2`). `evaluate_muster.load_model` sets none, so the model
   runs on whatever transformers defaults to.
2. Which *SDPA backend* torch then picks (math / efficient / flash / cudnn).
   Flash and the memory-efficient kernels do not accept fp32, so an fp32 model
   silently gets the math backend no matter what.
3. Whether the choice changes the model's output. For this checkpoint it does:
   `config.scale_attn_by_inverse_layer_idx=True`, and transformers' GPT-2 never
   forwards that per-layer scale to the sdpa/flash paths (it passes no `scaling`
   argument, so those use the plain 1/sqrt(head_dim)). Only `eager` applies it.
   The model was trained under sdpa, so sdpa is the correct path and eager is a
   different model -- see the numbers this prints.
"""

from __future__ import annotations

import json

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from bench_common import DEFAULT_CHECKPOINT, load_bench_model  # noqa: E402

from anticipation.packed_sequence import ALTERNATING_START  # noqa: E402


def which_backend(dtype, batch=8, heads=16, cache=1020, head_dim=64):
    """Report which SDPA backends accept our decode shape at `dtype`."""
    device = "cuda"
    query = torch.randn(batch, heads, 1, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch, heads, cache, head_dim, device=device, dtype=dtype)
    value = torch.randn_like(key)
    mask = torch.ones(1, 1, 1, cache, dtype=torch.bool, device=device)
    accepted = {}
    for name, backend in [
        ("math", SDPBackend.MATH),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("cudnn", SDPBackend.CUDNN_ATTENTION),
    ]:
        for label, attn_mask in (("no_mask", None), ("bool_mask", mask)):
            try:
                with sdpa_kernel(backend):
                    torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, attn_mask=attn_mask
                    )
                accepted[f"{name}/{label}"] = "ok"
            except Exception as exc:  # noqa: BLE001
                accepted[f"{name}/{label}"] = f"{type(exc).__name__}"
    return accepted


def implementation_divergence(checkpoint):
    """Max |logit| difference between attention implementations on a real prefix."""
    model, device = load_bench_model(checkpoint)
    ids = torch.randint(0, model.config.vocab_size, (2, ALTERNATING_START), device=device)
    outputs = {}
    for impl in ("sdpa", "eager"):
        model.set_attn_implementation(impl)
        with torch.inference_mode():
            outputs[impl] = model(ids, use_cache=False).logits.float().clone()
    delta = (outputs["sdpa"] - outputs["eager"]).abs().max().item()
    scale = outputs["sdpa"].abs().max().item()
    return {
        "scale_attn_by_inverse_layer_idx": model.config.scale_attn_by_inverse_layer_idx,
        "max_abs_delta_sdpa_vs_eager": delta,
        "max_abs_logit": scale,
        "relative": delta / scale if scale else float("nan"),
    }


def main():
    try:
        import flash_attn  # noqa: F401

        flash = getattr(flash_attn, "__version__", "installed")
    except Exception as exc:  # noqa: BLE001
        flash = f"UNAVAILABLE: {type(exc).__name__}"
    report = {"flash_attn_package": flash, "gpu": torch.cuda.get_device_name(0)}
    for name, dtype in (("float32", torch.float32), ("float16", torch.float16),
                        ("bfloat16", torch.bfloat16)):
        report[f"sdpa_backends_{name}"] = which_backend(dtype)
    report["implementation_divergence"] = implementation_divergence(DEFAULT_CHECKPOINT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
