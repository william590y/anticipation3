#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into its base model and save a plain HF checkpoint.

The eval chain (`scripts/fullsong_ours.slurm` -> `full_song_rollout.py` ->
`evaluate_muster_asap.load_model`) needs a NORMAL HuggingFace directory
(config.json + model.safetensors); a PEFT checkpoint only contains the adapter.
This script does the conversion:

    base = AutoModelForCausalLM.from_pretrained(<base ckpt>)   # + vocab resize
    model = PeftModel.from_pretrained(base, <adapter dir>).merge_and_unload()
    model.save_pretrained(<out dir>)

For a Bayesian LoRA adapter (see bayes_lora.py) the merge uses the posterior
MEAN: `lora_B.weight` *is* mu, and `merge_and_unload()` reads exactly that
weight, so the merged checkpoint is the mean model. The adapter's extra `rho`
tensors are simply unused here (PEFT reports them as unexpected keys); we load
the model in eval() mode as well, which is the same convention the sampling
forward uses to switch off noise.

Usage:
    python merge_lora.py --base run_paper_split_v2/checkpoint-2500 \
        --adapter run_bayes_lora_r512/final \
        --output run_bayes_lora_r512/merged
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from anticipation.vocab import VOCAB_SIZE


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="run_paper_split_v2/checkpoint-2500",
                    help="Base HF checkpoint the adapter was trained on top of.")
    ap.add_argument("--adapter", required=True,
                    help="PEFT adapter directory (adapter_config.json + adapter_model.safetensors).")
    ap.add_argument("--output", required=True,
                    help="Destination directory for the merged, plain HF checkpoint.")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                    help="dtype to load/merge/save in (default float32, matching training).")
    return ap.parse_args()


def main():
    args = parse_args()
    adapter = Path(args.adapter)
    out_dir = Path(args.output)

    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"No adapter_config.json in {adapter}")

    dtype = getattr(torch, args.dtype)
    print(f"Loading base model from {args.base} ({args.dtype}) ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, local_files_only=True, dtype=dtype
    )
    if base.config.vocab_size != VOCAB_SIZE:
        print(f"Resizing base embeddings {base.config.vocab_size} -> {VOCAB_SIZE}")
        base.resize_token_embeddings(VOCAB_SIZE)

    from peft import PeftModel

    print(f"Loading adapter from {adapter} ...")
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    # eval() == posterior mean for a Bayesian adapter; also disables LoRA dropout.
    model.eval()

    print("Merging adapter into the base weights (posterior mean) ...")
    merged = model.merge_and_unload()
    merged.eval()

    n_params = sum(p.numel() for p in merged.parameters())
    print(f"Merged model: {type(merged).__name__}, {n_params:,} parameters, "
          f"vocab_size={merged.config.vocab_size}")

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    print(f"Saved merged checkpoint to {out_dir}")

    # The eval chain calls AutoModelForCausalLM.from_pretrained on this directory;
    # fail loudly here rather than inside an 8-way decode array.
    if not (out_dir / "config.json").exists():
        raise RuntimeError(f"config.json missing from {out_dir}")
    weight_files = list(out_dir.glob("model*.safetensors")) + list(out_dir.glob("pytorch_model*.bin"))
    if not weight_files:
        raise RuntimeError(f"No weight file written to {out_dir}")
    print("Verifying the merged checkpoint reloads ...")
    reloaded = AutoModelForCausalLM.from_pretrained(str(out_dir), local_files_only=True)
    assert reloaded.config.vocab_size == VOCAB_SIZE, reloaded.config.vocab_size
    print("MERGE_OK")


if __name__ == "__main__":
    main()
