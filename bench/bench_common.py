"""Shared fixtures for the inference benchmark + correctness gate.

Everything in `bench/` measures exactly one thing: the cost of the KV-cached
autoregressive score decode (`onpolicy_rollout.rollout_score_slots` /
`inference.batched_autoregressive_generate_score`), which is the hot path for
every RL arm, for `nbest/generate_nbest.py`, and for `train.py`'s validation.

The window fixture must be *identical* across runs and across processes, since
a greedy rollout is chaotic (one flipped argmax cascades through the rest of the
window -- the same reason `onpolicy_rollout.evaluate_autoregressive` shards whole
batches and never items). Sampling windows by evenly spaced byte offsets gives a
fixed, seek-only fixture that does not depend on scanning a 1.5 GB token file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CHECKPOINT = "run_paper_split_v2/checkpoint-2500"
DEFAULT_TOKEN_FILE = "data/val_paper.txt"
PACKED_LENGTH = 1020


def load_bench_windows(token_file=DEFAULT_TOKEN_FILE, count=256, expect_length=PACKED_LENGTH):
    """`count` packed windows read at evenly spaced byte offsets -- deterministic.

    Consecutive lines of a token file are overlapping sliding windows of one
    performance, so taking the first N lines would benchmark (and gate) N nearly
    identical windows. Spreading the offsets over the whole file costs one seek
    per window instead of a full scan.
    """
    path = Path(token_file)
    if not path.is_absolute():
        path = REPO_ROOT / path
    size = path.stat().st_size
    windows = []
    with path.open("rb") as handle:
        for i in range(count):
            handle.seek((size // max(count, 1)) * i)
            if i > 0:
                handle.readline()  # discard the partial line we landed inside
            line = handle.readline()
            if not line:  # ran off the end: restart from the top
                handle.seek(0)
                line = handle.readline()
            text = line.decode("utf-8").split("|")[0]
            tokens = [int(tok) for tok in text.split()]
            if expect_length and len(tokens) != expect_length:
                continue
            windows.append(tokens)
    if not windows:
        raise RuntimeError(f"No usable windows read from {path}")
    return torch.tensor(windows, dtype=torch.long)


def load_bench_model(checkpoint=DEFAULT_CHECKPOINT, attn_implementation=None, dtype=None):
    """`evaluate_muster.load_model` (the canonical loader), with optional overrides.

    `attn_implementation=None` and `dtype=None` reproduce production exactly:
    fp32 weights, whatever attention backend transformers picks by default.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from evaluate_muster import load_model

    path = Path(checkpoint)
    if not path.is_absolute():
        path = REPO_ROOT / path
    model, device = load_model(str(path))
    if attn_implementation is not None:
        model.set_attn_implementation(attn_implementation)
    if dtype is not None:
        model = model.to(dtype)
    model.eval()
    return model, device


def describe_model(model):
    """One-line provenance for a results table: dtype + attention backend."""
    return {
        "dtype": str(next(model.parameters()).dtype),
        "attn_implementation": getattr(model.config, "_attn_implementation", "?"),
        "n_layer": model.config.n_layer,
        "n_embd": model.config.n_embd,
        "vocab_size": model.config.vocab_size,
    }
