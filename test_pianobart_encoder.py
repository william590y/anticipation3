#!/usr/bin/env python
"""CPU checks on the frozen PianoBART plan encoder.

Runs the real released checkpoint: the failure modes worth guarding here are
all silent ones, where a wrong load or a wrong grid still produces
plausible-looking features.

    python test_pianobart_encoder.py
"""
from __future__ import annotations

import sys

import torch

from pianobart_encoder import (
    BEATS_PER_BAR,
    BINS_PER_BEAT,
    OCTUPLE_FIELDS,
    POS_RESOLUTION,
    load_pianobart,
    score_features,
    score_to_octuple,
)

FAILURES = []


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}{f'  [{detail}]' if detail else ''}")
    if not condition:
        FAILURES.append(name)


print("=== loading the released checkpoint ===")
model, meta = load_pianobart(device="cpu", dtype=torch.float32)
check("hidden size is 1024", meta.hidden_size == 1024, str(meta.hidden_size))
check("grid is 50 bins per beat", BINS_PER_BEAT == 50, str(BINS_PER_BEAT))

# The projection in front of the encoder is saved only under its tied alias
# `decoder_linear`. If it is ever left at its random initialization the encoder
# is pretrained in name only, so compare against the checkpoint directly.
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

released = load_file(hf_hub_download("RS2002/PianoBART", "model.safetensors"))
check(
    "encoder_linear holds the pretrained (tied) projection",
    torch.equal(model.encoder_linear.weight, released["model.decoder_linear.weight"]),
)
check(
    "word embeddings are the pretrained ones",
    torch.equal(model.word_emb[3].lut.weight, released["model.word_emb.3.lut.weight"]),
)
check("frozen", not any(p.requires_grad for p in model.parameters()))

print("\n=== Octuple conversion ===")
field = {name: i for i, name in enumerate(OCTUPLE_FIELDS)}
# beat 0, beat 2.5, beat 4 (= bar 1 in 4/4), with a quarter and a sixteenth.
quarter, sixteenth = BINS_PER_BEAT, BINS_PER_BEAT // 4
score = torch.tensor(
    [[
        [0, quarter, 60],
        [round(2.5 * BINS_PER_BEAT), sixteenth, 64],
        [BEATS_PER_BAR * BINS_PER_BEAT, quarter, 67],
    ]],
    dtype=torch.long,
)
valid = torch.ones(1, 3, dtype=torch.bool)
octuple, mask = score_to_octuple(score, valid, meta)

check("shape is (B, N, 8)", tuple(octuple.shape) == (1, 3, 8), str(tuple(octuple.shape)))
check("bars", octuple[0, :, field["Bar"]].tolist() == [0, 0, 1], str(octuple[0, :, field["Bar"]].tolist()))
check(
    "positions in 1/16-beat units",
    octuple[0, :, field["Position"]].tolist() == [0, int(2.5 * POS_RESOLUTION), 0],
    str(octuple[0, :, field["Position"]].tolist()),
)
check(
    "pitches pass through",
    octuple[0, :, field["Pitch"]].tolist() == [60, 64, 67],
    str(octuple[0, :, field["Pitch"]].tolist()),
)
# dur_enc is linear over its first 16 buckets, so a quarter (16 units) lands on
# bucket 16 and a sixteenth (4 units) on bucket 4.
check(
    "duration buckets",
    octuple[0, :, field["Duration"]].tolist() == [16, 4, 16],
    str(octuple[0, :, field["Duration"]].tolist()),
)
check(
    "constant fields are single-valued",
    len({tuple(octuple[0, i, [field["Instrument"], field["TimeSig"], field["Tempo"], field["Velocity"]]].tolist()) for i in range(3)}) == 1,
)

padded_valid = valid.clone()
padded_valid[0, 2] = False
padded, padded_mask = score_to_octuple(score, padded_valid, meta)
check("invalid slots get <PAD>", torch.equal(padded[0, 2], meta.pad_word))
check("invalid slots are masked", padded_mask[0].tolist() == [1, 1, 0])

print("\n=== encoder features ===")
notes = torch.stack([
    torch.arange(24) * (BINS_PER_BEAT // 2),
    torch.full((24,), BINS_PER_BEAT // 2),
    60 + torch.arange(24) % 12,
], dim=-1).unsqueeze(0)
valid = torch.ones(1, 24, dtype=torch.bool)
features = score_features(model, meta, notes, valid)
check("feature shape", tuple(features.shape) == (1, 24, 1024), str(tuple(features.shape)))
check("features are finite", bool(torch.isfinite(features).all()))
check("features vary across notes", float(features.std(dim=1).mean()) > 1e-3, f"{float(features.std(dim=1).mean()):.4f}")

# Bidirectional context is the reason for using an encoder rather than a causal
# LM: editing the LAST note must move the FIRST note's state.
edited = notes.clone()
edited[0, -1, 2] = 30
edited_features = score_features(model, meta, edited, valid)
moved = float((features[0, 0] - edited_features[0, 0]).abs().max())
check("a later note changes an earlier note's state (bidirectional)", moved > 1e-4, f"{moved:.4f}")

print("\n=== on real packed windows ===")
try:
    from plan_vq import split_packed_batch

    with open("data/val_paper.txt") as handle:
        lines = [next(handle) for _ in range(2)]
    ids = torch.tensor([[int(t) for t in line.split("|")[0].split()] for line in lines])
    perf, score, valid = split_packed_batch(ids)
    octuple, mask = score_to_octuple(score, valid, meta)
    bars = octuple[..., field["Bar"]]
    check("real windows stay inside the bar vocabulary", int(bars.max()) <= 255, f"max bar {int(bars.max())}")
    real = score_features(model, meta, score, valid)
    check("real features are finite", bool(torch.isfinite(real).all()))
    check(
        "distinct windows get distinct features",
        float((real[0] - real[1]).abs().mean()) > 1e-3,
        f"{float((real[0] - real[1]).abs().mean()):.4f}",
    )
except (FileNotFoundError, StopIteration) as error:
    print(f"SKIP  real packed windows ({error})")

print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
    sys.exit(1)
print("all checks passed")
