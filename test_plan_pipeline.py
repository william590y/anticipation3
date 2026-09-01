#!/usr/bin/env python
"""Checks on the two-stage plan pipeline (CPU, tiny models).

The two failure modes worth building a test around, because both are silent:

  * the plan being *invisible* to the packed window (a masking or position-id
    mistake would leave the LM training on a plan it cannot read, and the loss
    curve would look completely normal);
  * the KV-cached, position-id-overridden autoregressive decode disagreeing with
    a single batched forward.

Run: python test_plan_pipeline.py
"""
from __future__ import annotations

import sys

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from anticipation.config import MAX_DUR, MAX_PITCH, MAX_TIME
from anticipation.packed_sequence import ALTERNATING_START, dummy_rest_triplet
from anticipation.score_constraints import constrain_score_token_logits
from anticipation.vocab import (
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    CONTROL_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    REST,
    TIME_OFFSET,
    VOCAB_SIZE,
)
from plan_lm import (
    PACKED_LENGTH,
    PLAN_CODE_OFFSET,
    PLAN_POSITION_BASE,
    PlanLayout,
    assemble_inputs,
    assemble_labels,
    plan_autoregressive_generate_score,
    prepare_plan_model,
    strip_plan,
)
from plan_vq import (
    BODY_SLOTS,
    NUM_PERF_NOTES,
    PlanVQ,
    PlanVQConfig,
    split_packed_batch,
)

# PianoBART features are stand-ins here so these checks stay CPU-only and need
# no download. The real encoder is covered by test_pianobart_encoder.py.
PIANOBART_DIM = 24

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        failures.append(name)


# Onset spacings are derived from MAX_TIME, not picked: an onset token is
# TIME_OFFSET/ATIME_OFFSET + t with t < MAX_TIME, so a spacing that overruns the
# window emits tokens in the *duration* range instead, and split_packed_batch's
# range clamp then quietly turns the round-trip check into a test of the clamp.
PERF_SPACING = (MAX_TIME - 20) // NUM_PERF_NOTES
SCORE_SPACING = (MAX_TIME - 3) // BODY_SLOTS


def make_packed_window(seed):
    """A synthetic but structurally exact packed window."""
    rng = torch.Generator().manual_seed(seed)

    def rand(high):
        return int(torch.randint(0, high, (1,), generator=rng).item())

    perf = [
        (PERF_SPACING * i + rand(20), 20 + rand(40), 21 + rand(80))
        for i in range(NUM_PERF_NOTES)
    ]
    score = [
        (SCORE_SPACING * i + rand(3), 25 + rand(50), 21 + rand(80))
        for i in range(BODY_SLOTS)
    ]
    for onset, duration, pitch in perf + score:
        assert onset < MAX_TIME and duration < MAX_DUR and pitch < MAX_PITCH

    tokens = []
    for i in range(ALTERNATING_START // 6):
        onset, duration, pitch = perf[i]
        tokens += [ATIME_OFFSET + onset, ADUR_OFFSET + duration, ANOTE_OFFSET + pitch]
        tokens += dummy_rest_triplet(0)
    for slot in range(BODY_SLOTS):
        onset, duration, pitch = score[slot]
        tokens += [TIME_OFFSET + onset, DUR_OFFSET + duration, NOTE_OFFSET + pitch]
        onset, duration, pitch = perf[ALTERNATING_START // 6 + slot]
        tokens += [ATIME_OFFSET + onset, ADUR_OFFSET + duration, ANOTE_OFFSET + pitch]

    assert len(tokens) == PACKED_LENGTH, len(tokens)
    return torch.tensor(tokens, dtype=torch.long), perf, score


def tiny_lm():
    config = GPT2Config(
        vocab_size=VOCAB_SIZE, n_positions=1024, n_embd=64, n_layer=2, n_head=2,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False,
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
print("packed window -> note tensors")

packed, perf_notes, score_notes = make_packed_window(1)
batch = torch.stack([packed, make_packed_window(2)[0]])
perf, score, valid = split_packed_batch(batch)

check("performance shape", tuple(perf.shape) == (2, NUM_PERF_NOTES, 3), str(perf.shape))
check("score shape", tuple(score.shape) == (2, BODY_SLOTS, 3), str(score.shape))
check("every body score slot is real", bool(valid.all()))
check(
    "performance notes round-trip",
    perf[0].tolist() == [list(note) for note in perf_notes],
)
check(
    "score notes round-trip",
    score[0].tolist() == [list(note) for note in score_notes],
)
check(
    "score slot k pairs with control k",
    perf[0, :BODY_SLOTS].shape[0] == BODY_SLOTS and perf.shape[1] == NUM_PERF_NOTES,
)

rest_window = packed.clone()
rest_window[ALTERNATING_START + 2] = REST
_, _, rest_valid = split_packed_batch(rest_window.unsqueeze(0))
check("a REST score slot is marked invalid", not bool(rest_valid[0, 0]))

check("raw values stay in the embedding ranges",
      bool((perf[..., 0] < MAX_TIME).all() and (perf[..., 1] < MAX_DUR).all()
           and (perf[..., 2] < MAX_PITCH).all()
           and (score[..., 0] < MAX_TIME).all() and (score[..., 1] < MAX_DUR).all()
           and (score[..., 2] < MAX_PITCH).all()))

# ---------------------------------------------------------------------------
print("\nstage 1: the VQ head over (stand-in) PianoBART features")

vq = PlanVQ(PlanVQConfig(num_codes=6, codebook_size=32, code_dim=8, d_model=32,
                         n_head=2, perf_layers=1, encoder_layers=1, decoder_layers=1,
                         fourier_bands=4, dropout=0.0, pianobart_dim=PIANOBART_DIM))
torch.manual_seed(3)
score_features = torch.randn(2, BODY_SLOTS, PIANOBART_DIM)
vq.train()
outputs = vq(perf, score_features, score, valid)
check("codes shape", tuple(outputs["codes"].shape) == (2, 6), str(outputs["codes"].shape))
check("codes land inside the codebook",
      bool((outputs["codes"] >= 0).all() and (outputs["codes"] < 32).all()))
check("loss is finite", bool(torch.isfinite(outputs["loss"])))

outputs["loss"].backward()
grad = vq.encoder_queries.grad
check("the straight-through path reaches the encoder",
      grad is not None and bool(grad.abs().sum() > 0))

vq.eval()
codes_a = vq.encode_codes(perf, score_features, valid)
codes_b = vq.encode_codes(perf, score_features, valid)
check("encode_codes is deterministic in eval mode", bool((codes_a == codes_b).all()))
check("encode_codes agrees with forward",
      bool((codes_a == vq(perf, score_features, score, valid)["codes"]).all()))
# Probed on the pre-quantization latents rather than the codes. The wiring
# question is whether the pooling queries read the performance at all; at
# initialization the codebook is 32 random vectors, so a genuine ~5% shift in the
# latent usually keeps the same nearest neighbour and an argmax-level probe comes
# out identical for 15 of 20 random seeds -- it tests the init, not the model.
with torch.no_grad():
    latents = vq.encode(vq.encode_performance(perf), score_features, valid)
    rolled = vq.encode(vq.encode_performance(perf.roll(1, dims=0)), score_features, valid)
relative_shift = float((latents - rolled).norm() / latents.norm())
check("the plan depends on the performance, not only the score",
      relative_shift > 1e-3, f"relative latent shift {relative_shift:.4f}")

predicted, _ = vq.reconstruct(perf, score_features, valid)
check("reconstruct returns score-shaped predictions",
      tuple(predicted.shape) == (2, BODY_SLOTS, 3), str(predicted.shape))
check("predictions stay in range",
      bool((predicted[..., 0] < MAX_TIME).all() and (predicted[..., 1] < MAX_DUR).all()
           and (predicted[..., 2] < MAX_PITCH).all()))

shuffled = vq(perf, score_features, score, valid, shuffle_codes=True)
check("the shuffled-code diagnostic runs", bool(torch.isfinite(shuffled["loss"])))

# The dead-code test is scale-free (a share of an equal share), so the EMA keeps
# moving instead of parking every code on a restart floor.
vq.train()
before = vq.quantizer.cluster_size.clone()
for _ in range(3):
    vq(perf, score_features, score, valid)
check("EMA statistics move during training",
      bool((vq.quantizer.cluster_size != before).any()))
check("codebook stays finite", bool(torch.isfinite(vq.quantizer.codebook).all()))

# ---------------------------------------------------------------------------
print("\nplan layout and sequence assembly")

for placement in ("front", "after_prefix"):
    layout = PlanLayout(num_codes=6, codebook_size=32, placement=placement)
    codes = torch.arange(12).view(2, 6) % 32
    labels = batch.clone()

    tokens, positions = assemble_inputs(batch, codes, layout)
    assembled_labels = assemble_labels(labels, codes, layout)

    check(f"[{placement}] assembled length", tokens.shape[1] == layout.total_length)
    check(f"[{placement}] labels match input length",
          assembled_labels.shape == tokens.shape)
    check(f"[{placement}] strip_plan inverts assembly",
          bool((strip_plan(tokens, layout) == batch).all()))

    bos_index, code_indices = layout.plan_token_positions()
    check(f"[{placement}] PLAN_BOS sits where the layout says",
          bool((tokens[:, bos_index] == layout.plan_bos).all()))
    check(f"[{placement}] codes are lifted above the base vocabulary",
          bool((tokens[:, code_indices] == PLAN_CODE_OFFSET + codes).all()))
    check(f"[{placement}] PLAN_BOS is never a target",
          bool((assembled_labels[:, bos_index] == -100).all()))
    check(f"[{placement}] the codes ARE targets (the model learns a prior)",
          bool((assembled_labels[:, code_indices] == PLAN_CODE_OFFSET + codes).all()))

    packed_positions = torch.cat(
        [positions[:, :layout.insert_at],
         positions[:, layout.insert_at + layout.plan_length:]], dim=1
    )
    check(f"[{placement}] the packed window keeps positions 0..1019",
          bool((packed_positions[0] == torch.arange(PACKED_LENGTH)).all()))
    check(f"[{placement}] the plan uses dedicated position rows",
          bool((positions[0, layout.insert_at:layout.insert_at + layout.plan_length]
                >= PLAN_POSITION_BASE).all()))
    check(f"[{placement}] score_start_idx points at a score triplet",
          bool((tokens[:, layout.score_start_idx] < CONTROL_OFFSET).all()))

# ---------------------------------------------------------------------------
print("\nmodel surgery")

layout = PlanLayout(num_codes=6, codebook_size=32, placement="front")
base = tiny_lm()
with torch.no_grad():
    reference_logits = base(
        input_ids=batch,
        attention_mask=torch.ones_like(batch),
        position_ids=torch.arange(PACKED_LENGTH).unsqueeze(0).expand(2, -1),
    ).logits.clone()

model = prepare_plan_model(base, layout, verbose=False)
check("vocabulary widened for the plan codes",
      model.config.vocab_size == layout.vocab_size, str(model.config.vocab_size))
check("position table widened for the plan slots",
      model.transformer.wpe.num_embeddings == layout.n_positions)
check("the causal mask buffer was rebuilt to match",
      model.transformer.h[0].attn.bias.shape[-1] == layout.n_positions)

# resize_token_embeddings' default mean-resizing produced near-zero,
# near-identical rows for the plan codes (row norm ~265x too small), which would
# have made the plan invisible at initialization.
embeddings = model.get_input_embeddings().weight.detach()
base_norm = embeddings[:VOCAB_SIZE].norm(dim=1).mean()
plan_norm = embeddings[VOCAB_SIZE:].norm(dim=1).mean()
check("plan token embeddings are on the base vocabulary's scale",
      0.5 < float(plan_norm / base_norm) < 2.0,
      f"plan {float(plan_norm):.4f} vs base {float(base_norm):.4f}")
code_gap = (embeddings[PLAN_CODE_OFFSET] - embeddings[PLAN_CODE_OFFSET + 31]).norm()
check("different codes have genuinely different embeddings",
      float(code_gap / base_norm) > 0.5, f"gap {float(code_gap):.5f}")
new_positions = model.transformer.wpe.weight.detach()[PLAN_POSITION_BASE:]
old_positions = model.transformer.wpe.weight.detach()[:PLAN_POSITION_BASE]
check("plan position rows are on the existing rows' scale",
      0.5 < float(new_positions.norm(dim=1).mean() / old_positions.norm(dim=1).mean()) < 2.0)

with torch.no_grad():
    after_logits = model(
        input_ids=batch,
        attention_mask=torch.ones_like(batch),
        position_ids=torch.arange(PACKED_LENGTH).unsqueeze(0).expand(2, -1),
    ).logits[..., :VOCAB_SIZE]
check("surgery leaves the base model's behaviour on a plain window untouched",
      torch.allclose(after_logits, reference_logits, atol=1e-5),
      f"max diff {float((after_logits - reference_logits).abs().max()):.2e}")

# ---------------------------------------------------------------------------
print("\nTHE plan actually conditions the body")

codes_lo = torch.zeros(2, 6, dtype=torch.long)
codes_hi = torch.full((2, 6), 31, dtype=torch.long)


def body_logits(codes, placement="front"):
    layout = PlanLayout(num_codes=6, codebook_size=32, placement=placement)
    tokens, positions = assemble_inputs(batch, codes, layout)
    with torch.no_grad():
        return model(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            position_ids=positions,
        ).logits[:, layout.score_start_idx - 1, :]


# Measured against a single-token baseline rather than an absolute threshold:
# an untrained model's logits have no meaningful scale, but "changing the plan
# should move a distant prediction at least as much as changing one ordinary
# token the same distance away" is scale-free and is what actually matters.
plan_effect = (body_logits(codes_lo) - body_logits(codes_hi)).abs().max()

altered = batch.clone()
altered[:, 0] = ATIME_OFFSET + 999
layout_front = PlanLayout(num_codes=6, codebook_size=32, placement="front")


def logits_at_body(packed, codes):
    tokens, positions = assemble_inputs(packed, codes, layout_front)
    with torch.no_grad():
        return model(
            input_ids=tokens, attention_mask=torch.ones_like(tokens), position_ids=positions
        ).logits[:, layout_front.score_start_idx - 1, :]


token_effect = (logits_at_body(batch, codes_lo) - logits_at_body(altered, codes_lo)).abs().max()
check("changing the plan changes a distant body prediction",
      bool(plan_effect > token_effect),
      f"plan {float(plan_effect):.2e} vs one-token baseline {float(token_effect):.2e}")

# The plan has to survive all the way to the far end of the window, not just
# the first few slots.
def logits_at_end(codes):
    tokens, positions = assemble_inputs(batch, codes, layout_front)
    with torch.no_grad():
        return model(
            input_ids=tokens, attention_mask=torch.ones_like(tokens), position_ids=positions
        ).logits[:, -2, :]


end_effect = (logits_at_end(codes_lo) - logits_at_end(codes_hi)).abs().max()
check("the plan still reaches the last score slot",
      bool(end_effect > 0.5 * token_effect),
      f"end-of-window effect {float(end_effect):.2e}")

# The same check WITHOUT an attention mask is the regression test for the
# packed-sequence trap: transformers reads position_ids for document
# boundaries when attention_mask is None, and the plan's dedicated position
# rows look exactly like one.
layout = PlanLayout(num_codes=6, codebook_size=32, placement="front")
tokens_lo, positions_lo = assemble_inputs(batch, codes_lo, layout)
tokens_hi, _ = assemble_inputs(batch, codes_hi, layout)
with torch.no_grad():
    unmasked_lo = model(input_ids=tokens_lo, position_ids=positions_lo).logits[
        :, layout.score_start_idx - 1, :]
    unmasked_hi = model(input_ids=tokens_hi, position_ids=positions_lo).logits[
        :, layout.score_start_idx - 1, :]
severed = float((unmasked_lo - unmasked_hi).abs().max()) < 1e-6
check("without an attention mask the plan would be severed (documents the trap)",
      severed,
      "the trap did not reproduce; the explicit mask in PlanBatcher may no longer be needed")

# ---------------------------------------------------------------------------
print("\nautoregressive decode with a plan prefix")

for placement in ("front", "after_prefix"):
    layout = PlanLayout(num_codes=6, codebook_size=32, placement=placement)
    codes = torch.arange(12).view(2, 6) % 32
    generated, used_codes = plan_autoregressive_generate_score(
        model, batch, layout, "cpu", codes=codes
    )

    check(f"[{placement}] output keeps the packed shape",
          tuple(generated.shape) == tuple(batch.shape))
    check(f"[{placement}] oracle codes are passed through", bool((used_codes == codes).all()))

    control_positions = [
        ALTERNATING_START + 6 * k + 3 + j for k in range(BODY_SLOTS) for j in range(3)
    ] + list(range(ALTERNATING_START))
    control_positions = torch.tensor(control_positions)
    check(f"[{placement}] controls and prefix are teacher-forced through unchanged",
          bool((generated[:, control_positions] == batch[:, control_positions]).all()))

    score_positions = [ALTERNATING_START + 6 * k for k in range(BODY_SLOTS)]
    onsets = generated[:, torch.tensor(score_positions)]
    durations = generated[:, torch.tensor([p + 1 for p in score_positions])]
    pitches = generated[:, torch.tensor([p + 2 for p in score_positions])]
    check(f"[{placement}] onsets decode inside the onset range",
          bool(((onsets >= TIME_OFFSET) & (onsets < DUR_OFFSET)).all()))
    check(f"[{placement}] durations decode inside the duration range",
          bool(((durations >= DUR_OFFSET) & (durations < NOTE_OFFSET)).all()))
    check(f"[{placement}] pitches decode inside the pitch range, never REST",
          bool(((pitches >= NOTE_OFFSET) & (pitches < REST)).all()))

    # The cached, position-overridden decode must equal one batched forward over
    # what it produced.
    tokens, positions = assemble_inputs(generated, codes, layout)
    with torch.no_grad():
        logits = model(
            input_ids=tokens, attention_mask=torch.ones_like(tokens), position_ids=positions
        ).logits
    mismatches = 0
    for slot in range(BODY_SLOTS):
        base_index = layout.score_start_idx + 6 * slot
        for role in range(3):
            step_logits = logits[:, base_index + role - 1, :]
            expected = constrain_score_token_logits(step_logits, role).argmax(dim=-1)
            actual = generated[:, ALTERNATING_START + 6 * slot + role]
            mismatches += int((expected != actual).sum().item())
    check(f"[{placement}] cached decode == one batched forward",
          mismatches == 0, f"{mismatches}/{BODY_SLOTS * 3 * 2} token mismatches")

    own_generated, own_codes = plan_autoregressive_generate_score(
        model, batch, layout, "cpu", generate_plan=True
    )
    check(f"[{placement}] a self-generated plan stays inside the codebook",
          bool((own_codes >= 0).all() and (own_codes < 32).all()))
    check(f"[{placement}] self-generated plan shape", tuple(own_codes.shape) == (2, 6))

# ---------------------------------------------------------------------------
print("\nstage-2 batching")

sys.path.insert(0, ".")
from train_plan_lm import PlanBatcher, forward_plan_batch, packed_index_map  # noqa: E402

layout = PlanLayout(num_codes=6, codebook_size=32, placement="front")
mapping = packed_index_map(layout, torch.device("cpu"))
check("index map covers every packed position exactly once",
      sorted(int(v) for v in mapping if v >= 0) == list(range(PACKED_LENGTH)))
check("index map marks exactly the plan tokens", int((mapping < 0).sum()) == layout.plan_length)

batcher = PlanBatcher(lambda indices: torch.zeros(len(indices), 6, dtype=torch.long), layout)
prepared = batcher.build(
    {"input_ids": batch, "labels": batch.clone(), "index": torch.arange(2),
     "score_mask": torch.zeros_like(batch, dtype=torch.bool)}
)
check("the batcher always supplies an attention mask",
      prepared["attention_mask"] is not None and bool(prepared["attention_mask"].all()))
outputs = forward_plan_batch(model, prepared)
check("stage-2 forward produces a finite loss", bool(torch.isfinite(outputs.loss)))
check("stage-2 logits cover the plan vocabulary",
      outputs.logits.shape[-1] == layout.vocab_size)

print()
if failures:
    print(f"{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("all checks passed")
