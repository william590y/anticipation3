#!/usr/bin/env python
"""Stage 2 plumbing: plan tokens in front of the packed alternating sequence.

The LM sequence becomes::

    [PLAN_BOS] c_1 ... c_K  <1020-token packed window>          (placement=front)
    <packed[:192]> [PLAN_BOS] c_1 ... c_K <packed[192:]>        (placement=after_prefix)

where ``c_j`` are the stage-1 VQ codes lifted into new vocabulary entries above
``VOCAB_SIZE``. The model predicts them like any other token, so it learns a
prior over plans and then conditions on the plan for the rest of the window.

Two details that are easy to get wrong:

**Position ids.** The plan is *inserted*, which would otherwise shift the whole
packed window by ``K+1`` positions and scramble the learned absolute-position
structure (the packed format's meaning is periodic with period 6, and the
prefix/body boundary sits at a fixed absolute position). Instead the plan gets
its own dedicated position rows appended past ``CONTEXT_SIZE`` and the packed
window keeps positions ``0..1019`` exactly. Causal ordering comes from sequence
order, not from the position ids, so the plan still precedes what conditions on
it.

**PLAN_BOS is not a target.** It is a marker we insert, never something the
model must predict, so its label is ``-100``.

Known limitation of ``placement="front"``: a causal LM at position 0 has not
seen any of the performance, so ``p(plan)`` it learns there is the *marginal*
prior, not ``p(plan | performance)``. Training feeds the true plan, inference
cannot recover it, and the gap shows up directly in
``val/plan_code_accuracy``. ``placement="after_prefix"`` puts the plan after
the 32 lookahead controls so it is at least conditioned on the first 32
performance notes. Both are supported; the validation metrics report the
oracle-plan ceiling alongside the predicted-plan number so the size of that gap
is never a guess.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from anticipation.config import CONTEXT_SIZE
from anticipation.packed_sequence import ALTERNATING_START, is_score_slot_start
from anticipation.score_constraints import constrain_score_token_logits
from anticipation.vocab import VOCAB_SIZE

PACKED_LENGTH = CONTEXT_SIZE - 4                  # 1020
PLAN_CODE_OFFSET = VOCAB_SIZE                     # first new vocabulary entry
PLAN_POSITION_BASE = CONTEXT_SIZE                 # first new wpe row
PLACEMENTS = ("front", "after_prefix")


@dataclass(frozen=True)
class PlanLayout:
    """Where the plan sits and how wide the model's vocabulary/positions must be."""

    num_codes: int
    codebook_size: int
    placement: str = "front"

    def __post_init__(self):
        if not 1 <= self.num_codes <= 64:
            raise ValueError(f"num_codes out of range: {self.num_codes}")
        if self.placement not in PLACEMENTS:
            raise ValueError(f"placement must be one of {PLACEMENTS}, got {self.placement!r}")

    @property
    def plan_bos(self):
        return PLAN_CODE_OFFSET + self.codebook_size

    @property
    def vocab_size(self):
        return PLAN_CODE_OFFSET + self.codebook_size + 1

    @property
    def plan_length(self):
        """Tokens the plan occupies, including PLAN_BOS."""
        return self.num_codes + 1

    @property
    def insert_at(self):
        return 0 if self.placement == "front" else ALTERNATING_START

    @property
    def total_length(self):
        return PACKED_LENGTH + self.plan_length

    @property
    def score_start_idx(self):
        """Index of the first body score triplet in the assembled sequence.

        The same for both placements: the plan is always inserted at or before
        ``ALTERNATING_START``.
        """
        return ALTERNATING_START + self.plan_length

    @property
    def n_positions(self):
        return PLAN_POSITION_BASE + self.plan_length

    def plan_token_positions(self):
        """Indices of PLAN_BOS and the code slots within the assembled sequence."""
        start = self.insert_at
        return start, list(range(start + 1, start + 1 + self.num_codes))


# ---------------------------------------------------------------------------
# sequence assembly
# ---------------------------------------------------------------------------

def _plan_tokens(codes, layout):
    batch = codes.shape[0]
    bos = torch.full((batch, 1), layout.plan_bos, dtype=torch.long, device=codes.device)
    return torch.cat([bos, PLAN_CODE_OFFSET + codes.long()], dim=1)


def assemble_inputs(packed, codes, layout):
    """Packed windows + plan codes -> (input_ids, position_ids)."""
    if packed.shape[1] != PACKED_LENGTH:
        raise ValueError(f"expected {PACKED_LENGTH}-token windows, got {packed.shape[1]}")
    if codes.shape[1] != layout.num_codes:
        raise ValueError(f"expected {layout.num_codes} codes, got {codes.shape[1]}")

    batch = packed.shape[0]
    device = packed.device
    plan = _plan_tokens(codes.to(device), layout)
    plan_positions = torch.arange(
        PLAN_POSITION_BASE, PLAN_POSITION_BASE + layout.plan_length, device=device
    ).unsqueeze(0).expand(batch, -1)
    packed_positions = torch.arange(PACKED_LENGTH, device=device).unsqueeze(0).expand(batch, -1)

    cut = layout.insert_at
    input_ids = torch.cat([packed[:, :cut], plan, packed[:, cut:]], dim=1)
    position_ids = torch.cat(
        [packed_positions[:, :cut], plan_positions, packed_positions[:, cut:]], dim=1
    )
    return input_ids, position_ids


def assemble_labels(labels, codes, layout):
    """Packed labels + plan codes -> labels for the assembled sequence.

    PLAN_BOS gets ``-100`` (we insert it, the model never has to predict it);
    the code slots are real targets, which is how the model learns its prior
    over plans.
    """
    batch = labels.shape[0]
    device = labels.device
    ignore = torch.full((batch, 1), -100, dtype=labels.dtype, device=device)
    plan_labels = torch.cat([ignore, (PLAN_CODE_OFFSET + codes.long()).to(device)], dim=1)
    cut = layout.insert_at
    return torch.cat([labels[:, :cut], plan_labels, labels[:, cut:]], dim=1)


def strip_plan(sequence, layout):
    """Assembled sequence -> the 1020-token packed window it contains."""
    cut = layout.insert_at
    return torch.cat(
        [sequence[:, :cut], sequence[:, cut + layout.plan_length :]], dim=1
    )


# ---------------------------------------------------------------------------
# model surgery
# ---------------------------------------------------------------------------

def _transformer_of(model):
    if hasattr(model, "transformer"):
        return model.transformer
    raise AttributeError("plan_lm expects a GPT-2 style model exposing `.transformer`")


def prepare_plan_model(model, layout, verbose=True):
    """Widen the vocabulary for plan tokens and the position table for plan slots.

    New position rows are appended past ``CONTEXT_SIZE`` so rows ``0..1023``
    (everything the packed window uses) keep their pretrained values untouched.

    Both sets of new rows are initialized explicitly rather than left to
    ``resize_token_embeddings``: its default mean-resizing draws from the old
    embeddings' estimated covariance, which on this vocabulary comes out ~265x
    too small and near-degenerate, leaving every plan code with essentially the
    same near-zero vector. The plan would then start out invisible and have to
    grow from nothing against the weight anchor.
    """
    import torch.nn as nn

    if model.config.vocab_size != layout.vocab_size:
        old_vocab = model.config.vocab_size
        if verbose:
            print(
                f"Resizing vocabulary {old_vocab} -> {layout.vocab_size} "
                f"({layout.codebook_size} plan codes + PLAN_BOS)"
            )
        model.resize_token_embeddings(layout.vocab_size, mean_resizing=False)
        embeddings = model.get_input_embeddings().weight
        with torch.no_grad():
            scale = embeddings[:old_vocab].std()
            embeddings[old_vocab:].normal_(mean=0.0, std=float(scale))
        if verbose:
            print(
                f"  plan rows initialized at the base vocabulary's scale "
                f"(std {float(scale):.4f})"
            )

    transformer = _transformer_of(model)
    needed = layout.n_positions
    current = transformer.wpe.num_embeddings
    if current < needed:
        if verbose:
            print(f"Extending position embeddings {current} -> {needed} (plan slots)")
        old = transformer.wpe
        new = nn.Embedding(needed, old.embedding_dim)
        new.to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight[:current] = old.weight
            # Warm-start the plan slots from the first real positions: the right
            # scale by construction, and a sensible prior ("these come first").
            # They are separate parameters receiving different gradients, so
            # they differentiate during training.
            new.weight[current:] = old.weight[: needed - current]
        transformer.wpe = new

        model.config.n_positions = needed
        model.config.max_position_embeddings = needed
        # GPT2Attention caches a triangular mask sized from the config at
        # construction time (a non-persistent buffer), so it has to be rebuilt.
        causal = torch.tril(torch.ones((needed, needed), dtype=torch.bool)).view(
            1, 1, needed, needed
        )
        for block in transformer.h:
            block.attn.register_buffer(
                "bias", causal.to(block.attn.bias.device), persistent=False
            )
    elif current < layout.n_positions:  # pragma: no cover - defensive
        raise RuntimeError("position table too small and could not be extended")

    return model


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------

def constrain_plan_token_logits(logits, layout):
    """Mask logits down to the codebook range."""
    constrained = torch.full_like(logits, -float("inf"))
    lo, hi = PLAN_CODE_OFFSET, PLAN_CODE_OFFSET + layout.codebook_size
    constrained[..., lo:hi] = logits[..., lo:hi]
    return constrained


@torch.inference_mode()
def plan_autoregressive_generate_score(
    model,
    packed,
    layout,
    device,
    codes=None,
    generate_plan=False,
    plan_temperature=0.0,
    constrain_score_tokens=True,
):
    """Greedy score decode with a plan prefix; controls are teacher-forced.

    Mirrors ``inference.batched_autoregressive_generate_score`` -- same slot
    schedule, same constraints -- with two additions: explicit ``position_ids``
    (so the packed window keeps positions ``0..1019``) and an optional plan
    generation phase.

    ``generate_plan=False`` feeds the oracle ``codes`` (the ceiling); ``True``
    lets the model produce its own plan (what inference actually gets).

    Returns ``(generated_packed, plan_codes)`` where ``generated_packed`` has
    the same shape and layout as ``packed`` -- controls copied through, body
    score slots replaced by what the model emitted -- so it drops straight into
    the existing slot-by-slot comparisons.
    """
    if codes is None and not generate_plan:
        raise ValueError("pass codes=... or set generate_plan=True")

    packed = packed.to(device)
    batch = packed.shape[0]
    if codes is None:
        codes = torch.zeros((batch, layout.num_codes), dtype=torch.long, device=device)
    else:
        codes = codes.to(device)

    past = None
    next_logits = None

    def feed(tokens, positions):
        """Run one chunk through the model, updating the cache."""
        nonlocal past, next_logits
        out = model(
            input_ids=tokens,
            position_ids=positions,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]

    def column(value):
        return torch.full((batch, 1), int(value), dtype=torch.long, device=device)

    def positions_for(values):
        return torch.tensor(values, dtype=torch.long, device=device).unsqueeze(0).expand(batch, -1)

    packed_positions = torch.arange(PACKED_LENGTH, device=device).unsqueeze(0).expand(batch, -1)

    # --- leading context -----------------------------------------------------
    if layout.placement == "after_prefix":
        feed(packed[:, :ALTERNATING_START], packed_positions[:, :ALTERNATING_START])
        feed(column(layout.plan_bos), positions_for([PLAN_POSITION_BASE]))
    else:
        feed(column(layout.plan_bos), positions_for([PLAN_POSITION_BASE]))

    # --- the plan ------------------------------------------------------------
    emitted_codes = []
    for j in range(layout.num_codes):
        if generate_plan:
            logits = constrain_plan_token_logits(next_logits, layout)
            if plan_temperature > 0:
                probs = torch.softmax(logits.float() / plan_temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            else:
                token = logits.argmax(dim=-1, keepdim=True)
        else:
            token = (PLAN_CODE_OFFSET + codes[:, j]).unsqueeze(1)
        emitted_codes.append(token.squeeze(1) - PLAN_CODE_OFFSET)
        feed(token, positions_for([PLAN_POSITION_BASE + 1 + j]))

    plan_codes = torch.stack(emitted_codes, dim=1)

    # --- the packed window ---------------------------------------------------
    if layout.placement == "front":
        feed(packed[:, :ALTERNATING_START], packed_positions[:, :ALTERNATING_START])

    generated = packed.clone()
    pos = ALTERNATING_START
    while pos + 5 < PACKED_LENGTH:
        if is_score_slot_start(pos, ALTERNATING_START):
            for slot in range(3):
                logits = next_logits
                if constrain_score_tokens:
                    logits = constrain_score_token_logits(logits, slot)
                token = logits.argmax(dim=-1)
                generated[:, pos + slot] = token
                feed(token.unsqueeze(1), packed_positions[:, pos + slot : pos + slot + 1])
            pos += 3

            if pos + 2 < PACKED_LENGTH:
                feed(packed[:, pos : pos + 3], packed_positions[:, pos : pos + 3])
                pos += 3
        else:  # pragma: no cover - the packed layout never lands here
            feed(packed[:, pos : pos + 1], packed_positions[:, pos : pos + 1])
            pos += 1

    return generated, plan_codes
