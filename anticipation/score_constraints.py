"""Shared score-slot decoding constraints."""

from __future__ import annotations

from anticipation.vocab import (
    CONTROL_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    REST,
    TIME_OFFSET,
    VOCAB_SIZE,
)


def constrain_score_token_logits(logits, slot, allow_rest=False):
    """Mask ``logits`` down to the legal token range for a body score slot.

    ``slot`` is the position within the triplet (0=onset, 1=duration, 2=pitch).
    Since unmatched performance notes are dropped at tokenization time, body
    score slots always hold real notes, so REST is excluded from the pitch slot
    unless ``allow_rest`` is set (needed only for decoding legacy dummy-slot
    sequences).

    ``logits`` may be a single 1-D vocab vector or a batched (batch, vocab)
    tensor -- indexing uses ``...`` on the last axis so both shapes work
    unchanged (batched autoregressive decoding needs the latter).
    """
    constrained = logits.clone()
    constrained[..., CONTROL_OFFSET:VOCAB_SIZE] = -float("inf")

    if slot == 0:
        constrained[..., DUR_OFFSET:CONTROL_OFFSET] = -float("inf")
    elif slot == 1:
        constrained[..., TIME_OFFSET:DUR_OFFSET] = -float("inf")
        constrained[..., NOTE_OFFSET:CONTROL_OFFSET] = -float("inf")
    elif slot == 2:
        constrained[..., TIME_OFFSET:NOTE_OFFSET] = -float("inf")
        if not allow_rest:
            constrained[..., REST] = -float("inf")
    else:
        raise ValueError(f"Invalid score slot: {slot}")

    return constrained
