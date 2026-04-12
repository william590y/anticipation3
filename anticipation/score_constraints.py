"""Shared score-slot decoding constraints."""

from __future__ import annotations

from anticipation.vocab import CONTROL_OFFSET, DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET, VOCAB_SIZE


def constrain_score_token_logits(logits, slot):
    constrained = logits.clone()
    constrained[CONTROL_OFFSET:VOCAB_SIZE] = -float("inf")

    if slot == 0:
        constrained[DUR_OFFSET:CONTROL_OFFSET] = -float("inf")
    elif slot == 1:
        constrained[TIME_OFFSET:DUR_OFFSET] = -float("inf")
        constrained[NOTE_OFFSET:CONTROL_OFFSET] = -float("inf")
    elif slot == 2:
        constrained[TIME_OFFSET:NOTE_OFFSET] = -float("inf")
        constrained[CONTROL_OFFSET:VOCAB_SIZE] = -float("inf")
    else:
        raise ValueError(f"Invalid score slot: {slot}")

    return constrained
