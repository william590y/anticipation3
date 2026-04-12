"""Helpers for reading and traversing packed ASAP training sequences."""

from __future__ import annotations

from anticipation.config import EVENT_SIZE
from anticipation.vocab import (
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    CONTROL_OFFSET,
    REST,
    SEPARATOR,
)


PREFIX_CONTROLS = 33
ALTERNATING_START = PREFIX_CONTROLS * 2 * EVENT_SIZE


def _to_int(token) -> int:
    if hasattr(token, "item"):
        return int(token.item())
    return int(token)


def triplet_values(tokens, pos: int) -> tuple[int, int, int]:
    return _to_int(tokens[pos]), _to_int(tokens[pos + 1]), _to_int(tokens[pos + 2])


def is_control_triplet_tokens(tok0: int, tok1: int, tok2: int) -> bool:
    return (
        tok0 >= CONTROL_OFFSET
        and tok1 >= CONTROL_OFFSET
        and tok2 >= CONTROL_OFFSET
        and tok0 != SEPARATOR
    )


def is_control_triplet(tokens, pos: int) -> bool:
    if pos + 2 >= len(tokens):
        return False
    tok0, tok1, tok2 = triplet_values(tokens, pos)
    return is_control_triplet_tokens(tok0, tok1, tok2)


def is_score_slot_start(pos: int, score_start_idx: int = ALTERNATING_START) -> bool:
    return pos >= score_start_idx and (pos - score_start_idx) % 6 == 0


def is_control_slot_start(pos: int, score_start_idx: int = ALTERNATING_START) -> bool:
    return pos >= score_start_idx and (pos - score_start_idx) % 6 == 3


def is_prefix_placeholder_triplet(tokens, pos: int, score_start_idx: int = ALTERNATING_START) -> bool:
    if pos + 2 >= len(tokens) or pos >= score_start_idx or pos % 6 != 3:
        return False
    tok0, tok1, tok2 = triplet_values(tokens, pos)
    return tok0 < CONTROL_OFFSET and tok1 < CONTROL_OFFSET and tok2 == REST


def is_score_triplet(tokens, pos: int, score_start_idx: int = ALTERNATING_START) -> bool:
    if pos + 2 >= len(tokens) or not is_score_slot_start(pos, score_start_idx):
        return False
    tok0, tok1, tok2 = triplet_values(tokens, pos)
    return tok0 < CONTROL_OFFSET and tok1 < CONTROL_OFFSET and tok2 < CONTROL_OFFSET


def is_dummy_score_triplet(tokens, pos: int, score_start_idx: int = ALTERNATING_START) -> bool:
    return is_score_triplet(tokens, pos, score_start_idx) and _to_int(tokens[pos + 2]) == REST


def is_real_score_triplet(tokens, pos: int, score_start_idx: int = ALTERNATING_START) -> bool:
    return is_score_triplet(tokens, pos, score_start_idx) and _to_int(tokens[pos + 2]) != REST


def iter_score_slot_positions(sequence_length: int, score_start_idx: int = ALTERNATING_START):
    for pos in range(score_start_idx, max(sequence_length - 2, score_start_idx), 6):
        yield pos


def control_triplet_to_raw(triplet: list[int] | tuple[int, int, int]) -> list[int]:
    return [
        _to_int(triplet[0]) - ATIME_OFFSET,
        _to_int(triplet[1]) - ADUR_OFFSET,
        _to_int(triplet[2]) - ANOTE_OFFSET,
    ]


def extract_packed_components(
    tokens,
    score_start_idx: int = ALTERNATING_START,
    include_dummy_score: bool = False,
) -> tuple[list[list[int]], list[list[int]]]:
    performance_raw = []
    score_triplets = []

    prefix_limit = min(score_start_idx, len(tokens))
    for pos in range(0, max(prefix_limit - 2, 0), 6):
        if not is_control_triplet(tokens, pos):
            break
        performance_raw.append(control_triplet_to_raw(triplet_values(tokens, pos)))

    for pos in iter_score_slot_positions(len(tokens), score_start_idx):
        if pos + 2 >= len(tokens) or not is_score_triplet(tokens, pos, score_start_idx):
            break

        score_triplet = list(triplet_values(tokens, pos))
        if include_dummy_score or score_triplet[2] != REST:
            score_triplets.append(score_triplet)

        control_pos = pos + 3
        if control_pos + 2 >= len(tokens) or not is_control_triplet(tokens, control_pos):
            break
        performance_raw.append(control_triplet_to_raw(triplet_values(tokens, control_pos)))

    return performance_raw, score_triplets
