"""A launch-free table ("n-gram"/lookup) draft for the packed format.

Why a table can work here at all
--------------------------------
Incremental decode is kernel-launch bound, so the cheapest useful proposal is
the one that needs no model forward.  The packed format hands us two pieces of
structure that a table can exploit:

1. **Fixed roles.**  Token role is ``index % 3`` -- onset, duration, pitch --
   so three small conditional tables cover everything the decoder ever emits.
2. **The performance is known 32 notes ahead.**  Score slot ``k`` is paired with
   *control* ``k``, and control ``k`` sits at position ``6k`` (prefix) or
   ``6k + 3`` (body) -- always ~189 tokens *before* slot ``k``'s own position.
   So when the decoder is about to emit slot ``k``, the aligned performance note
   is already in the context.  On the filtered paper-split data the aligned
   control's pitch equals the ground-truth score pitch **exactly** (verified:
   5520/5520 slots over 40 val windows; 70.3% on the unfiltered variant, where
   the pairing is deliberately broken).  Onset and duration are not copies --
   score onsets live on a 0.5 s beat grid while control times keep the real
   performance tempo -- but the *inter-onset interval* of the aligned controls
   predicts the score's onset delta well enough to be a proposal.

Tables (all conditioned on the aligned control, top-M candidates each):

  role 0 (onset)     key = clipped inter-onset interval of the aligned controls
                     value = score onset DELTA from the previous score onset
                     (slot 0 gets its own key and predicts the absolute onset)
  role 1 (duration)  key = clipped aligned control duration -> absolute duration
  role 2 (pitch)     key = aligned control pitch -> absolute score pitch

The tables are fitted to the **target's own rollouts**, not to ground truth:
what a draft is rewarded for is agreeing with the target, and this checkpoint's
rollouts are far from the data (teacher-forced score CE 0.266 vs 10.76 along its
own greedy rollout).  Fitting to GT would be fitting the wrong distribution.

Cost: building `q` is a gather plus a scatter_add plus a renormalise -- about
half a dozen kernels, versus ~10 x n_layer for a model forward.  It is the
cheapest proposal mechanism available, and therefore the lower bound on drafting
overhead in this regime.
"""

from __future__ import annotations

from collections import Counter

import torch

from anticipation.config import MAX_DUR, MAX_NOTE, MAX_TIME
from anticipation.packed_sequence import ALTERNATING_START, PREFIX_CONTROLS
from anticipation.vocab import (
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    REST,
    TIME_OFFSET,
    VOCAB_SIZE,
)

# Key spaces.  Row 0 of every table is the unconditional marginal, used as the
# backoff for a key never seen while fitting.
ONSET_GAP_CLIP = 256      # aligned-control inter-onset interval, in 10 ms bins
ONSET_FIRST_KEY = ONSET_GAP_CLIP        # slot 0 has no previous score onset
ONSET_KEY_SPACE = ONSET_GAP_CLIP + 1
DUR_KEY_CLIP = 256
PITCH_KEY_SPACE = MAX_NOTE
ONSET_DELTA_CLIP = 1024   # score onset deltas beyond this are clipped when fit


def aligned_control_position(slot_index: int) -> int:
    """Where control `k` (the one paired with score slot `k`) sits in the window.

    The prefix holds controls 0..31 at 6k; the body's controls continue at
    ``ALTERNATING_START + 6*(k-32) + 3 == 6k + 3``.  Either way it is well before
    slot ``k``'s own position ``ALTERNATING_START + 6k`` -- that ~32-note
    lookahead is the whole point of the anticipatory format.
    """
    return 6 * slot_index if slot_index < PREFIX_CONTROLS else 6 * slot_index + 3


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _finalise(counters, key_space, top_m, device):
    """Counters keyed by int -> (key_index, values, probs) dense tensors."""
    marginal = Counter()
    for counter in counters.values():
        marginal.update(counter)
    rows = [marginal]
    key_index = torch.zeros(key_space, dtype=torch.long)
    for key, counter in counters.items():
        if 0 <= key < key_space and sum(counter.values()) >= 4:
            key_index[key] = len(rows)
            rows.append(counter)

    values = torch.zeros(len(rows), top_m, dtype=torch.long)
    probs = torch.zeros(len(rows), top_m, dtype=torch.float32)
    for i, counter in enumerate(rows):
        common = counter.most_common(top_m)
        total = float(sum(count for _, count in common)) or 1.0
        for j, (value, count) in enumerate(common):
            values[i, j] = int(value)
            probs[i, j] = count / total
    return key_index.to(device), values.to(device), probs.to(device)


def fit_ngram_tables(sequences, top_m=16, device="cpu", max_slots=None):
    """Fit the three tables from packed sequences (N, length), int64.

    `sequences` should be the *target's* rollouts (see the module docstring).
    """
    sequences = sequences.to("cpu", torch.long)
    length = sequences.shape[1]
    n_slots = (length - ALTERNATING_START) // 6
    if max_slots is not None:
        n_slots = min(n_slots, max_slots)

    onset_counts, dur_counts, pitch_counts = {}, {}, {}
    prev_control_time = None
    prev_onset = None
    for k in range(n_slots):
        start = ALTERNATING_START + 6 * k
        if start + 2 >= length:
            break
        control = aligned_control_position(k)
        control_time = sequences[:, control] - ATIME_OFFSET
        control_dur = sequences[:, control + 1] - ADUR_OFFSET
        control_pitch = sequences[:, control + 2] - ANOTE_OFFSET
        onset = sequences[:, start] - TIME_OFFSET
        duration = sequences[:, start + 1] - DUR_OFFSET
        pitch = sequences[:, start + 2] - NOTE_OFFSET

        if k == 0:
            keys0 = torch.full_like(onset, ONSET_FIRST_KEY)
            deltas = onset
        else:
            keys0 = (control_time - prev_control_time).clamp(0, ONSET_GAP_CLIP - 1)
            deltas = (onset - prev_onset).clamp(0, ONSET_DELTA_CLIP - 1)
        for key, value in zip(keys0.tolist(), deltas.tolist()):
            onset_counts.setdefault(key, Counter())[value] += 1
        for key, value in zip(
            control_dur.clamp(0, DUR_KEY_CLIP - 1).tolist(), duration.clamp(0, MAX_DUR - 1).tolist()
        ):
            dur_counts.setdefault(key, Counter())[value] += 1
        for key, value in zip(
            control_pitch.clamp(0, PITCH_KEY_SPACE - 1).tolist(),
            pitch.clamp(0, MAX_NOTE - 1).tolist(),
        ):
            pitch_counts.setdefault(key, Counter())[value] += 1

        prev_control_time = control_time
        prev_onset = onset

    return {
        0: _finalise(onset_counts, ONSET_KEY_SPACE, top_m, device),
        1: _finalise(dur_counts, DUR_KEY_CLIP, top_m, device),
        2: _finalise(pitch_counts, PITCH_KEY_SPACE, top_m, device),
    }


def save_tables(tables, path):
    torch.save({str(k): [t.cpu() for t in v] for k, v in tables.items()}, str(path))


def load_tables(path, device="cpu"):
    raw = torch.load(str(path), map_location=device)
    return {int(k): [t.to(device) for t in v] for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Proposer
# ---------------------------------------------------------------------------


class NgramProposer:
    """Proposer interface (see `nbest/speculative.py`) backed by the tables.

    No KV cache, no model forward: `prime`/`rollback` are no-ops and every
    proposal is a gather + scatter_add.  Its "forwards" are counted under the
    level name so the report can show that this level costs no model launch.
    """

    level = "ngram"

    def __init__(self, tables, temperature=1.0, generator=None, level=None):
        self.tables = tables
        self.temperature = temperature
        self.generator = generator
        if level is not None:
            self.level = level

    def prime(self, out, upto, stats):
        return None

    def rollback(self, length):
        return None

    def _distribution(self, out, pos):
        batch = out.shape[0]
        device = out.device
        role = pos % 3
        start = pos - role
        slot = (start - ALTERNATING_START) // 6
        control = aligned_control_position(slot)
        key_index, values, probs = self.tables[role]

        if role == 0:
            previous_onset = out[:, start - 6] - TIME_OFFSET
            if slot == 0:
                keys = torch.full((batch,), ONSET_FIRST_KEY, dtype=torch.long, device=device)
                base = torch.zeros_like(previous_onset)
            else:
                previous_control = aligned_control_position(slot - 1)
                gap = out[:, control] - out[:, previous_control]
                keys = gap.clamp(0, ONSET_GAP_CLIP - 1)
                base = previous_onset
            rows = key_index[keys]
            tokens = (base.unsqueeze(1) + values[rows]).clamp(0, MAX_TIME - 1) + TIME_OFFSET
        elif role == 1:
            keys = (out[:, control + 1] - ADUR_OFFSET).clamp(0, DUR_KEY_CLIP - 1)
            rows = key_index[keys]
            tokens = values[rows].clamp(0, MAX_DUR - 1) + DUR_OFFSET
        else:
            keys = (out[:, control + 2] - ANOTE_OFFSET).clamp(0, PITCH_KEY_SPACE - 1)
            rows = key_index[keys]
            tokens = values[rows].clamp(0, MAX_NOTE - 1) + NOTE_OFFSET

        weights = probs[rows]
        if self.temperature is not None and self.temperature > 0 and self.temperature != 1.0:
            weights = weights.clamp_min(1e-12).pow(1.0 / self.temperature)
        distribution = torch.zeros(batch, VOCAB_SIZE, device=device, dtype=torch.float32)
        # Distinct table entries can collide after clipping to the legal range,
        # so accumulate rather than overwrite.
        distribution.scatter_add_(1, tokens, weights)
        # Every candidate is clamped into its role's legal range before this
        # scatter (pitch to MAX_NOTE-1, i.e. strictly below REST), so the row can
        # never be all-zero and the renormalisation is always well defined.
        total = distribution.sum(dim=-1, keepdim=True)
        return distribution / total.clamp_min(1e-30)

    def propose(self, out, positions, frontier, geom, stats):
        batch = out.shape[0]
        probs_by_pos = {}
        for pos in positions:
            probs = self._distribution(out, pos)
            stats.count_forward(self.level, batch)
            token = torch.multinomial(probs, num_samples=1, generator=self.generator).squeeze(1)
            token = torch.where(frontier > pos, out[:, pos], token)
            out[:, pos] = token
            probs_by_pos[pos] = probs
        return probs_by_pos
