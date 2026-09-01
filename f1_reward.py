"""Incremental note-level F1 as a dense per-triplet reward.

A terminal F1 would be a perfectly valid reward but gives one scalar for 138
decisions. Instead each emitted triplet is paid the *change* it caused:

    r_s = F1(notes emitted through slot s) - F1(notes emitted through slot s-1)

which telescopes to the final F1 exactly (F1 of the empty prefix is 0), so the
undiscounted return of a rollout **is** its F1 -- the same quantity the results
table reports -- while credit lands on the triplet that moved it.

Matching criterion: pitch exact, onset within +-1 bin, duration ignored -- the
identical rule behind the results table's third column (`onset_pitch_tol1` in
compute_f1.py, which buckets candidates by pitch alone). Training therefore moves
a number that is already published rather than a private variant.

`require_duration=True` switches to the stricter pitch+duration exact / onset +-1
rule. That is a genuinely different criterion (13.7% versus 22.7% for the base
checkpoint on unfiltered input) and matches no column of the table, so it is
opt-in only.

Matching is one-to-one and **online**: notes are matched in emission order and a
consumed ground-truth note is never returned to the pool. That is what makes the
increments well defined -- an offline greedy could revoke an earlier match and the
telescoping would break.

Note a mild consequence of the literal definition: with n_pred growing, an early
hit is worth more than a late one (F1_1 = 2/(1+n_gt) for the first note, versus
~1/n_gt of marginal credit near the end). This is inherent to "change in F1 per
triplet" and is left in deliberately -- it also gives a miss a genuinely negative
reward, which a fixed-denominator variant would not.
"""
from __future__ import annotations

from anticipation.config import MAX_DUR, MAX_PITCH, MAX_TIME
from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, REST, TIME_OFFSET

ONSET_TOLERANCE = 1


def score_triplet_to_note(onset_token, dur_token, pitch_token):
    """Decode a generated score triplet into (onset_bins, duration_bins, pitch).

    Returns None for a rest / out-of-range triplet: those are not notes and must
    not be matched against ground truth.
    """
    onset = int(onset_token) - TIME_OFFSET
    duration = int(dur_token) - DUR_OFFSET
    pitch = int(pitch_token) - NOTE_OFFSET
    if int(pitch_token) == REST:
        return None
    if not (0 <= onset < MAX_TIME and 0 <= duration < MAX_DUR and 0 <= pitch < MAX_PITCH):
        return None
    return onset, duration, pitch


class IncrementalF1:
    """One-to-one online matcher; `add` returns the change in F1 it caused."""

    def __init__(self, gt_notes, tolerance=ONSET_TOLERANCE, require_duration=False):
        self.tolerance = tolerance
        self.require_duration = require_duration
        self.n_gt = len(gt_notes)
        # Bucket on whatever must match exactly; the tolerance applies only to
        # the onset. Pitch alone reproduces the table's third column.
        self.pool = {}
        for onset, duration, pitch in gt_notes:
            key = (pitch, duration) if require_duration else pitch
            self.pool.setdefault(key, []).append([onset, False])
        self.n_pred = 0
        self.tp = 0
        self.f1 = 0.0

    def _score(self):
        denominator = self.n_pred + self.n_gt
        return (2.0 * self.tp / denominator) if denominator else 0.0

    def add(self, note):
        """Consume one emitted note (or None for a rest); return delta F1."""
        previous = self.f1
        self.n_pred += 1
        if note is not None:
            onset, duration, pitch = note
            key = (pitch, duration) if self.require_duration else pitch
            candidates = self.pool.get(key)
            if candidates:
                best, best_distance = None, None
                for entry in candidates:
                    if entry[1]:
                        continue
                    distance = abs(entry[0] - onset)
                    if distance <= self.tolerance and (
                        best_distance is None or distance < best_distance
                    ):
                        best, best_distance = entry, distance
                if best is not None:
                    best[1] = True
                    self.tp += 1
        self.f1 = self._score()
        return self.f1 - previous


def f1_delta_rewards(pred_notes, gt_notes, tolerance=ONSET_TOLERANCE,
                     require_duration=False):
    """Per-triplet rewards for one rollout. Sums to the rollout's final F1."""
    matcher = IncrementalF1(gt_notes, tolerance, require_duration)
    return [matcher.add(note) for note in pred_notes], matcher.f1


def final_f1(pred_notes, gt_notes, tolerance=ONSET_TOLERANCE, require_duration=False):
    """Final F1 only, under the identical online one-to-one matching."""
    matcher = IncrementalF1(gt_notes, tolerance, require_duration)
    for note in pred_notes:
        matcher.add(note)
    return matcher.f1


def decode_rollout_notes(tokens, slot_starts):
    """Decode a rolled-out sequence's score slots into notes (None where a rest)."""
    notes = []
    for start in slot_starts:
        notes.append(
            score_triplet_to_note(tokens[start], tokens[start + 1], tokens[start + 2])
        )
    return notes
