#!/usr/bin/env python
"""Checks on the incremental F1 reward: telescoping, the matching criterion, and
agreement with an independent batch recomputation."""
from __future__ import annotations

import random
import sys

from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, REST, TIME_OFFSET
from f1_reward import (
    IncrementalF1,
    f1_delta_rewards,
    final_f1,
    score_triplet_to_note,
)

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        failures.append(name)


def batch_f1(pred, gt, tolerance=1):
    """Independent reimplementation: one-to-one, emission order, same criterion.

    Buckets by pitch alone, matching the table's third column."""
    pool = {}
    for onset, duration, pitch in gt:
        pool.setdefault(pitch, []).append([onset, False])
    tp = 0
    for note in pred:
        if note is None:
            continue
        onset, duration, pitch = note
        best, best_d = None, None
        for entry in pool.get(pitch, []):
            if entry[1]:
                continue
            d = abs(entry[0] - onset)
            if d <= tolerance and (best_d is None or d < best_d):
                best, best_d = entry, d
        if best is not None:
            best[1] = True
            tp += 1
    denominator = len(pred) + len(gt)
    return (2.0 * tp / denominator) if denominator else 0.0


print("telescoping and agreement")
random.seed(7)
for trial in range(200):
    n_gt = random.randint(1, 40)
    gt = [
        (random.randrange(0, 500), random.choice([12, 37, 100]), random.randrange(21, 108))
        for _ in range(n_gt)
    ]
    pred = []
    for _ in range(random.randint(1, 50)):
        if random.random() < 0.1:
            pred.append(None)  # a rest
        elif gt and random.random() < 0.5:
            # perturb a real note: sometimes inside the tolerance, sometimes out
            onset, duration, pitch = random.choice(gt)
            pred.append((onset + random.choice([0, 1, -1, 2, -3]), duration, pitch))
        else:
            pred.append(
                (random.randrange(0, 500), random.choice([12, 37, 100]),
                 random.randrange(21, 108))
            )
    rewards, achieved = f1_delta_rewards(pred, gt)
    if abs(sum(rewards) - achieved) > 1e-9:
        check(f"telescoping trial {trial}", False, f"{sum(rewards)} vs {achieved}")
        break
    if abs(achieved - batch_f1(pred, gt)) > 1e-9:
        check(f"batch agreement trial {trial}", False,
              f"{achieved} vs {batch_f1(pred, gt)}")
        break
    if len(rewards) != len(pred):
        check(f"one reward per triplet trial {trial}", False)
        break
else:
    check("increments telescope to the final F1 (200 random trials)", True)
    check("matches an independent batch implementation (200 trials)", True)
    check("emits exactly one reward per emitted triplet", True)

print("\nmatching criterion (default = table column 3, onset_pitch_tol1)")
gt = [(100, 37, 60)]
check("exact note matches", final_f1([(100, 37, 60)], gt) == 1.0)
check("onset +1 bin matches", final_f1([(101, 37, 60)], gt) == 1.0)
check("onset -1 bin matches", final_f1([(99, 37, 60)], gt) == 1.0)
check("onset +2 bins does not", final_f1([(102, 37, 60)], gt) == 0.0)
check("wrong pitch does not", final_f1([(100, 37, 61)], gt) == 0.0)
check("wrong duration STILL matches (duration is ignored)",
      final_f1([(100, 12, 60)], gt) == 1.0)

print("\nrequire_duration=True is the stricter opt-in variant")
check("wrong duration does not match",
      final_f1([(100, 12, 60)], gt, require_duration=True) == 0.0)
check("right duration still matches",
      final_f1([(101, 37, 60)], gt, require_duration=True) == 1.0)

print("\none-to-one and sign")
gt2 = [(100, 37, 60)]
rewards, achieved = f1_delta_rewards([(100, 37, 60), (100, 37, 60)], gt2)
check("a duplicate cannot match the same GT note twice", achieved == 2 * 1 / 3)
check("the duplicate earns negative reward", rewards[1] < 0, f"{rewards[1]}")
check("the hit earns positive reward", rewards[0] > 0, f"{rewards[0]}")

rewards, _ = f1_delta_rewards([(0, 12, 30)], [(500, 37, 60)])
check("a pure miss is negative-or-zero", rewards[0] <= 0, f"{rewards[0]}")

print("\nnearest-onset preference")
gt3 = [(100, 37, 60), (102, 37, 60)]
matcher = IncrementalF1(gt3)
matcher.add((101, 37, 60))          # equidistant-ish; must take 100 (first best)
matcher.add((102, 37, 60))          # must still find 102 free
check("closest free GT note is taken first", matcher.tp == 2, f"tp={matcher.tp}")

print("\ntoken decoding")
check("decodes a score triplet",
      score_triplet_to_note(TIME_OFFSET + 5, DUR_OFFSET + 12, NOTE_OFFSET + 60) == (5, 12, 60))
check("a REST decodes to None",
      score_triplet_to_note(TIME_OFFSET + 5, DUR_OFFSET + 12, REST) is None)
check("an out-of-range triplet decodes to None",
      score_triplet_to_note(TIME_OFFSET - 1, DUR_OFFSET + 12, NOTE_OFFSET + 60) is None)

print()
if failures:
    print(f"{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("all checks passed")
