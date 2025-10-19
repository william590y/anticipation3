"""
CORRECTED understanding of what generate4 should produce.

KEY INSIGHT from tokenization:
- matched_tuples = [(perf_0, score_0), (perf_1, score_1), ..., (perf_N, score_N)]
- Prefix: perf_0 to perf_32 (with rests)
- Alternating: score_0 paired with perf_33, score_1 paired with perf_34, etc.

So:
- score_0 corresponds to perf_0 (which is in the prefix)
- score_1 corresponds to perf_1 (which is in the prefix)
- ...
- score_32 corresponds to perf_32 (which is in the prefix)
- score_33 corresponds to perf_33 (which is the first alternating control)
- score_N corresponds to perf_N

Therefore, in generation:
- Given all controls (perf_0 to perf_N)
- Generate performances (which should match score_0 to score_N)
- But we don't have the scores! We only have the controls!

WAIT - I think I misunderstood the task!

Let me re-read the alignment:
- matched_tuples[i] = [perf_tuple_i, i, score_tuple_i, best_index]
- perf_tuple_i and score_tuple_i are MATCHED (same pitch)
- So perf_i and score_i are the SAME NOTE played differently

In training:
- Given: perf controls (how it was performed)
- Generate: score (how it's written)

In testing (generate4):
- Given: perf controls (how it should be performed)
- Generate: ??? What are we supposed to generate?

The model is trained to generate SCORES from PERFORMANCE controls.
But in testing, we're giving it PERFORMANCE controls and expecting... what?

Let me check what the actual task is supposed to be...
"""

print(__doc__)

print("=" * 80)
print("UNDERSTANDING THE TASK")
print("=" * 80)
print()

print("From align_tokens2:")
print("  - Finds pairs of (performance_note, score_note) with same pitch")
print("  - matched_tuples[i] = [perf_i, idx, score_i, idx]")
print("  - perf_i and score_i have IDENTICAL pitch (matching enforced)")
print()

print("Training format (tokenize-asap.py):")
print("  PREFIX:")
print("    perf_0, rest, perf_1, rest, ..., perf_32, rest")
print("  ALTERNATING:")
print("    score_0, perf_33, score_1, perf_34, ..., score_N-33, perf_N")
print("  TRAILING:")
print("    score_N-32, score_N-31, ..., score_N")
print()

print("Model learns:")
print("  Given PREFIX of performance controls → Generate scores")
print("  Scores alternate with FUTURE performance controls")
print()

print("Inference (generate4):")
print("  Given: Performance controls extracted from test data")
print("  Generate: ???")
print()

print("QUESTION: What SHOULD we be generating and comparing?")
print()
print("Option A: Generate scores (like training)")
print("  - Makes sense: model trained to generate scores from perf controls")
print("  - Problem: We don't have ground truth scores in test data!")
print("  - Test data only has performance controls, not matched scores")
print()

print("Option B: Generate performances (reverse task)")
print("  - Doesn't make sense: model not trained for this")
print("  - Would require different training")
print()

print("Option C: The test data DOES have scores!")
print("  - Let me check the test sequence structure...")
print()

# Load and analyze test sequence
with open('data/test_output.txt', 'r') as f:
    line = f.readline()

sequence_tokens = [int(tok) for tok in line.strip().split()]

from anticipation.vocab import *
from anticipation.config import *

tokens = sequence_tokens[4:]  # Skip header

# Count controls vs non-controls
controls_count = 0
scores_count = 0

for i in range(0, len(tokens), 3):
    if i + 3 > len(tokens):
        break
    triplet = tokens[i:i+3]
    if triplet[0] >= CONTROL_OFFSET:
        controls_count += 1
    else:
        scores_count += 1

print(f"Test sequence analysis:")
print(f"  Total events: {(len(tokens)) // 3}")
print(f"  Controls (with CONTROL_OFFSET): {controls_count}")
print(f"  Scores (without CONTROL_OFFSET): {scores_count}")
print()

print("=" * 80)
print("REVELATION")
print("=" * 80)
print()

print("The test data DOES contain both controls AND scores!")
print("It's in the SAME format as training data!")
print()

print("So the task is:")
print("  1. Extract controls from test sequence")
print("  2. Generate new scores using generate4")
print("  3. Compare generated scores to ACTUAL scores in test sequence")
print()

print("But we're currently comparing:")
print("  Generated performances (output of generate4)")
print("  vs")
print("  Performance controls (input to generate4)")
print()

print("We SHOULD be comparing:")
print("  Generated performances (output of generate4)")
print("  vs")
print("  Score events from test sequence (ground truth)")
print()

print("TO FIX: Extract BOTH controls AND scores from test sequence!")
