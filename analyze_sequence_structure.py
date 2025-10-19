"""
More careful analysis of what's in the test sequences.
"""

from anticipation.vocab import *
from anticipation.config import *

# Load first test sequence
with open('data/test_output.txt', 'r') as f:
    line = f.readline()

sequence_tokens = [int(tok) for tok in line.strip().split()]
tokens = sequence_tokens[4:]  # Skip ANTICIPATE + 3 SEPs

print("=" * 80)
print("DETAILED SEQUENCE STRUCTURE ANALYSIS")
print("=" * 80)
print()
print(f"Total sequence tokens: {len(sequence_tokens)}")
print(f"Body tokens (after header): {len(tokens)}")
print()

# Count controls vs scores
controls_found = []
scores_found = []
rest_found = []

for i in range(0, len(tokens), 3):
    if i + 3 > len(tokens):
        break
    
    triplet = tokens[i:i+3]
    time_val = triplet[0]
    
    if time_val >= CONTROL_OFFSET:
        controls_found.append(i // 3)
    elif triplet[2] == REST:
        rest_found.append(i // 3)
    else:
        scores_found.append(i // 3)

print(f"Controls (time >= {CONTROL_OFFSET}): {len(controls_found)}")
print(f"Rests (note == {REST}): {len(rest_found)}")
print(f"Scores (time < {CONTROL_OFFSET}, note != {REST}): {len(scores_found)}")
print()

# Show first 50 triplets
print("First 50 triplets:")
for i in range(min(50, len(tokens) // 3)):
    triplet = tokens[i*3:i*3+3]
    time_val = triplet[0]
    note_val = triplet[2]
    
    if time_val >= CONTROL_OFFSET:
        label = "CTRL"
    elif note_val == REST:
        label = "REST"
    else:
        label = "SCORE"
    
    print(f"  [{i:3d}] {label:5s} {triplet}")

print()
print("..." )
print()

# Show last 50 triplets
print("Last 50 triplets:")
start_idx = max(0, len(tokens) // 3 - 50)
for i in range(start_idx, len(tokens) // 3):
    triplet = tokens[i*3:i*3+3]
    time_val = triplet[0]
    note_val = triplet[2]
    
    if time_val >= CONTROL_OFFSET:
        label = "CTRL"
    elif note_val == REST:
        label = "REST"
    else:
        label = "SCORE"
    
    print(f"  [{i:3d}] {label:5s} {triplet}")
