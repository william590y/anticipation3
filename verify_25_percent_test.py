"""
Verify that the 25% context test is implemented correctly.
Check sequence format and pitch extraction logic.
"""
import torch
from anticipation.vocab import *

print("="*80)
print("VERIFYING 25% CONTEXT TEST IMPLEMENTATION")
print("="*80)
print()

# Load one test sequence
print("Loading one test sequence...")
with open('data/test_normalized.txt', 'r') as f:
    line = f.readline()

# Parse sequence
if '|' in line:
    token_str, _ = line.split('|')
    tokens = [int(t) for t in token_str.strip().split()]
else:
    tokens = [int(t) for t in line.strip().split()]

print(f"Sequence length: {len(tokens)}")
print(f"First 20 tokens: {tokens[:20]}")
print()

# Check sequence format
print("Sequence format analysis:")
print(f"  Token 0 (should be ANTICIPATE={ANTICIPATE}): {tokens[0]}")
print(f"  CONTROL_OFFSET: {CONTROL_OFFSET}")
print(f"  REST: {REST}")
print()

# Find all score triplets
print("Finding all score triplets...")
score_positions = []
i = 0
while i + 2 < len(tokens):
    if (tokens[i] < CONTROL_OFFSET and 
        tokens[i+1] < CONTROL_OFFSET and 
        tokens[i+2] < CONTROL_OFFSET and
        tokens[i+2] != REST):
        score_positions.append(i)
        i += 3
    else:
        i += 1

print(f"Found {len(score_positions)} score triplets")
if len(score_positions) > 0:
    print(f"First score triplet position: {score_positions[0]}")
    first_pos = score_positions[0]
    print(f"  Tokens: {tokens[first_pos:first_pos+3]}")
    print(f"  Time: {tokens[first_pos]}")
    print(f"  Duration: {tokens[first_pos+1]}")
    print(f"  Pitch: {tokens[first_pos+2]}")
    print()
    
    if len(score_positions) > 1:
        print(f"Second score triplet position: {score_positions[1]}")
        second_pos = score_positions[1]
        print(f"  Tokens: {tokens[second_pos:second_pos+3]}")
        print()

# Calculate 25% cutoff
if len(score_positions) > 0:
    num_score_triplets = len(score_positions)
    cutoff_triplets = max(1, num_score_triplets // 4)
    cutoff_position = score_positions[cutoff_triplets - 1]
    cutoff_idx = cutoff_position + 3
    
    print(f"25% cutoff calculation:")
    print(f"  Total score triplets: {num_score_triplets}")
    print(f"  25% cutoff triplets: {cutoff_triplets}")
    print(f"  Last included triplet position: {cutoff_position}")
    print(f"  Context cutoff index: {cutoff_idx}")
    print(f"  Context length: {cutoff_idx} tokens")
    print(f"  Remaining length: {len(tokens) - cutoff_idx} tokens")
    print()
    
    # Extract pitches from remaining part
    remaining_tokens = tokens[cutoff_idx:]
    print(f"Extracting pitches from remaining {len(remaining_tokens)} tokens...")
    
    remaining_pitches = []
    i = 0
    while i + 2 < len(remaining_tokens):
        if (remaining_tokens[i] < CONTROL_OFFSET and 
            remaining_tokens[i+1] < CONTROL_OFFSET and 
            remaining_tokens[i+2] < CONTROL_OFFSET and
            remaining_tokens[i+2] != REST):
            pitch = remaining_tokens[i+2]
            remaining_pitches.append(pitch)
            i += 3
        else:
            i += 1
    
    print(f"  Extracted {len(remaining_pitches)} pitches from remaining tokens")
    expected_remaining = num_score_triplets - cutoff_triplets
    print(f"  Expected remaining score triplets: {expected_remaining}")
    
    if len(remaining_pitches) == expected_remaining:
        print(f"  ✓ CORRECT: Extracted count matches expected")
    else:
        print(f"  ✗ ERROR: Mismatch! Got {len(remaining_pitches)}, expected {expected_remaining}")
    
    print()
    print(f"First 10 remaining pitches: {remaining_pitches[:10]}")
    print()

# Now test the actual sequence format more carefully
print("="*80)
print("DETAILED SEQUENCE STRUCTURE")
print("="*80)
print()

# Look at the sequence structure
print("Scanning sequence structure...")
control_count = 0
score_count = 0
rest_count = 0
other_count = 0

i = 1  # Skip first token (ANTICIPATE)
while i + 2 < len(tokens):
    t0, t1, t2 = tokens[i], tokens[i+1], tokens[i+2]
    
    if t0 >= CONTROL_OFFSET and t1 >= CONTROL_OFFSET and t2 >= CONTROL_OFFSET:
        control_count += 1
        triplet_type = "control"
    elif t0 < CONTROL_OFFSET and t1 < CONTROL_OFFSET and t2 < CONTROL_OFFSET:
        if t2 == REST:
            rest_count += 1
            triplet_type = "rest"
        else:
            score_count += 1
            triplet_type = "score"
    else:
        other_count += 1
        triplet_type = "mixed/other"
        
    if i < 50:  # Print first few
        print(f"  Position {i}: [{t0}, {t1}, {t2}] -> {triplet_type}")
    
    i += 3

print()
print(f"Triplet counts:")
print(f"  Score triplets: {score_count}")
print(f"  Control triplets: {control_count}")
print(f"  Rest triplets: {rest_count}")
print(f"  Other/Mixed: {other_count}")
print()

# Check if sequence follows expected format
print("Expected format check:")
print("  Format: ANTICIPATE + SEP SEP SEP + control+rest pairs + alternating score/control")
if tokens[0] == ANTICIPATE:
    print("  ✓ Starts with ANTICIPATE")
else:
    print(f"  ✗ Token 0 is {tokens[0]}, expected ANTICIPATE ({ANTICIPATE})")

if len(tokens) > 3:
    sep_check = tokens[1] == SEPARATOR and tokens[2] == SEPARATOR and tokens[3] == SEPARATOR
    if sep_check:
        print("  ✓ Has SEP SEP SEP")
    else:
        print(f"  ? Tokens 1-3: {tokens[1:4]}, expected [SEP, SEP, SEP] = [{SEPARATOR}, {SEPARATOR}, {SEPARATOR}]")

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print()
print("If the score triplet extraction is correct, then the issue might be:")
print("  1. Model not generating valid score triplets")
print("  2. Model generating different sequence structure")
print("  3. Generation stopping early")
print("  4. Need to align properly with interleaved format")
