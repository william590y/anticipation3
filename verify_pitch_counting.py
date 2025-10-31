"""
Verify what we're actually counting for pitch accuracy
"""
import torch
from anticipation.vocab import CONTROL_OFFSET, TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET, REST, SEPARATOR, ANTICIPATE

print("="*80)
print("PITCH ACCURACY VERIFICATION - What are we counting?")
print("="*80)

# Load a sample sequence
print("\nLoading sample sequence from train_clean.txt...")
with open('data/train_clean.txt', 'r') as f:
    first_line = f.readline().strip()

if '|' in first_line:
    token_str, _ = first_line.split('|')
    tokens = [int(t) for t in token_str.strip().split()]
else:
    tokens = [int(t) for t in first_line.split()]

print(f"Sequence length: {len(tokens)}")
print(f"First token: {tokens[0]} (ANTICIPATE={ANTICIPATE})")

# Analyze the structure
print("\n" + "-"*80)
print("SEQUENCE STRUCTURE ANALYSIS")
print("-"*80)

score_triplets = []
control_triplets = []
rest_triplets = []
separators = []
other_tokens = []

i = 0
while i < len(tokens):
    if tokens[i] == SEPARATOR:
        separators.append(i)
        i += 1
    elif tokens[i] == ANTICIPATE:
        i += 1
    elif i < len(tokens) - 2:
        # Check for triplets
        if (tokens[i] >= CONTROL_OFFSET and 
            tokens[i+1] >= CONTROL_OFFSET and 
            tokens[i+2] >= CONTROL_OFFSET):
            # Control or rest triplet
            if tokens[i+2] == REST:
                rest_triplets.append((i, tokens[i:i+3]))
            else:
                control_triplets.append((i, tokens[i:i+3]))
            i += 3
        elif (tokens[i] < CONTROL_OFFSET and 
              tokens[i+1] < CONTROL_OFFSET and 
              tokens[i+2] < CONTROL_OFFSET):
            # Score triplet
            score_triplets.append((i, tokens[i:i+3]))
            i += 3
        else:
            other_tokens.append((i, tokens[i]))
            i += 1
    else:
        other_tokens.append((i, tokens[i]))
        i += 1

print(f"Separators: {len(separators)} at positions {separators[:10]}")
print(f"Score triplets: {len(score_triplets)}")
print(f"Control triplets: {len(control_triplets)}")
print(f"Rest triplets: {len(rest_triplets)}")
print(f"Other tokens: {len(other_tokens)}")

# Show the structure visually
print("\n" + "-"*80)
print("STRUCTURE BREAKDOWN (first 300 tokens)")
print("-"*80)

if score_triplets:
    print(f"\nFirst score triplet at position {score_triplets[0][0]}:")
    print(f"  Tokens: {score_triplets[0][1]}")
    print(f"  Time: {score_triplets[0][1][0] - TIME_OFFSET}")
    print(f"  Duration: {score_triplets[0][1][1] - DUR_OFFSET}")
    print(f"  Note: {score_triplets[0][1][2] - NOTE_OFFSET}")
    
    print(f"\nLast score triplet at position {score_triplets[-1][0]}:")
    print(f"  Tokens: {score_triplets[-1][1]}")

if control_triplets:
    print(f"\nFirst control triplet at position {control_triplets[0][0]}:")
    print(f"  Tokens: {control_triplets[0][1]}")
    print(f"  Time: {control_triplets[0][1][0] - CONTROL_OFFSET}")
    print(f"  Duration: {control_triplets[0][1][1] - CONTROL_OFFSET}")
    print(f"  Note: {control_triplets[0][1][2] - CONTROL_OFFSET}")
    
    print(f"\nLast control triplet at position {control_triplets[-1][0]}:")
    print(f"  Tokens: {control_triplets[-1][1]}")

if rest_triplets:
    print(f"\nFirst rest triplet at position {rest_triplets[0][0]}:")
    print(f"  Tokens: {rest_triplets[0][1]}")
    
    print(f"\nLast rest triplet at position {rest_triplets[-1][0]}:")
    print(f"  Tokens: {rest_triplets[-1][1]}")

# Check the interleaving pattern
print("\n" + "-"*80)
print("INTERLEAVING PATTERN (positions)")
print("-"*80)
print("Expected format from tokenize-asap.py:")
print("  1. Mode token (ANTICIPATE)")
print("  2. Separators (3x)")
print("  3. Prefix: 33 control+rest pairs")
print("  4. Body: alternating [score, control]")
print()

# Show first 20 triplets
all_triplets = []
if separators:
    for pos in separators:
        all_triplets.append(('SEP', pos, None))
for pos, triplet in rest_triplets:
    all_triplets.append(('REST', pos, triplet))
for pos, triplet in control_triplets:
    all_triplets.append(('CTRL', pos, triplet))
for pos, triplet in score_triplets:
    all_triplets.append(('SCORE', pos, triplet))

all_triplets.sort(key=lambda x: x[1])

print("First 30 items:")
for i, (typ, pos, triplet) in enumerate(all_triplets[:30]):
    print(f"  {i:2d}. Position {pos:4d}: {typ:5s} {triplet if triplet else ''}")

# Verify what pitch accuracy counts
print("\n" + "-"*80)
print("WHAT PITCH ACCURACY COUNTS")
print("-"*80)
print("Our evaluation code counts:")
print("  - Score triplets where all 3 tokens < CONTROL_OFFSET")
print("  - Position i+2 is the NOTE token")
print("  - Only non-masked positions (labels != -100)")
print()
print(f"In this sequence, we would count {len(score_triplets)} score notes")
print()
print("We do NOT count:")
print("  - Control triplets (performance timing)")
print("  - Rest triplets (rests in prefix)")
print("  - Separator tokens")
print("  - Masked positions")

# Double-check our counting logic matches
print("\n" + "-"*80)
print("VERIFICATION: Does our counting logic match?")
print("-"*80)

detected_score_count = 0
i = 1  # Skip first token (ANTICIPATE)
while i < len(tokens) - 2:
    if (tokens[i] < CONTROL_OFFSET and 
        tokens[i+1] < CONTROL_OFFSET and 
        tokens[i+2] < CONTROL_OFFSET):
        detected_score_count += 1
        i += 3
    else:
        i += 1

print(f"Score triplets found by structure analysis: {len(score_triplets)}")
print(f"Score triplets detected by eval logic: {detected_score_count}")

if len(score_triplets) == detected_score_count:
    print("✓ PASS: Counting logic is correct!")
else:
    print("❌ FAIL: Mismatch in counting logic!")

print("\n" + "="*80)
