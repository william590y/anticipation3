"""
Verify tokenization structure for masked loss training.

This script analyzes the tokenization format to determine which tokens
should be predicted (scores) vs which should be masked (controls, rests, separators).
"""

from anticipation.vocab import *
from anticipation.config import *

def analyze_sequence_structure(sequence_tokens):
    """
    Analyze a tokenized sequence to identify which tokens should be predicted.
    
    Returns:
        dict with token positions and types
    """
    if len(sequence_tokens) < 4:
        return None
    
    # Expected format: [ANTICIPATE, SEP, SEP, SEP, body_tokens...]
    header = sequence_tokens[:4]
    body = sequence_tokens[4:]
    
    analysis = {
        'total_tokens': len(sequence_tokens),
        'header': header,
        'body_length': len(body),
        'should_predict': [],  # Indices of tokens to predict (scores)
        'should_mask': [],      # Indices of tokens to mask (controls, rests, SEPs)
    }
    
    # Header tokens should all be masked
    for i in range(4):
        analysis['should_mask'].append(i)
    
    # Analyze body tokens
    k = 33  # prefix_controls
    pos = 4  # Start after header
    
    # Phase 1: Prefix (k pairs of [ctrl, rest])
    print(f"\nPhase 1: PREFIX ({k} control+rest pairs)")
    for i in range(k):
        if pos + 6 <= len(sequence_tokens):
            ctrl = sequence_tokens[pos:pos+3]
            rest = sequence_tokens[pos+3:pos+6]
            
            # All 6 tokens should be masked (controls and rests are given)
            for j in range(6):
                analysis['should_mask'].append(pos + j)
            
            if i < 3:  # Show first few
                print(f"  Pair {i}: ctrl={ctrl}, rest={rest} -> MASK")
            
            pos += 6
        else:
            break
    
    print(f"  ... (total {k} pairs)")
    print(f"  Masked tokens: {pos - 4}")
    
    # Phase 2: Body (alternating [score, ctrl] or trailing [score])
    print(f"\nPhase 2: BODY (alternating score/control)")
    
    body_start = pos
    score_count = 0
    ctrl_count = 0
    
    while pos + 3 <= len(sequence_tokens):
        triplet = sequence_tokens[pos:pos+3]
        
        # Check if it's a control or score
        if triplet[0] >= CONTROL_OFFSET:
            # Control - should be masked
            for j in range(3):
                analysis['should_mask'].append(pos + j)
            ctrl_count += 1
            label = "CTRL"
        elif triplet[2] == REST:
            # Rest - should be masked (shouldn't happen in body, but handle it)
            for j in range(3):
                analysis['should_mask'].append(pos + j)
            label = "REST"
        else:
            # Score - should be predicted!
            for j in range(3):
                analysis['should_predict'].append(pos + j)
            score_count += 1
            label = "SCORE"
        
        if pos - body_start < 30:  # Show first few
            print(f"  Pos {pos}: {triplet} -> {label}")
        elif pos - body_start == 30:
            print(f"  ...")
        
        pos += 3
    
    print(f"  Scores (PREDICT): {score_count}")
    print(f"  Controls (MASK): {ctrl_count}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total tokens: {analysis['total_tokens']}")
    print(f"To PREDICT (scores): {len(analysis['should_predict'])} tokens ({len(analysis['should_predict'])/analysis['total_tokens']*100:.1f}%)")
    print(f"To MASK (ctrl/rest/sep): {len(analysis['should_mask'])} tokens ({len(analysis['should_mask'])/analysis['total_tokens']*100:.1f}%)")
    print(f"\nScore triplets: {len(analysis['should_predict'])//3}")
    print(f"Masked triplets: {len(analysis['should_mask'])//3}")
    
    return analysis


# Load a sample sequence
print("="*60)
print("TOKENIZATION STRUCTURE ANALYSIS")
print("="*60)

with open('data/train_output.txt', 'r') as f:
    line = f.readline()

sequence = [int(tok) for tok in line.strip().split()]
print(f"\nAnalyzing sample sequence ({len(sequence)} tokens)...")

analysis = analyze_sequence_structure(sequence)

# Verify the analysis is correct
print(f"\n{'='*60}")
print(f"VERIFICATION")
print(f"{'='*60}")

# Check a few specific positions
print(f"\nChecking specific positions:")
print(f"  Token 0 (should be ANTICIPATE={ANTICIPATE}): {sequence[0]} {'✓' if sequence[0] == ANTICIPATE else '✗'}")
print(f"  Token 1-3 (should be SEP={SEPARATOR}): {sequence[1:4]} {'✓' if all(t == SEPARATOR for t in sequence[1:4]) else '✗'}")

# Check first control in prefix
first_ctrl = sequence[4:7]
print(f"  Token 4-6 (first ctrl, should have CONTROL_OFFSET): {first_ctrl}")
print(f"    Has CONTROL_OFFSET: {'✓' if first_ctrl[0] >= CONTROL_OFFSET else '✗'}")

# Check first rest
first_rest = sequence[7:10]
print(f"  Token 7-9 (first rest, should end with REST={REST}): {first_rest}")
print(f"    Ends with REST: {'✓' if first_rest[2] == REST else '✗'}")

# Find first score (should be after 66 tokens of prefix)
prefix_end = 4 + 33*6  # Header + 33 control+rest pairs
if prefix_end < len(sequence):
    first_score = sequence[prefix_end:prefix_end+3]
    print(f"  Token {prefix_end}-{prefix_end+2} (first score, should NOT have CONTROL_OFFSET): {first_score}")
    print(f"    Is score: {'✓' if first_score[0] < CONTROL_OFFSET and first_score[2] != REST else '✗'}")

print(f"\n{'='*60}")
print(f"READY FOR MASKED LOSS TRAINING")
print(f"{'='*60}")
print(f"\nStrategy:")
print(f"  1. Set labels = input_ids.clone()")
print(f"  2. For each token position:")
print(f"     - If token >= CONTROL_OFFSET: labels[pos] = -100 (mask)")
print(f"     - If token == REST: labels[pos] = -100 (mask)")
print(f"     - If token == SEPARATOR: labels[pos] = -100 (mask)")
print(f"     - If token == ANTICIPATE: labels[pos] = -100 (mask)")
print(f"     - Otherwise: keep label (predict score)")
print(f"\nThis will train the model to predict ONLY score tokens (~{len(analysis['should_predict'])//3} per sequence)")
