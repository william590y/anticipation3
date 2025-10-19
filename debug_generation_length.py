"""
Debug why generation is stopping early.
Check the number of controls vs expected generation length.
"""

from anticipation.vocab import *
from anticipation.config import *

def extract_from_sequence(sequence_tokens, prefix_controls=33):
    """Extract controls and scores from test sequence."""
    if len(sequence_tokens) < 4:
        return [], []
    
    tokens = sequence_tokens[4:]  # Skip ANTICIPATE + 3 SEPs
    
    prefix_controls_list = []
    body_scores = []
    body_controls = []
    trailing_scores = []
    
    i = 0
    k = prefix_controls
    
    # Extract prefix - k pairs of [ctrl, rest]
    for _ in range(k):
        if i + 6 <= len(tokens):
            control_triplet = tokens[i:i+3]
            rest_triplet = tokens[i+3:i+6]
            
            if control_triplet[0] >= CONTROL_OFFSET:
                prefix_controls_list.extend(control_triplet)
            else:
                break
            
            i += 6
        else:
            break
    
    # Extract body - alternating [score, future_ctrl]
    while i + 6 <= len(tokens):
        score_triplet = tokens[i:i+3]
        future_ctrl_triplet = tokens[i+3:i+6]
        
        if score_triplet[0] < CONTROL_OFFSET and future_ctrl_triplet[0] >= CONTROL_OFFSET:
            body_scores.extend(score_triplet)
            body_controls.extend(future_ctrl_triplet)
            i += 6
        else:
            break
    
    # Extract trailing scores
    while i + 3 <= len(tokens):
        trailing_triplet = tokens[i:i+3]
        if trailing_triplet[0] < CONTROL_OFFSET:
            trailing_scores.extend(trailing_triplet)
        i += 3
    
    controls = prefix_controls_list + body_controls
    scores = body_scores + trailing_scores
    
    return controls, scores


# Load test sequences
with open('data/test_output.txt', 'r') as f:
    test_lines = f.readlines()

print("=" * 80)
print("GENERATION LENGTH DEBUG")
print("=" * 80)
print()

for seq_idx in range(min(5, len(test_lines))):
    line = test_lines[seq_idx]
    sequence_tokens = [int(tok) for tok in line.strip().split()]
    
    controls, scores = extract_from_sequence(sequence_tokens, prefix_controls=33)
    
    num_controls = len(controls) // 3
    num_scores = len(scores) // 3
    
    print(f"Sequence {seq_idx}:")
    print(f"  Total sequence tokens: {len(sequence_tokens)}")
    print(f"  Body tokens (after SEPs): {len(sequence_tokens) - 4}")
    print(f"  Extracted controls: {num_controls}")
    print(f"  Extracted scores: {num_scores}")
    print()
    
    # Calculate expected generation
    k = 33
    num_to_generate_body = num_controls - k
    num_to_generate_trailing = k
    total_expected = num_to_generate_body + num_to_generate_trailing
    
    print(f"  Expected generation:")
    print(f"    Body (positions 0 to {num_controls-k-1}): {num_to_generate_body}")
    print(f"    Trailing (positions {num_controls-k} to {num_controls-1}): {num_to_generate_trailing}")
    print(f"    Total: {total_expected}")
    print()
    
    if total_expected != num_scores:
        print(f"  ⚠️ MISMATCH: Expected to generate {total_expected} but have {num_scores} ground truth scores")
        print(f"     Difference: {num_scores - total_expected}")
    else:
        print(f"  ✓ Expected generation matches ground truth scores")
    
    print()
    print("-" * 80)
    print()
