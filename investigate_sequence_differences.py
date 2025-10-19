"""
Investigate why sequence 2 has 100% match while others don't.
This will help us understand if there's still a format mismatch.
"""

from anticipation.vocab import *
from anticipation.config import *


def analyze_sequence_detailed(sequence_tokens, seq_idx):
    """Detailed analysis of a single sequence."""
    print(f"\n{'='*80}")
    print(f"SEQUENCE {seq_idx} DETAILED ANALYSIS")
    print(f"{'='*80}\n")
    
    tokens = sequence_tokens[4:]  # Skip header
    
    # Analyze prefix
    print("PREFIX ANALYSIS:")
    for i in range(min(5, 33)):
        pos = i * 6
        if pos + 6 <= len(tokens):
            ctrl = tokens[pos:pos+3]
            rest = tokens[pos+3:pos+6]
            
            ctrl_time = ctrl[0] - CONTROL_OFFSET
            ctrl_note = ctrl[2] - NOTE_OFFSET - CONTROL_OFFSET
            
            print(f"  Pair {i}: ctrl_note={ctrl_note}, rest={rest[2]==REST}")
    
    # Skip prefix
    pos = 33 * 6
    
    # Analyze alternating section
    print(f"\nALTERNATING SECTION (starting at position {pos}):")
    alternating_items = []
    
    while pos + 3 <= len(tokens):
        triplet = tokens[pos:pos+3]
        
        if triplet[0] >= CONTROL_OFFSET:
            note = triplet[2] - NOTE_OFFSET - CONTROL_OFFSET
            alternating_items.append(('ctrl', note))
        else:
            note = triplet[2] - NOTE_OFFSET
            alternating_items.append(('score', note))
        
        pos += 3
    
    # Print first 10
    print("  First 10 items:")
    for i, (typ, note) in enumerate(alternating_items[:10]):
        print(f"    {i}: {typ:6s} note={note}")
    
    # Check pattern
    print(f"\n  Total alternating items: {len(alternating_items)}")
    
    # Count pattern
    pattern = []
    for i in range(min(20, len(alternating_items))):
        pattern.append(alternating_items[i][0])
    
    print(f"  Pattern: {' -> '.join(pattern)}")
    
    # Separate scores and controls
    scores = [note for typ, note in alternating_items if typ == 'score']
    ctrls = [note for typ, note in alternating_items if typ == 'ctrl']
    
    print(f"\n  Scores in alternating: {len(scores)}")
    print(f"  Controls in alternating: {len(ctrls)}")
    
    return scores, ctrls


# Load sequences
with open('data/test_output.txt', 'r') as f:
    lines = f.readlines()

# Analyze sequence 2 (100% match) and sequence 3 (0.73% match)
for seq_idx in [2, 3]:
    sequence_tokens = [int(tok) for tok in lines[seq_idx].strip().split()]
    scores, ctrls = analyze_sequence_detailed(sequence_tokens, seq_idx)
    
    print(f"\nSCORES vs CONTROLS:")
    print(f"  First 5 scores: {scores[:5]}")
    print(f"  First 5 alternating controls: {ctrls[:5]}")
    
    # Check if they match
    matches = sum(1 for i in range(min(len(scores), len(ctrls))) if scores[i] == ctrls[i])
    print(f"  Matches if we compare score_i to ctrl_i: {matches}/{min(len(scores), len(ctrls))}")

print(f"\n{'='*80}")
print("HYPOTHESIS")
print(f"{'='*80}\n")

print("If sequence 2 has perfect match, it might be because:")
print("1. The alternating pattern is different")
print("2. The scores happen to match controls")
print("3. Our generation is doing something different")
print()
print("Let me check the actual structure more carefully...")
