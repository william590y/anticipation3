"""
NOW WE UNDERSTAND THE PATTERN!

The sequence is interleaved: CONTROL, SCORE, CONTROL, SCORE, ...

Given a control triplet at position i, the model should predict the score triplet at position i+3.
The score triplet can be:
- (time, duration, REST) if the score note doesn't match
- (time, duration, NOTE) if there's a corresponding score note

The model's task is to:
1. Predict if there should be a note or REST
2. If note, predict the correct pitch (same as control)
3. Predict the timing and duration

Let's verify:
- Control at position 4: pitch=55
- Score at position 7: REST
- Control at position 10: pitch=71
- Score at position 13: REST
...later...
- Control at position 100: pitch=?
- Score at position 103: pitch=? (should match if not REST)
"""

from anticipation.config import *
from anticipation.vocab import SEPARATOR, NOTE_OFFSET, CONTROL_OFFSET, REST

# Load test data
with open('data/test_clean.txt', 'r') as f:
    lines = f.readlines()

# Parse first sequence
line = lines[0].strip()
sequence = line.split('|')[0].strip()
tokens = [int(x) for x in sequence.split()]

print("="*100)
print("VERIFYING THE INTERLEAVED PATTERN")
print("="*100)
print()

# Find first non-REST score note
start_pos = 4
for i in range(100):
    control_pos = start_pos + i * 6  # Every other triplet
    score_pos = control_pos + 3
    
    if score_pos + 2 >= len(tokens):
        break
    
    control_note = tokens[control_pos + 2] - CONTROL_OFFSET
    score_note = tokens[score_pos + 2]
    
    if score_note != REST and score_note >= NOTE_OFFSET:
        control_pitch = (control_note - NOTE_OFFSET) % MAX_PITCH
        score_pitch = (score_note - NOTE_OFFSET) % MAX_PITCH
        
        match = "✓" if control_pitch == score_pitch else "✗"
        print(f"Pair {i}: Control pos={control_pos}, Score pos={score_pos}")
        print(f"  Control pitch={control_pitch:3d}, Score pitch={score_pitch:3d} {match}")
        print()
        
        if i >= 10:  # Just show first 10 matches
            break

print("="*100)
print("CONCLUSION:")
print("="*100)
print()
print("The model's task:")
print("  1. Given control triplet at position i")
print("  2. Predict score triplet at position i+3")
print("  3. Score can be REST or a NOTE")
print("  4. If NOTE, pitch should match control pitch")
print("  5. But timing and duration may differ")
print()
print("The 'simple pattern' is actually:")
print("  - Look back 3 tokens to the control")
print("  - Decide if score should be REST or NOTE")
print("  - If NOTE, copy the pitch from control")
print("  - Predict timing adjustments and duration")
print()
print("This is NOT a simple copy task because:")
print("  - Must decide REST vs NOTE")
print("  - Must predict timing/duration differences")
print("  - The correspondence is NOT one-to-one")
print("  - Some controls have REST, some have notes")
