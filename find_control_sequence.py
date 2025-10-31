"""
Find the actual control sequence in the full sequence.

The bootstrap is just 1 control note. But there should be a full control sequence
somewhere that matches the score sequence.
"""

import torch
from anticipation.config import *
from anticipation.vocab import SEPARATOR, NOTE_OFFSET, CONTROL_OFFSET

SEP = SEPARATOR

# Load test data
with open('data/test_clean.txt', 'r') as f:
    lines = f.readlines()

# Parse first sequence - sequences are separated by |
line = lines[0].strip()
# Split by | and take first sequence
sequence = line.split('|')[0].strip()
tokens = [int(x) for x in sequence.split()]
print(f"Total sequence length: {len(tokens)} tokens")
print()

# Find where different parts start
mode_token = tokens[0]
print(f"Mode token: {mode_token}")

# Count SEP tokens
sep_indices = [i for i, t in enumerate(tokens) if t == SEP]
print(f"SEP tokens at positions: {sep_indices[:10]}...")  # First 10
print(f"Total SEP tokens: {len(sep_indices)}")
print()

# The structure should be:
# MODE, bootstrap_control, SEP, SEP, SEP, control_sequence, SEP, SEP, SEP, score_sequence
# Bootstrap has 1 control note = 3 tokens (onset, duration, note)
# Then 3 SEPs
# Then control sequence should start

bootstrap_end = 1 + 3  # mode + 1 triplet
first_seps = 4  # mode + bootstrap
control_start = first_seps + 3  # After 3 SEPs

print(f"Bootstrap: tokens[1:4] = {tokens[1:4]}")
print(f"First 3 SEPs: tokens[4:7] = {tokens[4:7]}")
print()

# Count notes in different sections
def count_notes(start_pos, max_len=300):
    """Count note tokens (not onset/duration)"""
    count = 0
    pos = start_pos + 2  # Skip to first note token
    while pos < len(tokens) and pos < start_pos + max_len:
        if tokens[pos] == SEP:
            break
        if tokens[pos] >= NOTE_OFFSET:
            count += 1
        pos += 3  # Move to next triplet
    return count, pos

# Count control notes
control_count, control_end = count_notes(control_start)
print(f"Control sequence starts at position: {control_start}")
print(f"Control notes: {control_count}")
print(f"Control sequence ends at position: {control_end}")
print()

# After control there should be 3 more SEPs
score_start = control_end + 3
score_count, score_end = count_notes(score_start)
print(f"Score sequence starts at position: {score_start}")
print(f"Score notes: {score_count}")
print(f"Score sequence ends at position: {score_end}")
print()

print("="*80)
print("CHECKING PITCH MATCHING")
print("="*80)

# Now check if score pitches match control pitches
matches = 0
mismatches = 0

print(f"Checking first 10 notes (control has {control_count}, score has {score_count})")
print()

for note_idx in range(min(10, score_count, control_count)):
    control_pos = control_start + note_idx * 3 + 2
    score_pos = score_start + note_idx * 3 + 2
    
    control_token = tokens[control_pos]
    score_token = tokens[score_pos]
    
    print(f"Note {note_idx}: control_token={control_token}, score_token={score_token}")
    
    # Check if they are in valid ranges
    is_control = control_token >= CONTROL_OFFSET
    is_score = score_token >= NOTE_OFFSET
    
    print(f"  is_control={is_control}, is_score={is_score}")
    
    if is_control and is_score:
        control_pitch = (control_token - CONTROL_OFFSET - NOTE_OFFSET) % MAX_PITCH
        score_pitch = (score_token - NOTE_OFFSET) % MAX_PITCH
        
        match = "✓" if control_pitch == score_pitch else "✗"
        offset = score_pos - control_pos
        
        print(f"  control_pos={control_pos}, score_pos={score_pos}, offset={offset}")
        print(f"  control_pitch={control_pitch:3d}, score_pitch={score_pitch:3d} {match}")
        
        if control_pitch == score_pitch:
            matches += 1
        else:
            mismatches += 1
    print()

print()
print(f"Matches: {matches}/{matches+mismatches}")
print()

# Calculate the offset
if score_count > 0 and control_count > 0:
    offset = score_start - control_start
    print(f"ACTUAL OFFSET: {offset} tokens ({offset // 3} triplets)")
    print()
    print("This is the offset the model needs to learn!")
