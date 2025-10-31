"""
Properly analyze the sequence structure.

Structure should be:
MODE, SEP, SEP, SEP, control_sequence, SEP, SEP, SEP, score_sequence

Each event is (time, duration, note) triplet.
Control events have all 3 tokens >= CONTROL_OFFSET.
Score events have tokens < CONTROL_OFFSET (but note might be REST).
"""

from anticipation.config import *
from anticipation.vocab import SEPARATOR, TIME_OFFSET, NOTE_OFFSET, CONTROL_OFFSET, REST

# Load test data
with open('data/test_clean.txt', 'r') as f:
    lines = f.readlines()

# Parse first sequence
line = lines[0].strip()
sequence = line.split('|')[0].strip()
tokens = [int(x) for x in sequence.split()]

print(f"Total tokens: {len(tokens)}")
print()

# First token is mode
mode = tokens[0]
print(f"Mode: {mode}")
print()

# Find SEP positions
sep_positions = [i for i, t in enumerate(tokens) if t == SEPARATOR]
print(f"SEPARATOR = {SEPARATOR}")
print(f"SEP positions: {sep_positions[:20]}")  # First 20
print()

# The structure should be:
# [0]: mode
# [1:4]: 3 SEPs (bootstrap)
# [4:?]: control sequence
# [...]: 3 SEPs
# [...:]: score sequence

# Find where control sequence ends (first non-control token after position 4)
control_start = 4
i = control_start
while i < len(tokens) - 2:
    t0, t1, t2 = tokens[i], tokens[i+1], tokens[i+2]
    
    # Check if this is a control triplet
    # All 3 tokens should be >= CONTROL_OFFSET OR be SEPARATOR
    if t0 == SEPARATOR:
        # Found the separators between control and score
        if tokens[i+1] == SEPARATOR and tokens[i+2] == SEPARATOR:
            control_end = i
            score_start = i + 3
            break
    elif t0 < CONTROL_OFFSET:
        # Found first non-control token
        print(f"ERROR: Found non-control token at position {i}: {t0}")
        print(f"This should be the separator section!")
        break
    
    i += 3
else:
    print("ERROR: Could not find separator between control and score!")
    control_end = -1
    score_start = -1

print(f"Control sequence: positions {control_start} to {control_end}")
control_triplets = (control_end - control_start) // 3
print(f"Control triplets: {control_triplets}")
print()

print(f"Score sequence starts at: {score_start}")
score_triplets = (len(tokens) - score_start) // 3
print(f"Score triplets (approximate): {score_triplets}")
print()

# Print first few tokens of each section
print("="*80)
print("FIRST CONTROL TRIPLETS:")
print("="*80)
for i in range(min(5, control_triplets)):
    pos = control_start + i * 3
    t0, t1, t2 = tokens[pos], tokens[pos+1], tokens[pos+2]
    
    # Decode
    time = t0 - CONTROL_OFFSET
    dur = t1 - CONTROL_OFFSET
    note = t2 - CONTROL_OFFSET
    
    # Extract pitch and instrument
    pitch = (note - NOTE_OFFSET) % MAX_PITCH
    instr = (note - NOTE_OFFSET) // MAX_PITCH
    
    print(f"Triplet {i}: time={time:4d}, dur={dur:4d}, note={note:5d} (pitch={pitch:3d}, instr={instr})")

print()
print("="*80)
print("FIRST SCORE TRIPLETS:")
print("="*80)
for i in range(min(10, score_triplets)):
    pos = score_start + i * 3
    t0, t1, t2 = tokens[pos], tokens[pos+1], tokens[pos+2]
    
    # Decode
    time = t0  # Not offset for score
    dur = t1
    note = t2
    
    if note == REST:
        print(f"Triplet {i}: time={time:4d}, dur={dur:5d}, note=REST")
    elif note >= NOTE_OFFSET:
        # Extract pitch and instrument  
        pitch = (note - NOTE_OFFSET) % MAX_PITCH
        instr = (note - NOTE_OFFSET) // MAX_PITCH
        print(f"Triplet {i}: time={time:4d}, dur={dur:5d}, note={note:5d} (pitch={pitch:3d}, instr={instr})")
    else:
        print(f"Triplet {i}: time={time:4d}, dur={dur:5d}, note={note:5d} (INVALID?)")

print()
print("="*80)
print("NOW CHECK IF PITCHES MATCH (ignoring REST):")
print("="*80)

# Find matching pitches (skip REST in score)
score_idx = 0
matches = 0
total = 0

for control_idx in range(control_triplets):
    # Get control pitch
    control_note_pos = control_start + control_idx * 3 + 2
    control_note = tokens[control_note_pos] - CONTROL_OFFSET
    control_pitch = (control_note - NOTE_OFFSET) % MAX_PITCH
    
    # Find next non-REST score note
    while score_idx < score_triplets:
        score_note_pos = score_start + score_idx * 3 + 2
        score_note = tokens[score_note_pos]
        
        if score_note == REST:
            score_idx += 1
            continue
        
        # Found a real note
        score_pitch = (score_note - NOTE_OFFSET) % MAX_PITCH
        
        match = "✓" if control_pitch == score_pitch else "✗"
        if control_idx < 10:  # Print first 10
            print(f"Control {control_idx} (pitch {control_pitch:3d}) -> Score {score_idx} (pitch {score_pitch:3d}) {match}")
        
        if control_pitch == score_pitch:
            matches += 1
        total += 1
        
        score_idx += 1
        break
    else:
        # No more score notes
        break

print()
print(f"Matches: {matches}/{total} = {100*matches/total:.1f}%")
