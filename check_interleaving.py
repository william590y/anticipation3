"""
Analyze if the sequence is interleaved (control, score, control, score, ...)
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

print(f"Total tokens: {len(tokens)}\n")

# First token is mode, then 3 SEPs bootstrap
mode = tokens[0]
print(f"Mode: {mode}")
print(f"Separators (bootstrap): {tokens[1:4]}")
print()

# Now analyze triplets
print("First 30 triplets (after bootstrap SEPs):")
print("="*100)
print(f"{'Triplet':<8} {'Pos':<5} {'T0':<7} {'T1':<7} {'T2':<7} {'Type':<12} {'Details'}")
print("="*100)

start_pos = 4  # After mode + 3 SEPs

for i in range(30):
    pos = start_pos + i * 3
    if pos + 2 >= len(tokens):
        break
    
    t0, t1, t2 = tokens[pos], tokens[pos+1], tokens[pos+2]
    
    # Determine type
    if t0 >= CONTROL_OFFSET:
        typ = "CONTROL"
        time = t0 - CONTROL_OFFSET
        dur = t1 - CONTROL_OFFSET
        note = t2 - CONTROL_OFFSET
        pitch = (note - NOTE_OFFSET) % MAX_PITCH if note >= NOTE_OFFSET else -1
        details = f"t={time:4d}, d={dur:4d}, pitch={pitch:3d}"
    elif t2 == REST:
        typ = "SCORE (REST)"
        details = f"t={t0:4d}, d={t1:5d}"
    elif t2 >= NOTE_OFFSET:
        typ = "SCORE (NOTE)"
        pitch = (t2 - NOTE_OFFSET) % MAX_PITCH
        details = f"t={t0:4d}, d={t1:5d}, pitch={pitch:3d}"
    else:
        typ = "UNKNOWN"
        details = f"t0={t0}, t1={t1}, t2={t2}"
    
    print(f"{i:<8} {pos:<5} {t0:<7} {t1:<7} {t2:<7} {typ:<12} {details}")

print()
print("="*100)
print("PATTERN ANALYSIS:")
print("="*100)

# Count the pattern
types = []
start_pos = 4
for i in range(100):
    pos = start_pos + i * 3
    if pos + 2 >= len(tokens):
        break
    
    t0, t1, t2 = tokens[pos], tokens[pos+1], tokens[pos+2]
    
    if t0 >= CONTROL_OFFSET:
        types.append('C')
    elif t2 == REST:
        types.append('R')
    elif t2 >= NOTE_OFFSET:
        types.append('S')
    else:
        types.append('?')

print("First 100 triplets:", ''.join(types))
print()

# Find the pattern
from collections import Counter
counter = Counter(types)
print(f"Counts: {dict(counter)}")
print(f"Total triplets analyzed: {len(types)}")
