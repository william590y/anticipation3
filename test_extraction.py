from anticipation.vocab import *
import sys
sys.path.insert(0, '.')
from test import extract_controls_from_sequence

# Read first test sequence
with open('data/test_output.txt') as f:
    tokens = list(map(int, f.readline().split()))

print("Testing CORRECTED extraction logic...")
print()

controls = extract_controls_from_sequence(tokens)

print(f"Extracted {len(controls)//3} control events")
print(f"Total control tokens: {len(controls)}")
print()

# Show first few and last few
print("First 5 control events:")
for i in range(min(5, len(controls)//3)):
    triplet = controls[i*3:i*3+3]
    has_offset = triplet[0] >= CONTROL_OFFSET
    print(f"  {i}: {triplet}, has_CONTROL_OFFSET={has_offset}")

print()
print("Last 5 control events:")
start_idx = max(0, len(controls)//3 - 5)
for i in range(start_idx, len(controls)//3):
    triplet = controls[i*3:i*3+3]
    has_offset = triplet[0] >= CONTROL_OFFSET
    print(f"  {i}: {triplet}, has_CONTROL_OFFSET={has_offset}")

print()

# Validate ALL
print("Validation:")
invalid = []
for i in range(0, len(controls), 3):
    triplet = controls[i:i+3]
    if triplet[0] < CONTROL_OFFSET:
        invalid.append((i//3, triplet, "Missing CONTROL_OFFSET"))
    if any(tok < 0 or tok >= VOCAB_SIZE for tok in triplet):
        invalid.append((i//3, triplet, "Out of vocab range"))

if invalid:
    print(f"  ✗ Found {len(invalid)} invalid control triplets:")
    for idx, triplet, reason in invalid[:10]:
        print(f"    Event {idx}: {triplet} - {reason}")
else:
    print(f"  ✓ All {len(controls)//3} control events are valid!")
    print(f"  ✓ All tokens in range [{CONTROL_OFFSET}, {VOCAB_SIZE})")
