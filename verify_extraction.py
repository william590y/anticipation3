from anticipation.vocab import *

# Read first test sequence
with open('data/test_output.txt') as f:
    tokens = list(map(int, f.readline().split()))

print("Testing extraction logic...")
print()

# Skip [ANTICIPATE, SEP, SEP, SEP]
data = tokens[4:]

print("PREFIX extraction:")
controls_prefix = []
for i in range(33):
    ctrl = data[i*6:i*6+3]
    rest = data[i*6+3:i*6+6]
    controls_prefix.extend(ctrl)
    if i < 3:
        print(f"  {i}: ctrl={ctrl} (has CONTROL_OFFSET: {ctrl[0] >= CONTROL_OFFSET})")

print(f"\nExtracted {len(controls_prefix)//3} prefix controls")
print()

print("ALTERNATING extraction:")
start = 33 * 6
controls_alt = []
for i in range(min(5, (len(data) - start) // 6)):
    score = data[start + i*6:start + i*6+3]
    perf_ctrl = data[start + i*6+3:start + i*6+6]
    controls_alt.extend(perf_ctrl)
    print(f"  {i}: score={score} (is score: {score[0] < CONTROL_OFFSET}), "
          f"perf_ctrl={perf_ctrl} (has CONTROL_OFFSET: {perf_ctrl[0] >= CONTROL_OFFSET})")

print(f"\nExtracted {len(controls_alt)//3} alternating controls (first 5)")
print()

all_controls = controls_prefix + controls_alt
print(f"Total controls extracted: {len(all_controls)//3} events")
print()

# Validate
print("Validation:")
invalid = []
for i in range(0, len(all_controls), 3):
    triplet = all_controls[i:i+3]
    if triplet[0] < CONTROL_OFFSET:
        invalid.append((i//3, triplet))
    if any(tok >= VOCAB_SIZE for tok in triplet):
        invalid.append((i//3, triplet))

if invalid:
    print(f"  ✗ Found {len(invalid)} invalid control triplets:")
    for idx, triplet in invalid[:5]:
        print(f"    Event {idx}: {triplet}")
else:
    print(f"  ✓ All {len(all_controls)//3} control events are valid!")
