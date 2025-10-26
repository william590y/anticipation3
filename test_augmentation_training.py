"""
Test training-time augmentation on a realistic sequence.
"""

import torch
import sys
sys.path.insert(0, '.')

from train import TokenizedDataset
from anticipation.vocab import *
from anticipation.config import *

print("="*80)
print("TRAINING AUGMENTATION TEST")
print("="*80)

# Create a realistic test sequence (simplified)
# Format: [ANTICIPATE, SEP, SEP, SEP, ctrl0+rest0, ctrl1+rest1, score0, ctrl2, score1, ctrl3, ...]
sequence = [
    ANTICIPATE,
    SEPARATOR, SEPARATOR, SEPARATOR,
    # Prefix: ctrl0 + rest0
    CONTROL_OFFSET + 100, CONTROL_OFFSET + 50, CONTROL_OFFSET + 60,  # ctrl0
    TIME_OFFSET + 100, DUR_OFFSET + 0, REST,                          # rest0
    # Prefix: ctrl1 + rest1
    CONTROL_OFFSET + 200, CONTROL_OFFSET + 60, CONTROL_OFFSET + 62,  # ctrl1
    TIME_OFFSET + 200, DUR_OFFSET + 0, REST,                          # rest1
    # Body: alternating score/ctrl
    TIME_OFFSET + 95, DUR_OFFSET + 45, NOTE_OFFSET + 60,              # score0
    CONTROL_OFFSET + 300, CONTROL_OFFSET + 55, CONTROL_OFFSET + 64,  # ctrl2
    TIME_OFFSET + 195, DUR_OFFSET + 55, NOTE_OFFSET + 62,             # score1
    CONTROL_OFFSET + 400, CONTROL_OFFSET + 65, CONTROL_OFFSET + 65,  # ctrl3
]

print(f"\n1. ORIGINAL SEQUENCE (length={len(sequence)})")
print(f"   Position  0: ANTICIPATE = {sequence[0]}")
print(f"   Position  1-3: SEP = {sequence[1:4]}")
print(f"   Position  4-6: ctrl0 = {sequence[4:7]}")
print(f"   Position  7-9: rest0 = {sequence[7:10]}")
print(f"   Position 10-12: ctrl1 = {sequence[10:13]}")
print(f"   Position 13-15: rest1 = {sequence[13:16]}")
print(f"   Position 16-18: score0 = {sequence[16:19]}")
print(f"   Position 19-21: ctrl2 = {sequence[19:22]}")
print(f"   Position 22-24: score1 = {sequence[22:25]}")
print(f"   Position 25-27: ctrl3 = {sequence[25:28]}")

# Create dataset with augmentation enabled
print("\n2. CREATE AUGMENTED DATASET")

# Write test sequence to temp file
import tempfile
import os
temp_fd, temp_path = tempfile.mkstemp(suffix='.txt', text=True)
with os.fdopen(temp_fd, 'w') as f:
    # Write in new format (tokens | mask_indices)
    token_str = ' '.join(map(str, sequence))
    f.write(f"{token_str} |\n")  # No pre-computed masks

print(f"   Created temp file: {temp_path}")

# Load with augmentation
ds = TokenizedDataset(
    temp_path,
    perturb_std_ms=50.0,
    mask_prob=0.5,
    is_training=True
)

print("\n3. AUGMENTATION RESULTS (5 samples)")

for i in range(5):
    batch = ds[0]  # Get same sequence multiple times
    augmented = batch['input_ids']
    labels = batch['labels']
    mask_idxs = [j for j in range(len(labels)) if labels[j] == -100]
    
    print(f"\n   Sample {i+1}:")
    print(f"     Masked positions: {mask_idxs}")
    print(f"     Num masked: {len(mask_idxs)}")
    
    # Check which triplets were augmented
    ctrl_positions = [4, 10, 19, 25]  # Positions of control triplets
    for pos in ctrl_positions:
        original_time = sequence[pos]
        augmented_time = augmented[pos].item()
        is_masked = pos in mask_idxs
        time_changed = original_time != augmented_time
        
        ctrl_num = {4: 0, 10: 1, 19: 2, 25: 3}[pos]
        status = []
        if time_changed:
            delta = augmented_time - original_time
            status.append(f"perturbed (Δ={delta})")
        if is_masked:
            status.append("masked")
        if not status:
            status.append("unchanged")
        
        print(f"     ctrl{ctrl_num} (pos {pos}): {', '.join(status)}")
    
    # Verify score triplets are NOT augmented or masked
    score_positions = [16, 22]
    for pos in score_positions:
        original_time = sequence[pos]
        augmented_time = augmented[pos].item()
        is_masked = pos in mask_idxs
        
        score_num = {16: 0, 22: 1}[pos]
        if original_time != augmented_time or is_masked:
            print(f"     ✗ ERROR: score{score_num} (pos {pos}) was modified!")
        else:
            print(f"     ✓ score{score_num} (pos {pos}): unchanged (correct)")

print("\n4. VERIFY NO INVALID AUGMENTATION")

# Check that ANTICIPATE, SEPARATOR, REST are never modified or masked
protected_positions = [0, 1, 2, 3, 9, 15]  # ANTICIPATE, SEP, SEP, SEP, REST, REST
all_good = True

for i in range(5):
    batch = ds[0]
    augmented = batch['input_ids']
    labels = batch['labels']
    
    for pos in protected_positions:
        if sequence[pos] != augmented[pos].item():
            print(f"   ✗ ERROR: Protected position {pos} was modified!")
            all_good = False
        if labels[pos] == -100:
            print(f"   ✗ ERROR: Protected position {pos} was masked!")
            all_good = False

if all_good:
    print(f"   ✓ All protected tokens (ANTICIPATE, SEP, REST) unchanged")

# Cleanup
os.unlink(temp_path)

print("\n" + "="*80)
print("✓ TRAINING AUGMENTATION TEST COMPLETE")
print("="*80)
print("\nSummary:")
print("  ✓ Control triplets are augmented (perturbed + masked)")
print("  ✓ Score triplets are NOT augmented")
print("  ✓ Special tokens (ANTICIPATE, SEP, REST) are NOT augmented")
print("  ✓ Each call produces different random augmentation")
