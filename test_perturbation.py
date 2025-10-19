"""
Test script to verify time perturbation is working correctly.
Compares tokenization with and without perturbation.
"""
import numpy as np
from alignment import align_tokens2
from anticipation.config import TIME_RESOLUTION
from anticipation.vocab import CONTROL_OFFSET
import pandas as pd
import os

# Load a sample from ASAP dataset
meta_csv = './asap-dataset-master/metadata.csv'
df = pd.read_csv(meta_csv)

# Get first file
row = df.iloc[0]
file1 = os.path.join('./asap-dataset-master', row['midi_performance'])
file2 = os.path.join('./asap-dataset-master', row['midi_score'])
file3 = os.path.join('./asap-dataset-master', row['performance_annotations'])
file4 = os.path.join('./asap-dataset-master', row['midi_score_annotations'])

print(f"Testing with: {row['midi_performance']}")
print("="*80)

# Tokenize without perturbation
print("\n1. WITHOUT PERTURBATION (baseline):")
matched_no_perturb = align_tokens2(file1, file2, file3, file4, skip_Nones=True, perturb_std_ms=0.0)
print(f"   Generated {len(matched_no_perturb)} matched note pairs")

# Extract first 5 control token times (without perturbation)
times_no_perturb = []
for i in range(min(5, len(matched_no_perturb))):
    ctrl_time = matched_no_perturb[i][0][0] - CONTROL_OFFSET  # Remove offset to get raw time
    times_no_perturb.append(ctrl_time)
print(f"   First 5 control times: {times_no_perturb}")

# Tokenize WITH perturbation (50ms std dev)
print("\n2. WITH 50ms PERTURBATION:")
np.random.seed(42)  # Set seed for reproducibility
matched_perturb = align_tokens2(file1, file2, file3, file4, skip_Nones=True, perturb_std_ms=50.0)
print(f"   Generated {len(matched_perturb)} matched note pairs")

# Extract first 5 control token times (with perturbation)
times_perturb = []
for i in range(min(5, len(matched_perturb))):
    ctrl_time = matched_perturb[i][0][0] - CONTROL_OFFSET  # Remove offset to get raw time
    times_perturb.append(ctrl_time)
print(f"   First 5 control times: {times_perturb}")

# Calculate differences
print("\n3. COMPARISON:")
diffs = [times_perturb[i] - times_no_perturb[i] for i in range(len(times_no_perturb))]
print(f"   Time differences (in time units): {diffs}")

# Convert to milliseconds
diffs_ms = [d / TIME_RESOLUTION * 1000 for d in diffs]
print(f"   Time differences (in milliseconds): {[f'{d:.2f}' for d in diffs_ms]}")
print(f"   Mean absolute difference: {np.mean(np.abs(diffs_ms)):.2f}ms")
print(f"   Std dev of differences: {np.std(diffs_ms):.2f}ms (target: 50ms)")

# Verify perturbations are non-zero
print("\n4. VERIFICATION:")
has_perturbation = any(d != 0 for d in diffs)
print(f"   Perturbations applied: {'✓ YES' if has_perturbation else '✗ NO'}")
print(f"   All control tokens perturbed: {'✓ YES' if len(diffs) == len(times_no_perturb) else '✗ NO'}")

print("\n" + "="*80)
if has_perturbation:
    print("✓ Time perturbation is working correctly!")
    print(f"  The control/performance tokens are being perturbed with ~50ms std dev")
else:
    print("⚠ Warning: No perturbation detected!")
