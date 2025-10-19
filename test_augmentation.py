"""
Test script to verify augmentation (perturbation + masking) is working correctly.
"""
import numpy as np
from alignment import align_tokens2
from anticipation.config import TIME_RESOLUTION
from anticipation.vocab import CONTROL_OFFSET, MASK
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

print(f"Testing augmentation with: {row['midi_performance']}")
print("="*80)

# 1. Baseline without augmentation
print("\n1. BASELINE (no perturbation, no masking):")
np.random.seed(42)
matched_baseline = align_tokens2(file1, file2, file3, file4, skip_Nones=True, 
                                 perturb_std_ms=0.0, mask_prob=0.0)
print(f"   Generated {len(matched_baseline)} matched note pairs")
print(f"   First 5 control tokens: {[matched_baseline[i][0] for i in range(5)]}")

# 2. With perturbation only
print("\n2. WITH PERTURBATION ONLY (50ms std, no masking):")
np.random.seed(42)
matched_perturb = align_tokens2(file1, file2, file3, file4, skip_Nones=True, 
                                perturb_std_ms=50.0, mask_prob=0.0)
n_perturbed = sum(1 for i in range(len(matched_baseline)) 
                  if matched_perturb[i][0][0] != matched_baseline[i][0][0])
print(f"   Perturbed tokens: {n_perturbed}/{len(matched_baseline)} ({100*n_perturbed/len(matched_baseline):.1f}%)")
print(f"   First 5 control tokens: {[matched_perturb[i][0] for i in range(5)]}")

# 3. With masking only
print("\n3. WITH MASKING ONLY (50% mask prob, no perturbation):")
np.random.seed(42)
matched_mask = align_tokens2(file1, file2, file3, file4, skip_Nones=True, 
                             perturb_std_ms=0.0, mask_prob=0.5)
n_masked = sum(1 for i in range(len(matched_mask)) if matched_mask[i][0][0] == MASK)
print(f"   Masked tokens: {n_masked}/{len(matched_mask)} ({100*n_masked/len(matched_mask):.1f}%)")
print(f"   First 5 control tokens: {[matched_mask[i][0] for i in range(5)]}")
print(f"   MASK token value: {MASK}")

# 4. With both augmentations
print("\n4. WITH BOTH AUGMENTATIONS (50ms std + 50% masking):")
np.random.seed(42)
matched_both = align_tokens2(file1, file2, file3, file4, skip_Nones=True, 
                             perturb_std_ms=50.0, mask_prob=0.5)
n_masked = sum(1 for i in range(len(matched_both)) if matched_both[i][0][0] == MASK)
n_perturbed = sum(1 for i in range(len(matched_both)) 
                  if matched_both[i][0][0] != MASK and matched_both[i][0][0] != matched_baseline[i][0][0])
print(f"   Masked tokens: {n_masked}/{len(matched_both)} ({100*n_masked/len(matched_both):.1f}%)")
print(f"   Perturbed (non-masked) tokens: {n_perturbed}/{len(matched_both)-n_masked}")
print(f"   First 10 control tokens: {[matched_both[i][0] for i in range(10)]}")

# 5. Multiple augmentations have different results
print("\n5. UNIQUENESS CHECK (3 different augmentations):")
np.random.seed(100)
aug1 = align_tokens2(file1, file2, file3, file4, skip_Nones=True, 
                     perturb_std_ms=50.0, mask_prob=0.5)
aug2 = align_tokens2(file1, file2, file3, file4, skip_Nones=True, 
                     perturb_std_ms=50.0, mask_prob=0.5)
aug3 = align_tokens2(file1, file2, file3, file4, skip_Nones=True, 
                     perturb_std_ms=50.0, mask_prob=0.5)

diff_1_2 = sum(1 for i in range(len(aug1)) if aug1[i][0] != aug2[i][0])
diff_1_3 = sum(1 for i in range(len(aug1)) if aug1[i][0] != aug3[i][0])
diff_2_3 = sum(1 for i in range(len(aug2)) if aug2[i][0] != aug3[i][0])

print(f"   Aug1 vs Aug2: {diff_1_2} differences ({100*diff_1_2/len(aug1):.1f}%)")
print(f"   Aug1 vs Aug3: {diff_1_3} differences ({100*diff_1_3/len(aug1):.1f}%)")
print(f"   Aug2 vs Aug3: {diff_2_3} differences ({100*diff_2_3/len(aug2):.1f}%)")

print("\n" + "="*80)
success = n_masked > 0 and (40 <= 100*n_masked/len(matched_both) <= 60) and diff_1_2 > 0
print("✓ Augmentation is working correctly!" if success else "⚠ Warning: augmentation may not be working as expected")
