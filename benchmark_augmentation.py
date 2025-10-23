"""
Benchmark old vs new augmentation approach
"""
import time
import numpy as np
from alignment import align_tokens2
from anticipation.vocab import MASK, CONTROL_OFFSET
from anticipation.config import TIME_RESOLUTION
import pandas as pd
import os

meta_csv = './asap-dataset-master/metadata.csv'
df = pd.read_csv(meta_csv)
row = df.iloc[0]

file1 = os.path.join('./asap-dataset-master', row['midi_performance'])
file2 = os.path.join('./asap-dataset-master', row['midi_score'])
file3 = os.path.join('./asap-dataset-master', row['performance_annotations'])
file4 = os.path.join('./asap-dataset-master', row['midi_score_annotations'])

num_augmentations = 20
perturb_std_ms = 50.0
mask_prob = 0.5

print("Benchmarking augmentation approaches")
print("="*80)
print(f"File: {row['midi_performance']}")
print(f"Augmentations: {num_augmentations}")
print()

# OLD APPROACH: Call align_tokens2 every time
print("OLD APPROACH: Re-align for each augmentation")
start = time.time()
for aug_idx in range(num_augmentations):
    seed = hash((file1, aug_idx)) % (2**32)
    np.random.seed(seed)
    matched_tuples = align_tokens2(file1, file2, file3, file4, skip_Nones=True,
                                   perturb_std_ms=perturb_std_ms, mask_prob=mask_prob)
old_time = time.time() - start
print(f"  Time: {old_time:.3f} seconds")
print(f"  Per augmentation: {old_time/num_augmentations*1000:.1f} ms")

# NEW APPROACH: Align once, then apply augmentation
print("\nNEW APPROACH: Align once, augment cheaply")
start = time.time()

# Align once
matched_tuples_base = align_tokens2(file1, file2, file3, file4, skip_Nones=True,
                                   perturb_std_ms=0.0, mask_prob=0.0)

# Augment multiple times
for aug_idx in range(num_augmentations):
    seed = hash((file1, aug_idx)) % (2**32)
    np.random.seed(seed)
    
    matched_tuples = []
    for match in matched_tuples_base:
        perf_tuple = match[0]
        
        if mask_prob > 0 and np.random.random() < mask_prob:
            perf_tuple = [MASK, MASK, MASK]
        elif perturb_std_ms > 0:
            perturb_std_units = (perturb_std_ms / 1000.0) * TIME_RESOLUTION
            time_perturbation = np.random.normal(0, perturb_std_units)
            base_time = perf_tuple[0] - CONTROL_OFFSET
            perturbed_time = max(0, int(base_time + time_perturbation))
            perf_tuple = [CONTROL_OFFSET + perturbed_time, perf_tuple[1], perf_tuple[2]]
        
        matched_tuples.append([perf_tuple, match[1], match[2], match[3]])

new_time = time.time() - start
print(f"  Time: {new_time:.3f} seconds")
print(f"  Per augmentation: {new_time/num_augmentations*1000:.1f} ms")

print("\n" + "="*80)
print(f"SPEEDUP: {old_time/new_time:.1f}x faster")
print(f"Time saved: {old_time - new_time:.3f} seconds for {num_augmentations} augmentations")
print(f"For full dataset (~230 pieces): {(old_time - new_time) * 230 / 60:.1f} minutes saved")
