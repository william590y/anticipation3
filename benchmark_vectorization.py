"""
Benchmark vectorized vs loop-based augmentation
"""
import time
import numpy as np
from anticipation.vocab import MASK, CONTROL_OFFSET
from anticipation.config import TIME_RESOLUTION

# Simulate matched_tuples_base
n_tuples = 1000
matched_tuples_base = [
    [[CONTROL_OFFSET + i*10, 100, 5000 + i], i, [i*10, 100, 4000 + i], i]
    for i in range(n_tuples)
]

perturb_std_ms = 50.0
mask_prob = 0.5
num_tests = 100

print(f"Benchmarking augmentation on {n_tuples} tuples, {num_tests} runs")
print("="*80)

# OLD APPROACH: Loop with individual random calls
print("\nOLD APPROACH: Individual random calls per tuple")
start = time.time()
for _ in range(num_tests):
    np.random.seed(42)
    matched_tuples = []
    for match in matched_tuples_base:
        perf_tuple = list(match[0])
        
        if mask_prob > 0 and np.random.random() < mask_prob:
            perf_tuple = [MASK, MASK, MASK]
        elif perturb_std_ms > 0:
            perturb_std_units = (perturb_std_ms / 1000.0) * TIME_RESOLUTION
            time_perturbation = np.random.normal(0, perturb_std_units)
            base_time = perf_tuple[0] - CONTROL_OFFSET
            perturbed_time = max(0, int(base_time + time_perturbation))
            perf_tuple = [CONTROL_OFFSET + perturbed_time, perf_tuple[1], perf_tuple[2]]
        
        matched_tuples.append([perf_tuple, match[1], match[2], match[3]])

old_time = time.time() - start
print(f"  Time: {old_time:.3f} seconds")
print(f"  Per run: {old_time/num_tests*1000:.2f} ms")

# NEW APPROACH: Vectorized random generation
print("\nNEW APPROACH: Vectorized random generation")
start = time.time()
for _ in range(num_tests):
    np.random.seed(42)
    n = len(matched_tuples_base)
    
    # Generate all random values at once
    mask_decisions = np.random.random(n) < mask_prob
    perturb_std_units = (perturb_std_ms / 1000.0) * TIME_RESOLUTION
    time_perturbations = np.random.normal(0, perturb_std_units, n).astype(int)
    
    matched_tuples = []
    for i, match in enumerate(matched_tuples_base):
        perf_tuple = list(match[0])
        
        if mask_decisions[i]:
            perf_tuple = [MASK, MASK, MASK]
        elif time_perturbations[i] != 0:
            base_time = perf_tuple[0] - CONTROL_OFFSET
            perturbed_time = max(0, base_time + time_perturbations[i])
            perf_tuple = [CONTROL_OFFSET + perturbed_time, perf_tuple[1], perf_tuple[2]]
        
        matched_tuples.append([perf_tuple, match[1], match[2], match[3]])

new_time = time.time() - start
print(f"  Time: {new_time:.3f} seconds")
print(f"  Per run: {new_time/num_tests*1000:.2f} ms")

# FULLY VECTORIZED: No loop for applying augmentation
print("\nFULLY VECTORIZED: Numpy array operations")
start = time.time()
for _ in range(num_tests):
    np.random.seed(42)
    n = len(matched_tuples_base)
    
    # Extract control tokens as numpy array
    control_tokens = np.array([match[0] for match in matched_tuples_base])
    
    # Generate all random values at once
    mask_decisions = np.random.random(n) < mask_prob
    perturb_std_units = (perturb_std_ms / 1000.0) * TIME_RESOLUTION
    time_perturbations = np.random.normal(0, perturb_std_units, n).astype(int)
    
    # Copy and apply augmentation (vectorized)
    augmented_controls = control_tokens.copy()
    augmented_controls[mask_decisions] = MASK
    
    # Apply time perturbation to non-masked tokens
    non_masked = ~mask_decisions
    base_times = augmented_controls[non_masked, 0] - CONTROL_OFFSET
    perturbed_times = np.maximum(0, base_times + time_perturbations[non_masked])
    augmented_controls[non_masked, 0] = CONTROL_OFFSET + perturbed_times
    
    # Convert back to list format
    matched_tuples = [
        [augmented_controls[i].tolist(), matched_tuples_base[i][1],
         matched_tuples_base[i][2], matched_tuples_base[i][3]]
        for i in range(n)
    ]

fully_vec_time = time.time() - start
print(f"  Time: {fully_vec_time:.3f} seconds")
print(f"  Per run: {fully_vec_time/num_tests*1000:.2f} ms")

print("\n" + "="*80)
print(f"Semi-vectorized vs Old: {old_time/new_time:.2f}x faster")
print(f"Fully-vectorized vs Old: {old_time/fully_vec_time:.2f}x faster")
print(f"Fully-vectorized vs Semi: {new_time/fully_vec_time:.2f}x faster")
