"""Check if vectorization speedup is realistic or if we're skipping work"""
import time
import numpy as np

print("=" * 70)
print("VECTORIZATION SPEEDUP VERIFICATION")
print("=" * 70)

# Test 1: Random generation speedup
print("\n1. Random number generation (10000 values):")
n = 10000

start = time.time()
for i in range(n):
    x = np.random.random() < 0.5
elapsed_loop = time.time() - start
print(f"   Loop: {elapsed_loop:.4f}s")

start = time.time()
x = np.random.random(n) < 0.5
elapsed_vec = time.time() - start
print(f"   Vectorized: {elapsed_vec:.6f}s")
print(f"   Speedup: {elapsed_loop / elapsed_vec:.1f}x")

# Test 2: Normal distribution generation
print("\n2. Normal distribution (10000 values, mean=0, std=50):")
start = time.time()
for i in range(n):
    x = np.random.normal(0, 50)
elapsed_loop = time.time() - start
print(f"   Loop: {elapsed_loop:.4f}s")

start = time.time()
x = np.random.normal(0, 50, n)
elapsed_vec = time.time() - start
print(f"   Vectorized: {elapsed_vec:.6f}s")
print(f"   Speedup: {elapsed_loop / elapsed_vec:.1f}x")

# Test 3: Simulate actual augmentation workload
print("\n3. Realistic augmentation simulation (1000 pieces, 734 tuples each, 20 augmentations):")
n_pieces = 1000
n_tuples = 734
n_aug = 20

# OLD METHOD: Generate random values per augmentation, per tuple
print("\n   OLD METHOD (generate in loop):")
start = time.time()
for piece in range(n_pieces):
    for aug in range(n_aug):
        np.random.seed(piece * 1000 + aug)
        for i in range(n_tuples):
            mask_decision = np.random.random() < 0.5
            time_pert = np.random.normal(0, 50)
elapsed_old = time.time() - start
print(f"   Time: {elapsed_old:.2f}s")

# NEW METHOD: Generate all random values at once per augmentation
print("\n   NEW METHOD (vectorized per augmentation):")
start = time.time()
for piece in range(n_pieces):
    for aug in range(n_aug):
        np.random.seed(piece * 1000 + aug)
        mask_decisions = np.random.random(n_tuples) < 0.5
        time_perts = np.random.normal(0, 50, n_tuples)
elapsed_new = time.time() - start
print(f"   Time: {elapsed_new:.2f}s")
print(f"   Speedup: {elapsed_old / elapsed_new:.1f}x")

# Test 4: Check if we're actually doing the same work
print("\n4. Verify same random values generated:")
np.random.seed(42)
loop_vals = []
for i in range(10):
    loop_vals.append(np.random.random() < 0.5)

np.random.seed(42)
vec_vals = (np.random.random(10) < 0.5).tolist()

print(f"   Loop values:      {loop_vals}")
print(f"   Vectorized values: {vec_vals}")
print(f"   Match: {loop_vals == vec_vals}")

# Test 5: Estimate total tokenization time
print("\n5. Estimate full tokenization time:")
print(f"   Dataset: ~1067 pieces")
print(f"   Augmentations: 20 per piece")
print(f"   Average tuples per piece: ~734")
print(f"   Workers: 200")

# Estimate random generation time only
pieces = 1067
aug_per_piece = 20
tuples_per_piece = 734

# OLD: loop-based random generation
old_time_per_piece = (elapsed_old / n_pieces)  
old_total = old_time_per_piece * pieces / 200  # Parallel workers
print(f"\n   OLD METHOD:")
print(f"   - Random generation per piece: {old_time_per_piece*1000:.1f}ms")
print(f"   - Total with 200 workers: {old_total:.1f}s = {old_total/60:.1f}min")

# NEW: vectorized random generation
new_time_per_piece = (elapsed_new / n_pieces)
new_total = new_time_per_piece * pieces / 200
print(f"\n   NEW METHOD:")
print(f"   - Random generation per piece: {new_time_per_piece*1000:.1f}ms")
print(f"   - Total with 200 workers: {new_total:.1f}s = {new_total/60:.1f}min")

print(f"\n   NOTE: This only measures random generation!")
print(f"   Actual tokenization includes:")
print(f"   - MIDI parsing (expensive)")
print(f"   - Alignment matching (expensive)")
print(f"   - Token packing")
print(f"   - File I/O")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
if elapsed_old / elapsed_new > 100:
    print("⚠ WARNING: Speedup seems too good to be true!")
    print("  Possible issues:")
    print("  - Not doing the same amount of work")
    print("  - Skipping some computations")
    print("  - Measurement error")
elif elapsed_old / elapsed_new > 10:
    print("✓ Speedup is realistic for vectorized operations")
    print("  - 10-100x speedup is normal for numpy vectorization")
    print("  - Python loops have significant overhead")
else:
    print("✓ Modest speedup as expected")
