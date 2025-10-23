"""
Comprehensive verification of tokenization correctness:
1. Verify we get N augmentations
2. Verify augmentations are different
3. Verify format matches expected structure
4. Verify MASK tokens are present at ~50% rate
5. Verify time perturbations are present
"""
import numpy as np
import pandas as pd
import os
from collections import Counter
from anticipation.vocab import MASK, CONTROL_OFFSET, SEPARATOR, ANTICIPATE, SPECIAL_OFFSET, TIME_OFFSET, DUR_OFFSET, REST
from anticipation.config import EVENT_SIZE, M, MAX_TIME
from anticipation import ops
from alignment import align_tokens2

# Import the tokenization function
def _interleave_tokenize4_single(filegroup, skip_Nones=True, prefix_controls=33, perturb_std_ms=0.0, mask_prob=0.0, num_augmentations=1):
    """Same as in tokenize-asap.py"""
    file1, file2, file3, file4 = filegroup
    
    all_lines = []
    total_seqs = 0
    total_discards = 0
    
    # DO ALIGNMENT ONCE
    try:
        matched_tuples_base = align_tokens2(file1, file2, file3, file4, skip_Nones=skip_Nones, 
                                           perturb_std_ms=0.0, mask_prob=0.0)
    except Exception as e:
        return [], {"seq": 0, "discarded": 1, "err": str(e)}
    
    # Generate multiple augmented versions
    for aug_idx in range(num_augmentations):
        seed = hash((file1, aug_idx)) % (2**32)
        np.random.seed(seed)
        
        matched_tuples = []
        for match in matched_tuples_base:
            perf_tuple = list(match[0])
            
            if mask_prob > 0 and np.random.random() < mask_prob:
                perf_tuple = [MASK, MASK, MASK]
            elif perturb_std_ms > 0:
                from anticipation.config import TIME_RESOLUTION
                perturb_std_units = (perturb_std_ms / 1000.0) * TIME_RESOLUTION
                time_perturbation = np.random.normal(0, perturb_std_units)
                base_time = perf_tuple[0] - CONTROL_OFFSET
                perturbed_time = max(0, int(base_time + time_perturbation))
                perf_tuple = [CONTROL_OFFSET + perturbed_time, perf_tuple[1], perf_tuple[2]]
            
            matched_tuples.append([perf_tuple, match[1], match[2], match[3]])

        interleaved_tokens = []
        k = min(prefix_controls, len(matched_tuples))
        for t in matched_tuples[:k]:
            cc = t[0]
            interleaved_tokens.extend(cc)
            if cc[0] == MASK:
                interleaved_tokens.extend([TIME_OFFSET + 0, DUR_OFFSET + 0, REST])
            else:
                cc_time = cc[0] - CONTROL_OFFSET
                interleaved_tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])

        for i, t in enumerate(matched_tuples):
            sc = t[2]
            if sc[0] is not None:
                interleaved_tokens.extend(sc)
            ii = i + k
            if ii < len(matched_tuples):
                interleaved_tokens.extend(matched_tuples[ii][0])

        interleaved_tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]
        concatenated_tokens = interleaved_tokens
        z = ANTICIPATE
        stats_discards = 0
        while len(concatenated_tokens) >= EVENT_SIZE * M:
            seq = concatenated_tokens[0:EVENT_SIZE * M]
            concatenated_tokens = concatenated_tokens[EVENT_SIZE * M:]
            seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
            if ops.min_time(seq, seconds=False) != 0:
                dt = -ops.min_time(seq, seconds=False)
                seq = ops.translate(seq, dt, seconds=False)
            if ops.max_time(seq, seconds=False) >= MAX_TIME:
                stats_discards += 1
                continue
            seq.insert(0, z)
            all_lines.append(' '.join(str(tok) for tok in seq))
            total_seqs += 1
        
        total_discards += stats_discards

    return all_lines, {"seq": total_seqs, "discarded": total_discards}

print("="*80)
print("TOKENIZATION VERIFICATION")
print("="*80)

# Load one test piece
meta_csv = './asap-dataset-master/metadata.csv'
df = pd.read_csv(meta_csv)
row = df.iloc[0]

file1 = os.path.join('./asap-dataset-master', row['midi_performance'])
file2 = os.path.join('./asap-dataset-master', row['midi_score'])
file3 = os.path.join('./asap-dataset-master', row['performance_annotations'])
file4 = os.path.join('./asap-dataset-master', row['midi_score_annotations'])

print(f"\nTest piece: {row['midi_performance']}")
print(f"Testing with num_augmentations=20")
print()

# Run tokenization
lines, stats = _interleave_tokenize4_single(
    (file1, file2, file3, file4),
    skip_Nones=True,
    prefix_controls=33,
    perturb_std_ms=50.0,
    mask_prob=0.5,
    num_augmentations=20
)

total_seqs = stats['seq']
discarded = stats['discarded']

print(f"Result: {total_seqs} sequences generated, {discarded} discarded")
print()

# TEST 1: Verify we get 20x augmentations
print("TEST 1: Verify augmentation multiplier")
print("-" * 40)

# Run with 1 augmentation to get baseline
lines_single, stats_single = _interleave_tokenize4_single(
    (file1, file2, file3, file4),
    skip_Nones=True,
    prefix_controls=33,
    perturb_std_ms=0.0,
    mask_prob=0.0,
    num_augmentations=1
)

expected = stats_single['seq'] * 20
actual = total_seqs
print(f"  Sequences with 1 aug: {stats_single['seq']}")
print(f"  Expected with 20 aug: {expected}")
print(f"  Actual with 20 aug: {actual}")
print(f"  Ratio: {actual / stats_single['seq']:.2f}x")
print(f"  ✓ PASS" if abs(actual - expected) <= 1 else f"  ✗ FAIL - expected {expected}, got {actual}")

# TEST 2: Verify augmentations are different
print("\nTEST 2: Verify augmentations differ")
print("-" * 40)

# Check if sequences are unique
unique_sequences = len(set(lines))
total_sequences = len(lines)
print(f"  Total sequences: {total_sequences}")
print(f"  Unique sequences: {unique_sequences}")
print(f"  Duplicate ratio: {(total_sequences - unique_sequences) / total_sequences * 100:.1f}%")
print(f"  ✓ PASS" if unique_sequences > total_sequences * 0.9 else f"  ✗ FAIL - too many duplicates")

# TEST 3: Verify format (1024 tokens, starts with ANTICIPATE)
print("\nTEST 3: Verify sequence format")
print("-" * 40)

sample_seq = lines[0].split()
print(f"  Sample sequence length: {len(sample_seq)} tokens")
print(f"  First token (should be ANTICIPATE={ANTICIPATE}): {sample_seq[0]}")
print(f"  Contains SEPARATOR ({SEPARATOR}): {str(SEPARATOR) in sample_seq}")

format_correct = (
    len(sample_seq) == 1024 and
    sample_seq[0] == str(ANTICIPATE) and
    str(SEPARATOR) in sample_seq
)
print(f"  ✓ PASS" if format_correct else f"  ✗ FAIL - format issues")

# TEST 4: Verify MASK tokens present at ~50% rate
print("\nTEST 4: Verify masking rate")
print("-" * 40)

mask_counts = []
control_counts = []

for line in lines[:10]:  # Check first 10 sequences
    tokens = [int(t) for t in line.split()]
    
    # Count MASK tokens
    n_mask = sum(1 for t in tokens if t == MASK)
    mask_counts.append(n_mask)
    
    # Count control tokens (CONTROL_OFFSET <= token < SPECIAL_OFFSET)
    from anticipation.vocab import SPECIAL_OFFSET
    n_control = sum(1 for t in tokens if CONTROL_OFFSET <= t < SPECIAL_OFFSET)
    control_counts.append(n_control)

avg_mask = np.mean(mask_counts)
avg_control = np.mean(control_counts)
total_control_related = avg_mask + avg_control

if total_control_related > 0:
    mask_ratio = avg_mask / total_control_related
else:
    mask_ratio = 0

print(f"  Avg MASK tokens per seq: {avg_mask:.1f}")
print(f"  Avg control tokens per seq: {avg_control:.1f}")
print(f"  Masking ratio: {mask_ratio * 100:.1f}% (target: ~50%)")
print(f"  ✓ PASS" if 0.4 <= mask_ratio <= 0.6 else f"  ✗ FAIL - masking ratio out of range")

# TEST 5: Verify perturbations differ across augmentations
print("\nTEST 5: Verify time perturbations differ")
print("-" * 40)

# Generate 2 augmentations in one call and compare them
lines_multi, _ = _interleave_tokenize4_single(
    (file1, file2, file3, file4),
    skip_Nones=True, prefix_controls=33,
    perturb_std_ms=50.0, mask_prob=0.0,  # No masking, only perturbation
    num_augmentations=2
)

if len(lines_multi) >= 2:
    # Compare first sequence from aug 0 vs first sequence from aug 1
    seq1 = lines_multi[0].split()  # First sequence from aug 0
    seq2 = lines_multi[4].split()  # First sequence from aug 1 (same piece position)
    
    # Count differences
    differences = sum(1 for t1, t2 in zip(seq1, seq2) if t1 != t2)
    print(f"  Comparing aug 0 vs aug 1 (same piece, different perturbations)")
    print(f"  Differences: {differences}/{len(seq1)} tokens")
    print(f"  Difference ratio: {differences / len(seq1) * 100:.1f}%")
    print(f"  ✓ PASS" if differences > 50 else f"  ✗ FAIL - augmentations too similar ({differences} diffs)")
else:
    print(f"  ✗ FAIL - could not generate comparison augmentations")

# FINAL SUMMARY
print("\n" + "="*80)
print("SUMMARY:")
print("  If all tests pass, tokenization is correct and ready for production")
print("="*80)
