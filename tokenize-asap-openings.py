"""
Tokenize ASAP dataset using only the opening 1024 tokens from each piece.

This avoids sequence packing, which can cause the control note to be outside
the 1024 token context window. Each training example is the beginning of a piece.

Uses parallel processing with 128 workers for efficiency.
"""

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

from anticipation.config import *
from anticipation.vocab import *
from anticipation import ops
from alignment import align_tokens2

# Number of parallel workers
NUM_WORKERS = 128

# ASAP dataset path
ASAP_PATH = 'asap-dataset-master'
META_CSV = os.path.join(ASAP_PATH, 'metadata.csv')

# Output paths
TRAIN_OUTPUT = 'data/train_openings.txt'
TEST_OUTPUT = 'data/test_openings.txt'

print(f"Tokenization configuration:")
print(f"  Workers: {NUM_WORKERS}")
print(f"  Context size: {CONTEXT_SIZE}")
print(f"  Anticipation interval: {DELTA}s")
print(f"  Output format: space-separated tokens (one sequence per line)")
print()

# Load metadata
df = pd.read_csv(META_CSV)
print(f"Found {len(df)} pieces in metadata")

# Build file tuples and track score IDs for split
datafiles = []
score_keys = []  # use midi_score path as the key

for _, row in df.iterrows():
    file1 = os.path.join(ASAP_PATH, row['midi_performance'])
    file2 = os.path.join(ASAP_PATH, row['midi_score'])
    file3 = os.path.join(ASAP_PATH, row['performance_annotations'])
    file4 = os.path.join(ASAP_PATH, row['midi_score_annotations'])
    
    # Check if all files exist
    if all(os.path.exists(f) for f in [file1, file2, file3, file4]):
        datafiles.append((file1, file2, file3, file4))
        score_keys.append(file2)

print(f"Found {len(datafiles)} valid pieces with all required files")

# Split by unique score to avoid data leakage (same as tokenize-asap.py)
rng = np.random.default_rng(42)
unique_scores = list(sorted(set(score_keys)))
rng.shuffle(unique_scores)
n_test = int(np.ceil(0.2 * len(unique_scores)))
test_scores = set(unique_scores[:n_test])
train_scores = set(unique_scores[n_test:])

train_pairs = []
test_pairs = []
for fg, score in zip(datafiles, score_keys):
    if score in test_scores:
        test_pairs.append(fg)
    else:
        train_pairs.append(fg)

print(f"Train: {len(train_pairs)} pieces")
print(f"Test: {len(test_pairs)} pieces")
print()


def tokenize_opening(filegroup, prefix_controls=33):
    """
    Tokenize a single performance-score pair, taking only the first 1024 tokens.
    
    Uses the same interleaving structure as tokenize-asap.py but without sequence packing.
    
    Args:
        filegroup: Tuple of (perf_midi, score_midi, perf_beats, score_beats)
        prefix_controls: Number of control notes in the prefix (default 33)
    
    Returns:
        String line "token1 token2 ... tokenN" or None if sequence is invalid
    """
    file1, file2, file3, file4 = filegroup
    
    try:
        # Align the performance and score (no augmentation)
        matched_tuples = align_tokens2(file1, file2, file3, file4, 
                                      skip_Nones=True, 
                                      perturb_std_ms=0.0, 
                                      mask_prob=0.0)
        
        if len(matched_tuples) < 20:  # Need at least 20 matched pairs
            return None
        
        # Build interleaved stream using same logic as tokenize-asap.py
        # Structure: prefix_controls control+rest pairs, then alternate score/control
        interleaved_tokens = []
        
        k = min(prefix_controls, len(matched_tuples))
        
        # Prefix: control + rest pairs
        for i in range(k):
            match = matched_tuples[i]
            perf_triplet = match[0]
            
            # Add control triplet
            interleaved_tokens.extend(perf_triplet)
            
            # Add rest triplet
            cc_time = perf_triplet[0] - CONTROL_OFFSET
            interleaved_tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])
        
        # Main body: alternate score/control
        for i in range(len(matched_tuples)):
            match = matched_tuples[i]
            score_triplet = match[2]
            
            # Add score triplet if it exists
            if score_triplet[0] is not None:
                interleaved_tokens.extend(score_triplet)
            
            # Add next control if available
            ii = i + k
            if ii < len(matched_tuples):
                perf_triplet = matched_tuples[ii][0]
                interleaved_tokens.extend(perf_triplet)
        
        # Prepend 3 SEPs
        interleaved_tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]
        
        # Take only first 1023 tokens (EVENT_SIZE * M)
        max_body = EVENT_SIZE * M  # 1023
        if len(interleaved_tokens) < max_body:
            # Pad if necessary
            while len(interleaved_tokens) < max_body:
                interleaved_tokens.append(SEPARATOR)
        else:
            # Trim to fit
            interleaved_tokens = interleaved_tokens[:max_body]
        
        # Translate to start at time 0
        interleaved_tokens = ops.translate(interleaved_tokens, 
                                          -ops.min_time(interleaved_tokens, seconds=False), 
                                          seconds=False)
        
        # Check if sequence is valid
        if ops.max_time(interleaved_tokens, seconds=False) >= MAX_TIME:
            return None
        
        # Add mode token
        sequence = [ANTICIPATE] + interleaved_tokens
        
        # Verify sequence length
        assert len(sequence) == CONTEXT_SIZE, f"Expected {CONTEXT_SIZE} tokens, got {len(sequence)}"
        
        # Return as space-separated string with mask indices (empty since no masking)
        # Format: "token1 token2 ... tokenN | " (matches tokenize-asap.py output)
        token_str = ' '.join(str(tok) for tok in sequence)
        return f"{token_str} | "
        
    except Exception as e:
        # Silently fail for problematic files
        return None


def process_single_piece(filegroup):
    """
    Worker function for multiprocessing.
    Returns: (sequence_string, success_flag)
    """
    result = tokenize_opening(filegroup)
    if result is not None:
        return (result, True)
    else:
        return (None, False)


# Process train set with multiprocessing
print("Processing training set...")
os.makedirs('data', exist_ok=True)

train_success = 0
train_failed = 0

with open(TRAIN_OUTPUT, 'w') as f_train:
    with Pool(processes=NUM_WORKERS) as pool:
        with tqdm(total=len(train_pairs), desc='Train', unit='piece') as pbar:
            for result, success in pool.imap_unordered(process_single_piece, train_pairs):
                if success:
                    f_train.write(result + '\n')
                    train_success += 1
                else:
                    train_failed += 1
                pbar.update(1)

print(f"Train: {train_success} sequences generated, {train_failed} failed")

# Process test set with multiprocessing
print("\nProcessing test set...")

test_success = 0
test_failed = 0

with open(TEST_OUTPUT, 'w') as f_test:
    with Pool(processes=NUM_WORKERS) as pool:
        with tqdm(total=len(test_pairs), desc='Test', unit='piece') as pbar:
            for result, success in pool.imap_unordered(process_single_piece, test_pairs):
                if success:
                    f_test.write(result + '\n')
                    test_success += 1
                else:
                    test_failed += 1
                pbar.update(1)

print(f"Test: {test_success} sequences generated, {test_failed} failed")

# Verify the sequences
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

if train_success > 0:
    with open(TRAIN_OUTPUT, 'r') as f:
        first_line = f.readline().strip()
        first_seq = [int(x) for x in first_line.split()]
    
    print(f"First training sequence length: {len(first_seq)} tokens")
    print(f"Mode token: {first_seq[0]} (expected {ANTICIPATE})")
    print(f"Bootstrap: {first_seq[1:4]} (expected {[SEPARATOR, SEPARATOR, SEPARATOR]})")
    
    # Count control vs score tokens in first 100 triplets
    control_count = 0
    score_count = 0
    rest_count = 0
    
    for i in range(min(100, (len(first_seq) - 4) // 3)):
        pos = 4 + i * 3  # After mode + 3 SEPs
        if pos + 2 >= len(first_seq):
            break
        
        t0 = first_seq[pos]
        t2 = first_seq[pos + 2]
        
        if t0 >= CONTROL_OFFSET:
            control_count += 1
        elif t2 == REST:
            rest_count += 1
        elif t2 >= NOTE_OFFSET:
            score_count += 1
    
    print(f"\nFirst 100 triplets breakdown:")
    print(f"  Control triplets: {control_count}")
    print(f"  Score triplets (notes): {score_count}")
    print(f"  Score triplets (REST): {rest_count}")
    print(f"  Expected pattern: CRCR...CSCS... (interleaved)")
else:
    print("No sequences generated!")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Training sequences: {train_success}/{len(train_pairs)} ({100*train_success/len(train_pairs):.1f}%)")
print(f"Test sequences: {test_success}/{len(test_pairs)} ({100*test_success/len(test_pairs):.1f}%)")
print(f"Total sequences: {train_success + test_success}")
print(f"\nOutput files:")
print(f"  {TRAIN_OUTPUT}")
print(f"  {TEST_OUTPUT}")
print("\nTokenization complete!")
