"""
Tokenize ASAP dataset using sliding window to extract all possible packed 1020-token sequences.

This creates multiple training examples from each piece by starting the interleaving
process at every valid position that has enough remaining notes to form a complete
packed sequence.

Score normalization ENFORCES 0.5 second beat spacing regardless of original tempo.
Performance/control times preserve original tempo but are shifted to start at 0.

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
from anticipation.asap_aligned_stream import (
    STREAM_CACHE_DIR,
    build_performance_anchored_stream,
)
from anticipation.packed_sequence import (
    ALTERNATING_START,
    PREFIX_CONTROLS,
    dummy_rest_triplet,
    is_dummy_score_triplet,
    is_prefix_placeholder_triplet,
    is_real_score_triplet,
)

# Number of parallel workers
NUM_WORKERS = 128

# ASAP dataset path
ASAP_PATH = 'asap-dataset-master'
META_CSV = os.path.join(ASAP_PATH, 'metadata.csv')

# Output paths
TRAIN_OUTPUT = 'data/train_normalized.txt'
TEST_OUTPUT = 'data/test_normalized.txt'
SPLIT_FILE = 'data/normalized_split.txt'

print(f"Tokenization configuration:")
print(f"  Workers: {NUM_WORKERS}")
print(f"  Context size: {CONTEXT_SIZE}")
print(f"  Serialized length: {CONTEXT_SIZE - 4}")
print(f"  Prefix controls: {PREFIX_CONTROLS} (fixed)")
print(f"  Strategy: Sliding window over all piece positions")
print(f"  Output format: space-separated tokens (one sequence per line)")
print(f"  Aligned-stream cache: {STREAM_CACHE_DIR}")
print()

# Load metadata
df = pd.read_csv(META_CSV)
print(f"Found {len(df)} pieces in metadata")

# Build file tuples and track score IDs for split
datafiles = []
score_keys = []  # use midi_score path as the key
piece_names = []  # track piece names for split file

for _, row in df.iterrows():
    file1 = os.path.join(ASAP_PATH, row['midi_performance'])
    file2 = os.path.join(ASAP_PATH, row['midi_score'])
    file3 = os.path.join(ASAP_PATH, row['performance_annotations'])
    file4 = os.path.join(ASAP_PATH, row['midi_score_annotations'])
    
    # Check if all files exist
    if all(os.path.exists(f) for f in [file1, file2, file3, file4]):
        datafiles.append((file1, file2, file3, file4))
        score_keys.append(file2)
        piece_names.append(row['midi_performance'])

print(f"Found {len(datafiles)} valid pieces with all required files")

# Split by unique score to avoid data leakage
rng = np.random.default_rng(42)
unique_scores = list(sorted(set(score_keys)))
rng.shuffle(unique_scores)
n_test = int(np.ceil(0.2 * len(unique_scores)))
test_scores = set(unique_scores[:n_test])
train_scores = set(unique_scores[n_test:])

train_pairs = []
test_pairs = []
train_piece_names = []
test_piece_names = []

for fg, score, piece_name in zip(datafiles, score_keys, piece_names):
    if score in test_scores:
        test_pairs.append(fg)
        test_piece_names.append(piece_name)
    else:
        train_pairs.append(fg)
        train_piece_names.append(piece_name)

print(f"Train: {len(train_pairs)} pieces")
print(f"Test: {len(test_pairs)} pieces")
print()

# Write split information
print(f"Writing split information to {SPLIT_FILE}...")
with open(SPLIT_FILE, 'w') as f:
    f.write(f"# Train/Test Split (seed=42, test_frac=0.2)\n")
    f.write(f"# Total pieces: {len(datafiles)} (train: {len(train_pairs)}, test: {len(test_pairs)})\n")
    f.write(f"# Split by unique scores to prevent data leakage\n")
    f.write(f"# Strategy: Sliding window - all possible packed 1020-token sequences from each piece\n\n")
    
    f.write(f"=== TRAINING PIECES ===\n")
    for piece_name in sorted(train_piece_names):
        f.write(f"./{piece_name}\n")
    
    f.write(f"\n=== TEST PIECES ===\n")
    for piece_name in sorted(test_piece_names):
        f.write(f"./{piece_name}\n")

print(f"Split file written: {SPLIT_FILE}\n")


def _strip_control_offsets(control_triplet):
    return [
        control_triplet[0] - ATIME_OFFSET,
        control_triplet[1] - ADUR_OFFSET,
        control_triplet[2] - ANOTE_OFFSET,
    ]


def _build_real_score_suffix_min(score_triplets):
    suffix_min = [0] * len(score_triplets)
    has_real = [False] * len(score_triplets)
    current = None
    for idx in range(len(score_triplets) - 1, -1, -1):
        score_triplet = score_triplets[idx]
        if score_triplet is not None:
            score_time_units = score_triplet[0] - TIME_OFFSET
            current = score_time_units if current is None else min(current, score_time_units)
        if current is not None:
            suffix_min[idx] = current
            has_real[idx] = True
    return suffix_min, has_real


def tokenize_sliding_windows(filegroup, prefix_controls=PREFIX_CONTROLS):
    """
    Tokenize a single performance-score pair, extracting all possible packed sequences
    using a sliding window approach.
    
    Matches the exact interleaving logic from tokenize-asap-openings.py but applied at
    multiple starting positions.
    
    Score times are ENFORCED to have exactly 0.5 seconds between beats (TARGET_BEAT_INTERVAL=0.5).
    This means beat[0]->0.0s, beat[1]->0.5s, beat[2]->1.0s, etc., regardless of original tempo.
    
    Performance/control times are normalized to start at 0 but keep original tempo.
    Score times are first beat-normalized globally, then shifted to a window-local
    timeline for each sliding window.
    
    Args:
        filegroup: Tuple of (perf_midi, score_midi, perf_beats, score_beats)
        prefix_controls: Number of control notes in the prefix (default PREFIX_CONTROLS)
    
    Returns:
        Dict with:
          - piece_id: piece identifier for warnings
          - sequences: list of "token1 token2 ... tokenN | " strings
          - count: number of sequences
          - pickup_warning: aggregated pickup warning metadata or None
          - failure_reason: human-readable reason when count == 0
    """
    file1, file2, file3, file4 = filegroup
    piece_id = file1
    
    try:
        aligned_items = build_performance_anchored_stream(file1, file2, file3, file4)

        if len(aligned_items) < 20:  # Need at least 20 performance notes
            return {
                "piece_id": piece_id,
                "sequences": [],
                "count": 0,
                "pickup_warning": None,
                "failure_reason": f"only {len(aligned_items)} performance notes after preprocessing (need at least 20)",
            }

        sequences = []
        num_items = len(aligned_items)
        k = min(prefix_controls, num_items)
        pickup_start_indices = []
        most_negative_score_time_units = None
        full_length_candidates = 0
        skipped_for_max_time = 0
        raw_perf_triplets = [_strip_control_offsets(item["control"]) for item in aligned_items]
        global_score_triplets = [item["score"] for item in aligned_items]
        score_suffix_min_times, score_suffix_has_real = _build_real_score_suffix_min(global_score_triplets)
        
        # Try different starting positions
        for start_idx in range(num_items):
            # Build interleaved stream starting from start_idx (same logic as openings)
            interleaved_tokens = []

            remaining = num_items - start_idx
            if remaining < k:
                break  # Not enough notes for even the prefix

            perf_anchor = raw_perf_triplets[start_idx][0]
            min_score_time_units = 0
            if score_suffix_has_real[start_idx]:
                min_score_time_units = score_suffix_min_times[start_idx]
                if min_score_time_units < 0:
                    pickup_start_indices.append(start_idx)
                    if (
                        most_negative_score_time_units is None
                        or min_score_time_units < most_negative_score_time_units
                    ):
                        most_negative_score_time_units = min_score_time_units
            # Prefix: control + rest pairs using first k notes from normalized subset
            for i in range(k):
                perf_triplet = raw_perf_triplets[start_idx + i]
                local_perf_time = perf_triplet[0] - perf_anchor

                # Add control triplet (use correct offsets for each token type)
                interleaved_tokens.extend([
                    local_perf_time + ATIME_OFFSET,   # time
                    perf_triplet[1] + ADUR_OFFSET,    # duration
                    perf_triplet[2] + ANOTE_OFFSET    # pitch
                ])

                # Add dummy REST score triplet (time zero in score space)
                interleaved_tokens.extend(dummy_rest_triplet(0))
            
            # Main body: alternate score/control
            # Uses notes [0:] for scores and notes [k:] for controls from subset
            for i in range(remaining):
                item_idx = start_idx + i
                perf_triplet = raw_perf_triplets[item_idx]
                local_perf_time = perf_triplet[0] - perf_anchor
                score_triplet = global_score_triplets[item_idx]
                if score_triplet is None:
                    interleaved_tokens.extend(dummy_rest_triplet(0))
                else:
                    interleaved_tokens.extend([
                        score_triplet[0] - min_score_time_units,
                        score_triplet[1],
                        score_triplet[2],
                    ])
                
                # Add next control if available
                ii = i + k
                if ii < remaining:
                    perf_triplet = raw_perf_triplets[start_idx + ii]
                    interleaved_tokens.extend([
                        (perf_triplet[0] - perf_anchor) + ATIME_OFFSET,   # time
                        perf_triplet[1] + ADUR_OFFSET,    # duration
                        perf_triplet[2] + ANOTE_OFFSET    # pitch
                    ])
            
            # Check if we have enough tokens
            max_body = CONTEXT_SIZE - 4
            if len(interleaved_tokens) < max_body:
                # Not enough tokens for a full sequence, stop trying later positions
                break
            full_length_candidates += 1
            
            # Trim to the packed serialized length
            interleaved_tokens = interleaved_tokens[:max_body]
            
            # Check if sequence is valid
            if ops.max_time(interleaved_tokens, seconds=False) >= MAX_TIME:
                skipped_for_max_time += 1
                continue  # Skip this sequence, try next position
            
            sequence = interleaved_tokens
            
            # Verify sequence length
            assert len(sequence) == max_body, f"Expected {max_body} tokens, got {len(sequence)}"
            
            # Return as space-separated string
            token_str = ' '.join(str(tok) for tok in sequence)
            sequences.append(f"{token_str} | ")

        pickup_warning = None
        if pickup_start_indices:
            pickup_warning = {
                "piece_id": piece_id,
                "affected_windows": len(pickup_start_indices),
                "sample_start_indices": pickup_start_indices[:5],
                "most_negative_time_units": most_negative_score_time_units,
            }

        failure_reason = None
        if not sequences:
            if full_length_candidates == 0:
                failure_reason = (
                    f"piece is too short to form a full packed sequence of {CONTEXT_SIZE - 4} tokens"
                )
            elif skipped_for_max_time == full_length_candidates:
                failure_reason = (
                    f"all {full_length_candidates} candidate windows exceeded MAX_TIME={MAX_TIME}"
                )
            else:
                failure_reason = "no valid packed windows generated"

        return {
            "piece_id": piece_id,
            "sequences": sequences,
            "count": len(sequences),
            "pickup_warning": pickup_warning,
            "failure_reason": failure_reason,
        }
        
    except Exception as e:
        return {
            "piece_id": piece_id,
            "sequences": [],
            "count": 0,
            "pickup_warning": None,
            "failure_reason": f"{type(e).__name__}: {e}",
        }


def format_pickup_warning(warning_info):
    preview_indices = ", ".join(str(idx) for idx in warning_info["sample_start_indices"])
    if warning_info["affected_windows"] > len(warning_info["sample_start_indices"]):
        preview_indices = f"{preview_indices}, ..."
    most_negative_seconds = warning_info["most_negative_time_units"] / TIME_RESOLUTION
    return (
        f"WARNING: {warning_info['piece_id']}: shifted score times for pickup preservation in "
        f"{warning_info['affected_windows']} window(s) (start_idx examples: {preview_indices}); "
        f"most negative pre-shift score onset was {warning_info['most_negative_time_units']} bins "
        f"({most_negative_seconds:.3f}s)."
    )


def format_failure_warning(piece_id, failure_reason):
    return f"WARNING: {piece_id}: tokenization produced no sequences - {failure_reason}"


def process_single_piece(filegroup):
    """
    Worker function for multiprocessing.
    Returns a dict from tokenize_sliding_windows().
    """
    return tokenize_sliding_windows(filegroup)


# Process train set with multiprocessing
print("Processing training set...")
os.makedirs('data', exist_ok=True)

train_sequences_total = 0
train_pieces_success = 0
train_pieces_failed = 0

with open(TRAIN_OUTPUT, 'w') as f_train:
    with Pool(processes=NUM_WORKERS) as pool:
        with tqdm(total=len(train_pairs), desc='Train', unit='piece') as pbar:
            for result in pool.imap_unordered(process_single_piece, train_pairs):
                sequences = result["sequences"]
                count = result["count"]
                if result["pickup_warning"] is not None:
                    tqdm.write(format_pickup_warning(result["pickup_warning"]))
                if count > 0:
                    for seq in sequences:
                        f_train.write(seq + '\n')
                    train_sequences_total += count
                    train_pieces_success += 1
                else:
                    train_pieces_failed += 1
                    tqdm.write(format_failure_warning(result["piece_id"], result["failure_reason"]))
                pbar.update(1)

print(f"Train: {train_sequences_total} sequences from {train_pieces_success} pieces, {train_pieces_failed} pieces failed")

# Process test set with multiprocessing
print("\nProcessing test set...")

test_sequences_total = 0
test_pieces_success = 0
test_pieces_failed = 0

with open(TEST_OUTPUT, 'w') as f_test:
    with Pool(processes=NUM_WORKERS) as pool:
        with tqdm(total=len(test_pairs), desc='Test', unit='piece') as pbar:
            for result in pool.imap_unordered(process_single_piece, test_pairs):
                sequences = result["sequences"]
                count = result["count"]
                if result["pickup_warning"] is not None:
                    tqdm.write(format_pickup_warning(result["pickup_warning"]))
                if count > 0:
                    for seq in sequences:
                        f_test.write(seq + '\n')
                    test_sequences_total += count
                    test_pieces_success += 1
                else:
                    test_pieces_failed += 1
                    tqdm.write(format_failure_warning(result["piece_id"], result["failure_reason"]))
                pbar.update(1)

print(f"Test: {test_sequences_total} sequences from {test_pieces_success} pieces, {test_pieces_failed} pieces failed")

# Verify the sequences
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

if train_sequences_total > 0:
    with open(TRAIN_OUTPUT, 'r') as f:
        first_line = f.readline().strip()
        # Split and take only tokens before the | separator
        tokens_part = first_line.split('|')[0].strip()
        first_seq = [int(x) for x in tokens_part.split()]
    
    print(f"First training sequence length: {len(first_seq)} tokens")
    print("No mode/bootstrap tokens are serialized")
    
    # Count control vs score tokens in the first 100 triplets
    control_count = 0
    score_count = 0
    dummy_score_count = 0
    prefix_rest_count = 0
    
    for i in range(min(100, len(first_seq) // 3)):
        pos = i * 3
        if pos + 2 >= len(first_seq):
            break
        
        if first_seq[pos] >= CONTROL_OFFSET:
            control_count += 1
        elif is_prefix_placeholder_triplet(first_seq, pos, ALTERNATING_START):
            prefix_rest_count += 1
        elif is_dummy_score_triplet(first_seq, pos, ALTERNATING_START):
            dummy_score_count += 1
        elif is_real_score_triplet(first_seq, pos, ALTERNATING_START):
            score_count += 1
    
    print(f"\nFirst 100 triplets breakdown:")
    print(f"  Control triplets: {control_count}")
    print(f"  Score triplets (notes): {score_count}")
    print(f"  Score triplets (dummy REST): {dummy_score_count}")
    print(f"  Prefix placeholders (REST): {prefix_rest_count}")
else:
    print("No sequences generated!")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Training sequences: {train_sequences_total} from {train_pieces_success}/{len(train_pairs)} pieces")
print(f"  Average sequences per piece: {train_sequences_total/train_pieces_success:.1f}" if train_pieces_success > 0 else "")
print(f"Test sequences: {test_sequences_total} from {test_pieces_success}/{len(test_pairs)} pieces")
print(f"  Average sequences per piece: {test_sequences_total/test_pieces_success:.1f}" if test_pieces_success > 0 else "")
print(f"Total sequences: {train_sequences_total + test_sequences_total}")
print(f"\nOutput files:")
print(f"  {TRAIN_OUTPUT}")
print(f"  {TEST_OUTPUT}")
print(f"  {SPLIT_FILE}")
print("\nTokenization complete!")
