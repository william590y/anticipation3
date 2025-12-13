"""
Tokenize ASAP dataset using sliding window to extract all possible 1024-token sequences.

This creates multiple training examples from each piece by starting the interleaving
process at every valid position that has enough remaining notes to form a complete
1024-token sequence.

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
from alignment import align_tokens2, load_annotation_file

# Number of parallel workers
NUM_WORKERS = 200

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
print(f"  Prefix controls: 33 (fixed)")
print(f"  Strategy: Sliding window over all piece positions")
print(f"  Output format: space-separated tokens (one sequence per line)")
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
    f.write(f"# Strategy: Sliding window - all possible 1024-token sequences from each piece\n\n")
    
    f.write(f"=== TRAINING PIECES ===\n")
    for piece_name in sorted(train_piece_names):
        f.write(f"./{piece_name}\n")
    
    f.write(f"\n=== TEST PIECES ===\n")
    for piece_name in sorted(test_piece_names):
        f.write(f"./{piece_name}\n")

print(f"Split file written: {SPLIT_FILE}\n")


def tokenize_sliding_windows(filegroup, prefix_controls=33):
    """
    Tokenize a single performance-score pair, extracting ALL possible 1024-token sequences
    using a sliding window approach.
    
    Matches the exact interleaving logic from tokenize-asap-openings.py but applied at
    multiple starting positions.
    
    Score times are ENFORCED to have exactly 0.5 seconds between beats (TARGET_BEAT_INTERVAL=0.5).
    This means beat[0]->0.0s, beat[1]->0.5s, beat[2]->1.0s, etc., regardless of original tempo.
    
    Performance/control times are normalized to start at 0 but keep original tempo.
    
    Args:
        filegroup: Tuple of (perf_midi, score_midi, perf_beats, score_beats)
        prefix_controls: Number of control notes in the prefix (default 33)
    
    Returns:
        List of string lines, each: "token1 token2 ... tokenN | "
        Returns empty list if no valid sequences can be generated
    """
    file1, file2, file3, file4 = filegroup
    
    try:
        # Align the performance and score
        matched_tuples = align_tokens2(file1, file2, file3, file4, skip_Nones=True)
        
        if len(matched_tuples) < 20:  # Need at least 20 matched pairs
            return []
        
        # Load score beat annotations to create time normalization mapping - DO THIS ONCE
        score_annotations = load_annotation_file(file4)
        score_beat_times = [anno[0] for anno in score_annotations]  # Original beat times in seconds
        
        # ENFORCE 0.5 second beat spacing for score normalization
        # Map original beat times to enforced 0.5s intervals: beat[0]->0.0, beat[1]->0.5, beat[2]->1.0, etc.
        TARGET_BEAT_INTERVAL = 0.5  # seconds
        
        # Pre-normalize ALL score triplets once using beat mapping
        # This is much faster than normalizing per sliding window
        normalized_matched_tuples = []
        for match in matched_tuples:
            perf_triplet = match[0]
            score_triplet = match[2]
            
            if score_triplet[0] is not None:
                # Convert from quantized units back to seconds
                # Triplet format: [time, duration, pitch]
                original_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
                original_duration_sec = (score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION  # triplet[1] is duration!
                pitch = score_triplet[2]  # triplet[2] is pitch!
                
                # Normalize using beat mapping (ENFORCED 0.5 sec between beats)
                # Map first beat to 0.0, each subsequent beat to 0.5 sec apart
                normalized_time_sec = 0.0
                time_scale_factor = 1.0  # Track how much we scaled time to apply to duration
                
                if score_beat_times and len(score_beat_times) >= 2:
                    if original_time_sec < score_beat_times[0]:
                        # Before first beat - scale relative to first beat
                        beat_duration = score_beat_times[1] - score_beat_times[0]
                        if beat_duration > 0:
                            # How far before first beat as fraction of beat duration
                            progress = (original_time_sec - score_beat_times[0]) / beat_duration  # negative
                            time_scale_factor = TARGET_BEAT_INTERVAL / beat_duration
                        else:
                            progress = 0
                            time_scale_factor = 1.0
                        normalized_time_sec = 0.0 + progress * TARGET_BEAT_INTERVAL  # Will be negative
                    else:
                        # Find which beats this falls between
                        found = False
                        for i in range(len(score_beat_times) - 1):
                            if score_beat_times[i] <= original_time_sec <= score_beat_times[i + 1]:
                                beat_duration = score_beat_times[i + 1] - score_beat_times[i]
                                if beat_duration > 0:
                                    progress = (original_time_sec - score_beat_times[i]) / beat_duration
                                    time_scale_factor = TARGET_BEAT_INTERVAL / beat_duration  # ENFORCED 0.5 sec / original beat duration
                                else:
                                    progress = 0
                                    time_scale_factor = 1.0
                                # Beat index i (first beat) maps to 0.0, beat i+1 maps to TARGET_BEAT_INTERVAL, etc.
                                normalized_time_sec = i * TARGET_BEAT_INTERVAL + progress * TARGET_BEAT_INTERVAL
                                found = True
                                break
                        
                        if not found:
                            # After last beat: extrapolate
                            last_beat_idx = len(score_beat_times) - 1
                            if len(score_beat_times) >= 2:
                                last_beat_duration = score_beat_times[-1] - score_beat_times[-2]
                            else:
                                last_beat_duration = 1.0  # fallback
                            
                            if last_beat_duration > 0:
                                progress = (original_time_sec - score_beat_times[-1]) / last_beat_duration
                                time_scale_factor = TARGET_BEAT_INTERVAL / last_beat_duration
                            else:
                                progress = 0
                                time_scale_factor = 1.0
                            normalized_time_sec = last_beat_idx * TARGET_BEAT_INTERVAL + progress * TARGET_BEAT_INTERVAL
                else:
                    # Fallback if not enough beats - just shift to start at 0
                    normalized_time_sec = original_time_sec - (score_beat_times[0] if score_beat_times else 0)
                    time_scale_factor = 1.0
                
                # Scale duration by the same factor we scaled time (to maintain proportions)
                normalized_duration_sec = original_duration_sec * time_scale_factor
                
                # Convert back to quantized units
                # Triplet format: [time, duration, pitch]
                normalized_time_units = round(normalized_time_sec * TIME_RESOLUTION)
                normalized_duration_units = round(normalized_duration_sec * TIME_RESOLUTION)
                normalized_score = [
                    normalized_time_units + TIME_OFFSET,
                    normalized_duration_units + DUR_OFFSET,  # index 1 is duration!
                    pitch  # index 2 is pitch!
                ]
            else:
                normalized_score = score_triplet
            
            normalized_matched_tuples.append([perf_triplet, match[1], normalized_score, match[3]])
        
        sequences = []
        k = min(prefix_controls, len(normalized_matched_tuples))
        
        # Try different starting positions
        for start_idx in range(len(normalized_matched_tuples)):
            # Build interleaved stream starting from start_idx (same logic as openings)
            interleaved_tokens = []
            
            # Get subset starting from start_idx
            subset = normalized_matched_tuples[start_idx:]
            
            if len(subset) < k:
                break  # Not enough notes for even the prefix
            
            # Extract performance triplets from subset (remove offsets first)
            perf_triplets = [[match[0][0] - ATIME_OFFSET, match[0][1] - ADUR_OFFSET, match[0][2] - ANOTE_OFFSET] for match in subset]
            # Normalize performance to start at time 0
            if perf_triplets:
                perf_min_time = min(triplet[0] for triplet in perf_triplets)
                perf_triplets = [
                    [triplet[0] - perf_min_time, triplet[1], triplet[2]]
                    for triplet in perf_triplets
                ]
            
            # Extract already-normalized score triplets from subset
            score_triplets = [match[2] for match in subset]
            
            # Prefix: control + rest pairs using first k notes from normalized subset
            for i in range(k):
                perf_triplet = perf_triplets[i]
                
                # Add control triplet (use correct offsets for each token type)
                interleaved_tokens.extend([
                    perf_triplet[0] + ATIME_OFFSET,   # time
                    perf_triplet[1] + ADUR_OFFSET,    # duration
                    perf_triplet[2] + ANOTE_OFFSET    # pitch
                ])
                
                # Add rest triplet
                cc_time = perf_triplet[0]
                interleaved_tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])
            
            # Main body: alternate score/control
            # Uses notes [0:] for scores and notes [k:] for controls from subset
            for i in range(len(subset)):
                score_triplet = score_triplets[i]
                
                # Add score triplet if it exists
                if score_triplet[0] is not None:
                    interleaved_tokens.extend(score_triplet)
                
                # Add next control if available
                ii = i + k
                if ii < len(subset):
                    perf_triplet = perf_triplets[ii]
                    interleaved_tokens.extend([
                        perf_triplet[0] + ATIME_OFFSET,   # time
                        perf_triplet[1] + ADUR_OFFSET,    # duration
                        perf_triplet[2] + ANOTE_OFFSET    # pitch
                    ])
            
            # Prepend 3 SEPs
            interleaved_tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]
            
            # Check if we have enough tokens
            max_body = EVENT_SIZE * M  # 1023
            if len(interleaved_tokens) < max_body:
                # Not enough tokens for a full sequence, stop trying later positions
                break
            
            # Trim to exactly 1023 tokens
            interleaved_tokens = interleaved_tokens[:max_body]
            
            # Check if sequence is valid
            if ops.max_time(interleaved_tokens, seconds=False) >= MAX_TIME:
                continue  # Skip this sequence, try next position
            
            # Add mode token
            sequence = [ANTICIPATE] + interleaved_tokens
            
            # Verify sequence length
            assert len(sequence) == CONTEXT_SIZE, f"Expected {CONTEXT_SIZE} tokens, got {len(sequence)}"
            
            # Return as space-separated string
            token_str = ' '.join(str(tok) for tok in sequence)
            sequences.append(f"{token_str} | ")
        
        return sequences
        
    except Exception as e:
        # Silently fail for problematic files
        return []


def process_single_piece(filegroup):
    """
    Worker function for multiprocessing.
    Returns: (list_of_sequences, num_sequences)
    """
    sequences = tokenize_sliding_windows(filegroup)
    return (sequences, len(sequences))


# Process train set with multiprocessing
print("Processing training set...")
os.makedirs('data', exist_ok=True)

train_sequences_total = 0
train_pieces_success = 0
train_pieces_failed = 0

with open(TRAIN_OUTPUT, 'w') as f_train:
    with Pool(processes=NUM_WORKERS) as pool:
        with tqdm(total=len(train_pairs), desc='Train', unit='piece') as pbar:
            for sequences, count in pool.imap_unordered(process_single_piece, train_pairs):
                if count > 0:
                    for seq in sequences:
                        f_train.write(seq + '\n')
                    train_sequences_total += count
                    train_pieces_success += 1
                else:
                    train_pieces_failed += 1
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
            for sequences, count in pool.imap_unordered(process_single_piece, test_pairs):
                if count > 0:
                    for seq in sequences:
                        f_test.write(seq + '\n')
                    test_sequences_total += count
                    test_pieces_success += 1
                else:
                    test_pieces_failed += 1
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
