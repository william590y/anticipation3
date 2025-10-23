"""
Test tokenization on just 3 pieces to verify 5x augmentation multiplier
"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from anticipation.config import *
from anticipation.vocab import *
from anticipation import ops
from alignment import align_tokens2


def _interleave_tokenize4_single(filegroup, skip_Nones=True, prefix_controls=33, perturb_std_ms=0.0, mask_prob=0.0, num_augmentations=1):
    """Process a single piece with augmentations."""
    file1, file2, file3, file4 = filegroup
    
    all_lines = []
    total_seqs = 0
    total_discards = 0
    
    print(f"\n  Processing: {os.path.basename(file1)}")
    
    # Generate multiple augmented versions
    for aug_idx in range(num_augmentations):
        # Re-seed RNG for each augmentation to ensure different random values
        seed = hash((file1, aug_idx)) % (2**32)
        np.random.seed(seed)
        
        try:
            # Each augmentation gets different random perturbations and masks
            matched_tuples = align_tokens2(file1, file2, file3, file4, skip_Nones=skip_Nones, 
                                          perturb_std_ms=perturb_std_ms, mask_prob=mask_prob)
        except Exception as e:
            # Skip this augmentation but continue with others
            print(f"    WARNING: Aug {aug_idx} failed: {e}")
            total_discards += 1
            continue

        # Build interleaved stream: fixed-length control+pad prefix, then alternate score/control
        interleaved_tokens = []

        k = min(prefix_controls, len(matched_tuples))
        for t in matched_tuples[:k]:
            cc = t[0]
            interleaved_tokens.extend(cc)
            # Handle MASK tokens (they don't have CONTROL_OFFSET)
            if cc[0] == MASK:
                # For masked tokens, use a placeholder time of 0
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

        # Prepend separators
        interleaved_tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]

        # Chunk into sequences of 1023 body tokens and add global mode token
        concatenated_tokens = interleaved_tokens
        z = ANTICIPATE
        stats_discards = 0
        aug_seqs = 0
        while len(concatenated_tokens) >= EVENT_SIZE * M:
            seq = concatenated_tokens[0:EVENT_SIZE * M]
            concatenated_tokens = concatenated_tokens[EVENT_SIZE * M:]
            seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
            if ops.min_time(seq, seconds=False) != 0:
                # safety
                dt = -ops.min_time(seq, seconds=False)
                seq = ops.translate(seq, dt, seconds=False)
            if ops.max_time(seq, seconds=False) >= MAX_TIME:
                stats_discards += 1
                continue
            seq.insert(0, z)
            all_lines.append(' '.join(str(tok) for tok in seq))
            total_seqs += 1
            aug_seqs += 1
        
        print(f"    Aug {aug_idx}: {aug_seqs} sequences, {stats_discards} discarded")
        total_discards += stats_discards

    return all_lines, {"seq": total_seqs, "discarded": total_discards}


# Main test
print("="*80)
print("TOKENIZATION TEST: 3 pieces with 5 augmentations each")
print("="*80)

meta_csv = './asap-dataset-master/metadata.csv'
df = pd.read_csv(meta_csv)

# Take first 3 pieces
pieces = []
for idx in range(3):
    row = df.iloc[idx]
    file1 = os.path.join('./asap-dataset-master', row['midi_performance'])
    file2 = os.path.join('./asap-dataset-master', row['midi_score'])
    file3 = os.path.join('./asap-dataset-master', row['performance_annotations'])
    file4 = os.path.join('./asap-dataset-master', row['midi_score_annotations'])
    pieces.append((file1, file2, file3, file4, row['midi_performance']))

total_sequences = 0
all_results = []

for idx, (f1, f2, f3, f4, name) in enumerate(pieces, 1):
    print(f"\nPiece {idx}/3:")
    lines, stats = _interleave_tokenize4_single(
        (f1, f2, f3, f4),
        skip_Nones=True,
        prefix_controls=33,
        perturb_std_ms=50.0,
        mask_prob=0.5,
        num_augmentations=5
    )
    
    seqs = stats['seq']
    disc = stats['discarded']
    total_sequences += seqs
    all_results.append((name, seqs, disc))
    print(f"  TOTAL for this piece: {seqs} sequences ({disc} discarded)")

print("\n" + "="*80)
print("FINAL RESULTS:")
print("="*80)
for name, seqs, disc in all_results:
    print(f"  {name}: {seqs} sequences ({disc} discarded)")

print(f"\nGrand total: {total_sequences} sequences")
print(f"Average per piece: {total_sequences / 3:.1f} sequences")
print(f"Average per augmentation: {total_sequences / 15:.1f} sequences")
print("\nExpected behavior:")
print("  - Each augmentation of a piece should produce the same number of sequences")
print("  - Total should be ~5x the number from a single augmentation")
