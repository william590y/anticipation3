"""
Debug script to trace through augmentation logic on a single piece
"""
import numpy as np
import pandas as pd
import os
from alignment import align_tokens2
from anticipation.config import *
from anticipation.vocab import *
from anticipation import ops

# Load one piece
meta_csv = './asap-dataset-master/metadata.csv'
df = pd.read_csv(meta_csv)
row = df.iloc[0]

file1 = os.path.join('./asap-dataset-master', row['midi_performance'])
file2 = os.path.join('./asap-dataset-master', row['midi_score'])
file3 = os.path.join('./asap-dataset-master', row['performance_annotations'])
file4 = os.path.join('./asap-dataset-master', row['midi_score_annotations'])

print(f"Processing: {row['midi_performance']}")
print("="*80)

num_augmentations = 5
skip_Nones = True
prefix_controls = 33
perturb_std_ms = 50.0
mask_prob = 0.5

all_seqs_per_aug = []

for aug_idx in range(num_augmentations):
    print(f"\n--- Augmentation {aug_idx+1}/{num_augmentations} ---")
    
    # Re-seed RNG
    seed = hash((file1, aug_idx)) % (2**32)
    np.random.seed(seed)
    print(f"Seed: {seed}")
    
    # Align tokens
    matched_tuples = align_tokens2(file1, file2, file3, file4, skip_Nones=skip_Nones, 
                                   perturb_std_ms=perturb_std_ms, mask_prob=mask_prob)
    print(f"Matched tuples: {len(matched_tuples)}")
    
    # Build interleaved stream
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
    
    print(f"Total interleaved tokens: {len(interleaved_tokens)}")
    
    # Chunk into sequences
    concatenated_tokens = interleaved_tokens
    seqs_created = 0
    seqs_discarded = 0
    
    while len(concatenated_tokens) >= EVENT_SIZE * M:
        seq = concatenated_tokens[0:EVENT_SIZE * M]
        concatenated_tokens = concatenated_tokens[EVENT_SIZE * M:]
        
        try:
            seq = ops.translate(seq, -ops.min_time(seq, seconds=False), seconds=False)
            if ops.min_time(seq, seconds=False) != 0:
                dt = -ops.min_time(seq, seconds=False)
                seq = ops.translate(seq, dt, seconds=False)
            if ops.max_time(seq, seconds=False) >= MAX_TIME:
                seqs_discarded += 1
                continue
            seqs_created += 1
        except Exception as e:
            print(f"ERROR processing sequence: {e}")
            seqs_discarded += 1
            continue
    
    print(f"Sequences created: {seqs_created}")
    print(f"Sequences discarded: {seqs_discarded}")
    all_seqs_per_aug.append(seqs_created)

print("\n" + "="*80)
print("SUMMARY:")
print(f"Sequences per augmentation: {all_seqs_per_aug}")
print(f"Total sequences: {sum(all_seqs_per_aug)}")
print(f"Expected (5 augmentations): {all_seqs_per_aug[0] * 5 if all_seqs_per_aug else 0}")
print(f"Actual multiplier: {sum(all_seqs_per_aug) / all_seqs_per_aug[0] if all_seqs_per_aug[0] > 0 else 0:.2f}x")
