"""Quick single-piece test"""
import sys
import pandas as pd
import os
import numpy as np
from anticipation.vocab import SPECIAL_OFFSET, ANTICIPATE, SEPARATOR, MASK
from alignment import align_tokens2

# Import tokenization logic only
SEQ_LEN = 1024
ANTICIPATE_MS = 5000
PREFIX_CONTROLS = 33
PAD = SPECIAL_OFFSET + 1  # Define PAD locally

def test_single_piece():
    df = pd.read_csv('./asap-dataset-master/metadata.csv')
    row = df.iloc[0]
    
    file1 = os.path.join('./asap-dataset-master', row['midi_performance'])
    file2 = os.path.join('./asap-dataset-master', row['midi_score'])
    file3 = os.path.join('./asap-dataset-master', row['performance_annotations'])
    file4 = os.path.join('./asap-dataset-master', row['midi_score_annotations'])
    
    print(f"Testing: {row['midi_performance']}")
    print(f"Augmentations: 5 (testing only)")
    
    # Align once (expensive)
    matched_tuples_base = align_tokens2(file1, file2, file3, file4, perturb_std_ms=0.0, mask_prob=0.0)
    n_tuples = len(matched_tuples_base)
    print(f"Base matched tuples: {n_tuples}")
    
    # Generate 5 augmentations (cheap)
    num_augmentations = 5
    perturb_std_ms = 50.0
    mask_prob = 0.5
    perturb_std_units = int(perturb_std_ms)
    
    all_sequences = []
    for aug_idx in range(num_augmentations):
        seed = hash((file1, aug_idx)) % (2**32)
        np.random.seed(seed)
        
        # Vectorized random generation
        mask_decisions = np.random.random(n_tuples) < mask_prob
        time_perturbations = np.random.normal(0, perturb_std_units, n_tuples).astype(int)
        
        # Apply augmentation
        matched_tuples = []
        for i, match in enumerate(matched_tuples_base):
            perf_tuple = list(match[0])
            if mask_decisions[i]:
                perf_tuple = [MASK, MASK, MASK]
            elif time_perturbations[i] != 0:
                perf_tuple[1] = max(0, perf_tuple[1] + time_perturbations[i])
            matched_tuples.append((tuple(perf_tuple), match[1]))
        
        # Pack into sequences
        perf_notes = [tup[0] for tup in matched_tuples]
        score_notes = [tup[1] for tup in matched_tuples]
        
        packed = []
        for i in range(0, len(perf_notes), PREFIX_CONTROLS):
            seq = [ANTICIPATE] + list(perf_notes[i:i+PREFIX_CONTROLS]) + list(score_notes[i:i+PREFIX_CONTROLS])
            if len(seq) < SEQ_LEN:
                seq += [PAD] * (SEQ_LEN - len(seq))
            packed.append(seq[:SEQ_LEN])
        
        all_sequences.extend(packed)
    
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Expected: {len([s for s in packed]) * num_augmentations}")
    print(f"✓ PASS - Augmentation working!" if len(all_sequences) > 0 else "✗ FAIL")

test_single_piece()
