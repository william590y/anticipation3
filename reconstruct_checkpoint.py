"""
Reconstruct checkpoint file from existing tokenization output.

This allows you to resume with the new script even if you started with the old script.
The script analyzes the existing output files and determines which pieces were already processed.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm

def estimate_sequences_per_piece(num_augmentations=20):
    """Estimate how many sequences each piece generates on average."""
    # Based on ASAP dataset statistics:
    # - Average piece length: ~200 notes
    # - With 20 augmentations: ~200 * 20 = 4000 notes
    # - Sequences of 1024 tokens ≈ ~341 note triplets
    # - So roughly 4000/341 ≈ 11-12 sequences per piece
    return 12 * num_augmentations // 20  # Adjust for augmentation count

def reconstruct_checkpoint(train_file, test_file, checkpoint_file, asap_root, seed=0, test_frac=0.2, num_augmentations=20):
    """
    Reconstruct checkpoint by analyzing existing output files.
    
    Strategy:
    1. Count total sequences in train/test files
    2. Determine which pieces were processed based on expected sequence counts
    3. Write checkpoint file with processed pieces
    """
    
    print("="*80)
    print("RECONSTRUCTING CHECKPOINT FROM EXISTING TOKENIZATION")
    print("="*80)
    
    # Load ASAP metadata to get piece information
    meta_csv = os.path.join(asap_root, 'metadata.csv')
    df = pd.read_csv(meta_csv)
    
    print(f"\nLoading ASAP metadata: {len(df)} total performances")
    
    # Build file list and determine train/test split (must match original)
    import numpy as np
    datafiles = []
    score_keys = []
    for _, row in df.iterrows():
        file1 = os.path.join(asap_root, row['midi_performance'])
        file2 = os.path.join(asap_root, row['midi_score'])
        file3 = os.path.join(asap_root, row['performance_annotations'])
        file4 = os.path.join(asap_root, row['midi_score_annotations'])
        datafiles.append((file1, file2, file3, file4))
        score_keys.append(file2)
    
    # Recreate the exact same split
    rng = np.random.default_rng(seed)
    unique_scores = list(sorted(set(score_keys)))
    rng.shuffle(unique_scores)
    n_test = int(np.ceil(test_frac * len(unique_scores)))
    test_scores = set(unique_scores[:n_test])
    
    tasks_train = []
    tasks_test = []
    for fg, score in zip(datafiles, score_keys):
        if score in test_scores:
            tasks_test.append(fg)
        else:
            tasks_train.append(fg)
    
    print(f"Dataset split: {len(tasks_train)} train pieces, {len(tasks_test)} test pieces")
    
    # Count sequences in existing files
    train_seqs = 0
    test_seqs = 0
    
    if os.path.exists(train_file):
        print(f"\nCounting sequences in {train_file}...")
        with open(train_file, 'r') as f:
            train_seqs = sum(1 for _ in f)
        print(f"  Found {train_seqs} training sequences")
    else:
        print(f"\n⚠ Warning: {train_file} does not exist")
    
    if os.path.exists(test_file):
        print(f"Counting sequences in {test_file}...")
        with open(test_file, 'r') as f:
            test_seqs = sum(1 for _ in f)
        print(f"  Found {test_seqs} test sequences")
    else:
        print(f"⚠ Warning: {test_file} does not exist")
    
    # Estimate completion
    avg_seqs = estimate_sequences_per_piece(num_augmentations)
    estimated_total = (len(tasks_train) + len(tasks_test)) * avg_seqs
    actual_total = train_seqs + test_seqs
    completion_pct = (actual_total / estimated_total * 100) if estimated_total > 0 else 0
    
    print(f"\nEstimated completion:")
    print(f"  Average sequences per piece: ~{avg_seqs}")
    print(f"  Expected total sequences: ~{estimated_total}")
    print(f"  Actual sequences: {actual_total}")
    print(f"  Completion: ~{completion_pct:.1f}%")
    
    # Estimate which pieces were processed
    # Conservative estimate: assume each piece generated avg_seqs sequences
    processed_train_pieces = min(len(tasks_train), train_seqs // avg_seqs)
    processed_test_pieces = min(len(tasks_test), test_seqs // avg_seqs)
    
    print(f"\nEstimated processed pieces:")
    print(f"  Train: ~{processed_train_pieces}/{len(tasks_train)}")
    print(f"  Test: ~{processed_test_pieces}/{len(tasks_test)}")
    
    # WARNING: This is an approximation!
    print("\n" + "="*80)
    print("⚠ IMPORTANT: CHECKPOINT RECONSTRUCTION IS APPROXIMATE")
    print("="*80)
    print("Since imap_unordered processes pieces in random order, we cannot")
    print("determine exactly which pieces were processed.")
    print("")
    print("RECOMMENDED APPROACH:")
    print("1. Let the old script finish completely (no checkpoint needed)")
    print("2. Start fresh with new script for next tokenization")
    print("")
    print("ALTERNATIVE (if job is interrupted):")
    print("1. Note how many pieces were processed from progress bar")
    print("2. Use --resume with new script (may process some pieces twice)")
    print("3. Duplicates will increase dataset size but won't hurt training")
    print("="*80)
    
    # Ask user if they want to proceed anyway
    proceed = input("\nProceed with approximate checkpoint? (yes/no): ").strip().lower()
    
    if proceed != 'yes':
        print("Aborted. No checkpoint file created.")
        return
    
    # Write approximate checkpoint
    # Mark first N pieces as processed (conservative estimate)
    processed_pieces = []
    processed_pieces.extend([fg[0] for fg in tasks_test[:processed_test_pieces]])
    processed_pieces.extend([fg[0] for fg in tasks_train[:processed_train_pieces]])
    
    with open(checkpoint_file, 'w') as f:
        for piece_id in processed_pieces:
            f.write(piece_id + '\n')
    
    print(f"\n✓ Checkpoint created: {checkpoint_file}")
    print(f"  Marked {len(processed_pieces)} pieces as processed")
    print(f"\nYou can now resume with: python tokenize-asap.py --resume")

def main():
    ap = argparse.ArgumentParser(description='Reconstruct checkpoint from existing tokenization output')
    ap.add_argument('--train-file', default='./data/train_output.txt', help='Existing training output file')
    ap.add_argument('--test-file', default='./data/test_output.txt', help='Existing test output file')
    ap.add_argument('--checkpoint-file', default='./data/train_output.txt.checkpoint', help='Checkpoint file to create')
    ap.add_argument('--asap-root', default='./asap-dataset-master', help='Path to ASAP dataset root')
    ap.add_argument('--seed', type=int, default=0, help='Random seed (must match original)')
    ap.add_argument('--test-frac', type=float, default=0.2, help='Test fraction (must match original)')
    ap.add_argument('--num-augmentations', type=int, default=20, help='Number of augmentations (for estimation)')
    args = ap.parse_args()
    
    reconstruct_checkpoint(
        args.train_file,
        args.test_file,
        args.checkpoint_file,
        args.asap_root,
        args.seed,
        args.test_frac,
        args.num_augmentations
    )

if __name__ == '__main__':
    main()
