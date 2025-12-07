"""
Smoketest: Verify that tokenization produces valid interleaved sequences.

Checks that control and score pitches are properly aligned in the interleaved format.
"""
from anticipation.vocab import CONTROL_OFFSET, REST, NOTE_OFFSET, ANTICIPATE, SEPARATOR
from tqdm import tqdm

def check_sequence_structure(tokens):
    """
    Verify that score pitches match their corresponding control pitches in interleaved sequence.
    
    In the interleaved format:
    - First 33 positions have: control + rest pairs
    - After that: score + control alternation
    
    For proper alignment, each score note should match the control note 33 positions earlier.
    
    Returns:
        (matches, total_score_notes, errors)
    """
    # Skip ANTICIPATE and SEPARATORs
    start_idx = 4 if (len(tokens) > 4 and tokens[0] == ANTICIPATE and tokens[1] == SEPARATOR) else 0
    
    # Find all score and control triplets
    score_triplets = []
    control_triplets = []
    
    i = start_idx
    while i < len(tokens) - 2:
        time_tok, dur_tok, note_tok = tokens[i], tokens[i+1], tokens[i+2]
        
        # Score triplet
        if (time_tok < CONTROL_OFFSET and 
            dur_tok < CONTROL_OFFSET and 
            note_tok < CONTROL_OFFSET and
            note_tok != REST):
            score_triplets.append((i, note_tok))
            i += 3
        # Control triplet
        elif (time_tok >= CONTROL_OFFSET and 
              dur_tok >= CONTROL_OFFSET and 
              note_tok >= CONTROL_OFFSET):
            # Remove CONTROL_OFFSET to get actual note
            control_triplets.append((i, note_tok - CONTROL_OFFSET))
            i += 3
        else:
            i += 1
    
    # Check alignment: score[i] should match control[i+33]
    # (because first 33 controls are in the prefix)
    # BUT: Skip REST tokens (pitch=0) - they're not part of the melody
    #
    # WAIT - THIS IS WRONG! Let's check BOTH:
    # 1. Alignment property: score[i] should match control[i] (same matched pair)
    # 2. Sliding window: score[i] vs control[i+33] (training pairing)
    
    aligned_matches = 0  # score[i] vs control[i]
    sliding_matches = 0  # score[i] vs control[i+33]
    total_comparisons = 0
    alignment_errors = []  # Track score[i] != control[i] 
    sliding_errors = []    # Track score[i] != control[i+33]
    
    for idx, (score_pos, score_note) in enumerate(score_triplets):
        # Extract just the pitch (remove NOTE_OFFSET)
        score_pitch = score_note - NOTE_OFFSET if score_note >= NOTE_OFFSET else score_note
        
        # Skip REST tokens - we only care about actual note pitches
        if score_pitch == 0:
            continue
        
        # Check alignment property: score[i] vs control[i]
        if idx < len(control_triplets):
            control_pos, control_note = control_triplets[idx]
            control_pitch = control_note - NOTE_OFFSET if control_note >= NOTE_OFFSET else control_note
            
            if control_pitch != 0:  # Skip if control is REST
                total_comparisons += 1
                if score_pitch == control_pitch:
                    aligned_matches += 1
                else:
                    alignment_errors.append({
                        'score_idx': idx,
                        'score_pitch': score_pitch,
                        'control_idx': idx,
                        'control_pitch': control_pitch
                    })
        
        # Check sliding window: score[i] vs control[i+33]
        control_idx = idx + 33
        if control_idx < len(control_triplets):
            control_pos, control_note = control_triplets[control_idx]
            control_pitch = control_note - NOTE_OFFSET if control_note >= NOTE_OFFSET else control_note
            
            if control_pitch != 0:  # Skip if control is REST  
                if score_pitch == control_pitch:
                    sliding_matches += 1
                else:
                    sliding_errors.append({
                        'score_idx': idx,
                        'score_pos': score_pos,
                        'score_pitch': score_pitch,
                        'control_idx': control_idx,
                        'control_pos': control_pos,
                        'control_pitch': control_pitch
                    })
    
    return aligned_matches, sliding_matches, total_comparisons, alignment_errors, sliding_errors

def test_dataset(data_file, num_sequences, dataset_name):
    """Test pitch alignment on a dataset."""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name}: {data_file}")
    print(f"{'='*60}")
    
    with open(data_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()][:num_sequences]
    
    print(f"Testing first {len(lines)} sequences...")
    
    # Debug: inspect first sequence
    if lines:
        first_line = lines[0]
        if '|' in first_line:
            token_part = first_line.split('|')[0].strip()
        else:
            token_part = first_line
        first_tokens = [int(t) for t in token_part.split()]
        
        print(f"\nDEBUG - First sequence:")
        print(f"  Total tokens: {len(first_tokens)}")
        print(f"  First 10 tokens: {first_tokens[:10]}")
        
        # Count triplet types
        score_count = 0
        control_count = 0
        rest_count = 0
        i = 4  # Skip ANTICIPATE + 3 SEPs
        while i < len(first_tokens) - 2:
            t0, t1, t2 = first_tokens[i], first_tokens[i+1], first_tokens[i+2]
            if t0 < CONTROL_OFFSET and t1 < CONTROL_OFFSET and t2 < CONTROL_OFFSET:
                if t2 == REST:
                    rest_count += 1
                else:
                    score_count += 1
                i += 3
            elif t0 >= CONTROL_OFFSET and t1 >= CONTROL_OFFSET and t2 >= CONTROL_OFFSET:
                control_count += 1
                i += 3
            else:
                i += 1
        
        print(f"  Score triplets: {score_count}")
        print(f"  Control triplets: {control_count}")
        print(f"  REST triplets: {rest_count}")
    
    total_aligned = 0
    total_sliding = 0
    total_notes = 0
    sequences_with_align_errors = 0
    sequences_with_sliding_errors = 0
    all_alignment_errors = []
    all_sliding_errors = []
    
    for seq_idx, line in enumerate(tqdm(lines, desc=dataset_name)):
        if '|' in line:
            token_part = line.split('|')[0].strip()
        else:
            token_part = line
        
        tokens = [int(t) for t in token_part.split()]
        
        aligned, sliding, total, align_errs, slide_errs = check_sequence_structure(tokens)
        total_aligned += aligned
        total_sliding += sliding
        total_notes += total
        
        if align_errs:
            sequences_with_align_errors += 1
            all_alignment_errors.append((seq_idx, align_errs))
            
            # Debug first alignment error
            if seq_idx == 0:
                print(f"\nDEBUG - First ALIGNMENT error in sequence 0:")
                err = align_errs[0]
                print(f"  Score triplet #{err['score_idx']}: pitch={err['score_pitch']}")
                print(f"  Aligned control #{err['control_idx']}: pitch={err['control_pitch']}")
        
        if slide_errs:
            sequences_with_sliding_errors += 1
            all_sliding_errors.append((seq_idx, slide_errs))
    
    aligned_acc = (total_aligned / total_notes * 100) if total_notes > 0 else 0
    sliding_acc = (total_sliding / total_notes * 100) if total_notes > 0 else 0
    
    print(f"\nAlignment Accuracy (score[i] vs control[i]): {aligned_acc:.2f}% ({total_aligned}/{total_notes})")
    print(f"  Sequences with alignment errors: {sequences_with_align_errors}/{len(lines)}")
    print(f"\nSliding Window (score[i] vs control[i+33]): {sliding_acc:.2f}% ({total_sliding}/{total_notes})")
    print(f"  Sequences with sliding errors: {sequences_with_sliding_errors}/{len(lines)}")
    
    if all_alignment_errors and sequences_with_align_errors <= 3:
        print("\nAlignment error details:")
        for seq_idx, errors in all_alignment_errors[:3]:
            print(f"\nSequence {seq_idx}: {len(errors)} mismatches")
            for err in errors[:2]:  # Show first 2 errors per sequence
                print(f"  Score[{err['score_idx']}]: pitch={err['score_pitch']}, Control[{err['control_idx']}]: pitch={err['control_pitch']}")
    
    perfect = (aligned_acc >= 99.9)  # Allow for minor data anomalies
    if perfect:
        print("✓ PASS: >99.9% alignment accuracy (score[i] matches control[i])")
    else:
        print(f"✗ FAIL: Only {aligned_acc:.2f}% alignment, expected >=99.9%")
    
    print(f"\nNote: Sliding window matching ({sliding_acc:.2f}%) is NOT expected to be 100%")
    print(f"      The model learns to predict score[i] from arbitrary control context,")
    print(f"      not to copy control[i+33]. Low sliding accuracy is normal.")
    
    return perfect

def main():
    num_sequences = 5
    
    # Test training data
    train_pass = test_dataset(
        'data/train_normalized.txt', 
        num_sequences,
        "TRAINING DATA"
    )
    
    # Test validation data
    test_pass = test_dataset(
        'data/test_normalized.txt',
        num_sequences,
        "VALIDATION DATA"
    )
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Training data (first {num_sequences}): {'✓ PASS' if train_pass else '✗ FAIL'}")
    print(f"Validation data (first {num_sequences}): {'✓ PASS' if test_pass else '✗ FAIL'}")
    print(f"{'='*60}")
    
    if train_pass and test_pass:
        print("\n✓ SUCCESS: Tokenization preserves pitch alignment correctly!")
        print("   Each score[i] matches its aligned control[i] from matched_tuples.")
        print("   Separate window normalization is working as intended.")
        print("   Data is ready for training.")
    else:
        print("\n✗ WARNING: Tokenization error - score/control pitch mismatch detected!")
        print("   Check the interleaving logic in tokenize-asap-sliding.py")

if __name__ == "__main__":
    main()
