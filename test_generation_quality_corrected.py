"""
CORRECTED generation quality test.

The test data contains both controls (performance) and scores.
We should:
1. Extract controls from test sequence
2. Extract scores from test sequence (ground truth)
3. Generate using generate4
4. Compare generated output to ground truth scores
"""

import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from anticipation.sample import generate4
from anticipation.config import *
from anticipation.vocab import *


def extract_from_sequence(sequence_tokens, prefix_controls=33):
    """
    Extract controls and scores from a test sequence.
    
    ACTUAL training format (from tokenize-asap.py):
    1. Prefix: k=33 pairs of [ctrl_0, rest_0, ctrl_1, rest_1, ..., ctrl_32, rest_32]
    2. Body: For each matched tuple i (0 to N-1):
       - If score_i exists (not None): add score_i
       - If i+k < N: add ctrl_(i+k)
    3. Result: [score_0, ctrl_33, score_1, ctrl_34, ..., trailing_scores]
    
    BUT some scores may be None (unmatched), so actual pattern could be:
       [score_0, ctrl_33, ctrl_34, score_2, ctrl_35, ...]  (score_1 was skipped)
    
    Since we can't reconstruct which controls go with which scores without
    knowing the original matched_tuples, we'll use a simplified approach:
    - Extract ALL controls (in order from prefix + body)
    - Extract ALL scores (in order from body)
    - For generation: use only first len(scores) controls
    - This ensures we generate the same number as ground truth
    
    Returns:
        controls: List of control tokens for generation (trimmed to match score count)
        scores: List of ALL score tokens (ground truth)
    """
    if len(sequence_tokens) < 4:
        return [], []
    
    tokens = sequence_tokens[4:]  # Skip ANTICIPATE + 3 SEPs
    
    all_controls = []
    all_scores = []
    
    i = 0
    k = prefix_controls
    
    # Step 1: Extract prefix - k pairs of [ctrl, rest]
    for _ in range(k):
        if i + 6 <= len(tokens):
            control_triplet = tokens[i:i+3]
            rest_triplet = tokens[i+3:i+6]
            
            if control_triplet[0] >= CONTROL_OFFSET:
                all_controls.extend(control_triplet)
                i += 6
            else:
                break
        else:
            break
    
    # Step 2: Extract body - could be [score, ctrl] or just [ctrl] or just [score]
    # Simpler approach: just extract all remaining scores and controls
    while i + 3 <= len(tokens):
        triplet = tokens[i:i+3]
        
        if triplet[0] >= CONTROL_OFFSET:
            all_controls.extend(triplet)
        elif triplet[2] != REST:
            all_scores.extend(triplet)
        # Skip REST tokens
        
        i += 3
    
    # Trim controls to match number of scores (for fair comparison)
    num_scores = len(all_scores) // 3
    controls_trimmed = all_controls[:num_scores * 3]
    
    return controls_trimmed, all_scores


def compare_generated_to_scores(generated_events, ground_truth_scores):
    """
    Compare generated events to ground truth scores from test sequence.
    
    Args:
        generated_events: List of generated tokens [time+TIME_OFFSET, dur+DUR_OFFSET, note+NOTE_OFFSET]
        ground_truth_scores: List of score tokens [time+TIME_OFFSET, dur+DUR_OFFSET, note+NOTE_OFFSET]
    
    Returns:
        dict with comparison metrics
    """
    num_generated = len(generated_events) // 3
    num_scores = len(ground_truth_scores) // 3
    
    # Should generate same number as scores
    count_match = (num_generated == num_scores)
    
    # Compare pitch matches
    pitch_matches = 0
    pitch_mismatches = 0
    
    min_len = min(num_generated, num_scores)
    
    for i in range(min_len):
        # Both are without CONTROL_OFFSET
        gen_note = generated_events[i*3 + 2] - NOTE_OFFSET
        score_note = ground_truth_scores[i*3 + 2] - NOTE_OFFSET
        
        if gen_note == score_note:
            pitch_matches += 1
        else:
            pitch_mismatches += 1
    
    match_rate = pitch_matches / min_len if min_len > 0 else 0.0
    
    return {
        'num_generated': num_generated,
        'num_scores': num_scores,
        'count_match': count_match,
        'pitch_matches': pitch_matches,
        'pitch_mismatches': pitch_mismatches,
        'match_rate': match_rate,
        'min_len': min_len
    }


def test_generation_quality_corrected(checkpoint_path, test_data_path, num_sequences=50, top_p=0.95):
    """
    Test generation quality on test sequences (CORRECTED VERSION).
    """
    print("=" * 80)
    print("GENERATION QUALITY TEST (CORRECTED)")
    print("=" * 80)
    print()
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ No GPU available - using CPU (will be slow)")
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    device = next(model.parameters()).device
    print(f"✓ Model loaded on {device} with dtype {model.dtype}")
    print()
    
    # Load test sequences
    print(f"Loading test sequences from: {test_data_path}")
    with open(test_data_path, 'r') as f:
        test_lines = f.readlines()
    
    test_lines = test_lines[:num_sequences]
    print(f"✓ Testing on {len(test_lines)} sequences")
    print()
    
    # Process each sequence
    results = []
    errors = 0
    
    print("Generating and comparing...")
    for seq_idx, line in enumerate(tqdm(test_lines, desc="Testing sequences")):
        try:
            # Parse sequence
            sequence_tokens = [int(tok) for tok in line.strip().split()]
            
            # Extract controls AND scores
            controls, ground_truth_scores = extract_from_sequence(sequence_tokens, prefix_controls=33)
            
            if len(controls) == 0 or len(controls) % 3 != 0:
                errors += 1
                continue
            
            if len(ground_truth_scores) == 0 or len(ground_truth_scores) % 3 != 0:
                errors += 1
                continue
            
            # Generate
            events, tokens = generate4(model, controls=controls, top_p=top_p, prefix_controls=33)
            
            # Compare to ground truth scores
            result = compare_generated_to_scores(events, ground_truth_scores)
            result['seq_idx'] = seq_idx
            results.append(result)
            
        except Exception as e:
            print(f"\n⚠ Error in sequence {seq_idx}: {e}")
            errors += 1
            continue
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    if errors > 0:
        print(f"⚠ {errors} sequences had errors")
        print()
    
    if not results:
        print("❌ No valid results to analyze")
        return
    
    # Aggregate statistics
    total_generated = sum(r['num_generated'] for r in results)
    total_scores = sum(r['num_scores'] for r in results)
    total_pitch_matches = sum(r['pitch_matches'] for r in results)
    total_pitch_mismatches = sum(r['pitch_mismatches'] for r in results)
    total_notes = sum(r['min_len'] for r in results)
    
    overall_match_rate = total_pitch_matches / total_notes if total_notes > 0 else 0.0
    
    count_matches = sum(1 for r in results if r['count_match'])
    count_match_rate = count_matches / len(results) if results else 0.0
    
    print(f"Successfully tested: {len(results)} sequences")
    print(f"Total notes generated: {total_generated:,}")
    print(f"Total ground truth scores: {total_scores:,}")
    print(f"Count match rate: {count_match_rate*100:.2f}% (generated same number as ground truth)")
    print()
    
    print(f"Pitch comparison (generated vs ground truth scores):")
    print(f"  Matches: {total_pitch_matches:,} ({overall_match_rate*100:.2f}%)")
    print(f"  Mismatches: {total_pitch_mismatches:,} ({(1-overall_match_rate)*100:.2f}%)")
    print()
    
    # Per-sequence statistics
    match_rates = [r['match_rate'] for r in results]
    print("Per-sequence pitch match rate statistics:")
    print(f"  Mean: {np.mean(match_rates)*100:.2f}%")
    print(f"  Median: {np.median(match_rates)*100:.2f}%")
    print(f"  Std: {np.std(match_rates)*100:.2f}%")
    print(f"  Min: {np.min(match_rates)*100:.2f}%")
    print(f"  Max: {np.max(match_rates)*100:.2f}%")
    print()
    
    # Distribution
    print("Match rate distribution:")
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    hist, _ = np.histogram(match_rates, bins=bins)
    
    for i in range(len(bins)-1):
        count = hist[i]
        pct = count / len(results) * 100
        print(f"  {bins[i]*100:5.1f}% - {bins[i+1]*100:5.1f}%: {count:3d} sequences ({pct:5.1f}%)")
    print()
    
    # Find worst and best sequences
    sorted_results = sorted(zip(match_rates, results), key=lambda x: x[0])
    
    print("Bottom 5 sequences (lowest match rates):")
    for rate, r in sorted_results[:5]:
        print(f"  Seq {r['seq_idx']:3d}: {rate*100:6.2f}% ({r['pitch_matches']}/{r['min_len']} matches)")
    print()
    
    print("Top 5 sequences (highest match rates):")
    for rate, r in sorted_results[-5:]:
        print(f"  Seq {r['seq_idx']:3d}: {rate*100:6.2f}% ({r['pitch_matches']}/{r['min_len']} matches)")
    print()
    
    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    
    print("Generation Quality Assessment:")
    if overall_match_rate > 0.95:
        print("✅ EXCELLENT: >95% pitch match - model generates scores matching ground truth")
    elif overall_match_rate > 0.90:
        print("✅ GOOD: 90-95% pitch match - model follows training well")
    elif overall_match_rate > 0.80:
        print("⚠ FAIR: 80-90% pitch match - model mostly follows training")
    elif overall_match_rate > 0.50:
        print("⚠ POOR: 50-80% pitch match - model partially learned")
    else:
        print("❌ VERY POOR: <50% pitch match - model didn't learn properly")
    
    print()
    print("Note: Since alignment enforces pitch matching (perf and score have same pitch),")
    print("high match rate means the model learned to generate the score from performance controls.")
    print()
    
    # Save detailed results
    output_file = 'generation_quality_results_corrected.txt'
    with open(output_file, 'w') as f:
        f.write("Generation Quality Test Results (CORRECTED)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test sequences: {len(results)}\n")
        f.write(f"Overall pitch match rate: {overall_match_rate*100:.2f}%\n\n")
        
        f.write("Per-sequence results:\n")
        for r in results:
            f.write(f"Seq {r['seq_idx']:3d}: {r['match_rate']*100:6.2f}% "
                   f"({r['pitch_matches']}/{r['min_len']} matches, "
                   f"{r['num_generated']} generated vs {r['num_scores']} ground truth)\n")
    
    print(f"✓ Detailed results saved to: {output_file}")
    print()
    
    return {
        'overall_match_rate': overall_match_rate,
        'count_match_rate': count_match_rate,
        'results': results
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test generation quality (CORRECTED)')
    parser.add_argument('--checkpoint', default='hf-ckpt-3500/checkpoint-3500',
                       help='Path to model checkpoint')
    parser.add_argument('--test-data', default='data/test_output.txt',
                       help='Path to test data file')
    parser.add_argument('--num-sequences', type=int, default=50,
                       help='Number of test sequences to evaluate')
    parser.add_argument('--top-p', type=float, default=0.95,
                       help='Nucleus sampling parameter')
    
    args = parser.parse_args()
    
    test_generation_quality_corrected(
        checkpoint_path=args.checkpoint,
        test_data_path=args.test_data,
        num_sequences=args.num_sequences,
        top_p=args.top_p
    )


if __name__ == "__main__":
    main()
