"""
Test generation quality by comparing generated performance notes with control notes.

This evaluates how well generate4 produces performances that match the controls.
For a well-trained model, we expect:
- Generated performance notes to closely match the control notes (same pitch)
- Good timing alignment
- Reasonable duration predictions
"""

import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from anticipation.sample import generate4
from anticipation.config import *
from anticipation.vocab import *


def extract_controls_from_sequence(sequence_tokens, prefix_controls=33):
    """
    Extract control tokens from a tokenized sequence.
    (Same as in test.py)
    """
    if len(sequence_tokens) < 4:
        return []
    
    tokens = sequence_tokens[4:]  # Skip ANTICIPATE + 3 SEPs
    controls = []
    i = 0
    
    # Extract prefix controls
    for _ in range(prefix_controls):
        if i + 6 <= len(tokens):
            control_triplet = tokens[i:i+3]
            rest_triplet = tokens[i+3:i+6]
            
            if control_triplet[0] >= CONTROL_OFFSET:
                controls.extend(control_triplet)
            else:
                break
            
            i += 6
        else:
            break
    
    # Extract alternating controls
    while i + 3 <= len(tokens):
        triplet = tokens[i:i+3]
        
        if triplet[0] >= CONTROL_OFFSET:
            controls.extend(triplet)
        
        i += 3
    
    return controls


def compare_generated_to_controls(generated_events, controls):
    """
    Compare generated performance events to control events.
    
    Args:
        generated_events: List of generated tokens [time+TIME_OFFSET, dur+DUR_OFFSET, note+NOTE_OFFSET]
        controls: List of control tokens [time+CONTROL_OFFSET, dur+DUR_OFFSET, note+NOTE_OFFSET]
    
    Returns:
        dict with comparison metrics
    """
    num_generated = len(generated_events) // 3
    num_controls = len(controls) // 3
    
    # Should generate one event per control
    count_match = (num_generated == num_controls)
    
    # Compare pitch matches (most important)
    pitch_matches = 0
    pitch_mismatches = 0
    
    # Compare note by note
    min_len = min(num_generated, num_controls)
    
    for i in range(min_len):
        # Generated events: [time+TIME_OFFSET, dur+DUR_OFFSET, note+NOTE_OFFSET]
        gen_note = generated_events[i*3 + 2] - NOTE_OFFSET
        
        # Controls: ALL elements have CONTROL_OFFSET added (like in tokenization)
        # [time+CONTROL_OFFSET, dur+DUR_OFFSET+CONTROL_OFFSET, note+NOTE_OFFSET+CONTROL_OFFSET]
        ctrl_note = controls[i*3 + 2] - NOTE_OFFSET - CONTROL_OFFSET
        
        if gen_note == ctrl_note:
            pitch_matches += 1
        else:
            pitch_mismatches += 1
    
    match_rate = pitch_matches / min_len if min_len > 0 else 0.0
    
    return {
        'num_generated': num_generated,
        'num_controls': num_controls,
        'count_match': count_match,
        'pitch_matches': pitch_matches,
        'pitch_mismatches': pitch_mismatches,
        'match_rate': match_rate,
        'min_len': min_len
    }


def test_generation_quality(checkpoint_path, test_data_path, num_sequences=50, top_p=0.95):
    """
    Test generation quality on a subset of test sequences.
    """
    print("=" * 80)
    print("GENERATION QUALITY TEST")
    print("=" * 80)
    print()
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    
    # Check GPU availability
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
    
    # Limit to num_sequences
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
            
            # Extract controls
            controls = extract_controls_from_sequence(sequence_tokens, prefix_controls=33)
            
            if len(controls) == 0 or len(controls) % 3 != 0:
                errors += 1
                continue
            
            # Generate
            events, tokens = generate4(model, controls=controls, top_p=top_p, prefix_controls=33)
            
            # Compare
            result = compare_generated_to_controls(events, controls)
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
    total_controls = sum(r['num_controls'] for r in results)
    total_pitch_matches = sum(r['pitch_matches'] for r in results)
    total_pitch_mismatches = sum(r['pitch_mismatches'] for r in results)
    total_notes = sum(r['min_len'] for r in results)
    
    overall_match_rate = total_pitch_matches / total_notes if total_notes > 0 else 0.0
    
    count_matches = sum(1 for r in results if r['count_match'])
    count_match_rate = count_matches / len(results) if results else 0.0
    
    print(f"Successfully tested: {len(results)} sequences")
    print(f"Total notes generated: {total_generated:,}")
    print(f"Total control notes: {total_controls:,}")
    print(f"Count match rate: {count_match_rate*100:.2f}% (generated same number as controls)")
    print()
    
    print(f"Pitch comparison:")
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
        print("✅ EXCELLENT: >95% pitch match - model follows controls very closely")
    elif overall_match_rate > 0.90:
        print("✅ GOOD: 90-95% pitch match - model follows controls well")
    elif overall_match_rate > 0.80:
        print("⚠ FAIR: 80-90% pitch match - model follows controls reasonably")
    elif overall_match_rate > 0.50:
        print("⚠ POOR: 50-80% pitch match - model struggles to follow controls")
    else:
        print("❌ VERY POOR: <50% pitch match - model not following controls")
    
    print()
    print("Expected behavior:")
    print("  - Well-trained model: >90% pitch match (follows performance controls)")
    print("  - Undertrained model: 50-80% (partially follows controls)")
    print("  - Random model: ~10% (doesn't follow controls)")
    print()
    
    # Save detailed results
    output_file = 'generation_quality_results.txt'
    with open(output_file, 'w') as f:
        f.write("Generation Quality Test Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test sequences: {len(results)}\n")
        f.write(f"Overall pitch match rate: {overall_match_rate*100:.2f}%\n\n")
        
        f.write("Per-sequence results:\n")
        for r in results:
            f.write(f"Seq {r['seq_idx']:3d}: {r['match_rate']*100:6.2f}% "
                   f"({r['pitch_matches']}/{r['min_len']} matches, "
                   f"{r['num_generated']} generated vs {r['num_controls']} controls)\n")
    
    print(f"✓ Detailed results saved to: {output_file}")
    print()
    
    return {
        'overall_match_rate': overall_match_rate,
        'count_match_rate': count_match_rate,
        'results': results
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test generation quality')
    parser.add_argument('--checkpoint', default='hf-ckpt-3500/checkpoint-3500',
                       help='Path to model checkpoint')
    parser.add_argument('--test-data', default='data/test_output.txt',
                       help='Path to test data file')
    parser.add_argument('--num-sequences', type=int, default=50,
                       help='Number of test sequences to evaluate')
    parser.add_argument('--top-p', type=float, default=0.95,
                       help='Nucleus sampling parameter')
    
    args = parser.parse_args()
    
    test_generation_quality(
        checkpoint_path=args.checkpoint,
        test_data_path=args.test_data,
        num_sequences=args.num_sequences,
        top_p=args.top_p
    )


if __name__ == "__main__":
    main()
