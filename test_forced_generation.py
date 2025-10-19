"""
Test generation quality with pitch forcing.

This tests:
1. Normal generation (baseline)
2. Forced pitch generation (constrains model to correct pitch)

Comparison helps understand:
- If model learns timing and duration patterns
- If pitch selection is the main weakness
"""

import os
import torch
import numpy as np
import argparse
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from anticipation.sample import generate4, generate4_forced
from anticipation.config import *
from anticipation.vocab import *


def extract_from_sequence(sequence_tokens, prefix_controls=33):
    """Extract controls and scores from test sequence."""
    if len(sequence_tokens) < 4:
        return [], []
    
    tokens = sequence_tokens[4:]  # Skip ANTICIPATE + 3 SEPs
    
    all_controls = []
    all_scores = []
    
    i = 0
    k = prefix_controls
    
    # Extract prefix
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
    
    # Extract body
    while i + 3 <= len(tokens):
        triplet = tokens[i:i+3]
        
        if triplet[0] >= CONTROL_OFFSET:
            all_controls.extend(triplet)
        elif triplet[2] != REST:
            all_scores.extend(triplet)
        
        i += 3
    
    # Trim controls to match scores
    num_scores = len(all_scores) // 3
    controls_trimmed = all_controls[:num_scores * 3]
    
    return controls_trimmed, all_scores


def compare_events(generated, ground_truth, check_type="all"):
    """
    Compare generated to ground truth.
    
    Args:
        generated: [time+TIME, dur+DUR, note+NOTE]
        ground_truth: [time+TIME, dur+DUR, note+NOTE]
        check_type: "all", "timing", "duration", "pitch"
    """
    num_gen = len(generated) // 3
    num_gt = len(ground_truth) // 3
    min_len = min(num_gen, num_gt)
    
    matches = 0
    
    for i in range(min_len):
        gen_triplet = generated[i*3:i*3+3]
        gt_triplet = ground_truth[i*3:i*3+3]
        
        if check_type == "all":
            if gen_triplet == gt_triplet:
                matches += 1
        elif check_type == "timing":
            if gen_triplet[0] == gt_triplet[0]:
                matches += 1
        elif check_type == "duration":
            if gen_triplet[1] == gt_triplet[1]:
                matches += 1
        elif check_type == "pitch":
            if gen_triplet[2] == gt_triplet[2]:
                matches += 1
    
    return {
        'matches': matches,
        'total': min_len,
        'rate': matches / min_len if min_len > 0 else 0.0
    }


def test_forced_generation(checkpoint_path, test_data_path, num_sequences=10, top_p=0.95):
    """Test normal vs forced pitch generation."""
    print("=" * 80)
    print("FORCED PITCH GENERATION TEST")
    print("=" * 80)
    print()
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ No GPU available")
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    print(f"✓ Model loaded on {next(model.parameters()).device}")
    print()
    
    # Load test sequences
    with open(test_data_path, 'r') as f:
        test_lines = f.readlines()
    
    test_lines = test_lines[:num_sequences]
    print(f"Testing on {len(test_lines)} sequences")
    print()
    
    # Results storage
    normal_results = {
        'timing': [],
        'duration': [],
        'pitch': [],
        'all': []
    }
    
    forced_results = {
        'timing': [],
        'duration': [],
        'all': []
    }
    
    # Process sequences
    print("Generating...")
    for seq_idx, line in enumerate(tqdm(test_lines, desc="Testing sequences")):
        try:
            sequence_tokens = [int(tok) for tok in line.strip().split()]
            controls, ground_truth = extract_from_sequence(sequence_tokens, prefix_controls=33)
            
            if len(controls) == 0 or len(ground_truth) == 0:
                continue
            
            # Normal generation
            events_normal, _ = generate4(model, controls=controls, top_p=top_p, prefix_controls=33)
            
            # Forced pitch generation (rejection sampling with limited attempts)
            events_forced, _ = generate4_forced(
                model, 
                controls=controls, 
                ground_truth_scores=ground_truth,
                top_p=top_p, 
                prefix_controls=33,
                max_attempts=20  # Limit attempts to keep test reasonable
            )
            
            # Compare normal generation
            normal_results['timing'].append(compare_events(events_normal, ground_truth, "timing")['rate'])
            normal_results['duration'].append(compare_events(events_normal, ground_truth, "duration")['rate'])
            normal_results['pitch'].append(compare_events(events_normal, ground_truth, "pitch")['rate'])
            normal_results['all'].append(compare_events(events_normal, ground_truth, "all")['rate'])
            
            # Compare forced generation (pitch should be 100%)
            forced_results['timing'].append(compare_events(events_forced, ground_truth, "timing")['rate'])
            forced_results['duration'].append(compare_events(events_forced, ground_truth, "duration")['rate'])
            forced_results['all'].append(compare_events(events_forced, ground_truth, "all")['rate'])
            
        except Exception as e:
            print(f"\n⚠ Error in sequence {seq_idx}: {e}")
            continue
    
    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    print("NORMAL GENERATION (free sampling):")
    print(f"  Timing match:   {np.mean(normal_results['timing'])*100:.2f}% ± {np.std(normal_results['timing'])*100:.2f}%")
    print(f"  Duration match: {np.mean(normal_results['duration'])*100:.2f}% ± {np.std(normal_results['duration'])*100:.2f}%")
    print(f"  Pitch match:    {np.mean(normal_results['pitch'])*100:.2f}% ± {np.std(normal_results['pitch'])*100:.2f}%")
    print(f"  Perfect match:  {np.mean(normal_results['all'])*100:.2f}% ± {np.std(normal_results['all'])*100:.2f}%")
    print()
    
    print("FORCED PITCH GENERATION (pitch constrained to ground truth):")
    print(f"  Timing match:   {np.mean(forced_results['timing'])*100:.2f}% ± {np.std(forced_results['timing'])*100:.2f}%")
    print(f"  Duration match: {np.mean(forced_results['duration'])*100:.2f}% ± {np.std(forced_results['duration'])*100:.2f}%")
    print(f"  Pitch match:    100.00% (forced)")
    print(f"  Perfect match:  {np.mean(forced_results['all'])*100:.2f}% ± {np.std(forced_results['all'])*100:.2f}%")
    print()
    
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    
    timing_improvement = np.mean(forced_results['timing']) - np.mean(normal_results['timing'])
    duration_improvement = np.mean(forced_results['duration']) - np.mean(normal_results['duration'])
    
    print(f"By forcing correct pitch:")
    print(f"  Timing accuracy changes by:   {timing_improvement*100:+.2f}%")
    print(f"  Duration accuracy changes by: {duration_improvement*100:+.2f}%")
    print()
    
    if np.mean(forced_results['timing']) > 0.5:
        print("✓ GOOD: Model learned timing patterns (>50% accuracy with forced pitch)")
    else:
        print("❌ POOR: Model struggles with timing even with forced pitch")
    
    if np.mean(forced_results['duration']) > 0.5:
        print("✓ GOOD: Model learned duration patterns (>50% accuracy with forced pitch)")
    else:
        print("❌ POOR: Model struggles with duration even with forced pitch")
    
    if np.mean(normal_results['pitch']) < 0.3:
        print("❌ POOR: Pitch selection is very weak (<30% accuracy)")
    else:
        print("✓ OK: Pitch selection shows some learning (≥30% accuracy)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="hf-ckpt-3500/checkpoint-3500")
    parser.add_argument("--test-data", type=str, default="data/test_output.txt")
    parser.add_argument("--num-sequences", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.95)
    
    args = parser.parse_args()
    
    test_forced_generation(
        args.checkpoint,
        args.test_data,
        args.num_sequences,
        args.top_p
    )
