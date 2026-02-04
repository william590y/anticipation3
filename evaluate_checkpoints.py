"""
Evaluate models on test_combined.txt with autoregressive score generation.

Generates MIDI files for: input performance, output score, and ground truth score.
Also computes aggregate statistics for pitch, duration, and time token accuracy.

Usage:
    python evaluate_checkpoints.py
"""
import os
import sys
import json
import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM
from anticipation.vocab import *
from anticipation.config import *
from anticipation.convert import events_to_midi
from tqdm import tqdm

# Configuration
CHECKPOINTS = ['checkpoint-1000', 'checkpoint-1750']
TEST_FILE = 'data/test_combined.txt'
OUTPUT_BASE = 'evaluation_results'
NUM_EXAMPLES = 100  # Randomly sample 100 sequences
RANDOM_SEED = 42
K_PREFIX = 33  # Number of control+rest pairs in prefix


def load_model(checkpoint_path):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    print(f"  Model loaded on {device}")
    return model, device


def parse_sequence(line):
    """Parse a sequence from the test file."""
    if '|' in line:
        token_str, _ = line.split('|')
        tokens = [int(t) for t in token_str.strip().split()]
    else:
        tokens = [int(t) for t in line.strip().split()]
    return tokens


def find_score_start(tokens):
    """
    Find where score notes start in the sequence.
    
    Format:
    - Position 0: ANTICIPATE
    - Positions 1-3: SEP SEP SEP
    - Positions 4-201: 33 control+rest pairs (k=33, 6 tokens each = 198 tokens)
    - Position 202+: Alternating score/control triplets
    """
    # Score starts after: ANTICIPATE (1) + SEP SEP SEP (3) + 33 control+rest pairs (198)
    # = position 202, but let's verify by finding first score triplet
    expected_start = 4 + K_PREFIX * 6  # = 202
    
    # Verify it's a score triplet: time < CONTROL, dur < CONTROL, pitch in [NOTE_OFFSET, REST)
    if expected_start + 2 < len(tokens):
        t0, t1, t2 = tokens[expected_start], tokens[expected_start+1], tokens[expected_start+2]
        if t0 < CONTROL_OFFSET and t1 < CONTROL_OFFSET and t2 >= NOTE_OFFSET and t2 < REST:
            return expected_start
    
    # Fallback: search for first score triplet
    for i in range(4, len(tokens) - 2, 3):
        t0, t1, t2 = tokens[i], tokens[i+1], tokens[i+2]
        if t0 < CONTROL_OFFSET and t1 < CONTROL_OFFSET and t2 >= NOTE_OFFSET and t2 < REST:
            return i
    
    return None


def extract_components(tokens, score_start_idx):
    """
    Extract performance, score, and control information from interleaved sequence.
    
    Returns:
        performance_triplets: List of [time, dur, pitch] (raw values, no offset)
        score_triplets: List of [time, dur, pitch] (with offsets)
        control_positions: List of indices where control triplets occur after score_start
    """
    # Extract performance from control+rest pairs (positions 4 to score_start_idx)
    # Control triplets use ATIME_OFFSET, ADUR_OFFSET, ANOTE_OFFSET
    performance = []
    for i in range(4, score_start_idx, 6):
        if i + 2 < len(tokens):
            ctrl_time = tokens[i] - ATIME_OFFSET
            ctrl_dur = tokens[i + 1] - ADUR_OFFSET
            ctrl_pitch = tokens[i + 2] - ANOTE_OFFSET
            performance.append([ctrl_time, ctrl_dur, ctrl_pitch])
    
    # Extract from alternating section
    alternating = tokens[score_start_idx:]
    score_triplets = []
    control_positions = []  # positions relative to score_start_idx
    
    pos = 0
    while pos + 2 < len(alternating):
        t0, t1, t2 = alternating[pos], alternating[pos+1], alternating[pos+2]
        
        # Check if score triplet: time < CONTROL, dur < CONTROL, pitch in [NOTE_OFFSET, REST)
        if t0 < CONTROL_OFFSET and t1 < CONTROL_OFFSET and t2 >= NOTE_OFFSET and t2 < REST:
            score_triplets.append([t0, t1, t2])
            pos += 3
            
            # Check for following control triplet
            if pos + 2 < len(alternating):
                c0, c1, c2 = alternating[pos], alternating[pos+1], alternating[pos+2]
                if c0 >= CONTROL_OFFSET and c1 >= CONTROL_OFFSET and c2 >= CONTROL_OFFSET:
                    control_positions.append(pos)
                    # Add performance note (use proper offsets)
                    performance.append([c0 - ATIME_OFFSET, c1 - ADUR_OFFSET, c2 - ANOTE_OFFSET])
                    pos += 3
                else:
                    break  # End of valid sequence
            else:
                break
        else:
            break
    
    return performance, score_triplets, control_positions


def autoregressive_generate_score(model, tokens, score_start_idx, device):
    """
    Generate score tokens autoregressively while keeping control tokens fixed.
    
    The sequence format after score_start_idx is:
    [score_triplet, control_triplet, score_triplet, control_triplet, ...]
    
    We generate score triplets token-by-token, then insert the ground truth
    control triplets to maintain the interleaved structure.
    """
    # Context is everything before first score triplet
    context = tokens[:score_start_idx]
    
    # Parse the ground truth alternating section to get control triplet positions
    gt_alternating = tokens[score_start_idx:]
    
    # Count how many score/control pairs exist
    num_score_triplets = 0
    control_triplets = []
    
    pos = 0
    while pos + 2 < len(gt_alternating):
        t0, t1, t2 = gt_alternating[pos], gt_alternating[pos+1], gt_alternating[pos+2]
        
        # Check if this is a score triplet
        if t0 < CONTROL_OFFSET and t1 < CONTROL_OFFSET and t2 >= NOTE_OFFSET and t2 < REST:
            num_score_triplets += 1
            pos += 3
            
            # Check for following control triplet
            if pos + 2 < len(gt_alternating):
                c0, c1, c2 = gt_alternating[pos], gt_alternating[pos+1], gt_alternating[pos+2]
                if c0 >= CONTROL_OFFSET:
                    control_triplets.append([c0, c1, c2])
                    pos += 3
                else:
                    break
            else:
                break
        else:
            break
    
    # Generate autoregressively
    generated = list(context)
    
    with torch.no_grad():
        # Process context to build initial KV cache
        input_ids = torch.tensor([context], device=device)
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        
        for triplet_idx in range(num_score_triplets):
            # Generate 3 score tokens
            for token_in_triplet in range(3):
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).item()
                generated.append(next_token)
                
                # Forward pass for next token
                input_ids = torch.tensor([[next_token]], device=device)
                outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
            
            # Insert control triplet from ground truth (if available)
            if triplet_idx < len(control_triplets):
                ctrl = control_triplets[triplet_idx]
                generated.extend(ctrl)
                
                # Update KV cache with control tokens
                ctrl_ids = torch.tensor([ctrl], device=device)
                outputs = model(ctrl_ids, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
            
            if len(generated) >= CONTEXT_SIZE:
                break
    
    return generated


def compute_statistics(gt_score, pred_score):
    """
    Compute accuracy statistics for time, duration, and pitch tokens.
    
    Returns dict with:
        - time_correct, time_total, time_accuracy
        - dur_correct, dur_total, dur_accuracy  
        - pitch_correct, pitch_total, pitch_accuracy
        - overall_correct, overall_total, overall_accuracy
    """
    min_len = min(len(gt_score), len(pred_score))
    
    stats = {
        'time_correct': 0, 'time_total': 0,
        'dur_correct': 0, 'dur_total': 0,
        'pitch_correct': 0, 'pitch_total': 0,
        'overall_correct': 0, 'overall_total': 0,
        'num_gt_notes': len(gt_score),
        'num_pred_notes': len(pred_score),
    }
    
    for i in range(min_len):
        gt_time, gt_dur, gt_pitch = gt_score[i]
        pred_time, pred_dur, pred_pitch = pred_score[i]
        
        # Time accuracy
        stats['time_total'] += 1
        if gt_time == pred_time:
            stats['time_correct'] += 1
        
        # Duration accuracy
        stats['dur_total'] += 1
        if gt_dur == pred_dur:
            stats['dur_correct'] += 1
        
        # Pitch accuracy
        stats['pitch_total'] += 1
        if gt_pitch == pred_pitch:
            stats['pitch_correct'] += 1
        
        # Overall (all 3 must match)
        stats['overall_total'] += 1
        if gt_time == pred_time and gt_dur == pred_dur and gt_pitch == pred_pitch:
            stats['overall_correct'] += 1
    
    # Compute percentages
    for key in ['time', 'dur', 'pitch', 'overall']:
        total = stats[f'{key}_total']
        correct = stats[f'{key}_correct']
        stats[f'{key}_accuracy'] = 100.0 * correct / total if total > 0 else 0.0
    
    return stats


def triplets_to_events(triplets):
    """Convert list of [time, dur, pitch] triplets to flat event list."""
    events = []
    for t in triplets:
        events.extend(t)
    return events


def save_midi(events, filepath):
    """Save events as MIDI file."""
    try:
        midi = events_to_midi(events)
        midi.save(filepath)
        return True
    except Exception as e:
        print(f"    Warning: Could not save {filepath}: {e}")
        return False


def evaluate_checkpoint(checkpoint_path, test_lines, original_indices, output_dir):
    """
    Evaluate a single checkpoint on test sequences.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_lines: List of test sequence strings
        original_indices: Original indices in the full test set (for folder naming)
        output_dir: Output directory path
    
    Returns aggregate statistics.
    """
    model, device = load_model(checkpoint_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate statistics
    aggregate = {
        'time_correct': 0, 'time_total': 0,
        'dur_correct': 0, 'dur_total': 0,
        'pitch_correct': 0, 'pitch_total': 0,
        'overall_correct': 0, 'overall_total': 0,
        'num_sequences': 0,
        'num_failed': 0,
    }
    
    per_sequence_stats = []
    
    for i, (line, orig_idx) in enumerate(tqdm(zip(test_lines, original_indices), 
                                               total=len(test_lines),
                                               desc=f"Evaluating {checkpoint_path}")):
        tokens = parse_sequence(line)
        
        # Find score start
        score_start_idx = find_score_start(tokens)
        if score_start_idx is None:
            aggregate['num_failed'] += 1
            continue
        
        # Extract ground truth components
        gt_perf, gt_score, _ = extract_components(tokens, score_start_idx)
        
        if len(gt_score) < 10:
            aggregate['num_failed'] += 1
            continue
        
        # Generate predictions
        try:
            predicted_tokens = autoregressive_generate_score(model, tokens, score_start_idx, device)
            _, pred_score, _ = extract_components(predicted_tokens, score_start_idx)
        except Exception as e:
            print(f"  Sequence {orig_idx}: Generation failed - {e}")
            aggregate['num_failed'] += 1
            continue
        
        # Compute statistics
        stats = compute_statistics(gt_score, pred_score)
        stats['original_index'] = orig_idx
        per_sequence_stats.append(stats)
        
        # Update aggregate
        for key in ['time_correct', 'time_total', 'dur_correct', 'dur_total', 
                    'pitch_correct', 'pitch_total', 'overall_correct', 'overall_total']:
            aggregate[key] += stats[key]
        aggregate['num_sequences'] += 1
        
        # Save MIDI files for this sequence (use original index for folder name)
        seq_dir = os.path.join(output_dir, f'sequence_{orig_idx:04d}')
        os.makedirs(seq_dir, exist_ok=True)
        
        # Performance MIDI (convert to proper format with offsets)
        perf_events = []
        for p in gt_perf:
            perf_events.extend([p[0] + TIME_OFFSET, p[1] + DUR_OFFSET, p[2] + NOTE_OFFSET])
        save_midi(perf_events, os.path.join(seq_dir, 'input_performance.mid'))
        
        # Ground truth score MIDI
        gt_score_events = triplets_to_events(gt_score)
        save_midi(gt_score_events, os.path.join(seq_dir, 'ground_truth_score.mid'))
        
        # Predicted score MIDI
        pred_score_events = triplets_to_events(pred_score)
        save_midi(pred_score_events, os.path.join(seq_dir, 'output_score.mid'))
        
        # Save per-sequence stats
        with open(os.path.join(seq_dir, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
    
    # Compute aggregate percentages
    for key in ['time', 'dur', 'pitch', 'overall']:
        total = aggregate[f'{key}_total']
        correct = aggregate[f'{key}_correct']
        aggregate[f'{key}_accuracy'] = 100.0 * correct / total if total > 0 else 0.0
    
    # Save aggregate statistics
    with open(os.path.join(output_dir, 'aggregate_stats.json'), 'w') as f:
        json.dump(aggregate, f, indent=2)
    
    # Save per-sequence statistics
    with open(os.path.join(output_dir, 'per_sequence_stats.json'), 'w') as f:
        json.dump(per_sequence_stats, f, indent=2)
    
    return aggregate


def print_summary(checkpoint_name, stats):
    """Print summary statistics for a checkpoint."""
    print(f"\n{'='*60}")
    print(f"Results for {checkpoint_name}")
    print(f"{'='*60}")
    print(f"  Sequences evaluated: {stats['num_sequences']}")
    print(f"  Sequences failed: {stats['num_failed']}")
    print()
    print(f"  Time accuracy:     {stats['time_accuracy']:.2f}% ({stats['time_correct']}/{stats['time_total']})")
    print(f"  Duration accuracy: {stats['dur_accuracy']:.2f}% ({stats['dur_correct']}/{stats['dur_total']})")
    print(f"  Pitch accuracy:    {stats['pitch_accuracy']:.2f}% ({stats['pitch_correct']}/{stats['pitch_total']})")
    print(f"  Overall accuracy:  {stats['overall_accuracy']:.2f}% ({stats['overall_correct']}/{stats['overall_total']})")
    print()


def main():
    print("="*80)
    print("CHECKPOINT EVALUATION ON test_combined.txt")
    print("="*80)
    print()
    
    # Check test file exists
    if not os.path.exists(TEST_FILE):
        print(f"ERROR: Test file not found: {TEST_FILE}")
        sys.exit(1)
    
    # Load test data
    print(f"Loading test data from {TEST_FILE}...")
    with open(TEST_FILE, 'r') as f:
        all_lines = [line.strip() for line in f if line.strip()]
    print(f"  Found {len(all_lines)} total test sequences")
    
    # Randomly sample sequences
    random.seed(RANDOM_SEED)
    if NUM_EXAMPLES is not None and NUM_EXAMPLES < len(all_lines):
        sampled_indices = random.sample(range(len(all_lines)), NUM_EXAMPLES)
        sampled_indices.sort()  # Keep in order for reproducibility
        test_lines = [all_lines[i] for i in sampled_indices]
        print(f"  Randomly sampled {NUM_EXAMPLES} sequences (seed={RANDOM_SEED})")
    else:
        test_lines = all_lines
        sampled_indices = list(range(len(all_lines)))
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # Save sampled indices for reproducibility
    with open(os.path.join(OUTPUT_BASE, 'sampled_indices.json'), 'w') as f:
        json.dump({'seed': RANDOM_SEED, 'num_samples': len(sampled_indices), 'indices': sampled_indices}, f, indent=2)
    
    all_results = {}
    
    for checkpoint in CHECKPOINTS:
        if not os.path.exists(checkpoint):
            print(f"WARNING: Checkpoint not found: {checkpoint}, skipping...")
            continue
        
        output_dir = os.path.join(OUTPUT_BASE, checkpoint)
        stats = evaluate_checkpoint(checkpoint, test_lines, sampled_indices, output_dir)
        all_results[checkpoint] = stats
        print_summary(checkpoint, stats)
    
    # Print comparison if multiple checkpoints
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print()
        print(f"{'Checkpoint':<20} {'Time':>10} {'Duration':>10} {'Pitch':>10} {'Overall':>10}")
        print("-"*62)
        for ckpt, stats in all_results.items():
            print(f"{ckpt:<20} {stats['time_accuracy']:>9.2f}% {stats['dur_accuracy']:>9.2f}% "
                  f"{stats['pitch_accuracy']:>9.2f}% {stats['overall_accuracy']:>9.2f}%")
    
    # Save overall summary
    summary_file = os.path.join(OUTPUT_BASE, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nOutput directories:")
    for ckpt in all_results:
        print(f"  {OUTPUT_BASE}/{ckpt}/")
    print(f"\nEach sequence folder contains:")
    print("  • input_performance.mid - The input performance")
    print("  • ground_truth_score.mid - The actual score from test data")
    print("  • output_score.mid - Model's autoregressive predictions")
    print("  • stats.json - Per-sequence accuracy statistics")


if __name__ == '__main__':
    main()
