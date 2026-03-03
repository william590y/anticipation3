"""
Evaluate models on test data with autoregressive score generation.

Generates MIDI files for: input performance, output score, and ground truth score.
Also computes aggregate statistics for pitch, duration, and time token accuracy.

Alignment with train.py (so metrics are comparable):
  - Alternating section start: fixed 202 (ALTERNATING_START), same as train.py.
  - Default test file: data/test_combined.txt (match --val_file used when training).
  - Generation: GT control triplet added after each predicted score triplet (train-style).
  - No minimum-note filter: all sequences with len > 202 are evaluated (same as train.py/inference.py).

Usage:
    python evaluate_checkpoints.py [--test-file PATH]
    
    Default test file: data/test_combined.txt (match --val_file used when training)
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
# Default test file: use test_combined.txt to match --val_file used during training
TEST_FILE = 'data/test_combined.txt'
OUTPUT_BASE = 'evaluation_results_corrected'
NUM_EXAMPLES = 10  # Randomly sample sequences
RANDOM_SEED = 42
K_PREFIX = 33  # Number of control+rest pairs in prefix
# Must match train.py: alternating section starts at 202 (ANTICIPATE + SEP×3 + 33 control+rest pairs)
ALTERNATING_START = 4 + K_PREFIX * 6  # = 202


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
    """Parse a sequence from the test file.
    
    Matches train.py's TokenizedDataset preprocessing:
    - Replace negative tokens with 0 (TIME_OFFSET)
    """
    if '|' in line:
        token_str, _ = line.split('|')
        tokens = [int(t) for t in token_str.strip().split()]
    else:
        tokens = [int(t) for t in line.strip().split()]
    # Match train.py: replace invalid negative tokens with 0
    tokens = [max(0, t) for t in tokens]
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
    
    # Verify it's a score triplet: all 3 tokens < CONTROL_OFFSET, pitch != REST (matching train.py logic)
    if expected_start + 2 < len(tokens):
        t0, t1, t2 = tokens[expected_start], tokens[expected_start+1], tokens[expected_start+2]
        if t0 < CONTROL_OFFSET and t1 < CONTROL_OFFSET and t2 < CONTROL_OFFSET and t2 != REST:
            return expected_start
    
    # Fallback: search for first score triplet
    for i in range(4, len(tokens) - 2, 3):
        t0, t1, t2 = tokens[i], tokens[i+1], tokens[i+2]
        if t0 < CONTROL_OFFSET and t1 < CONTROL_OFFSET and t2 < CONTROL_OFFSET and t2 != REST:
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
        
        # Check if score triplet: all 3 tokens < CONTROL_OFFSET, pitch != REST (matching train.py)
        if t0 < CONTROL_OFFSET and t1 < CONTROL_OFFSET and t2 < CONTROL_OFFSET and t2 != REST:
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
    
    This exactly matches train.py's autoregressive evaluation logic:
    - Loop while pos + 5 < len(tokens) (need 6 tokens: score + control)
    - If score triplet: generate 3 tokens, add GT control triplet
    - If not score triplet: add single GT token and continue
    """
    # Start context with everything before alternating section (positions 0 to score_start_idx-1)
    context = list(tokens[:score_start_idx])
    
    pos = score_start_idx
    while pos + 5 < len(tokens):
        # Check if this is a score triplet (all 3 tokens < CONTROL_OFFSET, pitch != REST)
        if (tokens[pos] < CONTROL_OFFSET and 
            tokens[pos+1] < CONTROL_OFFSET and 
            tokens[pos+2] < CONTROL_OFFSET and
            tokens[pos+2] != REST):
            
            # This is a score triplet - generate it autoregressively
            with torch.no_grad():
                # Generate TIME token
                input_tensor = torch.tensor([context], device=device)
                outputs = model(input_tensor)
                pred_time = outputs.logits[0, -1, :].argmax().item()
                context.append(pred_time)
                
                # Generate DURATION token
                input_tensor = torch.tensor([context], device=device)
                outputs = model(input_tensor)
                pred_dur = outputs.logits[0, -1, :].argmax().item()
                context.append(pred_dur)
                
                # Generate PITCH token
                input_tensor = torch.tensor([context], device=device)
                outputs = model(input_tensor)
                pred_pitch = outputs.logits[0, -1, :].argmax().item()
                context.append(pred_pitch)
            
            pos += 3
            
            # After score triplet, add ground truth control triplet to context
            # (We're only testing score generation, not control generation)
            if pos + 2 < len(tokens):
                context.extend([tokens[pos], tokens[pos+1], tokens[pos+2]])
                pos += 3
        else:
            # Not a score triplet, add to context and continue (matching train.py)
            context.append(tokens[pos])
            pos += 1
    
    return context


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


def compute_statistics_by_position(gt_tokens, pred_tokens, score_start_idx):
    """
    Compute accuracy by comparing tokens at score triplet POSITIONS, not by parsing.
    This matches train.py's evaluation logic exactly.
    
    Uses same loop termination: pos + 5 < len (need 6 tokens for score + control)
    """
    stats = {
        'time_correct': 0, 'time_total': 0,
        'dur_correct': 0, 'dur_total': 0,
        'pitch_correct': 0, 'pitch_total': 0,
        'overall_correct': 0, 'overall_total': 0,
        'num_gt_notes': 0,
        'num_pred_notes': 0,
    }
    
    # Match train.py's loop exactly
    pos = score_start_idx
    while pos + 5 < len(gt_tokens) and pos + 5 < len(pred_tokens):
        # Check if GT has a score triplet here (all < CONTROL_OFFSET, pitch != REST)
        gt_t0, gt_t1, gt_t2 = gt_tokens[pos], gt_tokens[pos+1], gt_tokens[pos+2]
        
        if (gt_t0 < CONTROL_OFFSET and gt_t1 < CONTROL_OFFSET and 
            gt_t2 < CONTROL_OFFSET and gt_t2 != REST):
            # This is a score triplet position - compare with generated
            pred_t0, pred_t1, pred_t2 = pred_tokens[pos], pred_tokens[pos+1], pred_tokens[pos+2]
            
            stats['num_gt_notes'] += 1
            stats['num_pred_notes'] += 1
            
            # Time accuracy
            stats['time_total'] += 1
            if gt_t0 == pred_t0:
                stats['time_correct'] += 1
            
            # Duration accuracy
            stats['dur_total'] += 1
            if gt_t1 == pred_t1:
                stats['dur_correct'] += 1
            
            # Pitch accuracy (this is what train.py tracks)
            stats['pitch_total'] += 1
            if gt_t2 == pred_t2:
                stats['pitch_correct'] += 1
            
            # Overall
            stats['overall_total'] += 1
            if gt_t0 == pred_t0 and gt_t1 == pred_t1 and gt_t2 == pred_t2:
                stats['overall_correct'] += 1
            
            # Move past score triplet (3 tokens)
            pos += 3
            
            # Skip control triplet (3 tokens) - matching train.py
            if pos + 2 < len(gt_tokens):
                pos += 3
        else:
            # Not a score triplet - skip one token (matching train.py's else branch)
            pos += 1
    
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


def normalize_triplet_times(triplets):
    """Normalize triplet times to start at 0 and sort by time.
    
    Triplets have format [time+TIME_OFFSET, dur+DUR_OFFSET, pitch+NOTE_OFFSET].
    """
    if not triplets:
        return triplets
    # Sort by time first
    triplets = sorted(triplets, key=lambda t: t[0])
    # Find minimum time (subtract TIME_OFFSET to get raw time)
    min_time = min(t[0] - TIME_OFFSET for t in triplets)
    # Shift all times by min_time
    return [[t[0] - min_time, t[1], t[2]] for t in triplets]


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
        
        # Use fixed alternating start (202) to match train.py exactly; skip if sequence too short
        if len(tokens) <= ALTERNATING_START:
            aggregate['num_failed'] += 1
            continue
        score_start_idx = ALTERNATING_START
        
        # Extract ground truth components (for MIDI saving, not accuracy)
        gt_perf, gt_score, _ = extract_components(tokens, score_start_idx)
        
        # Do not filter by len(gt_score): train.py and inference.py evaluate all sequences
        # with len > 202. Filtering to len(gt_score) >= 10 biased accuracy downward.
        
        # Generate predictions
        try:
            predicted_tokens = autoregressive_generate_score(model, tokens, score_start_idx, device)
        except Exception as e:
            print(f"  Sequence {orig_idx}: Generation failed - {e}")
            aggregate['num_failed'] += 1
            continue
        
        # Compute statistics by POSITION (matching train.py logic)
        stats = compute_statistics_by_position(tokens, predicted_tokens, score_start_idx)
        stats['original_index'] = orig_idx
        per_sequence_stats.append(stats)
        
        # Extract predicted triplets for MIDI saving only
        _, pred_score, _ = extract_components(predicted_tokens, score_start_idx)
        
        # Update aggregate
        for key in ['time_correct', 'time_total', 'dur_correct', 'dur_total', 
                    'pitch_correct', 'pitch_total', 'overall_correct', 'overall_total']:
            aggregate[key] += stats[key]
        aggregate['num_sequences'] += 1
        
        # Save MIDI files for this sequence (use original index for folder name)
        seq_dir = os.path.join(output_dir, f'sequence_{orig_idx:04d}')
        os.makedirs(seq_dir, exist_ok=True)
        
        # Performance MIDI (convert to proper format with offsets, then normalize)
        perf_triplets = [[p[0] + TIME_OFFSET, p[1] + DUR_OFFSET, p[2] + NOTE_OFFSET] for p in gt_perf]
        perf_triplets = normalize_triplet_times(perf_triplets)
        perf_events = triplets_to_events(perf_triplets)
        save_midi(perf_events, os.path.join(seq_dir, 'input_performance.mid'))
        
        # Ground truth score MIDI (normalize times to start at 0)
        gt_score_normalized = normalize_triplet_times(gt_score)
        gt_score_events = triplets_to_events(gt_score_normalized)
        save_midi(gt_score_events, os.path.join(seq_dir, 'ground_truth_score.mid'))
        
        # Predicted score MIDI (normalize times to start at 0)
        pred_score_normalized = normalize_triplet_times(pred_score)
        pred_score_events = triplets_to_events(pred_score_normalized)
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


def main(test_file=None):
    if test_file is None:
        test_file = TEST_FILE
    
    print("="*80)
    print(f"CHECKPOINT EVALUATION ON {os.path.basename(test_file)}")
    print("="*80)
    print()
    
    # Check test file exists
    if not os.path.exists(test_file):
        print(f"ERROR: Test file not found: {test_file}")
        sys.exit(1)
    
    # Load test data
    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r') as f:
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
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate checkpoints with autoregressive generation')
    parser.add_argument('--test-file', type=str, default=TEST_FILE, 
                        help=f'Path to test file (default: {TEST_FILE})')
    args = parser.parse_args()
    main(test_file=args.test_file)
