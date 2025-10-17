"""
Evaluate model accuracy by comparing generated sequences against ground truth.
Checks:
1. Note accuracy (pitch)
2. Duration accuracy
3. Time accuracy (onset timing)
"""

import os
import numpy as np
from pathlib import Path

from anticipation.config import *
from anticipation.vocab import *

# Paths
TEST_DATA_PATH = r'data\test_output.txt'
OUTPUT_DIR = r'test_outputs'

def load_test_sequences(test_file_path):
    """Load all test sequences from the test output file."""
    sequences = []
    with open(test_file_path, 'r') as f:
        for line in f:
            tokens = list(map(int, line.strip().split()))
            sequences.append(tokens)
    return sequences


def extract_score_tokens(sequence_tokens):
    """
    Extract score tokens (ground truth performance) from a tokenized sequence.
    
    The tokenization format is:
    [ANTICIPATE, SEP, SEP, SEP, 
     perf_ctrl_0, rest_0, ..., perf_ctrl_32, rest_32,  # PREFIX (33 performance controls)
     score_0, perf_ctrl_33, score_1, perf_ctrl_34, ...]  # ALTERNATING (score, future performance)
    
    We want to extract the score tokens (even positions in alternating section).
    
    Returns:
        score_events: List of triplets [time, dur, note] for score (ground truth)
    """
    # Skip the control flag and 3 SEPARATORs
    if len(sequence_tokens) < 4:
        return []
    
    tokens = sequence_tokens[4:]
    
    # Skip the prefix (33 performance controls + 33 rests = 66 triplets = 198 tokens)
    prefix_length = 33 * 6  # 33 pairs of (perf_ctrl triplet + rest triplet)
    if len(tokens) < prefix_length:
        return []
    
    alternating_tokens = tokens[prefix_length:]
    
    # Extract score events from alternating pattern
    # Pattern: [score_triplet, perf_ctrl_triplet, score_triplet, perf_ctrl_triplet, ...]
    score_events = []
    i = 0
    while i + 3 <= len(alternating_tokens):
        triplet = alternating_tokens[i:i+3]
        
        # Score tokens have time < CONTROL_OFFSET
        if triplet[0] < CONTROL_OFFSET:
            score_events.extend(triplet)
        
        i += 3
    
    return score_events


def load_generated_tokens(sequence_idx):
    """
    Load generated tokens from saved token files.
    """
    tokens_path = os.path.join(OUTPUT_DIR, f'test_seq_{sequence_idx:04d}_tokens.txt')
    if not os.path.exists(tokens_path):
        return None
    
    with open(tokens_path, 'r') as f:
        tokens = list(map(int, f.read().strip().split()))
    
    return tokens


def decode_events(token_list):
    """
    Decode a list of tokens into human-readable events.
    
    Args:
        token_list: List of tokens (must be divisible by 3)
    
    Returns:
        events: List of dicts with keys: time, duration, note, instrument
    """
    if len(token_list) % 3 != 0:
        return []
    
    events = []
    for i in range(0, len(token_list), 3):
        time_tok = token_list[i]
        dur_tok = token_list[i+1]
        note_tok = token_list[i+2]
        
        # Decode based on whether it's a control or regular token
        if note_tok >= CONTROL_OFFSET:
            # Control token
            time = time_tok - ATIME_OFFSET
            duration = dur_tok - ADUR_OFFSET
            note_val = note_tok - ANOTE_OFFSET
            is_control = True
        else:
            # Regular score token
            time = time_tok - TIME_OFFSET
            duration = dur_tok - DUR_OFFSET
            note_val = note_tok - NOTE_OFFSET
            is_control = False
        
        # Extract pitch and instrument
        pitch = note_val % 128
        instrument = note_val // 128
        
        events.append({
            'time': time,
            'duration': duration,
            'pitch': pitch,
            'instrument': instrument,
            'is_control': is_control,
            'raw_tokens': (time_tok, dur_tok, note_tok)
        })
    
    return events


def compare_events(ground_truth_events, generated_events):
    """
    Compare ground truth and generated events.
    
    Returns:
        dict with accuracy metrics
    """
    gt = decode_events(ground_truth_events)
    gen = decode_events(generated_events)
    
    # Basic statistics
    num_gt = len(gt)
    num_gen = len(gen)
    
    # For now, we'll just compare what we can
    # Ideally, generated events should match ground truth in count
    min_len = min(num_gt, num_gen)
    
    if min_len == 0:
        return {
            'num_ground_truth': num_gt,
            'num_generated': num_gen,
            'note_accuracy': 0.0,
            'duration_accuracy': 0.0,
            'time_accuracy': 0.0,
            'exact_match': 0.0,
        }
    
    # Compare event by event
    note_matches = 0
    duration_matches = 0
    time_matches = 0
    exact_matches = 0
    
    time_errors = []
    duration_errors = []
    
    for i in range(min_len):
        gt_event = gt[i]
        gen_event = gen[i]
        
        # Note accuracy (pitch + instrument)
        if gt_event['pitch'] == gen_event['pitch'] and gt_event['instrument'] == gen_event['instrument']:
            note_matches += 1
        
        # Duration accuracy (exact match)
        if gt_event['duration'] == gen_event['duration']:
            duration_matches += 1
        
        # Time accuracy (exact match)
        if gt_event['time'] == gen_event['time']:
            time_matches += 1
        
        # Exact match (all three)
        if (gt_event['pitch'] == gen_event['pitch'] and 
            gt_event['instrument'] == gen_event['instrument'] and
            gt_event['duration'] == gen_event['duration'] and
            gt_event['time'] == gen_event['time']):
            exact_matches += 1
        
        # Track errors
        time_errors.append(abs(gt_event['time'] - gen_event['time']))
        duration_errors.append(abs(gt_event['duration'] - gen_event['duration']))
    
    return {
        'num_ground_truth': num_gt,
        'num_generated': num_gen,
        'num_compared': min_len,
        'note_accuracy': note_matches / min_len if min_len > 0 else 0.0,
        'duration_accuracy': duration_matches / min_len if min_len > 0 else 0.0,
        'time_accuracy': time_matches / min_len if min_len > 0 else 0.0,
        'exact_match_accuracy': exact_matches / min_len if min_len > 0 else 0.0,
        'mean_time_error': np.mean(time_errors) if time_errors else 0.0,
        'mean_duration_error': np.mean(duration_errors) if duration_errors else 0.0,
        'note_matches': note_matches,
        'duration_matches': duration_matches,
        'time_matches': time_matches,
        'exact_matches': exact_matches,
    }


def evaluate_sequence(sequence_idx, sequence_tokens, generated_tokens):
    """Evaluate a single sequence."""
    # Extract ground truth score from the test sequence
    ground_truth = extract_score_tokens(sequence_tokens)
    
    if len(ground_truth) == 0:
        return None
    
    # Generated tokens are already just the performance events
    # Compare directly
    metrics = compare_events(ground_truth, generated_tokens)
    metrics['sequence_idx'] = sequence_idx
    
    return metrics


def main():
    print(f"\n{'='*60}")
    print(f"Evaluating Model Accuracy")
    print(f"{'='*60}\n")
    
    # Load test sequences
    print("Loading test sequences...")
    test_sequences = load_test_sequences(TEST_DATA_PATH)
    print(f"Loaded {len(test_sequences)} test sequences\n")
    
    # Find which sequences have generated tokens
    generated_files = list(Path(OUTPUT_DIR).glob('test_seq_*_tokens.txt'))
    num_generated = len(generated_files)
    
    if num_generated == 0:
        print("ERROR: No generated token files found!")
        print("Please run test.py first to generate sequences.")
        return
    
    print(f"Found {num_generated} generated sequences\n")
    
    # Evaluate each sequence
    all_metrics = []
    
    for seq_idx in range(num_generated):
        generated_tokens = load_generated_tokens(seq_idx)
        
        if generated_tokens is None:
            continue
        
        if seq_idx >= len(test_sequences):
            print(f"Warning: Sequence {seq_idx} not found in test data")
            continue
        
        sequence_tokens = test_sequences[seq_idx]
        metrics = evaluate_sequence(seq_idx, sequence_tokens, generated_tokens)
        
        if metrics is not None:
            all_metrics.append(metrics)
            
            print(f"\nSequence {seq_idx}:")
            print(f"  Ground truth events: {metrics['num_ground_truth']}")
            print(f"  Generated events: {metrics['num_generated']}")
            print(f"  Compared events: {metrics['num_compared']}")
            print(f"  Note accuracy: {metrics['note_accuracy']*100:.1f}%")
            print(f"  Duration accuracy: {metrics['duration_accuracy']*100:.1f}%")
            print(f"  Time accuracy: {metrics['time_accuracy']*100:.1f}%")
            print(f"  Exact match: {metrics['exact_match_accuracy']*100:.1f}%")
            print(f"  Mean time error: {metrics['mean_time_error']:.2f} ticks")
            print(f"  Mean duration error: {metrics['mean_duration_error']:.2f} ticks")
    
    # Aggregate statistics
    if len(all_metrics) > 0:
        print(f"\n{'='*60}")
        print(f"AGGREGATE STATISTICS ({len(all_metrics)} sequences)")
        print(f"{'='*60}")
        
        avg_note_acc = np.mean([m['note_accuracy'] for m in all_metrics])
        avg_dur_acc = np.mean([m['duration_accuracy'] for m in all_metrics])
        avg_time_acc = np.mean([m['time_accuracy'] for m in all_metrics])
        avg_exact_acc = np.mean([m['exact_match_accuracy'] for m in all_metrics])
        avg_time_error = np.mean([m['mean_time_error'] for m in all_metrics])
        avg_dur_error = np.mean([m['mean_duration_error'] for m in all_metrics])
        
        print(f"\nAverage Note Accuracy: {avg_note_acc*100:.1f}%")
        print(f"Average Duration Accuracy: {avg_dur_acc*100:.1f}%")
        print(f"Average Time Accuracy: {avg_time_acc*100:.1f}%")
        print(f"Average Exact Match: {avg_exact_acc*100:.1f}%")
        print(f"\nAverage Time Error: {avg_time_error:.2f} ticks")
        print(f"Average Duration Error: {avg_dur_error:.2f} ticks")
        
        # Save detailed results
        results_path = os.path.join(OUTPUT_DIR, 'accuracy_evaluation.txt')
        with open(results_path, 'w') as f:
            f.write(f"Model Accuracy Evaluation\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Sequences evaluated: {len(all_metrics)}\n\n")
            f.write(f"AGGREGATE METRICS:\n")
            f.write(f"  Average Note Accuracy: {avg_note_acc*100:.2f}%\n")
            f.write(f"  Average Duration Accuracy: {avg_dur_acc*100:.2f}%\n")
            f.write(f"  Average Time Accuracy: {avg_time_acc*100:.2f}%\n")
            f.write(f"  Average Exact Match: {avg_exact_acc*100:.2f}%\n")
            f.write(f"  Average Time Error: {avg_time_error:.2f} ticks\n")
            f.write(f"  Average Duration Error: {avg_dur_error:.2f} ticks\n\n")
            
            f.write(f"\nPER-SEQUENCE DETAILS:\n")
            f.write(f"{'-'*60}\n")
            for m in all_metrics:
                f.write(f"\nSequence {m['sequence_idx']}:\n")
                f.write(f"  Events (GT/Gen/Compared): {m['num_ground_truth']}/{m['num_generated']}/{m['num_compared']}\n")
                f.write(f"  Note accuracy: {m['note_accuracy']*100:.2f}%\n")
                f.write(f"  Duration accuracy: {m['duration_accuracy']*100:.2f}%\n")
                f.write(f"  Time accuracy: {m['time_accuracy']*100:.2f}%\n")
                f.write(f"  Exact match: {m['exact_match_accuracy']*100:.2f}%\n")
                f.write(f"  Mean time error: {m['mean_time_error']:.2f} ticks\n")
                f.write(f"  Mean duration error: {m['mean_duration_error']:.2f} ticks\n")
        
        print(f"\nDetailed results saved to: {results_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
