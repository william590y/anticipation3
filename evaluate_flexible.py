"""
Improved accuracy evaluation with timing-flexible matching.
Uses a window-based approach to account for expressive timing variations.
"""

import os
import numpy as np
from pathlib import Path

from anticipation.config import *
from anticipation.vocab import *

# Paths
TEST_DATA_PATH = r'data\test_output.txt'
OUTPUT_DIR = r'test_outputs'

# Matching parameters
TIME_WINDOW = 20  # Allow ±20 ticks for matching (flexible timing)


def load_test_sequences(test_file_path):
    """Load all test sequences from the test output file."""
    sequences = []
    with open(test_file_path, 'r') as f:
        for line in f:
            tokens = list(map(int, line.strip().split()))
            sequences.append(tokens)
    return sequences


def extract_score_tokens(sequence_tokens):
    """Extract score tokens from a tokenized sequence."""
    if len(sequence_tokens) < 4:
        return []
    
    tokens = sequence_tokens[4:]
    prefix_length = 33 * 6
    if len(tokens) < prefix_length:
        return []
    
    alternating_tokens = tokens[prefix_length:]
    
    score_events = []
    i = 0
    while i + 3 <= len(alternating_tokens):
        triplet = alternating_tokens[i:i+3]
        if triplet[0] < CONTROL_OFFSET:
            score_events.extend(triplet)
        i += 3
    
    return score_events


def load_generated_tokens(sequence_idx):
    """Load generated tokens from saved token files."""
    tokens_path = os.path.join(OUTPUT_DIR, f'test_seq_{sequence_idx:04d}_tokens.txt')
    if not os.path.exists(tokens_path):
        return None
    
    with open(tokens_path, 'r') as f:
        tokens = list(map(int, f.read().strip().split()))
    
    return tokens


def decode_events(token_list):
    """Decode a list of tokens into event dictionaries."""
    if len(token_list) % 3 != 0:
        return []
    
    events = []
    for i in range(0, len(token_list), 3):
        time_tok = token_list[i]
        dur_tok = token_list[i+1]
        note_tok = token_list[i+2]
        
        if note_tok >= CONTROL_OFFSET:
            time = time_tok - ATIME_OFFSET
            duration = dur_tok - ADUR_OFFSET
            note_val = note_tok - ANOTE_OFFSET
        else:
            time = time_tok - TIME_OFFSET
            duration = dur_tok - DUR_OFFSET
            note_val = note_tok - NOTE_OFFSET
        
        pitch = note_val % 128
        instrument = note_val // 128
        
        events.append({
            'time': time,
            'duration': duration,
            'pitch': pitch,
            'instrument': instrument,
        })
    
    return events


def flexible_match_events(gt_events, gen_events, time_window=TIME_WINDOW):
    """
    Match events with flexible timing.
    For each GT event, find the closest generated event within a time window.
    """
    matches = {
        'note_matches': 0,
        'duration_matches': 0,
        'time_matches': 0,
        'exact_matches': 0,
        'total_gt': len(gt_events),
        'total_gen': len(gen_events),
        'matched': 0,
        'time_errors': [],
        'duration_errors': [],
    }
    
    used_gen_indices = set()
    
    for gt_event in gt_events:
        best_match = None
        best_time_diff = float('inf')
        best_idx = -1
        
        # Find closest generated event in time window
        for i, gen_event in enumerate(gen_events):
            if i in used_gen_indices:
                continue
            
            time_diff = abs(gt_event['time'] - gen_event['time'])
            
            if time_diff <= time_window and time_diff < best_time_diff:
                best_match = gen_event
                best_time_diff = time_diff
                best_idx = i
        
        if best_match is not None:
            matches['matched'] += 1
            used_gen_indices.add(best_idx)
            
            # Check note match (pitch + instrument)
            if (gt_event['pitch'] == best_match['pitch'] and 
                gt_event['instrument'] == best_match['instrument']):
                matches['note_matches'] += 1
            
            # Check duration match
            if gt_event['duration'] == best_match['duration']:
                matches['duration_matches'] += 1
            
            # Check time match (exact)
            if gt_event['time'] == best_match['time']:
                matches['time_matches'] += 1
            
            # Exact match
            if (gt_event['pitch'] == best_match['pitch'] and
                gt_event['instrument'] == best_match['instrument'] and
                gt_event['duration'] == best_match['duration'] and
                gt_event['time'] == best_match['time']):
                matches['exact_matches'] += 1
            
            # Track errors
            matches['time_errors'].append(abs(gt_event['time'] - best_match['time']))
            matches['duration_errors'].append(abs(gt_event['duration'] - best_match['duration']))
    
    return matches


def evaluate_sequence(sequence_idx, sequence_tokens, generated_tokens):
    """Evaluate a single sequence with flexible timing."""
    ground_truth = extract_score_tokens(sequence_tokens)
    
    if len(ground_truth) == 0:
        return None
    
    gt_events = decode_events(ground_truth)
    gen_events = decode_events(generated_tokens)
    
    # Flexible matching
    matches = flexible_match_events(gt_events, gen_events)
    
    # Calculate metrics
    matched = matches['matched']
    total = matches['total_gt']
    
    return {
        'sequence_idx': sequence_idx,
        'num_ground_truth': total,
        'num_generated': matches['total_gen'],
        'num_matched': matched,
        'note_accuracy': matches['note_matches'] / matched if matched > 0 else 0.0,
        'duration_accuracy': matches['duration_matches'] / matched if matched > 0 else 0.0,
        'time_accuracy': matches['time_matches'] / matched if matched > 0 else 0.0,
        'exact_match_accuracy': matches['exact_matches'] / matched if matched > 0 else 0.0,
        'match_rate': matched / total if total > 0 else 0.0,
        'mean_time_error': np.mean(matches['time_errors']) if matches['time_errors'] else 0.0,
        'mean_duration_error': np.mean(matches['duration_errors']) if matches['duration_errors'] else 0.0,
    }


def main():
    print(f"\n{'='*60}")
    print(f"Evaluating Model Accuracy (Flexible Timing)")
    print(f"Time window: ±{TIME_WINDOW} ticks")
    print(f"{'='*60}\n")
    
    # Load test sequences
    print("Loading test sequences...")
    test_sequences = load_test_sequences(TEST_DATA_PATH)
    print(f"Loaded {len(test_sequences)} test sequences\n")
    
    # Find generated sequences
    generated_files = list(Path(OUTPUT_DIR).glob('test_seq_*_tokens.txt'))
    num_generated = len(generated_files)
    
    if num_generated == 0:
        print("ERROR: No generated token files found!")
        return
    
    print(f"Found {num_generated} generated sequences\n")
    
    # Evaluate each sequence
    all_metrics = []
    
    for seq_idx in range(num_generated):
        generated_tokens = load_generated_tokens(seq_idx)
        
        if generated_tokens is None:
            continue
        
        if seq_idx >= len(test_sequences):
            continue
        
        sequence_tokens = test_sequences[seq_idx]
        metrics = evaluate_sequence(seq_idx, sequence_tokens, generated_tokens)
        
        if metrics is not None:
            all_metrics.append(metrics)
            
            print(f"\nSequence {seq_idx}:")
            print(f"  Events (GT/Gen/Matched): {metrics['num_ground_truth']}/{metrics['num_generated']}/{metrics['num_matched']}")
            print(f"  Match rate: {metrics['match_rate']*100:.1f}%")
            print(f"  Note accuracy (of matched): {metrics['note_accuracy']*100:.1f}%")
            print(f"  Duration accuracy: {metrics['duration_accuracy']*100:.1f}%")
            print(f"  Time accuracy (exact): {metrics['time_accuracy']*100:.1f}%")
            print(f"  Exact match: {metrics['exact_match_accuracy']*100:.1f}%")
            print(f"  Mean time error: {metrics['mean_time_error']:.2f} ticks")
            print(f"  Mean duration error: {metrics['mean_duration_error']:.2f} ticks")
    
    # Aggregate statistics
    if len(all_metrics) > 0:
        print(f"\n{'='*60}")
        print(f"AGGREGATE STATISTICS ({len(all_metrics)} sequences)")
        print(f"{'='*60}")
        
        avg_match_rate = np.mean([m['match_rate'] for m in all_metrics])
        avg_note_acc = np.mean([m['note_accuracy'] for m in all_metrics])
        avg_dur_acc = np.mean([m['duration_accuracy'] for m in all_metrics])
        avg_time_acc = np.mean([m['time_accuracy'] for m in all_metrics])
        avg_exact_acc = np.mean([m['exact_match_accuracy'] for m in all_metrics])
        avg_time_error = np.mean([m['mean_time_error'] for m in all_metrics])
        avg_dur_error = np.mean([m['mean_duration_error'] for m in all_metrics])
        
        print(f"\nAverage Match Rate: {avg_match_rate*100:.1f}% (events found within ±{TIME_WINDOW} ticks)")
        print(f"Average Note Accuracy (of matched): {avg_note_acc*100:.1f}%")
        print(f"Average Duration Accuracy: {avg_dur_acc*100:.1f}%")
        print(f"Average Time Accuracy (exact): {avg_time_acc*100:.1f}%")
        print(f"Average Exact Match: {avg_exact_acc*100:.1f}%")
        print(f"\nAverage Time Error: {avg_time_error:.2f} ticks")
        print(f"Average Duration Error: {avg_dur_error:.2f} ticks")
        
        print(f"\n{'='*60}")
        print("INTERPRETATION:")
        print(f"{'='*60}")
        print(f"- {avg_match_rate*100:.1f}% of score notes have a corresponding generated note nearby")
        print(f"- Of those matched notes, {avg_note_acc*100:.1f}% have the correct pitch")
        print(f"- Average timing deviation: {avg_time_error:.2f} ticks (~{avg_time_error/10:.1f} ms)")
        
        # Save results
        results_path = os.path.join(OUTPUT_DIR, 'accuracy_flexible.txt')
        with open(results_path, 'w') as f:
            f.write(f"Model Accuracy Evaluation (Flexible Timing)\n")
            f.write(f"Time window: ±{TIME_WINDOW} ticks\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Sequences evaluated: {len(all_metrics)}\n\n")
            f.write(f"AGGREGATE METRICS:\n")
            f.write(f"  Average Match Rate: {avg_match_rate*100:.2f}%\n")
            f.write(f"  Average Note Accuracy (of matched): {avg_note_acc*100:.2f}%\n")
            f.write(f"  Average Duration Accuracy: {avg_dur_acc*100:.2f}%\n")
            f.write(f"  Average Time Accuracy (exact): {avg_time_acc*100:.2f}%\n")
            f.write(f"  Average Exact Match: {avg_exact_acc*100:.2f}%\n")
            f.write(f"  Average Time Error: {avg_time_error:.2f} ticks\n")
            f.write(f"  Average Duration Error: {avg_dur_error:.2f} ticks\n\n")
            
            for m in all_metrics:
                f.write(f"\nSequence {m['sequence_idx']}:\n")
                f.write(f"  Events (GT/Gen/Matched): {m['num_ground_truth']}/{m['num_generated']}/{m['num_matched']}\n")
                f.write(f"  Match rate: {m['match_rate']*100:.2f}%\n")
                f.write(f"  Note accuracy: {m['note_accuracy']*100:.2f}%\n")
                f.write(f"  Duration accuracy: {m['duration_accuracy']*100:.2f}%\n")
                f.write(f"  Time accuracy: {m['time_accuracy']*100:.2f}%\n")
        
        print(f"\nDetailed results saved to: {results_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
