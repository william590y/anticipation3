"""
Debug script to examine what we're actually comparing.
"""

import os
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
    """Extract score tokens from a tokenized sequence."""
    if len(sequence_tokens) < 4:
        return []
    
    tokens = sequence_tokens[4:]
    
    # Skip the prefix (33 performance controls + 33 rests = 198 tokens)
    prefix_length = 33 * 6
    if len(tokens) < prefix_length:
        return []
    
    alternating_tokens = tokens[prefix_length:]
    
    # Extract score events from alternating pattern
    score_events = []
    i = 0
    while i + 3 <= len(alternating_tokens):
        triplet = alternating_tokens[i:i+3]
        
        # Score tokens have time < CONTROL_OFFSET
        if triplet[0] < CONTROL_OFFSET:
            score_events.extend(triplet)
        
        i += 3
    
    return score_events


def decode_event(time_tok, dur_tok, note_tok):
    """Decode a single event triplet."""
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
    
    pitch = note_val % 128
    instrument = note_val // 128
    
    return {
        'time': time,
        'duration': duration,
        'pitch': pitch,
        'instrument': instrument,
        'is_control': is_control,
        'raw_tokens': (time_tok, dur_tok, note_tok)
    }


def main():
    print(f"\n{'='*80}")
    print(f"DEBUG: Comparing Ground Truth vs Generated")
    print(f"{'='*80}\n")
    
    # Load test sequence 0
    test_sequences = load_test_sequences(TEST_DATA_PATH)
    sequence = test_sequences[0]
    
    # Extract ground truth
    gt_tokens = extract_score_tokens(sequence)
    print(f"Ground truth extracted: {len(gt_tokens)} tokens ({len(gt_tokens)//3} events)")
    
    # Load generated tokens
    gen_path = os.path.join(OUTPUT_DIR, 'test_seq_0000_tokens.txt')
    with open(gen_path, 'r') as f:
        gen_tokens = list(map(int, f.read().strip().split()))
    
    print(f"Generated tokens: {len(gen_tokens)} tokens ({len(gen_tokens)//3} events)\n")
    
    # Compare first 10 events
    print(f"{'='*80}")
    print(f"FIRST 10 EVENTS COMPARISON:")
    print(f"{'='*80}\n")
    
    for i in range(min(10, len(gt_tokens)//3, len(gen_tokens)//3)):
        gt_idx = i * 3
        gen_idx = i * 3
        
        gt_event = decode_event(gt_tokens[gt_idx], gt_tokens[gt_idx+1], gt_tokens[gt_idx+2])
        gen_event = decode_event(gen_tokens[gen_idx], gen_tokens[gen_idx+1], gen_tokens[gen_idx+2])
        
        match = "✓" if gt_event['pitch'] == gen_event['pitch'] and gt_event['instrument'] == gen_event['instrument'] else "✗"
        
        print(f"Event {i}: {match}")
        print(f"  Ground Truth: time={gt_event['time']:5d}, dur={gt_event['duration']:5d}, "
              f"pitch={gt_event['pitch']:3d}, inst={gt_event['instrument']}, "
              f"is_control={gt_event['is_control']}")
        print(f"  Generated:    time={gen_event['time']:5d}, dur={gen_event['duration']:5d}, "
              f"pitch={gen_event['pitch']:3d}, inst={gen_event['instrument']}, "
              f"is_control={gen_event['is_control']}")
        print(f"  GT Raw:  {gt_event['raw_tokens']}")
        print(f"  Gen Raw: {gen_event['raw_tokens']}")
        print()
    
    # Check if ground truth is actually performance (control) tokens
    print(f"\n{'='*80}")
    print(f"TOKEN TYPE ANALYSIS:")
    print(f"{'='*80}\n")
    
    gt_control_count = sum(1 for i in range(0, len(gt_tokens), 3) if gt_tokens[i] >= CONTROL_OFFSET)
    gen_control_count = sum(1 for i in range(0, len(gen_tokens), 3) if gen_tokens[i] >= CONTROL_OFFSET)
    
    print(f"Ground truth: {gt_control_count}/{len(gt_tokens)//3} are control tokens")
    print(f"Generated:    {gen_control_count}/{len(gen_tokens)//3} are control tokens")
    
    # Show raw format of first part of sequence
    print(f"\n{'='*80}")
    print(f"RAW SEQUENCE FORMAT (first 30 tokens of test sequence):")
    print(f"{'='*80}\n")
    print(sequence[:30])
    print(f"\nAfter skipping ANTICIPATE + 3 SEPs:")
    print(sequence[4:34])
    
    print(f"\n{'='*80}")
    print(f"EXTRACTED GROUND TRUTH (first 30 tokens):")
    print(f"{'='*80}\n")
    print(gt_tokens[:30])
    
    print(f"\n{'='*80}")
    print(f"GENERATED OUTPUT (first 30 tokens):")
    print(f"{'='*80}\n")
    print(gen_tokens[:30])


if __name__ == "__main__":
    main()
