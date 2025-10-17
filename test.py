"""
Test script for evaluating the model on the test dataset using generate4.
Processes the entire test set with the new tokenization scheme.
"""

import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import traceback

from anticipation.sample import generate4
from anticipation.convert import events_to_midi
from anticipation.config import *
from anticipation.vocab import *
from anticipation import ops

# Enable more detailed CUDA error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Path to your checkpoint model
CHECKPOINT_PATH = r'hf-ckpt-3500\checkpoint-3500'
TEST_DATA_PATH = r'data\test_output.txt'
OUTPUT_DIR = r'test_outputs'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ Load the model ------------------
print(f"Loading model from checkpoint: {CHECKPOINT_PATH}...")
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_PATH, 
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Use FP16 for faster inference
    device_map="auto"  # Let the model decide the best device mapping
)

# Set to eval mode for inference
model.eval()

if torch.cuda.is_available():
    print(f"Model loaded on GPU with FP16")
else:
    print(f"Model loaded on CPU")

print(f"Model dtype: {model.dtype}")
print(f"Model device: {next(model.parameters()).device}")


def extract_controls_from_sequence(sequence_tokens, prefix_controls=33):
    """
    Extract control tokens from a tokenized sequence.
    
    The tokenization format is:
    [ANTICIPATE, SEP, SEP, SEP, 
     perf_ctrl_0, rest_0, ..., perf_ctrl_32, rest_32,  # PREFIX (33 performance controls)
     score_0, perf_ctrl_33, score_1, perf_ctrl_34, ...]  # ALTERNATING (score, future performance)
    
    Where:
    - perf_ctrl = performance tokens already WITH CONTROL_OFFSET
    - score = score tokens without offset
    - rest = [TIME, DUR=0, REST]
    
    Args:
        sequence_tokens: List of tokens from the test dataset
        prefix_controls: Number of control tokens in the prefix (default 33)
    
    Returns:
        controls: List of performance tokens (already with CONTROL_OFFSET)
    """
    # First 4 tokens should be: ANTICIPATE/AUTOREGRESS, SEP, SEP, SEP
    if len(sequence_tokens) < 4:
        return []
    
    # Skip the control flag and 3 SEPARATORs
    tokens = sequence_tokens[4:]
    
    controls = []
    i = 0
    
    # Extract the prefix controls (first k performance triplets, each followed by rest triplet)
    # Performance tokens already have CONTROL_OFFSET added during alignment
    for _ in range(prefix_controls):
        if i + 6 <= len(tokens):
            # Extract performance control triplet (already has CONTROL_OFFSET)
            control_triplet = tokens[i:i+3]
            rest_triplet = tokens[i+3:i+6]
            
            # Validate this is actually a control token (should be >= CONTROL_OFFSET)
            if control_triplet[0] >= CONTROL_OFFSET:
                controls.extend(control_triplet)
            else:
                # Prefix ended early
                break
            
            i += 6
        else:
            break
    
    # After the prefix, extract future performance controls from alternating pattern
    # Pattern is: [future_perf_ctrl, score, future_perf_ctrl, score, ...]
    # Future performance tokens (odd positions) already have CONTROL_OFFSET - extract as-is!
    while i + 3 <= len(tokens):
        # tokens[i:i+3] is future performance triplet (extract if has CONTROL_OFFSET)
        triplet = tokens[i:i+3]
        
        # Check if this has CONTROL_OFFSET (future performance)
        if triplet[0] >= CONTROL_OFFSET:
            controls.extend(triplet)
        
        i += 3
    
    return controls


def process_test_sequence(sequence_idx, sequence_tokens, top_p=0.95, save_midi=True):
    """
    Process a single test sequence: extract controls, generate, and optionally save MIDI.
    
    Args:
        sequence_idx: Index of the sequence in the test set
        sequence_tokens: List of tokens for this sequence
        top_p: Nucleus sampling parameter
        save_midi: Whether to save the generated MIDI file
    
    Returns:
        dict with results: {'success': bool, 'error': str or None}
    """
    try:
        # Extract control tokens from the sequence
        controls = extract_controls_from_sequence(sequence_tokens)
        
        if len(controls) == 0:
            return {'success': False, 'error': 'No control tokens found'}
        
        if len(controls) % 3 != 0:
            return {'success': False, 'error': f'Invalid control tokens length: {len(controls)} (not divisible by 3)'}
        
        # Validate all control tokens are in valid range
        for i, tok in enumerate(controls):
            if tok < 0 or tok >= VOCAB_SIZE:
                return {'success': False, 'error': f'Invalid control token {tok} at index {i} (vocab_size={VOCAB_SIZE})'}
        
        print(f"\nSequence {sequence_idx}: Extracted {len(controls)//3} control events")
        
        # Generate using the new scheme
        events, tokens = generate4(model, controls=controls, top_p=top_p, prefix_controls=33)
        
        print(f"Sequence {sequence_idx}: Generated {len(events)//3} performance events")
        
        # Save MIDI if requested
        if save_midi:
            midi_path = os.path.join(OUTPUT_DIR, f'test_seq_{sequence_idx:04d}.mid')
            mid = events_to_midi(events)
            mid.save(midi_path)
            print(f"Sequence {sequence_idx}: Saved MIDI to {midi_path}")
        
        # Save generated tokens for evaluation
        tokens_path = os.path.join(OUTPUT_DIR, f'test_seq_{sequence_idx:04d}_tokens.txt')
        with open(tokens_path, 'w') as f:
            # Save the generated performance events (not the full sequence with controls)
            f.write(' '.join(map(str, events)))
        
        return {
            'success': True,
            'error': None,
            'num_control_events': len(controls) // 3,
            'num_generated_events': len(events) // 3,
            'generated_tokens': events  # Return events, not full tokens
        }
        
    except Exception as e:
        error_msg = f"Error processing sequence {sequence_idx}: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {'success': False, 'error': error_msg}


def load_test_sequences(test_file_path):
    """Load all test sequences from the test output file."""
    sequences = []
    with open(test_file_path, 'r') as f:
        for line in f:
            tokens = list(map(int, line.strip().split()))
            sequences.append(tokens)
    return sequences


def main():
    print(f"\n{'='*60}")
    print(f"Testing model on dataset: {TEST_DATA_PATH}")
    print(f"Model checkpoint: {CHECKPOINT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    # Load test sequences
    print("Loading test sequences...")
    test_sequences = load_test_sequences(TEST_DATA_PATH)
    print(f"Loaded {len(test_sequences)} test sequences\n")
    
    # LIMIT TO FIRST 15 SEQUENCES FOR TESTING
    num_sequences_to_test = 3  # Testing just 3 to verify the fix quickly
    test_sequences = test_sequences[:num_sequences_to_test]
    print(f"Running on first {num_sequences_to_test} sequences only\n")
    
    # Statistics
    results = []
    successful = 0
    failed = 0
    
    # Process each sequence
    for idx, sequence in enumerate(test_sequences):
        result = process_test_sequence(idx, sequence, top_p=0.95, save_midi=True)
        results.append(result)
        
        if result['success']:
            successful += 1
        else:
            failed += 1
        
        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"\nProgress: {idx + 1}/{len(test_sequences)} sequences processed")
            print(f"Success rate: {successful}/{idx + 1} ({100*successful/(idx+1):.1f}%)")
    
    # Final statistics
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total sequences: {len(test_sequences)}")
    print(f"Successful: {successful} ({100*successful/len(test_sequences):.1f}%)")
    print(f"Failed: {failed} ({100*failed/len(test_sequences):.1f}%)")
    
    # Save results to file
    results_file = os.path.join(OUTPUT_DIR, 'results_summary.txt')
    with open(results_file, 'w') as f:
        f.write(f"Test Results Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model: {CHECKPOINT_PATH}\n")
        f.write(f"Test Data: {TEST_DATA_PATH}\n")
        f.write(f"Total sequences: {len(test_sequences)}\n")
        f.write(f"Successful: {successful} ({100*successful/len(test_sequences):.1f}%)\n")
        f.write(f"Failed: {failed} ({100*failed/len(test_sequences):.1f}%)\n\n")
        
        f.write(f"\nDetailed Results:\n")
        f.write(f"{'-'*60}\n")
        for idx, result in enumerate(results):
            if result['success']:
                f.write(f"Seq {idx:04d}: SUCCESS - "
                       f"Controls: {result['num_control_events']}, "
                       f"Generated: {result['num_generated_events']}\n")
            else:
                f.write(f"Seq {idx:04d}: FAILED - {result['error']}\n")
    
    print(f"\nResults saved to: {results_file}")
    print(f"MIDI files saved to: {OUTPUT_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()

