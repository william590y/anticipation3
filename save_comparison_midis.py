"""
Extract test sequences and save as MIDI files.
Creates pairs of:
1. Ground truth score (from test data)
2. Generated performance (from model)
"""

import torch
from transformers import AutoModelForCausalLM
from anticipation.sample import generate4
from anticipation.vocab import *
from anticipation.config import *
from anticipation import ops, convert
import numpy as np

def extract_from_sequence(sequence_tokens, prefix_controls=33):
    """Extract controls and scores from test sequence."""
    if len(sequence_tokens) < 4:
        return [], []
    
    tokens = sequence_tokens[4:]  # Skip ANTICIPATE + 3 SEPs
    
    all_controls = []
    all_scores = []
    
    i = 0
    k = prefix_controls
    
    # Extract prefix controls
    for _ in range(k):
        if i + 6 <= len(tokens):
            control_triplet = tokens[i:i+3]
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


def tokens_to_midi(tokens, output_path, add_control_offset=False):
    """Convert tokens to MIDI file."""
    # Remove offsets
    events = []
    for i in range(0, len(tokens), 3):
        if i + 3 <= len(tokens):
            time_tok = tokens[i]
            dur_tok = tokens[i+1]
            note_tok = tokens[i+2]
            
            # Remove offsets
            if time_tok >= CONTROL_OFFSET:
                time = time_tok - CONTROL_OFFSET
            else:
                time = time_tok - TIME_OFFSET if time_tok >= TIME_OFFSET else time_tok
            
            if dur_tok >= ADUR_OFFSET:
                dur = dur_tok - ADUR_OFFSET
            else:
                dur = dur_tok - DUR_OFFSET if dur_tok >= DUR_OFFSET else dur_tok
            
            if note_tok >= ANOTE_OFFSET:
                note = note_tok - ANOTE_OFFSET
            elif note_tok >= NOTE_OFFSET:
                note = note_tok - NOTE_OFFSET
            else:
                note = note_tok
            
            events.extend([time, dur, note])
    
    # Convert to MIDI
    convert.to_midi(events, output_path)
    print(f"  Saved: {output_path}")


def main():
    print("="*80)
    print("EXTRACT TEST EXAMPLES TO MIDI")
    print("="*80)
    
    # Load model
    print("\nLoading model from: hf-ckpt-3500/checkpoint-3500")
    model = AutoModelForCausalLM.from_pretrained(
        'hf-ckpt-3500/checkpoint-3500',
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    print(f"✓ Model loaded")
    
    # Load test sequences
    print("\nLoading test sequences from: data/test_output.txt")
    with open('data/test_output.txt', 'r') as f:
        test_lines = f.readlines()
    
    print(f"✓ Loaded {len(test_lines)} test sequences")
    
    # Process first N sequences
    num_examples = 5
    print(f"\nGenerating {num_examples} examples...")
    
    for idx in range(num_examples):
        print(f"\n{'='*60}")
        print(f"Example {idx + 1}/{num_examples}")
        print(f"{'='*60}")
        
        # Parse sequence
        sequence_tokens = [int(tok) for tok in test_lines[idx].strip().split()]
        controls, ground_truth = extract_from_sequence(sequence_tokens, prefix_controls=33)
        
        print(f"Controls: {len(controls)//3} notes")
        print(f"Ground truth scores: {len(ground_truth)//3} notes")
        
        # Generate performance
        print("Generating performance...")
        generated, _ = generate4(model, controls=controls, top_p=0.95, prefix_controls=33)
        print(f"Generated: {len(generated)//3} notes")
        
        # Save ground truth score as MIDI
        print("\nSaving MIDI files...")
        score_path = f"test_outputs/example_{idx+1}_score.mid"
        tokens_to_midi(ground_truth, score_path)
        
        # Save generated performance as MIDI
        gen_path = f"test_outputs/example_{idx+1}_generated.mid"
        tokens_to_midi(generated, gen_path)
        
        # Also save the control (performance input)
        control_path = f"test_outputs/example_{idx+1}_control.mid"
        tokens_to_midi(controls, control_path)
        
        print(f"✓ Example {idx+1} complete")
    
    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")
    print(f"\nGenerated {num_examples} example pairs in test_outputs/")
    print("\nFiles:")
    print("  *_score.mid      - Ground truth score (what should be generated)")
    print("  *_control.mid    - Performance input (given to model)")
    print("  *_generated.mid  - Model output (what model generated)")


if __name__ == "__main__":
    import os
    os.makedirs('test_outputs', exist_ok=True)
    main()
