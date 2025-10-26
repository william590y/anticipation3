"""Generate MIDI output from model using bach_performance.mid using anticipation.sample"""
import torch
from transformers import AutoModelForCausalLM
import numpy as np
import pretty_midi
import argparse
import os

from anticipation.vocab import NOTE_OFFSET, TIME_OFFSET, DUR_OFFSET, CONTROL_OFFSET
from anticipation.config import TIME_RESOLUTION
from anticipation.sample import generate4
from anticipation.convert import events_to_midi

def midi_to_control_tokens(midi_file):
    """Convert MIDI to control token triplets with CONTROL_OFFSET applied"""
    midi = pretty_midi.PrettyMIDI(midi_file)
    
    # Get all notes
    notes = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append((note.start, note.pitch, note.duration))
    
    # Sort by time
    notes.sort()
    
    # Convert to control tokens (with CONTROL_OFFSET already applied)
    control_tokens = []
    for start_time, pitch, duration in notes:
        # Convert to time units
        time_units = int(start_time * TIME_RESOLUTION)
        dur_units = int(duration * TIME_RESOLUTION)
        
        # Control triplet: [time + CONTROL_OFFSET, duration + DUR_OFFSET, pitch + NOTE_OFFSET]
        control_tokens.extend([
            CONTROL_OFFSET + time_units,
            DUR_OFFSET + dur_units,
            NOTE_OFFSET + pitch
        ])
    
    return control_tokens, notes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./new_model')
    parser.add_argument('--input', type=str, default='./bach_example/bach_performance.mid')
    parser.add_argument('--score', type=str, default='./bach_example/bach_score.mid', help='Ground truth score MIDI for accuracy comparison')
    parser.add_argument('--output', type=str, default='./bach_example/generated_output.mid')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--top-p', type=float, default=1.0, help='Nucleus sampling parameter')
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ).to(args.device)
    
    print(f"\nTokenizing {args.input}...")
    control_tokens, original_notes = midi_to_control_tokens(args.input)
    print(f"Extracted {len(original_notes)} notes -> {len(control_tokens)} control tokens")
    
    # Load ground truth score if provided
    ground_truth_notes = None
    if args.score and os.path.exists(args.score):
        print(f"\nLoading ground truth score from {args.score}...")
        score_midi = pretty_midi.PrettyMIDI(args.score)
        ground_truth_notes = []
        for instrument in score_midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    ground_truth_notes.append((note.start, note.pitch, note.duration))
        ground_truth_notes.sort()
        print(f"Loaded {len(ground_truth_notes)} ground truth score notes")
    
    print(f"\nGenerating score using generate4 (top_p={args.top_p})...")
    generated_events, full_tokens = generate4(model, control_tokens, top_p=args.top_p, prefix_controls=33)
    
    print(f"Generated {len(generated_events)//3} score events")
    
    # Calculate pitch accuracy
    correct = 0
    total = 0
    errors = []
    
    # Compare generated score to ground truth score (if available)
    comparison_notes = ground_truth_notes if ground_truth_notes else original_notes
    comparison_label = "ground truth score" if ground_truth_notes else "original performance"
    
    for i in range(min(len(generated_events)//3, len(comparison_notes))):
        gen_pitch_token = generated_events[i*3 + 2]
        if NOTE_OFFSET <= gen_pitch_token < NOTE_OFFSET + 128:
            total += 1
            gen_pitch = gen_pitch_token - NOTE_OFFSET
            true_pitch = comparison_notes[i][1]  # MIDI pitch
            
            if gen_pitch == true_pitch:
                correct += 1
            else:
                errors.append(abs(gen_pitch - true_pitch))
    
    # Print results
    print("\n" + "=" * 70)
    print("PITCH ACCURACY RESULTS")
    print("=" * 70)
    print(f"Input (performance): {args.input}")
    print(f"Comparing to: {comparison_label}")
    print(f"Total notes predicted: {total}")
    print(f"Correct pitches: {correct}")
    
    if total > 0:
        accuracy = 100 * correct / total
        print(f"\nPitch Accuracy: {accuracy:.2f}%")
        
        if errors:
            print(f"\nError Statistics (when wrong):")
            print(f"  Mean error: {np.mean(errors):.2f} semitones")
            print(f"  Median error: {np.median(errors):.2f} semitones")
            print(f"  Max error: {max(errors)} semitones")
    
    print("=" * 70)
    
    # Generate MIDI output
    print(f"\nGenerating MIDI output...")
    midi_data = events_to_midi(generated_events)
    midi_data.save(args.output)
    print(f"Saved MIDI to {args.output}")
    print(f"\n✓ Complete! Output saved to: {args.output}")
