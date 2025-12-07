"""
Test greedy autoregressive pitch accuracy on model-experimental.

Simple script to evaluate model-experimental's greedy decoding performance
on validation sequences and save MIDI outputs.
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET, REST
from anticipation.convert import events_to_midi
from tqdm import tqdm
import random
import os

def extract_score_only(tokens):
    """Extract only score tokens (not performance/control tokens)."""
    from anticipation.vocab import ANTICIPATE, SEPARATOR
    
    # Skip ANTICIPATE token if present
    start_idx = 1 if (len(tokens) > 0 and tokens[0] == ANTICIPATE) else 0
    
    # Skip separator tokens (3 SEP tokens after ANTICIPATE)
    if start_idx == 1 and len(tokens) > 4:
        if tokens[1] == SEPARATOR:
            start_idx += 3
    
    events = []
    i = start_idx
    while i < len(tokens) - 2:
        time_tok, dur_tok, note_tok = tokens[i], tokens[i+1], tokens[i+2]
        
        # Score triplet only (not control) - include REST tokens for MIDI conversion
        if time_tok < CONTROL_OFFSET and dur_tok < CONTROL_OFFSET and note_tok < CONTROL_OFFSET:
            events.extend([time_tok, dur_tok, note_tok])
            i += 3
        else:
            i += 1
    
    return events

def extract_performance_only(tokens):
    """Extract only performance (control) tokens."""
    from anticipation.vocab import ANTICIPATE, SEPARATOR
    
    # Skip ANTICIPATE token if present
    start_idx = 1 if (len(tokens) > 0 and tokens[0] == ANTICIPATE) else 0
    
    # Skip separator tokens (3 SEP tokens after ANTICIPATE)
    if start_idx == 1 and len(tokens) > 4:
        if tokens[1] == SEPARATOR:
            start_idx += 3
    
    events = []
    i = start_idx
    while i < len(tokens) - 2:
        time_tok, dur_tok, note_tok = tokens[i], tokens[i+1], tokens[i+2]
        
        # Control triplet (performance) - remove CONTROL_OFFSET
        if time_tok >= CONTROL_OFFSET and dur_tok >= CONTROL_OFFSET and note_tok >= CONTROL_OFFSET:
            events.extend([time_tok - CONTROL_OFFSET, dur_tok - CONTROL_OFFSET, note_tok - CONTROL_OFFSET])
            i += 3
        else:
            i += 1
    
    return events

def greedy_pitch_accuracy(model, tokens, device):
    """
    Run greedy autoregressive generation and measure pitch accuracy.
    Uses ground truth control tokens, only generates score triplets.
    
    Returns:
        (correct_predictions, total_pitches, gt_aligned_count, generated_sequence)
        - correct_predictions: predicted pitches matching ground truth
        - total_pitches: total score pitches evaluated
        - gt_aligned_count: ground truth scores matching their aligned controls (data quality)
        - generated_sequence: full generated token sequence
    """
    # Find all score triplet positions and their aligned control triplets
    # Exclude REST tokens from both - only collect actual notes
    score_positions = []
    control_positions = []
    i = 1  # Skip mode token
    while i < len(tokens) - 2:
        time_tok, dur_tok, note_tok = tokens[i], tokens[i+1], tokens[i+2]
        
        if (time_tok < CONTROL_OFFSET and 
            dur_tok < CONTROL_OFFSET and 
            note_tok < CONTROL_OFFSET and
            note_tok != REST):  # Exclude REST tokens
            score_positions.append((i, i+1, i+2))
            i += 3
        elif (time_tok >= CONTROL_OFFSET and 
              dur_tok >= CONTROL_OFFSET and 
              note_tok >= CONTROL_OFFSET):
            # Only add if it's not a REST (shouldn't happen for controls, but be safe)
            if note_tok - CONTROL_OFFSET != REST:
                control_positions.append((i, i+1, i+2))
            i += 3
        else:
            i += 1
    
    if len(score_positions) == 0:
        return 0, 0, 0, 0, tokens
    
    # Validate ground truth alignment (score[i] should match control[i])
    # Both lists exclude REST tokens, so we compare by index
    from anticipation.vocab import NOTE_OFFSET
    gt_aligned = 0
    gt_total = min(len(score_positions), len(control_positions))
    
    for score_idx in range(gt_total):
        score_pitch_tok = tokens[score_positions[score_idx][2]]
        control_pitch_tok = tokens[control_positions[score_idx][2]]
        
        # Remove offsets to get actual pitch values
        score_pitch = score_pitch_tok - NOTE_OFFSET if score_pitch_tok >= NOTE_OFFSET else score_pitch_tok
        control_pitch = control_pitch_tok - CONTROL_OFFSET - NOTE_OFFSET if control_pitch_tok >= CONTROL_OFFSET + NOTE_OFFSET else control_pitch_tok - CONTROL_OFFSET
        
        if score_pitch == control_pitch:
            gt_aligned += 1
    
    # Find the first score triplet position
    first_score_pos = score_positions[0][0]
    
    # Start with context up to first score triplet
    context = tokens[:first_score_pos]
    
    correct_predictions = 0
    total = 0
    
    last_pos = first_score_pos
    
    model.eval()
    with torch.no_grad():
        # Initialize with context up to first score triplet (with KV caching)
        init_context = torch.tensor([context]).to(device)
        outputs = model(init_context, past_key_values=None, use_cache=True)
        past_key_values = outputs.past_key_values
        
        # Generate each score triplet
        for score_idx, (time_pos, dur_pos, pitch_pos) in enumerate(score_positions):
            # Add ground truth intermediate control tokens
            if time_pos > last_pos:
                intermediate = torch.tensor([tokens[last_pos:time_pos]]).to(device)
                outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                context = context + tokens[last_pos:time_pos]
            
            # Predict TIME
            logits = outputs.logits[0, -1]
            pred_time = logits.argmax().item()
            context = context + [pred_time]
            
            # Feed predicted TIME back
            next_token = torch.tensor([[pred_time]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Predict DURATION
            logits = outputs.logits[0, -1]
            pred_dur = logits.argmax().item()
            context = context + [pred_dur]
            
            # Feed predicted DURATION back
            next_token = torch.tensor([[pred_dur]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Predict PITCH
            logits = outputs.logits[0, -1]
            pred_pitch = logits.argmax().item()
            context = context + [pred_pitch]
            
            # Feed predicted PITCH back for next iteration
            next_token = torch.tensor([[pred_pitch]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Check pitch accuracy vs ground truth
            true_pitch = tokens[pitch_pos]
            if pred_pitch == true_pitch:
                correct_predictions += 1
            
            total += 1
            
            last_pos = pitch_pos + 1
    
    return correct_predictions, total, gt_aligned, gt_total, context

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'model-experimental'
    
    print(f"Loading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # Load test sequences
    test_file = 'data/test_normalized.txt'
    print(f"Loading test sequences from {test_file}...")
    
    with open(test_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(lines)} test sequences")
    
    # Sample random sequences
    num_sequences = 5
    random.seed(42)
    sampled_lines = random.sample(lines, min(num_sequences, len(lines)))
    
    # Create output directory for MIDI files
    output_dir = 'experimental_greedy_outputs'
    os.makedirs(output_dir, exist_ok=True)
    print(f"MIDI outputs will be saved to: {output_dir}/")
    
    total_correct = 0
    total_pitches = 0
    total_gt_aligned = 0
    total_gt_scores = 0
    
    # Save first 10 sequences as MIDI
    num_midi_saves = 10
    
    print(f"\nEvaluating on {len(sampled_lines)} sequences...")
    for seq_idx, line in enumerate(tqdm(sampled_lines, desc="Testing")):
        if '|' in line:
            token_part = line.split('|')[0].strip()
        else:
            token_part = line
        
        tokens = [int(t) for t in token_part.split()]
        
        correct, total, gt_aligned, gt_total, generated_seq = greedy_pitch_accuracy(model, tokens, device)
        total_correct += correct
        total_pitches += total
        total_gt_aligned += gt_aligned
        total_gt_scores += gt_total
        
        # Save MIDI files for first few sequences
        if seq_idx < num_midi_saves:
            # Extract and save performance (input control)
            perf_events = extract_performance_only(tokens)
            if perf_events:
                perf_midi_path = os.path.join(output_dir, f'seq{seq_idx:03d}_input_performance.mid')
                perf_midi = events_to_midi(perf_events)
                perf_midi.save(perf_midi_path)
            
            # Extract and save ground truth score
            gt_events = extract_score_only(tokens)
            if gt_events:
                gt_midi_path = os.path.join(output_dir, f'seq{seq_idx:03d}_ground_truth.mid')
                gt_midi = events_to_midi(gt_events)
                gt_midi.save(gt_midi_path)
            
            # Extract and save greedy generated score
            greedy_events = extract_score_only(generated_seq)
            if greedy_events:
                greedy_midi_path = os.path.join(output_dir, f'seq{seq_idx:03d}_greedy.mid')
                greedy_midi = events_to_midi(greedy_events)
                greedy_midi.save(greedy_midi_path)
    
    model_accuracy = (total_correct / total_pitches * 100) if total_pitches > 0 else 0
    gt_alignment_acc = (total_gt_aligned / total_gt_scores * 100) if total_gt_scores > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Test file: {test_file}")
    print(f"Sequences evaluated: {len(sampled_lines)}")
    print(f"\n--- MODEL PERFORMANCE ---")
    print(f"Model Pitch Accuracy: {model_accuracy:.2f}%")
    print(f"  Correct predictions: {total_correct}/{total_pitches}")
    print(f"  (How well the model reconstructs ground truth scores)")
    print(f"\n--- DATA QUALITY CHECK ---")
    print(f"Ground Truth Alignment: {gt_alignment_acc:.2f}%")
    print(f"  GT scores matching controls: {total_gt_aligned}/{total_gt_scores}")
    print(f"  (Verifies score[i] == control[i] in test data)")
    if gt_alignment_acc > 99:
        print(f"  ✓ Data alignment preserved correctly")
    else:
        print(f"  ⚠ Warning: Data may have alignment issues")
    print(f"\nMIDI outputs saved to: {output_dir}/")
    print(f"  - First {num_midi_saves} sequences saved")
    print(f"  - Files: input_performance.mid, ground_truth.mid, greedy.mid")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
