"""
Extract test examples using newest_model with greedy decoding (top_k=1).
Creates MIDI files for: ground_truth_score, performance, and model_predictions.
"""
import os
import torch
from transformers import AutoModelForCausalLM
from anticipation.vocab import *
from anticipation.config import *
from anticipation.convert import events_to_midi

def greedy_decode_sequence(model, input_ids, max_new_tokens=1024):
    """Greedy decoding with KV caching."""
    device = model.device
    input_ids = input_ids.to(device)
    
    generated = input_ids.clone()
    past_key_values = None
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if past_key_values is None:
                model_inputs = generated
            else:
                model_inputs = generated[:, -1:]
            
            outputs = model(model_inputs, past_key_values=past_key_values, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # Greedy: argmax
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if generated.shape[1] >= CONTEXT_SIZE:
                break
    
    return generated

def extract_parts_from_sequence(tokens):
    """Extract performance (control), ground truth score, and positions."""
    # Skip ANTICIPATE token
    tokens = tokens[1:]
    
    # Skip SEP tokens
    tokens = tokens[3:]
    
    performance_triplets = []
    score_triplets = []
    
    for i in range(0, len(tokens), 3):
        if i+2 >= len(tokens):
            break
        
        time_tok, dur_tok, note_tok = tokens[i], tokens[i+1], tokens[i+2]
        
        # Control triplet (performance)
        if time_tok >= CONTROL_OFFSET and dur_tok >= CONTROL_OFFSET and note_tok >= CONTROL_OFFSET:
            # Remove CONTROL_OFFSET to get original note values
            perf_time = time_tok - CONTROL_OFFSET
            perf_dur = dur_tok - CONTROL_OFFSET
            perf_note = note_tok - CONTROL_OFFSET
            performance_triplets.extend([perf_time, perf_dur, perf_note])
        
        # Score triplet (ground truth)
        elif (time_tok < CONTROL_OFFSET and dur_tok < CONTROL_OFFSET and 
              note_tok < CONTROL_OFFSET and note_tok != REST):
            score_triplets.extend([time_tok, dur_tok, note_tok])
    
    return performance_triplets, score_triplets

print("="*80)
print("EXTRACTING GREEDY EXAMPLES FROM newest_model")
print("="*80)
print()

# Load model
print("Loading model from newest_model/...")
model = AutoModelForCausalLM.from_pretrained('newest_model/')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")
print()

# Load test data
print("Loading test data...")
with open('data/test_clean.txt', 'r') as f:
    lines = f.readlines()

num_examples = 5
print(f"Extracting {num_examples} examples with greedy decoding (top_k=1)")
print()

total_correct = 0
total_notes = 0

for example_idx in range(num_examples):
    print(f"Processing example {example_idx + 1}/{num_examples}...")
    
    line = lines[example_idx]
    
    # Parse sequence
    if '|' in line:
        token_str, _ = line.split('|')
        tokens = [int(t) for t in token_str.strip().split()]
    else:
        tokens = [int(t) for t in line.strip().split()]
    
    # Find where score notes start
    score_start_idx = None
    for i in range(1, len(tokens), 3):
        if i+2 < len(tokens):
            if (tokens[i] < CONTROL_OFFSET and 
                tokens[i+1] < CONTROL_OFFSET and 
                tokens[i+2] < CONTROL_OFFSET and
                tokens[i+2] != REST):
                score_start_idx = i
                break
    
    if score_start_idx is None:
        print(f"  Skipping - no score notes found")
        continue
    
    # Use bootstrap prefix as context
    context_tokens = tokens[:score_start_idx]
    
    # Convert to tensor and generate
    input_ids = torch.tensor([context_tokens])
    generated = greedy_decode_sequence(model, input_ids, max_new_tokens=len(tokens) - score_start_idx)
    
    # Get predicted tokens (remove context)
    predicted_tokens = generated[0, len(context_tokens):].cpu().tolist()
    
    # Reconstruct full sequences for saving
    full_ground_truth = tokens
    full_predicted = tokens[:score_start_idx] + predicted_tokens[:len(tokens) - score_start_idx]
    
    # Extract performance, ground truth score, and predicted score
    performance, gt_score = extract_parts_from_sequence(full_ground_truth)
    _, pred_score = extract_parts_from_sequence(full_predicted)
    
    # Calculate accuracy for this example
    gt_pitches = [(note - NOTE_OFFSET) % MAX_PITCH for note in gt_score[2::3]]
    pred_pitches = [(note - NOTE_OFFSET) % MAX_PITCH for note in pred_score[2::3]]
    
    min_len = min(len(gt_pitches), len(pred_pitches))
    if min_len > 0:
        correct = sum(1 for i in range(min_len) if gt_pitches[i] == pred_pitches[i])
        accuracy = 100.0 * correct / min_len
        total_correct += correct
        total_notes += min_len
    else:
        accuracy = 0.0
    
    # Create output directory
    output_dir = f'test_examples/example_{example_idx + 1}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as MIDI files
    print(f"  Saving MIDI files to {output_dir}/")
    
    # Performance MIDI
    perf_midi_path = os.path.join(output_dir, 'performance.mid')
    perf_midi = events_to_midi(performance)
    perf_midi.save(perf_midi_path)
    
    # Ground truth score MIDI
    gt_midi_path = os.path.join(output_dir, 'ground_truth_score.mid')
    gt_midi = events_to_midi(gt_score)
    gt_midi.save(gt_midi_path)
    
    # Predicted score MIDI
    pred_midi_path = os.path.join(output_dir, 'model_predictions.mid')
    pred_midi = events_to_midi(pred_score)
    pred_midi.save(pred_midi_path)
    
    print(f"  Performance notes: {len(performance)//3}")
    print(f"  Ground truth score notes: {len(gt_score)//3}")
    print(f"  Predicted score notes: {len(pred_score)//3}")
    print(f"  Pitch accuracy: {accuracy:.2f}% ({correct}/{min_len})")
    print()

print("="*80)
print("DONE!")
print("="*80)
print(f"Saved {num_examples} examples to test_examples/")
print()
print(f"Overall pitch accuracy: {100.0 * total_correct / total_notes:.2f}% ({total_correct}/{total_notes})")
print()
print("Each example contains:")
print("  • performance.mid - The input performance")
print("  • ground_truth_score.mid - The actual score")
print("  • model_predictions.mid - Model's greedy predictions (top_k=1)")
