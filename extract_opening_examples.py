"""
Extract performance, model predictions, and ground truth from test sequences
using the opening_model and save as MIDI files for comparison
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET, TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET
from anticipation.convert import events_to_midi
import os

print("="*80)
print("EXTRACTING OPENING_MODEL TEST EXAMPLES FOR VIEWING")
print("="*80)

# Load model
print("\nLoading model from opening_model/...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained('opening_model/')
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Create output directories
os.makedirs('opening_examples', exist_ok=True)
for i in range(5):
    os.makedirs(f'opening_examples/example_{i+1}', exist_ok=True)

# Load data - use test_openings.txt
print("\nLoading validation data from data/test_openings.txt...")
with open('data/test_openings.txt', 'r') as f:
    lines = f.readlines()[:5]

print(f"Processing {len(lines)} sequences...\n")

for seq_idx, line in enumerate(lines):
    print(f"Processing sequence {seq_idx + 1}/5...")
    line = line.strip()
    if not line:
        continue
    
    # Handle both formats: "tokens | masks" or just "tokens"
    if '|' in line:
        token_part = line.split('|')[0].strip()
    else:
        token_part = line
    
    tokens = [int(t) for t in token_part.split()]
    
    # Extract ground truth score (all triplets where all 3 tokens < CONTROL_OFFSET)
    ground_truth_score = []
    i = 0
    print(f"  Extracting ground truth score triplets...")
    num_rest_notes = 0
    num_actual_notes = 0
    while i < len(tokens) - 2:
        if (tokens[i] < CONTROL_OFFSET and 
            tokens[i+1] < CONTROL_OFFSET and 
            tokens[i+2] < CONTROL_OFFSET):
            note_token = tokens[i+2]
            if note_token == 27512:  # REST
                num_rest_notes += 1
            else:
                num_actual_notes += 1
            ground_truth_score.extend([tokens[i], tokens[i+1], tokens[i+2]])
            i += 3
        else:
            i += 1
    print(f"  Ground truth: {num_actual_notes} actual notes, {num_rest_notes} REST tokens")
    
    # Extract performance (all triplets where all 3 tokens >= CONTROL_OFFSET)
    performance = []
    i = 0
    while i < len(tokens) - 2:
        if (tokens[i] >= CONTROL_OFFSET and tokens[i] < CONTROL_OFFSET + 27512 and
            tokens[i+1] >= CONTROL_OFFSET and tokens[i+1] < CONTROL_OFFSET + 27512 and
            tokens[i+2] >= CONTROL_OFFSET and tokens[i+2] < CONTROL_OFFSET + 27512):
            # Remove CONTROL_OFFSET to get actual performance tokens
            performance.extend([tokens[i] - CONTROL_OFFSET, 
                              tokens[i+1] - CONTROL_OFFSET, 
                              tokens[i+2] - CONTROL_OFFSET])
            i += 3
        else:
            i += 1
    
    # Generate model predictions FULLY AUTOREGRESSIVELY with KV cache
    # Predict ALL THREE TOKENS (time, dur, pitch) for each score triplet
    model_predictions = []
    past_key_values = None
    
    # Find score triplet positions
    score_triplet_positions = []
    i = 0
    while i < len(tokens) - 2:
        if (tokens[i] < CONTROL_OFFSET and 
            tokens[i+1] < CONTROL_OFFSET and 
            tokens[i+2] < CONTROL_OFFSET):
            score_triplet_positions.append((i, i+1, i+2))  # (time, dur, pitch)
            i += 3
        else:
            i += 1
    
    print(f"  Total score triplets: {len(score_triplet_positions)}")
    
    with torch.no_grad():
        # Find first score triplet position
        first_score_time_pos = score_triplet_positions[0][0] if score_triplet_positions else len(tokens)
        
        # Process mode token + everything before first score triplet
        if first_score_time_pos > 0:
            init_context = torch.tensor([tokens[:first_score_time_pos]]).to(device)
            outputs = model(init_context, past_key_values=None, use_cache=True)
            past_key_values = outputs.past_key_values
        
        # Statistics trackers
        stats = {
            'score_time': {'correct': 0, 'total': 0},
            'score_dur': {'correct': 0, 'total': 0},
            'score_pitch': {'correct': 0, 'total': 0}
        }
        
        predicted_score = []
        last_pos = first_score_time_pos
        
        print(f"  Predicting ALL THREE TOKENS autoregressively for each score triplet...")
        
        for triplet_idx, (time_pos, dur_pos, pitch_pos) in enumerate(score_triplet_positions):
            # Process intermediate control tokens between score triplets
            if time_pos > last_pos:
                intermediate = torch.tensor([tokens[last_pos:time_pos]]).to(device)
                outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
            
            # Predict TIME token
            logits = outputs.logits[0, -1]
            pred_time = logits.argmax().item()
            if pred_time == tokens[time_pos]:
                stats['score_time']['correct'] += 1
            stats['score_time']['total'] += 1
            
            # Feed predicted time back
            next_token = torch.tensor([[pred_time]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Predict DURATION token
            logits = outputs.logits[0, -1]
            pred_dur = logits.argmax().item()
            if pred_dur == tokens[dur_pos]:
                stats['score_dur']['correct'] += 1
            stats['score_dur']['total'] += 1
            
            # Feed predicted duration back
            next_token = torch.tensor([[pred_dur]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Predict PITCH token
            logits = outputs.logits[0, -1]
            pred_pitch = logits.argmax().item()
            if pred_pitch == tokens[pitch_pos]:
                stats['score_pitch']['correct'] += 1
            stats['score_pitch']['total'] += 1
            
            # Feed predicted pitch back
            next_token = torch.tensor([[pred_pitch]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Store predicted triplet
            predicted_score.extend([pred_time, pred_dur, pred_pitch])
            last_pos = pitch_pos + 1
    
    # Save as MIDI files
    example_dir = f'opening_examples/example_{seq_idx + 1}'
    
    # Ground truth score
    if ground_truth_score:
        midi = events_to_midi(ground_truth_score)
        midi.save(f'{example_dir}/ground_truth_score.mid')
        print(f"  Saved ground truth score ({len(ground_truth_score)//3} notes)")
    
    # Performance (controls)
    if performance:
        midi = events_to_midi(performance)
        midi.save(f'{example_dir}/performance.mid')
        print(f"  Saved performance ({len(performance)//3} notes)")
    
    # Model predictions
    if predicted_score:
        midi = events_to_midi(predicted_score)
        midi.save(f'{example_dir}/model_predictions.mid')
        print(f"  Saved model predictions ({len(predicted_score)//3} notes)")
        print(f"  AUTOREGRESSIVE ACCURACY (all 3 tokens predicted):")
        for token_type in ['score_time', 'score_dur', 'score_pitch']:
            correct = stats[token_type]['correct']
            total = stats[token_type]['total']
            acc = (correct / total * 100) if total > 0 else 0
            print(f"    {token_type:15s}: {acc:6.2f}% ({correct}/{total})")
        total_correct = sum(s['correct'] for s in stats.values())
        total_tokens = sum(s['total'] for s in stats.values())
        overall_acc = (total_correct / total_tokens * 100) if total_tokens > 0 else 0
        print(f"    Overall:        {overall_acc:6.2f}% ({total_correct}/{total_tokens})")

print("\n" + "="*80)
print("DONE! Files saved to opening_examples/")
print("="*80)
print("\nEach example folder contains:")
print("  - ground_truth_score.mid: The actual score")
print("  - performance.mid: The performance (controls)")
print("  - model_predictions.mid: What the model predicted")
print("\nYou can open these in any MIDI viewer/DAW to compare!")
