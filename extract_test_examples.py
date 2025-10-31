"""
Extract performance, model predictions, and ground truth from test sequences
and save as MIDI files for comparison
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET, TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET
from anticipation.convert import events_to_midi
import os

print("="*80)
print("EXTRACTING TEST EXAMPLES FOR VIEWING")
print("="*80)

# Load model
print("\nLoading model from new_model/...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained('new_model/')
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Create output directories
os.makedirs('test_examples', exist_ok=True)
for i in range(5):
    os.makedirs(f'test_examples/example_{i+1}', exist_ok=True)

# Load data
print("\nLoading validation data...")
with open('data/test_output.txt', 'r') as f:
    lines = f.readlines()[:5]

print(f"Processing {len(lines)} sequences...\n")

for seq_idx, line in enumerate(lines):
    print(f"Processing sequence {seq_idx + 1}/5...")
    parts = line.strip().split(' | ')
    if len(parts) < 1:
        continue
    
    tokens = [int(t) for t in parts[0].split()]
    
    # Extract ground truth score (all triplets where all 3 tokens < CONTROL_OFFSET)
    ground_truth_score = []
    i = 0
    while i < len(tokens) - 2:
        if (tokens[i] < CONTROL_OFFSET and 
            tokens[i+1] < CONTROL_OFFSET and 
            tokens[i+2] < CONTROL_OFFSET):
            ground_truth_score.extend([tokens[i], tokens[i+1], tokens[i+2]])
            i += 3
        else:
            i += 1
    
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
    
    # Generate model predictions autoregressively with KV cache
    model_predictions = []
    past_key_values = None
    
    score_positions = []
    i = 0
    while i < len(tokens) - 2:
        if (tokens[i] < CONTROL_OFFSET and 
            tokens[i+1] < CONTROL_OFFSET and 
            tokens[i+2] < CONTROL_OFFSET):
            score_positions.append(i + 2)  # Note position
            i += 3
        else:
            i += 1
    
    with torch.no_grad():
        # Process up to first score position
        first_score_pos = score_positions[0] if score_positions else len(tokens)
        if first_score_pos > 0:
            init_context = torch.tensor([tokens[:first_score_pos]]).to(device)
            outputs = model(init_context, past_key_values=None, use_cache=True)
            past_key_values = outputs.past_key_values
        
        # Process each score position
        last_pos = first_score_pos
        predicted_score = []
        
        for pos in score_positions:
            # Process intermediate tokens
            if pos > last_pos:
                intermediate = torch.tensor([tokens[last_pos:pos]]).to(device)
                outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
            
            # Get prediction
            logits = outputs.logits[0, -1]
            predicted_token = logits.argmax().item()
            
            # Store the predicted triplet (using ground truth for time/duration, prediction for note)
            triplet_start = pos - 2
            predicted_score.extend([
                tokens[triplet_start],      # Ground truth time
                tokens[triplet_start + 1],  # Ground truth duration
                predicted_token             # Predicted note
            ])
            
            # Update cache with ground truth token
            next_token = torch.tensor([[tokens[pos]]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            last_pos = pos + 1
    
    # Save as MIDI files
    example_dir = f'test_examples/example_{seq_idx + 1}'
    
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

print("\n" + "="*80)
print("DONE! Files saved to test_examples/")
print("="*80)
print("\nEach example folder contains:")
print("  - ground_truth_score.mid: The actual score")
print("  - performance.mid: The performance (controls)")
print("  - model_predictions.mid: What the model predicted")
print("\nYou can open these in any MIDI viewer/DAW to compare!")
