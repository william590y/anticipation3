"""
Extract performance, model predictions, and ground truth from TRAIN sequences
using the opening_model to check if it's overfitting to training data
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET, TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET
from anticipation.convert import events_to_midi
import os

print("="*80)
print("EXTRACTING TRAIN EXAMPLES - CHECKING FOR OVERFITTING")
print("="*80)

# Load model
print("\nLoading model from opening_model/...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained('opening_model/')
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Load TRAIN data
print("\nLoading TRAIN data from data/train_openings.txt...")
with open('data/train_openings.txt', 'r') as f:
    lines = f.readlines()

print(f"Total train sequences: {len(lines)}")
print(f"Evaluating on first 5 sequences...\n")

# Process first 5 train sequences
for seq_idx, line in enumerate(lines[:5]):
    print(f"Processing train sequence {seq_idx + 1}/5...")
    line = line.strip()
    if not line:
        continue
    
    # Handle both formats: "tokens | masks" or just "tokens"
    if '|' in line:
        token_part = line.split('|')[0].strip()
    else:
        token_part = line
    
    tokens = [int(t) for t in token_part.split()]
    
    # Find score note positions
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
    
    print(f"  Total score positions: {len(score_positions)}")
    
    # Generate predictions
    with torch.no_grad():
        # Process up to first score position
        first_score_pos = score_positions[0] if score_positions else len(tokens)
        if first_score_pos > 0:
            init_context = torch.tensor([tokens[:first_score_pos]]).to(device)
            outputs = model(init_context, past_key_values=None, use_cache=True)
            past_key_values = outputs.past_key_values
        
        # Predict each score note
        last_pos = first_score_pos
        num_correct = 0
        num_total = 0
        num_correct_after_prefix = 0
        num_total_after_prefix = 0
        
        for idx, pos in enumerate(score_positions):
            # Process intermediate tokens
            if pos > last_pos:
                intermediate = torch.tensor([tokens[last_pos:pos]]).to(device)
                outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
            
            # Predict
            logits = outputs.logits[0, -1]
            predicted_token = logits.argmax().item()
            ground_truth_token = tokens[pos]
            
            if predicted_token == ground_truth_token:
                num_correct += 1
            num_total += 1
            
            # Track after prefix
            if idx >= 33:
                if predicted_token == ground_truth_token:
                    num_correct_after_prefix += 1
                num_total_after_prefix += 1
            
            # Feed predicted token back
            next_token = torch.tensor([[predicted_token]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            last_pos = pos + 1
        
        accuracy = (num_correct / num_total * 100) if num_total > 0 else 0
        accuracy_after_prefix = (num_correct_after_prefix / num_total_after_prefix * 100) if num_total_after_prefix > 0 else 0
        
        print(f"  Overall accuracy: {accuracy:.1f}% ({num_correct}/{num_total})")
        print(f"  Accuracy after prefix (notes 34+): {accuracy_after_prefix:.1f}% ({num_correct_after_prefix}/{num_total_after_prefix})")
        print()

print("="*80)
print("SUMMARY")
print("="*80)
print("If train accuracy >> test accuracy, the model is overfitting.")
print("If train accuracy ≈ test accuracy, the model has good generalization.")
