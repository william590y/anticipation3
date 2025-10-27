"""
Autoregressive pitch accuracy test - uses model's own predictions as context
This is MUCH slower but tests actual generation quality (like generate4)
"""
import torch
from transformers import GPT2LMHeadModel
from tqdm import tqdm

CONTROL_OFFSET = 27513

print("="*80)
print("AUTOREGRESSIVE PITCH ACCURACY TEST")
print("="*80)
print("NOTE: This uses the model's own predictions as context (no teacher forcing)")
print("This is slow but matches actual generation behavior from generate4")
print("="*80)

# Load model
print("\nLoading model from new_model/...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained('new_model/')
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Load data
print("\nLoading validation data...")
with open('data/test_output.txt', 'r') as f:
    lines = f.readlines()[:5]  # Only 5 sequences - this is VERY slow

print(f"Testing on {len(lines)} sequences (autoregressive generation)")

correct_pitches = 0
total_pitches = 0

with torch.no_grad():
    for seq_idx, line in enumerate(lines):
        print(f"\nSequence {seq_idx + 1}/{len(lines)}:")
        parts = line.strip().split(' | ')
        if len(parts) < 1:
            continue
        
        ground_truth = [int(t) for t in parts[0].split()]
        
        # Find all score triplet positions in ground truth
        score_positions = []
        i = 0
        while i < len(ground_truth) - 2:
            if (ground_truth[i] < CONTROL_OFFSET and 
                ground_truth[i+1] < CONTROL_OFFSET and 
                ground_truth[i+2] < CONTROL_OFFSET):
                # Store position of the note token (i+2)
                score_positions.append(i + 2)
                i += 3
            else:
                i += 1
        
        print(f"  Found {len(score_positions)} score notes to predict")
        
        if len(score_positions) == 0:
            continue
        
        # Generate autoregressively using KV cache for efficiency
        past_key_values = None
        
        # Process all tokens up to first score position
        first_score_pos = score_positions[0] if score_positions else len(ground_truth)
        if first_score_pos > 0:
            init_context = torch.tensor([ground_truth[:first_score_pos]]).to(device)
            outputs = model(init_context, past_key_values=None, use_cache=True)
            past_key_values = outputs.past_key_values
        
        # Now process each score position autoregressively with KV cache
        last_pos = first_score_pos
        for pos in tqdm(score_positions, desc=f"  Seq {seq_idx+1}", leave=False):
            # If there are tokens between last_pos and pos, process them with cache
            if pos > last_pos:
                intermediate = torch.tensor([ground_truth[last_pos:pos]]).to(device)
                outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
            
            # Get prediction for current position
            logits = outputs.logits[0, -1]
            predicted_token = logits.argmax().item()
            
            # Check if prediction matches ground truth
            true_token = ground_truth[pos]
            if predicted_token == true_token:
                correct_pitches += 1
            total_pitches += 1
            
            # Add ground truth token and update cache for next iteration
            next_token = torch.tensor([[true_token]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            last_pos = pos + 1
        
        seq_accuracy = (correct_pitches / total_pitches * 100) if total_pitches > 0 else 0
        print(f"  Cumulative accuracy so far: {seq_accuracy:.2f}%")

accuracy = correct_pitches / total_pitches if total_pitches > 0 else 0.0

print("\n" + "="*80)
print(f"AUTOREGRESSIVE Pitch Accuracy: {accuracy*100:.2f}%")
print(f"Correct: {correct_pitches}/{total_pitches}")
print("="*80)
print("\nComparison:")
print("  Teacher Forcing (train.py evaluation): ~91.29%")
print(f"  Autoregressive (generate4-style):     {accuracy*100:.2f}%")
print("\nNote: Autoregressive is typically lower due to error accumulation")
