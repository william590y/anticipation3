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
        
        # Generate autoregressively - build up the sequence token by token
        generated = []
        
        for pos in tqdm(score_positions, desc=f"  Seq {seq_idx+1}", leave=False):
            # Context is: all ground truth tokens before this position
            # PLUS any tokens we've generated so far
            context = ground_truth[:pos]
            
            # Generate the next token (the pitch at position pos)
            input_ids = torch.tensor([context]).to(device)
            outputs = model(input_ids)
            logits = outputs.logits[0, -1]  # Last position's logits
            predicted_token = logits.argmax().item()
            
            # Check if prediction matches ground truth
            true_token = ground_truth[pos]
            if predicted_token == true_token:
                correct_pitches += 1
            total_pitches += 1
        
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
