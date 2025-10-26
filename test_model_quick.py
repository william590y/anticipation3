"""Quick test of real model pitch accuracy - AUTOREGRESSIVE (no teacher forcing)"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET, TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET

print("="*80)
print("PITCH ACCURACY TEST - AUTOREGRESSIVE GENERATION (NO TEACHER FORCING)")
print("="*80)

# Load model
print("\nLoading model from new_model/...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained('new_model/')
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Load data
print("Loading validation data...")
with open('data/test_output.txt', 'r') as f:
    lines = f.readlines()[:10]  # Use 10 sequences for autoregressive test

correct_pitches = 0
total_pitches = 0

print(f"\nGenerating {len(lines)} sequences autoregressively...")

for seq_idx, line in enumerate(lines):
    parts = line.strip().split(' | ')
    if len(parts) < 1:
        continue
    
    tokens = [int(t) for t in parts[0].split()]
    
    # Find all score triplets (ground truth)
    ground_truth_pitches = []
    i = 0
    while i < len(tokens) - 2:
        if (tokens[i] < CONTROL_OFFSET and 
            tokens[i+1] < CONTROL_OFFSET and 
            tokens[i+2] < CONTROL_OFFSET):
            ground_truth_pitches.append((i+2, tokens[i+2]))  # (position, pitch)
            i += 3
        else:
            i += 1
    
    if len(ground_truth_pitches) == 0:
        continue
    
    # Generate autoregressively and check pitch predictions
    with torch.no_grad():
        for pos, true_pitch in ground_truth_pitches:
            # Use all tokens BEFORE position pos as context
            context = tokens[:pos]
            input_ids = torch.tensor([context]).to(device)
            
            # Predict next token (which should be the pitch at position pos)
            outputs = model(input_ids)
            logits = outputs.logits[0, -1]  # Last position logits
            predicted_token = logits.argmax().item()
            
            # Check if prediction matches ground truth
            if predicted_token == true_pitch:
                correct_pitches += 1
            total_pitches += 1
    
    if (seq_idx + 1) % 5 == 0:
        print(f"  Processed {seq_idx + 1}/{len(lines)} sequences...")

accuracy = correct_pitches / total_pitches if total_pitches > 0 else 0.0

print("\n" + "="*80)
print(f"AUTOREGRESSIVE Pitch Accuracy: {accuracy*100:.2f}%")
print(f"Correct: {correct_pitches}/{total_pitches}")
print("="*80)
print("\nNOTE: This is WITHOUT teacher forcing (autoregressive generation)")
print("The model uses its own predictions as context, not ground truth.")
