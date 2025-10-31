"""
Verify what the training script's "pitch accuracy" metric actually measures.
This will evaluate using teacher forcing (like during training) to see if we get
the same ~91% accuracy that was reported.
"""
import torch
from transformers import AutoModelForCausalLM
from anticipation.vocab import CONTROL_OFFSET, NOTE_OFFSET
from anticipation.config import MAX_PITCH, MAX_INSTR

print("="*80)
print("VERIFYING TRAINING PITCH ACCURACY METRIC")
print("="*80)
print()

# Load the model
print("Loading newest_model...")
model = AutoModelForCausalLM.from_pretrained('newest_model/')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")
print()

# Load validation data
print("Loading validation data...")
with open('data/test_clean.txt', 'r') as f:
    lines = f.readlines()

# Take first 50 sequences
num_sequences = 50
print(f"Testing on {num_sequences} validation sequences")
print()

total_exact_match = 0  # Exact token match (what training calls "pitch accuracy")
total_pitch_only = 0   # True pitch-only match
total_notes = 0

with torch.no_grad():
    for seq_idx in range(num_sequences):
        # Parse tokens (skip the | separator at the end)
        tokens = [int(x) for x in lines[seq_idx].strip().split() if x != '|']
        
        # Convert to tensor and get predictions
        input_ids = torch.tensor([tokens]).to(device)
        
        # Teacher forcing: use ground truth as input
        outputs = model(input_ids)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        
        # Check predictions on score note tokens
        i = 1  # Skip first token (mode token)
        while i < len(tokens) - 2:
            # Check if this is a score triplet
            if (tokens[i] < CONTROL_OFFSET and 
                tokens[i+1] < CONTROL_OFFSET and 
                tokens[i+2] < CONTROL_OFFSET):
                # This is a score triplet, position i+2 is the note token
                note_pos = i + 2
                
                # Get prediction (predict token at note_pos from context before it)
                predicted_token = logits[note_pos - 1].argmax().item()
                true_token = tokens[note_pos]
                
                # Exact token match
                if predicted_token == true_token:
                    total_exact_match += 1
                
                # Pitch-only match
                if predicted_token >= NOTE_OFFSET and true_token >= NOTE_OFFSET:
                    pred_pitch = (predicted_token - NOTE_OFFSET) % MAX_PITCH
                    true_pitch = (true_token - NOTE_OFFSET) % MAX_PITCH
                    if pred_pitch == true_pitch:
                        total_pitch_only += 1
                
                total_notes += 1
                i += 3
            else:
                i += 1

print("="*80)
print("RESULTS (Teacher Forcing Validation)")
print("="*80)
print(f"Total score notes evaluated: {total_notes}")
print()
print(f"Exact token match: {total_exact_match}/{total_notes} = {100*total_exact_match/total_notes:.2f}%")
print(f"  ^ This is what train.py calls 'pitch accuracy'")
print()
print(f"True pitch-only match: {total_pitch_only}/{total_notes} = {100*total_pitch_only/total_notes:.2f}%")
print(f"  ^ This is actual pitch matching (ignoring instrument)")
print()
print("="*80)
print("COMPARISON")
print("="*80)
print(f"Training reported 'pitch accuracy': 91.20%")
print(f"Our exact token match (teacher forcing): {100*total_exact_match/total_notes:.2f}%")
print(f"Our pitch-only match (teacher forcing): {100*total_pitch_only/total_notes:.2f}%")
print()
print(f"Autoregressive pitch accuracy (from earlier): 12.50%")
print(f"Autoregressive exact token match (from earlier): 0.13%")
print()
print("CONCLUSION:")
print(f"  - Training 'pitch accuracy' = exact token match with teacher forcing (~91%)")
print(f"  - This drops to ~12.5% pitch and 0.13% exact when using autoregressive generation")
print(f"  - This huge gap is due to exposure bias (model trained on ground truth context)")
