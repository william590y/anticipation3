"""
Test greedy generation (top_k=1) to see if the model can achieve higher accuracy
when forced to always pick the most likely token.
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from anticipation.vocab import *
from anticipation.config import *
from tqdm import tqdm

def calculate_pitch_accuracy(predictions, ground_truth):
    """Calculate pitch accuracy between predicted and ground truth score notes."""
    # Extract score triplets (tokens < CONTROL_OFFSET)
    pred_score = []
    gt_score = []
    
    for i in range(0, len(predictions), 3):
        if i+2 < len(predictions):
            if (predictions[i] < CONTROL_OFFSET and 
                predictions[i+1] < CONTROL_OFFSET and 
                predictions[i+2] < CONTROL_OFFSET and
                predictions[i+2] != REST):
                pred_score.append(predictions[i+2])
    
    for i in range(0, len(ground_truth), 3):
        if i+2 < len(ground_truth):
            if (ground_truth[i] < CONTROL_OFFSET and 
                ground_truth[i+1] < CONTROL_OFFSET and 
                ground_truth[i+2] < CONTROL_OFFSET and
                ground_truth[i+2] != REST):
                gt_score.append(ground_truth[i+2])
    
    # Compare pitches (extract pitch from note tokens)
    min_len = min(len(pred_score), len(gt_score))
    if min_len == 0:
        return 0, 0, 0
    
    pred_pitches = [(note - NOTE_OFFSET) // MAX_INSTR for note in pred_score[:min_len]]
    gt_pitches = [(note - NOTE_OFFSET) // MAX_INSTR for note in gt_score[:min_len]]
    
    correct = sum(1 for p, g in zip(pred_pitches, gt_pitches) if p == g)
    return correct, min_len, 100.0 * correct / min_len if min_len > 0 else 0


def greedy_decode_sequence(model, input_ids, max_new_tokens=1024):
    """
    Greedy decoding with KV caching: always pick the most likely token (top_k=1).
    This is equivalent to argmax sampling.
    """
    device = model.device
    input_ids = input_ids.to(device)
    
    generated = input_ids.clone()
    past_key_values = None
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # On first iteration, use full sequence. After that, only use the last token
            if past_key_values is None:
                model_inputs = generated
            else:
                model_inputs = generated[:, -1:]
            
            # Get logits for next token with KV cache
            outputs = model(model_inputs, past_key_values=past_key_values, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # Greedy: pick the most likely token (argmax)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if we've filled the context
            if generated.shape[1] >= CONTEXT_SIZE:
                break
    
    return generated


print("="*80)
print("TESTING GREEDY GENERATION (top_k=1)")
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

# Load validation data
print("Loading validation data...")
with open('data/test_clean.txt', 'r') as f:
    lines = f.readlines()

num_sequences = 5  # Test on 5 sequences
lines = lines[:num_sequences]

print(f"Testing on {num_sequences} sequences with greedy decoding")
print()

total_correct = 0
total_notes = 0

for idx, line in enumerate(tqdm(lines, desc="Evaluating")):
    # Parse sequence
    if '|' in line:
        token_str, _ = line.split('|')
        tokens = [int(t) for t in token_str.strip().split()]
    else:
        tokens = [int(t) for t in line.strip().split()]
    
    # Create input (everything except the last token for autoregressive prediction)
    # For greedy, we'll feed in the prefix and let it generate the rest
    
    # Find where score notes start (after bootstrap prefix)
    score_start_idx = None
    for i in range(1, len(tokens), 3):  # Skip ANTICIPATE token
        if i+2 < len(tokens):
            if (tokens[i] < CONTROL_OFFSET and 
                tokens[i+1] < CONTROL_OFFSET and 
                tokens[i+2] < CONTROL_OFFSET and
                tokens[i+2] != REST):
                score_start_idx = i
                break
    
    if score_start_idx is None:
        continue
    
    # Use bootstrap prefix + first few score notes as context
    # Then let model generate the rest with greedy decoding
    context_tokens = tokens[:score_start_idx + 30]  # Use first 10 score triplets as context
    ground_truth = tokens[score_start_idx + 30:]
    
    # Convert to tensor
    input_ids = torch.tensor([context_tokens])
    
    # Greedy decode
    generated = greedy_decode_sequence(model, input_ids, max_new_tokens=len(ground_truth))
    
    # Extract generated tokens (remove input context)
    generated_tokens = generated[0, len(context_tokens):].cpu().tolist()
    
    # Trim to same length as ground truth for comparison
    generated_tokens = generated_tokens[:len(ground_truth)]
    
    # Calculate accuracy
    correct, total, accuracy = calculate_pitch_accuracy(generated_tokens, ground_truth)
    total_correct += correct
    total_notes += total

print()
print("="*80)
print("RESULTS")
print("="*80)
print(f"Total sequences: {num_sequences}")
print(f"Total score notes: {total_notes:,}")
print(f"Correct predictions: {total_correct:,}")
print(f"Pitch accuracy: {100.0 * total_correct / total_notes:.2f}%")
print()

print("="*80)
print("COMPARISON")
print("="*80)
print(f"Previous results (top_p sampling):")
print(f"  • Teacher forcing (validation): 91.29%")
print(f"  • Autoregressive (validation): 91.84%")
print()
print(f"Greedy decoding (top_k=1): {100.0 * total_correct / total_notes:.2f}%")
print()
print("Expected: Greedy should give HIGHER accuracy than sampling")
print("because it always picks the most confident prediction.")
