"""
Test checkpoint-750 by providing first 25% of score notes as ground truth context,
then generating the rest. Report pitch accuracy.
"""
import os
import torch
from transformers import AutoModelForCausalLM
from anticipation.vocab import *
from anticipation.config import *

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

def count_score_triplets(tokens):
    """Count number of score triplets in a sequence."""
    count = 0
    i = 0
    while i + 2 < len(tokens):
        if (tokens[i] < CONTROL_OFFSET and 
            tokens[i+1] < CONTROL_OFFSET and 
            tokens[i+2] < CONTROL_OFFSET and
            tokens[i+2] != REST):
            count += 1
            i += 3
        else:
            i += 1
    return count

def find_score_triplets(tokens):
    """Find all score triplet positions (start index of each triplet)."""
    positions = []
    i = 0
    while i + 2 < len(tokens):
        if (tokens[i] < CONTROL_OFFSET and 
            tokens[i+1] < CONTROL_OFFSET and 
            tokens[i+2] < CONTROL_OFFSET and
            tokens[i+2] != REST):
            positions.append(i)
            i += 3
        else:
            i += 1
    return positions

def extract_score_pitches(tokens):
    """Extract all score note pitches from a sequence."""
    pitches = []
    i = 0
    while i + 2 < len(tokens):
        if (tokens[i] < CONTROL_OFFSET and 
            tokens[i+1] < CONTROL_OFFSET and 
            tokens[i+2] < CONTROL_OFFSET and
            tokens[i+2] != REST):
            pitch = tokens[i+2]
            pitches.append(pitch)
            i += 3
        else:
            i += 1
    return pitches

print("="*80)
print("TESTING checkpoint-750 WITH 25% SCORE CONTEXT")
print("="*80)
print()

# Load model
model_path = 'checkpoint-750'
print(f"Loading model from {model_path}/...")
model = AutoModelForCausalLM.from_pretrained(model_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")
print()

# Load test data
print("Loading test data...")
with open('data/test_normalized.txt', 'r') as f:
    lines = f.readlines()

num_examples = 50  # Test on 50 examples
print(f"Testing on {num_examples} examples")
print()

total_correct = 0
total_notes = 0
examples_processed = 0

for example_idx in range(min(num_examples, len(lines))):
    line = lines[example_idx]
    
    # Parse sequence
    if '|' in line:
        token_str, _ = line.split('|')
        tokens = [int(t) for t in token_str.strip().split()]
    else:
        tokens = [int(t) for t in line.strip().split()]
    
    # Find all score triplet positions
    score_positions = find_score_triplets(tokens)
    
    if len(score_positions) == 0:
        print(f"Example {example_idx + 1}: Skipping - no score notes found")
        continue
    
    # Calculate 25% cutoff
    num_score_triplets = len(score_positions)
    cutoff_triplets = max(1, num_score_triplets // 4)  # At least 1 triplet
    
    # Get context up to and including the 25% mark
    # cutoff_idx is the position AFTER the last triplet in the 25%
    cutoff_position = score_positions[cutoff_triplets - 1]
    cutoff_idx = cutoff_position + 3  # Include the full triplet
    
    # Context: everything up to cutoff
    context_tokens = tokens[:cutoff_idx]
    
    # Ground truth: full sequence for comparison
    gt_all_pitches = extract_score_pitches(tokens)
    # Ground truth pitches after the 25% context
    gt_remaining_pitches = gt_all_pitches[cutoff_triplets:]
    
    # Generate from context
    input_ids = torch.tensor([context_tokens])
    generated = greedy_decode_sequence(model, input_ids, max_new_tokens=len(tokens) - cutoff_idx + 100)
    
    # Get full generated sequence
    generated_tokens = generated[0].cpu().tolist()
    
    # Extract ALL score pitches from generated sequence (including context)
    gen_all_pitches = extract_score_pitches(generated_tokens)
    
    # Skip the first cutoff_triplets pitches (they were given as context)
    # Compare only the generated part
    if len(gen_all_pitches) > cutoff_triplets:
        pred_pitches = gen_all_pitches[cutoff_triplets:]
    else:
        pred_pitches = []
    
    # Calculate pitch accuracy
    min_len = min(len(gt_remaining_pitches), len(pred_pitches))
    
    if min_len > 0:
        correct = sum(1 for i in range(min_len) if gt_remaining_pitches[i] == pred_pitches[i])
        accuracy = 100.0 * correct / min_len
        total_correct += correct
        total_notes += min_len
        examples_processed += 1
        
        if example_idx < 5:  # Print details for first 5 examples
            print(f"Example {example_idx + 1}:")
            print(f"  Total score triplets: {num_score_triplets}")
            print(f"  Context triplets (25%): {cutoff_triplets}")
            print(f"  GT remaining notes: {len(gt_remaining_pitches)}")
            print(f"  Generated total notes: {len(gen_all_pitches)}")
            print(f"  Predicted remaining notes: {len(pred_pitches)}")
            print(f"  Compared notes: {min_len}")
            print(f"  Pitch accuracy: {accuracy:.2f}% ({correct}/{min_len})")
            if example_idx == 0:
                print(f"  First 10 GT pitches: {gt_remaining_pitches[:10]}")
                print(f"  First 10 Pred pitches: {pred_pitches[:10]}")
            print()
    else:
        print(f"Example {example_idx + 1}: Skipping - no notes to compare (gen={len(gen_all_pitches)}, cutoff={cutoff_triplets})")

print("="*80)
print("RESULTS")
print("="*80)
print(f"Examples processed: {examples_processed}")
print(f"Total notes compared: {total_notes}")
print(f"Total correct: {total_correct}")
print()
if total_notes > 0:
    overall_accuracy = 100.0 * total_correct / total_notes
    print(f"Overall pitch accuracy: {overall_accuracy:.2f}%")
else:
    print("No notes to evaluate")
print()
print("="*80)
