"""
Compare teacher forcing vs autoregressive generation accuracy with top_k=1.
"""
import torch
from transformers import AutoModelForCausalLM
from anticipation.vocab import CONTROL_OFFSET, NOTE_OFFSET, REST
from anticipation.config import MAX_PITCH, MAX_INSTR, CONTEXT_SIZE
from tqdm import tqdm

print("="*80)
print("TEACHER FORCING vs AUTOREGRESSIVE ACCURACY (top_k=1)")
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

# Take first 5 sequences
num_sequences = 5
print(f"Testing on {num_sequences} validation sequences")
print()

def greedy_decode_sequence(model, input_ids, max_new_tokens):
    """Generate with greedy decoding (top_k=1)."""
    generated = input_ids.clone()
    
    with torch.no_grad():
        past_key_values = None
        for _ in range(max_new_tokens):
            if past_key_values is None:
                outputs = model(generated, use_cache=True)
            else:
                outputs = model(generated[:, -1:], past_key_values=past_key_values, use_cache=True)
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # Greedy: argmax
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if generated.shape[1] >= CONTEXT_SIZE:
                break
    
    return generated

# TEACHER FORCING EVALUATION
print("="*80)
print("1. TEACHER FORCING EVALUATION")
print("="*80)

tf_exact = 0
tf_pitch = 0
tf_total = 0

with torch.no_grad():
    for seq_idx in tqdm(range(num_sequences), desc="Teacher forcing eval"):
        tokens = [int(x) for x in lines[seq_idx].strip().split() if x != '|']
        input_ids = torch.tensor([tokens]).to(device)
        
        # Teacher forcing: use ground truth as input
        outputs = model(input_ids)
        logits = outputs.logits[0]
        
        # Check predictions on score note tokens
        i = 1
        while i < len(tokens) - 2:
            if (tokens[i] < CONTROL_OFFSET and 
                tokens[i+1] < CONTROL_OFFSET and 
                tokens[i+2] < CONTROL_OFFSET):
                note_pos = i + 2
                
                predicted_token = logits[note_pos - 1].argmax().item()
                true_token = tokens[note_pos]
                
                # Exact token match
                if predicted_token == true_token:
                    tf_exact += 1
                
                # Pitch-only match
                if predicted_token >= NOTE_OFFSET and true_token >= NOTE_OFFSET:
                    pred_pitch = (predicted_token - NOTE_OFFSET) % MAX_PITCH
                    true_pitch = (true_token - NOTE_OFFSET) % MAX_PITCH
                    if pred_pitch == true_pitch:
                        tf_pitch += 1
                
                tf_total += 1
                i += 3
            else:
                i += 1

print(f"Total score notes: {tf_total}")
print(f"Exact token match: {tf_exact}/{tf_total} = {100*tf_exact/tf_total:.2f}%")
print(f"Pitch-only match: {tf_pitch}/{tf_total} = {100*tf_pitch/tf_total:.2f}%")
print()

# AUTOREGRESSIVE EVALUATION
print("="*80)
print("2. AUTOREGRESSIVE EVALUATION (top_k=1)")
print("="*80)

ar_exact = 0
ar_pitch = 0
ar_total = 0

with torch.no_grad():
    for seq_idx in tqdm(range(num_sequences), desc="Autoregressive eval"):
        tokens = [int(x) for x in lines[seq_idx].strip().split() if x != '|']
        
        # Find where score notes start (after bootstrap prefix)
        score_start_idx = None
        i = 1  # Skip ANTICIPATE token
        i += 3  # Skip SEP tokens
        
        # Skip control+rest pairs in bootstrap
        while i < len(tokens) - 2:
            if (tokens[i] >= CONTROL_OFFSET and 
                tokens[i+1] >= CONTROL_OFFSET and 
                tokens[i+2] >= CONTROL_OFFSET):
                i += 3
            else:
                score_start_idx = i
                break
        
        if score_start_idx is None:
            continue
        
        # Use bootstrap as context, generate the rest
        context_tokens = tokens[:score_start_idx]
        input_ids = torch.tensor([context_tokens]).to(device)
        
        # Greedy decode
        generated = greedy_decode_sequence(model, input_ids, len(tokens) - score_start_idx)
        predicted_tokens = generated[0, len(context_tokens):].cpu().tolist()
        
        # Compare predictions to ground truth
        gt_tokens = tokens[score_start_idx:]
        
        # Extract score note tokens
        i = 0
        while i < len(gt_tokens) - 2 and i < len(predicted_tokens) - 2:
            if (gt_tokens[i] < CONTROL_OFFSET and 
                gt_tokens[i+1] < CONTROL_OFFSET and 
                gt_tokens[i+2] < CONTROL_OFFSET):
                # This is a score triplet
                note_pos = i + 2
                
                if note_pos < len(predicted_tokens):
                    pred_token = predicted_tokens[note_pos]
                    true_token = gt_tokens[note_pos]
                    
                    # Exact token match
                    if pred_token == true_token:
                        ar_exact += 1
                    
                    # Pitch-only match
                    if pred_token >= NOTE_OFFSET and true_token >= NOTE_OFFSET:
                        pred_pitch = (pred_token - NOTE_OFFSET) % MAX_PITCH
                        true_pitch = (true_token - NOTE_OFFSET) % MAX_PITCH
                        if pred_pitch == true_pitch:
                            ar_pitch += 1
                    
                    ar_total += 1
                
                i += 3
            else:
                i += 1

print(f"Total score notes: {ar_total}")
print(f"Exact token match: {ar_exact}/{ar_total} = {100*ar_exact/ar_total:.2f}%")
print(f"Pitch-only match: {ar_pitch}/{ar_total} = {100*ar_pitch/ar_total:.2f}%")
print()

# COMPARISON
print("="*80)
print("COMPARISON (top_k=1)")
print("="*80)
print(f"{'Metric':<30} {'Teacher Forcing':<20} {'Autoregressive':<20} {'Gap':<10}")
print("-"*80)
print(f"{'Exact token match':<30} {100*tf_exact/tf_total:>6.2f}% {'':<13} {100*ar_exact/ar_total:>6.2f}% {'':<13} {100*(tf_exact/tf_total - ar_exact/ar_total):>6.2f}%")
print(f"{'Pitch-only match':<30} {100*tf_pitch/tf_total:>6.2f}% {'':<13} {100*ar_pitch/ar_total:>6.2f}% {'':<13} {100*(tf_pitch/tf_total - ar_pitch/ar_total):>6.2f}%")
print()
print("INTERPRETATION:")
print(f"  - Model achieves ~{100*tf_exact/tf_total:.1f}% accuracy when given ground truth context")
print(f"  - Model achieves only ~{100*ar_pitch/ar_total:.1f}% pitch accuracy when using its own predictions")
print(f"  - The {100*(tf_exact/tf_total - ar_pitch/ar_total):.1f}% gap shows severe exposure bias")
print(f"  - Even with greedy decoding (top_k=1), errors compound during generation")
