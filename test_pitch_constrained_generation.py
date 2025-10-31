"""
Test autoregressive generation with pitch constraint:
Keep regenerating triplets until we get the correct pitch.
This tests if the model can predict correct timing/duration when pitch is constrained.
"""
import torch
from transformers import AutoModelForCausalLM
from anticipation.vocab import CONTROL_OFFSET, NOTE_OFFSET, REST
from anticipation.config import MAX_PITCH, MAX_INSTR, CONTEXT_SIZE
from tqdm import tqdm

print("="*80)
print("PITCH-CONSTRAINED AUTOREGRESSIVE GENERATION")
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
with open('data/test_clean.txt', 'r') as f:
    lines = f.readlines()

num_sequences = 5
print(f"Testing on {num_sequences} validation sequences")
print()

def pitch_constrained_decode(model, input_ids, ground_truth_tokens, score_start_idx, max_attempts=100):
    """
    Generate sequence but keep regenerating each note token until we get the correct pitch.
    This tests if the model can predict timing/duration correctly when pitch is constrained.
    """
    generated = input_ids.clone()
    gt_tokens = ground_truth_tokens[score_start_idx:]
    
    stats = {
        'total_notes': 0,
        'first_attempt_correct': 0,
        'attempts_needed': [],
        'exact_match': 0,
        'time_match': 0,
        'dur_match': 0,
    }
    
    with torch.no_grad():
        past_key_values = None
        gt_idx = 0
        
        while gt_idx < len(gt_tokens):
            # Check if we need to generate a score triplet
            if (gt_idx + 2 < len(gt_tokens) and
                gt_tokens[gt_idx] < CONTROL_OFFSET and 
                gt_tokens[gt_idx+1] < CONTROL_OFFSET and 
                gt_tokens[gt_idx+2] < CONTROL_OFFSET):
                
                # This is a score triplet - generate all 3 tokens
                gt_time = gt_tokens[gt_idx]
                gt_dur = gt_tokens[gt_idx + 1]
                gt_note = gt_tokens[gt_idx + 2]
                gt_pitch = (gt_note - NOTE_OFFSET) % MAX_PITCH
                
                # Generate time token
                if past_key_values is None:
                    outputs = model(generated, use_cache=True)
                else:
                    outputs = model(generated[:, -1:], past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                time_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, time_token], dim=1)
                
                # Generate duration token
                outputs = model(generated[:, -1:], past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                dur_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, dur_token], dim=1)
                
                # Generate note token with pitch constraint
                attempts = 0
                note_token = None
                
                for attempt in range(max_attempts):
                    outputs = model(generated[:, -1:], past_key_values=past_key_values, use_cache=True)
                    
                    # Get top-k tokens to sample from
                    logits = outputs.logits[0, -1, :]
                    top_k = 50
                    top_logits, top_indices = torch.topk(logits, top_k)
                    probs = torch.softmax(top_logits, dim=-1)
                    
                    # Sample from top-k
                    sampled_idx = torch.multinomial(probs, 1)
                    candidate = top_indices[sampled_idx].item()
                    
                    # Check if it has the correct pitch
                    if candidate >= NOTE_OFFSET:
                        pred_pitch = (candidate - NOTE_OFFSET) % MAX_PITCH
                        if pred_pitch == gt_pitch:
                            note_token = candidate
                            attempts = attempt + 1
                            break
                
                if note_token is None:
                    # Fallback: couldn't find correct pitch in max_attempts
                    # Just use ground truth
                    note_token = gt_note
                    attempts = max_attempts
                
                # Update stats
                stats['total_notes'] += 1
                stats['attempts_needed'].append(attempts)
                
                if attempts == 1:
                    stats['first_attempt_correct'] += 1
                
                if (time_token.item() == gt_time and 
                    dur_token.item() == gt_dur and 
                    note_token == gt_note):
                    stats['exact_match'] += 1
                
                if time_token.item() == gt_time:
                    stats['time_match'] += 1
                
                if dur_token.item() == gt_dur:
                    stats['dur_match'] += 1
                
                # Add note token to generated sequence
                generated = torch.cat([generated, torch.tensor([[note_token]]).to(device)], dim=1)
                
                # Update KV cache with the note token
                outputs = model(generated[:, -1:], past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                gt_idx += 3
                
            else:
                # Not a score triplet, just copy ground truth (shouldn't happen)
                next_token = torch.tensor([[gt_tokens[gt_idx]]]).to(device)
                generated = torch.cat([generated, next_token], dim=1)
                
                if past_key_values is None:
                    outputs = model(generated, use_cache=True)
                else:
                    outputs = model(generated[:, -1:], past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                gt_idx += 1
            
            if generated.shape[1] >= CONTEXT_SIZE:
                break
    
    return generated, stats

# Run evaluation
print("="*80)
print("PITCH-CONSTRAINED GENERATION")
print("="*80)
print()

total_stats = {
    'total_notes': 0,
    'first_attempt_correct': 0,
    'attempts_needed': [],
    'exact_match': 0,
    'time_match': 0,
    'dur_match': 0,
}

for seq_idx in tqdm(range(num_sequences), desc="Generating"):
    tokens = [int(x) for x in lines[seq_idx].strip().split() if x != '|']
    
    # Find where score notes start
    score_start_idx = None
    i = 1  # Skip ANTICIPATE token
    i += 3  # Skip SEP tokens
    
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
    
    # Generate with pitch constraint
    context_tokens = tokens[:score_start_idx]
    input_ids = torch.tensor([context_tokens]).to(device)
    
    generated, stats = pitch_constrained_decode(model, input_ids, tokens, score_start_idx)
    
    # Aggregate stats
    total_stats['total_notes'] += stats['total_notes']
    total_stats['first_attempt_correct'] += stats['first_attempt_correct']
    total_stats['attempts_needed'].extend(stats['attempts_needed'])
    total_stats['exact_match'] += stats['exact_match']
    total_stats['time_match'] += stats['time_match']
    total_stats['dur_match'] += stats['dur_match']

# Print results
print()
print("="*80)
print("RESULTS")
print("="*80)
print(f"Total notes generated: {total_stats['total_notes']}")
print()
print(f"Pitch accuracy: 100.00% (by constraint)")
print(f"Exact triplet match: {100*total_stats['exact_match']/total_stats['total_notes']:.2f}%")
print(f"Time match: {100*total_stats['time_match']/total_stats['total_notes']:.2f}%")
print(f"Duration match: {100*total_stats['dur_match']/total_stats['total_notes']:.2f}%")
print()
print("Pitch constraint stats:")
print(f"  First attempt correct: {100*total_stats['first_attempt_correct']/total_stats['total_notes']:.2f}%")
print(f"  Average attempts needed: {sum(total_stats['attempts_needed'])/len(total_stats['attempts_needed']):.2f}")
print(f"  Max attempts needed: {max(total_stats['attempts_needed'])}")
print()
print("INTERPRETATION:")
print("  - This shows model performance when pitch is constrained to be correct")
print("  - Time/duration accuracy reveals how well model predicts timing when pitch is fixed")
print(f"  - Only {100*total_stats['first_attempt_correct']/total_stats['total_notes']:.1f}% of notes had correct pitch on first attempt")
print(f"  - Time accuracy of {100*total_stats['time_match']/total_stats['total_notes']:.1f}% shows timing prediction quality")
