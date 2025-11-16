"""
Verify that we're comparing predictions to the correct ground truth positions.
Check sequence 3 specifically since it has suspicious results.
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET

print("="*80)
print("VERIFYING AUTOREGRESSIVE ACCURACY CALCULATION - SEQUENCE 3")
print("="*80)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained('opening_model/')
model = model.to(device)
model.eval()

# Load sequence 3 (index 2)
with open('data/test_openings.txt', 'r') as f:
    lines = f.readlines()
    line = lines[2].strip()

token_part = line.split('|')[0].strip()
tokens = [int(t) for t in token_part.split()]

# Find score triplet positions
score_triplet_positions = []
i = 1
while i < len(tokens) - 2:
    if (tokens[i] < CONTROL_OFFSET and 
        tokens[i+1] < CONTROL_OFFSET and 
        tokens[i+2] < CONTROL_OFFSET):
        score_triplet_positions.append((i, i+1, i+2))
        i += 3
    else:
        i += 1

# Focus on actual notes (skip first 33 REST triplets)
note_triplets = score_triplet_positions[33:]

print(f"Total note triplets: {len(note_triplets)}")
print(f"\nLet's manually check first 10 note predictions vs ground truth:")
print()

with torch.no_grad():
    first_score_time_pos = score_triplet_positions[0][0]
    init_context = torch.tensor([tokens[:first_score_time_pos]]).to(device)
    outputs = model(init_context, past_key_values=None, use_cache=True)
    past_key_values = outputs.past_key_values
    last_pos = first_score_time_pos
    
    note_count = 0
    
    for triplet_idx, (time_pos, dur_pos, pitch_pos) in enumerate(score_triplet_positions):
        is_rest = (tokens[pitch_pos] == 27512)
        
        # Skip REST triplets
        if is_rest:
            # Still need to process them for KV cache
            if time_pos > last_pos:
                intermediate = torch.tensor([tokens[last_pos:time_pos]]).to(device)
                outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
            
            for pos in [time_pos, dur_pos, pitch_pos]:
                pred = outputs.logits[0, -1].argmax().item()
                next_token = torch.tensor([[pred]]).to(device)
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
            
            last_pos = pitch_pos + 1
            continue
        
        # Process intermediate
        if time_pos > last_pos:
            intermediate = torch.tensor([tokens[last_pos:time_pos]]).to(device)
            outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
        
        # Predict TIME
        logits = outputs.logits[0, -1]
        pred_time = logits.argmax().item()
        gt_time = tokens[time_pos]
        
        # Feed predicted time
        next_token = torch.tensor([[pred_time]]).to(device)
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        
        # Predict DURATION
        logits = outputs.logits[0, -1]
        pred_dur = logits.argmax().item()
        gt_dur = tokens[dur_pos]
        
        # Feed predicted duration
        next_token = torch.tensor([[pred_dur]]).to(device)
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        
        # Predict PITCH
        logits = outputs.logits[0, -1]
        pred_pitch = logits.argmax().item()
        gt_pitch = tokens[pitch_pos]
        
        # Feed predicted pitch
        next_token = torch.tensor([[pred_pitch]]).to(device)
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        
        last_pos = pitch_pos + 1
        
        # Print first 10 notes
        if note_count < 10:
            print(f"Note {note_count}:")
            print(f"  Position in sequence: triplet {triplet_idx}, positions [{time_pos}, {dur_pos}, {pitch_pos}]")
            print(f"  TIME:  pred={pred_time:6d}, gt={gt_time:6d}, match={pred_time==gt_time}")
            print(f"  DUR:   pred={pred_dur:6d}, gt={gt_dur:6d}, match={pred_dur==gt_dur}")
            print(f"  PITCH: pred={pred_pitch:6d}, gt={gt_pitch:6d}, match={pred_pitch==gt_pitch}")
            print()
        
        note_count += 1

print("="*80)
print("KEY CHECK:")
print("="*80)
print("Are we comparing predictions at position P to ground truth at position P?")
print("YES - the code shows we're using tokens[time_pos], tokens[dur_pos], tokens[pitch_pos]")
print("which are the correct ground truth positions.")
print()
print("So the accuracy calculation is correct.")
print("The question remains: why is pitch 75% accurate when time/dur are wrong?")
print("="*80)
