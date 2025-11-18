"""Debug beam search vs greedy on a single sequence."""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained('150_model')
model = model.to(device)
model.eval()

# Load one sequence
with open('data/test_sliding.txt') as f:
    line = f.readline().strip()
    tokens = [int(t) for t in line.split('|')[0].split()]

# Find score triplets
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

print(f"Sequence has {len(score_triplet_positions)} score triplets")
print(f"Testing on first 3 triplets only\n")

# Take only first 3 triplets for debugging
score_triplet_positions = score_triplet_positions[:3]

first_score_time_pos = score_triplet_positions[0][0]
print(f"First score position: {first_score_time_pos}")
print(f"Context length: {first_score_time_pos}\n")

# === GREEDY ===
print("="*60)
print("GREEDY DECODING")
print("="*60)

init_context = torch.tensor([tokens[:first_score_time_pos]]).to(device)
outputs = model(init_context, past_key_values=None, use_cache=True)
past_key_values = outputs.past_key_values
last_pos = first_score_time_pos

greedy_predictions = []

for triplet_idx, (time_pos, dur_pos, pitch_pos) in enumerate(score_triplet_positions):
    print(f"\nTriplet {triplet_idx}:")
    
    # Handle intermediate tokens
    if time_pos > last_pos:
        intermediate = torch.tensor([tokens[last_pos:time_pos]]).to(device)
        outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
    
    # TIME
    logits = outputs.logits[0, -1]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    pred_time = logits.argmax().item()
    gt_time = tokens[time_pos]
    time_log_prob = log_probs[pred_time].item()
    
    print(f"  TIME: pred={pred_time}, gt={gt_time}, match={pred_time==gt_time}, log_prob={time_log_prob:.4f}")
    
    next_token = torch.tensor([[pred_time]]).to(device)
    outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    
    # DURATION
    logits = outputs.logits[0, -1]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    pred_dur = logits.argmax().item()
    gt_dur = tokens[dur_pos]
    dur_log_prob = log_probs[pred_dur].item()
    
    print(f"  DUR:  pred={pred_dur}, gt={gt_dur}, match={pred_dur==gt_dur}, log_prob={dur_log_prob:.4f}")
    
    next_token = torch.tensor([[pred_dur]]).to(device)
    outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    
    # PITCH
    logits = outputs.logits[0, -1]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    pred_pitch = logits.argmax().item()
    gt_pitch = tokens[pitch_pos]
    pitch_log_prob = log_probs[pred_pitch].item()
    
    print(f"  PITCH: pred={pred_pitch}, gt={gt_pitch}, match={pred_pitch==gt_pitch}, log_prob={pitch_log_prob:.4f}")
    print(f"  Triplet log prob: {time_log_prob + dur_log_prob + pitch_log_prob:.4f}")
    
    greedy_predictions.append((pred_time, pred_dur, pred_pitch))
    
    next_token = torch.tensor([[pred_pitch]]).to(device)
    outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    
    last_pos = pitch_pos + 1

# === BEAM SEARCH (num_beams=5, k=3,2,1) ===
print("\n" + "="*60)
print("BEAM SEARCH (num_beams=5)")
print("="*60)

beams = [(0.0, tokens[:first_score_time_pos])]
last_pos = first_score_time_pos

for triplet_idx, (time_pos, dur_pos, pitch_pos) in enumerate(score_triplet_positions):
    print(f"\nTriplet {triplet_idx}:")
    
    # Add intermediate
    if time_pos > last_pos:
        intermediate = tokens[last_pos:time_pos]
        beams = [(score, seq + intermediate) for score, seq in beams]
    
    k_time, k_dur, k_pitch = 3, 2, 1
    new_beams = []
    
    # Get TIME candidates from each beam
    for beam_idx, (beam_score, beam_seq) in enumerate(beams):
        seq_tensor = torch.tensor([beam_seq], device=device)
        outputs = model(seq_tensor)
        time_logits = outputs.logits[0, -1, :]
        time_log_probs = torch.nn.functional.log_softmax(time_logits, dim=-1)
        
        top_k_time_lp, top_k_time_idx = torch.topk(time_log_probs, k_time)
        
        if beam_idx == 0:
            print(f"  Beam 0 top-{k_time} TIME: {top_k_time_idx.tolist()} (log_probs: {top_k_time_lp.tolist()})")
            print(f"  GT TIME: {tokens[time_pos]}, in top-k: {tokens[time_pos] in top_k_time_idx}")
        
        # Expand TIME
        for time_idx in range(k_time):
            time_token = top_k_time_idx[time_idx].item()
            time_lp = top_k_time_lp[time_idx].item()
            
            seq_with_time = beam_seq + [time_token]
            score_with_time = beam_score + time_lp
            
            # Get DURATION candidates
            seq_tensor = torch.tensor([seq_with_time], device=device)
            outputs = model(seq_tensor)
            dur_logits = outputs.logits[0, -1, :]
            dur_log_probs = torch.nn.functional.log_softmax(dur_logits, dim=-1)
            
            top_k_dur_lp, top_k_dur_idx = torch.topk(dur_log_probs, k_dur)
            
            # Expand DURATION
            for dur_idx in range(k_dur):
                dur_token = top_k_dur_idx[dur_idx].item()
                dur_lp = top_k_dur_lp[dur_idx].item()
                
                seq_with_dur = seq_with_time + [dur_token]
                score_with_dur = score_with_time + dur_lp
                
                # Get PITCH candidates
                seq_tensor = torch.tensor([seq_with_dur], device=device)
                outputs = model(seq_tensor)
                pitch_logits = outputs.logits[0, -1, :]
                pitch_log_probs = torch.nn.functional.log_softmax(pitch_logits, dim=-1)
                
                top_k_pitch_lp, top_k_pitch_idx = torch.topk(pitch_log_probs, k_pitch)
                
                # Expand PITCH
                for pitch_idx in range(k_pitch):
                    pitch_token = top_k_pitch_idx[pitch_idx].item()
                    pitch_lp = top_k_pitch_lp[pitch_idx].item()
                    
                    final_seq = seq_with_dur + [pitch_token]
                    final_score = score_with_dur + pitch_lp
                    
                    new_beams.append((final_score, final_seq))
    
    new_beams.sort(key=lambda x: x[0], reverse=True)
    beams = new_beams[:5]
    
    print(f"  Generated {len(new_beams)} candidates, kept top 5")
    print(f"  Best beam score: {beams[0][0]:.4f}")
    
    last_pos = pitch_pos + 1

# Extract predictions from best beam
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)

best_score, best_seq = beams[0]
print(f"\nBest beam total score: {best_score:.4f}")
print(f"Best beam sequence length: {len(best_seq)}")

pred_idx = first_score_time_pos
prev_pos = first_score_time_pos

beam_predictions = []
for time_pos, dur_pos, pitch_pos in score_triplet_positions:
    if time_pos > prev_pos:
        pred_idx += (time_pos - prev_pos)
    
    pred_time = best_seq[pred_idx]
    pred_dur = best_seq[pred_idx + 1]
    pred_pitch = best_seq[pred_idx + 2]
    
    beam_predictions.append((pred_time, pred_dur, pred_pitch))
    
    pred_idx += 3
    prev_pos = pitch_pos + 1

print("\nPredictions:")
print(f"{'Triplet':<10} {'GT':<20} {'Greedy':<20} {'Beam':<20} {'Greedy Match':<15} {'Beam Match'}")
print("-" * 100)
for i, (time_pos, dur_pos, pitch_pos) in enumerate(score_triplet_positions):
    gt = (tokens[time_pos], tokens[dur_pos], tokens[pitch_pos])
    greedy = greedy_predictions[i]
    beam = beam_predictions[i]
    
    greedy_match = "✓" if greedy == gt else "✗"
    beam_match = "✓" if beam == gt else "✗"
    
    print(f"{i:<10} {str(gt):<20} {str(greedy):<20} {str(beam):<20} {greedy_match:<15} {beam_match}")
