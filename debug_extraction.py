"""Debug beam extraction logic."""
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

# Take first 5 triplets
score_triplet_positions = score_triplet_positions[:5]
first_score_time_pos = score_triplet_positions[0][0]

print(f"First score position: {first_score_time_pos}")
print(f"Score triplet positions: {score_triplet_positions}")
print()

# Build beam manually
beams = [(0.0, tokens[:first_score_time_pos])]
last_pos = first_score_time_pos

print("Building beam sequences...")
for triplet_idx, (time_pos, dur_pos, pitch_pos) in enumerate(score_triplet_positions):
    # Add intermediate
    if time_pos > last_pos:
        intermediate = tokens[last_pos:time_pos]
        beams = [(score, seq + intermediate) for score, seq in beams]
        print(f"Triplet {triplet_idx}: Added {len(intermediate)} intermediate tokens")
    
    # Just take greedy for simplicity
    beam_seq = beams[0][1]
    
    # TIME
    seq_tensor = torch.tensor([beam_seq], device=device)
    outputs = model(seq_tensor)
    pred_time = outputs.logits[0, -1].argmax().item()
    
    # DURATION
    seq_tensor = torch.tensor([beam_seq + [pred_time]], device=device)
    outputs = model(seq_tensor)
    pred_dur = outputs.logits[0, -1].argmax().item()
    
    # PITCH
    seq_tensor = torch.tensor([beam_seq + [pred_time, pred_dur]], device=device)
    outputs = model(seq_tensor)
    pred_pitch = outputs.logits[0, -1].argmax().item()
    
    # Update beam
    beams = [(0.0, beam_seq + [pred_time, pred_dur, pred_pitch])]
    
    print(f"  Added triplet: TIME={pred_time}, DUR={pred_dur}, PITCH={pred_pitch}")
    print(f"  Sequence length: {len(beams[0][1])}")
    
    last_pos = pitch_pos + 1

# Now extract using the same logic as analyze_triplet_beam_search.py
print("\n" + "="*60)
print("EXTRACTION")
print("="*60)

best_score, best_seq = beams[0]
print(f"Best sequence length: {len(best_seq)}")
print(f"First score time pos: {first_score_time_pos}")
print()

pred_idx = first_score_time_pos
prev_pos = first_score_time_pos

extracted_predictions = []

for triplet_idx, (time_pos, dur_pos, pitch_pos) in enumerate(score_triplet_positions):
    print(f"\nTriplet {triplet_idx}:")
    print(f"  Original positions: time={time_pos}, dur={dur_pos}, pitch={pitch_pos}")
    print(f"  prev_pos={prev_pos}, pred_idx={pred_idx}")
    
    if time_pos > prev_pos:
        num_intermediate = time_pos - prev_pos
        print(f"  Skipping {num_intermediate} intermediate tokens")
        pred_idx += num_intermediate
    
    print(f"  Extracting from indices: {pred_idx}, {pred_idx+1}, {pred_idx+2}")
    
    # Extract
    pred_time = best_seq[pred_idx]
    pred_dur = best_seq[pred_idx + 1]
    pred_pitch = best_seq[pred_idx + 2]
    
    # Ground truth
    gt_time = tokens[time_pos]
    gt_dur = tokens[dur_pos]
    gt_pitch = tokens[pitch_pos]
    
    print(f"  Predicted: TIME={pred_time}, DUR={pred_dur}, PITCH={pred_pitch}")
    print(f"  Ground truth: TIME={gt_time}, DUR={gt_dur}, PITCH={gt_pitch}")
    print(f"  Match: TIME={pred_time==gt_time}, DUR={pred_dur==gt_dur}, PITCH={pred_pitch==gt_pitch}")
    
    extracted_predictions.append((pred_time, pred_dur, pred_pitch))
    
    pred_idx += 3
    prev_pos = pitch_pos + 1

print("\n" + "="*60)
print("Extraction looks correct!" if all(
    best_seq[first_score_time_pos + i*3:first_score_time_pos + i*3 + 3] == 
    list(extracted_predictions[i]) 
    for i in range(len(extracted_predictions))
) else "ERROR in extraction!")
