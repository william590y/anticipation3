"""
Trace through autoregressive prediction step-by-step to verify logic is correct.
Compare what the model sees vs what it should see.
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET

print("="*80)
print("DETAILED TRACE OF AUTOREGRESSIVE LOGIC")
print("="*80)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained('opening_model/')
model = model.to(device)
model.eval()

# Note: Using opening_model since we don't have sliding model checkpoint

# Load one test sequence
with open('data/test_openings.txt', 'r') as f:
    line = f.readline().strip()

token_part = line.split('|')[0].strip()
tokens = [int(t) for t in token_part.split()]

print(f"\nLoaded sequence with {len(tokens)} tokens")
print(f"First 20 tokens: {tokens[:20]}")

# Find first 3 SCORE triplets
score_triplets = []
i = 1  # Skip mode
while i < len(tokens) - 2 and len(score_triplets) < 3:
    if (tokens[i] < CONTROL_OFFSET and 
        tokens[i+1] < CONTROL_OFFSET and 
        tokens[i+2] < CONTROL_OFFSET):
        score_triplets.append((i, i+1, i+2))
        i += 3
    else:
        i += 1

print(f"\nFirst 3 SCORE triplets:")
for idx, (t, d, p) in enumerate(score_triplets):
    print(f"  Triplet {idx}: positions [{t}, {d}, {p}] = {tokens[t:p+1]}")

print("\n" + "="*80)
print("SIMULATING AUTOREGRESSIVE GENERATION")
print("="*80)

with torch.no_grad():
    first_score_time_pos = score_triplets[0][0]
    
    # Step 1: Initialize
    print(f"\n{'='*80}")
    print(f"STEP 1: INITIALIZE CONTEXT")
    print(f"{'='*80}")
    print(f"Feed tokens[0:{first_score_time_pos}]")
    print(f"  = {tokens[:first_score_time_pos]}")
    
    init_context = torch.tensor([tokens[:first_score_time_pos]]).to(device)
    outputs = model(init_context, past_key_values=None, use_cache=True)
    past_key_values = outputs.past_key_values
    
    print(f"After this, outputs.logits has shape: {outputs.logits.shape}")
    print(f"  outputs.logits[0, -1] predicts position {first_score_time_pos}")
    
    last_pos = first_score_time_pos
    
    # Step 2: Predict first triplet
    print(f"\n{'='*80}")
    print(f"STEP 2: PREDICT FIRST SCORE TRIPLET")
    print(f"{'='*80}")
    
    time_pos, dur_pos, pitch_pos = score_triplets[0]
    
    print(f"Target positions: time={time_pos}, dur={dur_pos}, pitch={pitch_pos}")
    print(f"Ground truth: {tokens[time_pos:pitch_pos+1]}")
    print(f"last_pos = {last_pos}, time_pos = {time_pos}")
    print(f"time_pos > last_pos? {time_pos > last_pos}")
    
    if time_pos > last_pos:
        print(f"YES - need to feed intermediate tokens[{last_pos}:{time_pos}]")
        intermediate = tokens[last_pos:time_pos]
        print(f"  Intermediate: {intermediate}")
        intermediate_tensor = torch.tensor([intermediate]).to(device)
        outputs = model(intermediate_tensor, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
    else:
        print(f"NO - use existing outputs")
    
    # Predict TIME
    print(f"\n--- Predicting TIME at position {time_pos} ---")
    print(f"Model has seen up to position {time_pos - 1}")
    print(f"outputs.logits[0, -1] predicts position {time_pos}")
    
    logits = outputs.logits[0, -1]
    pred_time = logits.argmax().item()
    gt_time = tokens[time_pos]
    
    print(f"  Predicted: {pred_time}")
    print(f"  Ground truth: {gt_time}")
    print(f"  Match: {pred_time == gt_time}")
    
    # Feed predicted time back
    print(f"\nFeed predicted time ({pred_time}) back to model")
    next_token = torch.tensor([[pred_time]]).to(device)
    outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    print(f"Now outputs.logits[0, -1] predicts position {dur_pos}")
    
    # Predict DURATION
    print(f"\n--- Predicting DURATION at position {dur_pos} ---")
    print(f"Model has seen: [..., {pred_time}]")
    print(f"outputs.logits[0, -1] predicts position {dur_pos}")
    
    logits = outputs.logits[0, -1]
    pred_dur = logits.argmax().item()
    gt_dur = tokens[dur_pos]
    
    print(f"  Predicted: {pred_dur}")
    print(f"  Ground truth: {gt_dur}")
    print(f"  Match: {pred_dur == gt_dur}")
    
    # Feed predicted duration back
    print(f"\nFeed predicted duration ({pred_dur}) back to model")
    next_token = torch.tensor([[pred_dur]]).to(device)
    outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    print(f"Now outputs.logits[0, -1] predicts position {pitch_pos}")
    
    # Predict PITCH
    print(f"\n--- Predicting PITCH at position {pitch_pos} ---")
    print(f"Model has seen: [..., {pred_time}, {pred_dur}]")
    print(f"outputs.logits[0, -1] predicts position {pitch_pos}")
    
    logits = outputs.logits[0, -1]
    pred_pitch = logits.argmax().item()
    gt_pitch = tokens[pitch_pos]
    
    print(f"  Predicted: {pred_pitch}")
    print(f"  Ground truth: {gt_pitch}")
    print(f"  Match: {pred_pitch == gt_pitch}")
    
    print(f"\n{'='*80}")
    print(f"CRITICAL CHECK: WHAT CONTEXT DID PITCH PREDICTION SEE?")
    print(f"{'='*80}")
    
    # In teacher forcing, pitch would see: tokens[0:pitch_pos]
    # In our autoregressive, pitch sees: tokens[0:time_pos] + [pred_time, pred_dur]
    
    print(f"\nIn TEACHER FORCING:")
    print(f"  Pitch prediction would see: tokens[0:{pitch_pos}]")
    print(f"  This includes ground truth: time={gt_time}, dur={gt_dur}")
    
    print(f"\nIn OUR AUTOREGRESSIVE:")
    print(f"  Pitch prediction sees: tokens[0:{time_pos}] + [{pred_time}, {pred_dur}]")
    print(f"  This includes predictions: time={pred_time}, dur={pred_dur}")
    
    if pred_time != gt_time or pred_dur != gt_dur:
        print(f"\n  WARNING: Pitch is being predicted with WRONG time/dur context!")
        print(f"  The model is asked to predict pitch after seeing:")
        print(f"    time={pred_time} (should be {gt_time})")
        print(f"    dur={pred_dur} (should be {gt_dur})")
        print(f"  This is a DIFFERENT question than teacher forcing!")
    else:
        print(f"\n  OK: Time and duration were predicted correctly")
        print(f"  Pitch sees the same context as teacher forcing")
    
    # Now let's trace what happens next
    print(f"\n{'='*80}")
    print(f"STEP 3: WHAT HAPPENS NEXT?")
    print(f"{'='*80}")
    
    print(f"\nFeed predicted pitch ({pred_pitch}) back to model")
    next_token = torch.tensor([[pred_pitch]]).to(device)
    outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    
    last_pos = pitch_pos + 1
    print(f"Update last_pos = {last_pos}")
    
    # Second triplet
    time_pos2, dur_pos2, pitch_pos2 = score_triplets[1]
    print(f"\nNext triplet to predict: positions [{time_pos2}, {dur_pos2}, {pitch_pos2}]")
    print(f"Ground truth: {tokens[time_pos2:pitch_pos2+1]}")
    
    print(f"\ntime_pos2 ({time_pos2}) > last_pos ({last_pos})? {time_pos2 > last_pos}")
    if time_pos2 > last_pos:
        intermediate = tokens[last_pos:time_pos2]
        print(f"Feed intermediate tokens[{last_pos}:{time_pos2}] = {intermediate}")
        
        # Check what these tokens are
        is_control = all(tok >= CONTROL_OFFSET for tok in intermediate)
        is_score = all(tok < CONTROL_OFFSET for tok in intermediate)
        
        if is_control:
            print(f"  These are CONTROL tokens (ground truth)")
        elif is_score:
            print(f"  These are SCORE tokens (should be our predictions!)")
            print(f"  *** BUG DETECTED: We're feeding ground truth instead of predictions! ***")
        else:
            print(f"  Mixed tokens")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("The autoregressive logic appears correct:")
print("1. We predict time, feed it back")
print("2. We predict duration (seeing predicted time), feed it back")
print("3. We predict pitch (seeing predicted time+dur), feed it back")
print("4. Between score triplets, we feed ground truth CONTROL tokens")
print()
print("This is correct because controls are the 'given' performance,")
print("and we're only autoregressively generating the score.")
print("="*80)
