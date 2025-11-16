"""
Autoregressive evaluation where we force the pitch to ground truth.
Only TIME and DURATION are predicted autoregressively.
This tests if pitch errors were cascading and affecting timing predictions.
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET
import random
from tqdm import tqdm

print("="*80)
print("AUTOREGRESSIVE EVALUATION - PITCH FORCED TO GROUND TRUTH")
print("="*80)

# Load model
print("\nLoading model from opening_model/...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained('opening_model/')
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Load test data
print("\nLoading test data from data/test_openings.txt...")
with open('data/test_openings.txt', 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

# Sample sequences
sample_size = min(15, len(lines))
random.seed(42)
sampled_indices = random.sample(range(len(lines)), sample_size)
sampled_lines = [lines[i] for i in sampled_indices]

print(f"Sampled {sample_size} sequences for evaluation")

# Statistics trackers
stats = {
    'score_time': {'correct': 0, 'total': 0, 'errors': []},
    'score_dur': {'correct': 0, 'total': 0, 'errors': []},
    'score_pitch': {'correct': 0, 'total': 0, 'errors': []},  # Should be 100% since forced
}

print(f"\nProcessing sequences (pitch forced to ground truth)...")

for seq_idx, line in enumerate(tqdm(sampled_lines, desc="Evaluating", unit="seq")):
    if '|' in line:
        token_part = line.split('|')[0].strip()
    else:
        token_part = line
    
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
    
    if not score_triplet_positions:
        continue
    
    # Skip REST prefix, focus on actual notes
    note_triplets = [(t, d, p) for t, d, p in score_triplet_positions 
                     if tokens[p] != 27512]
    
    if not note_triplets:
        continue
    
    # Autoregressive evaluation with PITCH FORCED
    with torch.no_grad():
        first_score_time_pos = score_triplet_positions[0][0]
        init_context = torch.tensor([tokens[:first_score_time_pos]]).to(device)
        outputs = model(init_context, past_key_values=None, use_cache=True)
        past_key_values = outputs.past_key_values
        last_pos = first_score_time_pos
        
        for time_pos, dur_pos, pitch_pos in score_triplet_positions:
            is_rest = (tokens[pitch_pos] == 27512)
            
            # Only track actual notes
            if is_rest:
                # Still process REST for KV cache
                if time_pos > last_pos:
                    intermediate = torch.tensor([tokens[last_pos:time_pos]]).to(device)
                    outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                
                # For REST, use ground truth for all
                for pos in [time_pos, dur_pos, pitch_pos]:
                    gt_token = tokens[pos]
                    next_token = torch.tensor([[gt_token]]).to(device)
                    outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                
                last_pos = pitch_pos + 1
                continue
            
            # Process intermediate control tokens
            if time_pos > last_pos:
                intermediate = torch.tensor([tokens[last_pos:time_pos]]).to(device)
                outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
            
            # Predict TIME
            logits = outputs.logits[0, -1]
            pred_time = logits.argmax().item()
            gt_time = tokens[time_pos]
            
            if pred_time == gt_time:
                stats['score_time']['correct'] += 1
            else:
                stats['score_time']['errors'].append(abs(pred_time - gt_time))
            stats['score_time']['total'] += 1
            
            # Feed predicted time back
            next_token = torch.tensor([[pred_time]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Predict DURATION
            logits = outputs.logits[0, -1]
            pred_dur = logits.argmax().item()
            gt_dur = tokens[dur_pos]
            
            if pred_dur == gt_dur:
                stats['score_dur']['correct'] += 1
            else:
                stats['score_dur']['errors'].append(abs(pred_dur - gt_dur))
            stats['score_dur']['total'] += 1
            
            # Feed predicted duration back
            next_token = torch.tensor([[pred_dur]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # *** FORCE PITCH TO GROUND TRUTH ***
            gt_pitch = tokens[pitch_pos]
            
            # Check what model would have predicted
            logits = outputs.logits[0, -1]
            pred_pitch = logits.argmax().item()
            
            if pred_pitch == gt_pitch:
                stats['score_pitch']['correct'] += 1
            stats['score_pitch']['total'] += 1
            
            # Feed GROUND TRUTH pitch back (FORCED)
            next_token = torch.tensor([[gt_pitch]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            last_pos = pitch_pos + 1

# Print results
print("\n" + "="*80)
print("RESULTS - PITCH FORCED TO GROUND TRUTH")
print("="*80)

print("\nScore TIME (autoregressive):")
time_correct = stats['score_time']['correct']
time_total = stats['score_time']['total']
time_acc = (time_correct / time_total * 100) if time_total > 0 else 0
time_mean_err = sum(stats['score_time']['errors']) / len(stats['score_time']['errors']) if stats['score_time']['errors'] else 0
time_median_err = sorted(stats['score_time']['errors'])[len(stats['score_time']['errors'])//2] if stats['score_time']['errors'] else 0
print(f"  Accuracy: {time_acc:6.2f}% ({time_correct}/{time_total})")
print(f"  Mean error: {time_mean_err:.2f}, Median error: {time_median_err}")

print("\nScore DURATION (autoregressive):")
dur_correct = stats['score_dur']['correct']
dur_total = stats['score_dur']['total']
dur_acc = (dur_correct / dur_total * 100) if dur_total > 0 else 0
dur_mean_err = sum(stats['score_dur']['errors']) / len(stats['score_dur']['errors']) if stats['score_dur']['errors'] else 0
dur_median_err = sorted(stats['score_dur']['errors'])[len(stats['score_dur']['errors'])//2] if stats['score_dur']['errors'] else 0
print(f"  Accuracy: {dur_acc:6.2f}% ({dur_correct}/{dur_total})")
print(f"  Mean error: {dur_mean_err:.2f}, Median error: {dur_median_err}")

print("\nScore PITCH (forced to ground truth, but showing what model predicted):")
pitch_correct = stats['score_pitch']['correct']
pitch_total = stats['score_pitch']['total']
pitch_acc = (pitch_correct / pitch_total * 100) if pitch_total > 0 else 0
print(f"  Model would have predicted: {pitch_acc:6.2f}% ({pitch_correct}/{pitch_total})")
print(f"  But we forced it to 100% by using ground truth")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print("\nRECALL - Normal autoregressive (from previous run):")
print("  Score Time:     ~1.5-2.2% (sequences 1-4), 98.5% (sequence 5)")
print("  Score Duration: ~0-21% (sequences 1-4), 83.2% (sequence 5)")
print("  Score Pitch:    ~10-75% (sequences 1-4), 100% (sequence 5)")
print()
print("WITH PITCH FORCED (this run):")
print(f"  Score Time:     {time_acc:6.2f}%")
print(f"  Score Duration: {dur_acc:6.2f}%")
print(f"  Score Pitch:    100% (forced)")
print()

if time_acc > 5:
    print("FINDING: Forcing pitch IMPROVED timing predictions!")
    print("This suggests pitch errors were cascading and corrupting context.")
else:
    print("FINDING: Forcing pitch did NOT improve timing predictions significantly.")
    print("This suggests the timing prediction problem is fundamental,")
    print("not caused by cascading pitch errors.")

print("="*80)
