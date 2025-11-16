"""
Evaluate sliding-model with autoregressive score prediction.
Controls (performance) are given as ground truth, model predicts score tokens.
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

print("="*80)
print("AUTOREGRESSIVE EVALUATION OF SLIDING-MODEL")
print("="*80)
print("Controls (performance) = Ground Truth (given)")
print("Score tokens = Model Predictions (autoregressive)")
print("="*80)

# Load model
print("\nLoading model from sliding-model/...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained('sliding-model/')
model = model.to(device)
model.eval()
print(f"Model loaded on {device}\n")

# Load test data
print("Loading test data from data/test_sliding.txt...")
with open('data/test_sliding.txt', 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Total test sequences: {len(lines)}")

# Sample a subset for faster evaluation
random.seed(42)
sample_size = min(15, len(lines))  # Start with just 15 sequences
lines = random.sample(lines, sample_size)
print(f"Sampling {sample_size} sequences for evaluation\n")

# Statistics trackers
stats = {
    'score_time': {'correct': 0, 'total': 0, 'errors': []},
    'score_dur': {'correct': 0, 'total': 0, 'errors': []},
    'score_pitch': {'correct': 0, 'total': 0, 'errors': []},
}

# Process sequences
print("Processing sequences autoregressively...")
for line in tqdm(lines):
    # Parse tokens
    if '|' in line:
        token_part = line.split('|')[0].strip()
    else:
        token_part = line
    
    tokens = [int(t) for t in token_part.split()]
    
    # Find all triplet positions with their types
    all_triplets = []  # (type, time_pos, dur_pos, pitch_pos)
    
    i = 1  # Skip mode token
    while i < len(tokens) - 2:
        if (tokens[i] < CONTROL_OFFSET and 
            tokens[i+1] < CONTROL_OFFSET and 
            tokens[i+2] < CONTROL_OFFSET):
            all_triplets.append(('score', i, i+1, i+2))
            i += 3
        elif (tokens[i] >= CONTROL_OFFSET and 
              tokens[i+1] >= CONTROL_OFFSET and 
              tokens[i+2] >= CONTROL_OFFSET and
              tokens[i] < CONTROL_OFFSET + 27512):
            all_triplets.append(('control', i, i+1, i+2))
            i += 3
        else:
            i += 1
    
    if not all_triplets:
        continue
    
    # Autoregressive evaluation
    # Feed controls as ground truth, predict scores autoregressively
    with torch.no_grad():
        past_key_values = None
        last_pos = 0
        
        for triplet_type, time_pos, dur_pos, pitch_pos in all_triplets:
            if triplet_type == 'control':
                # Control triplet - feed all 3 tokens as ground truth
                if time_pos > last_pos:
                    # Feed tokens from last_pos to pitch_pos (inclusive)
                    context = torch.tensor([tokens[last_pos:pitch_pos+1]], dtype=torch.long).to(device)
                    outputs = model(context, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                    last_pos = pitch_pos + 1
                else:
                    # Already at this position, feed the triplet
                    context = torch.tensor([tokens[time_pos:pitch_pos+1]], dtype=torch.long).to(device)
                    outputs = model(context, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                    last_pos = pitch_pos + 1
                    
            else:  # score triplet
                # Score triplet - predict each token autoregressively
                
                # Predict time token
                if time_pos > last_pos:
                    # Feed intermediate tokens first
                    context = torch.tensor([tokens[last_pos:time_pos]], dtype=torch.long).to(device)
                    outputs = model(context, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                
                # Now predict time token
                logits = outputs.logits[0, -1]
                pred_time = logits.argmax().item()
                gt_time = tokens[time_pos]
                
                error = abs(pred_time - gt_time)
                stats['score_time']['errors'].append(error)
                if pred_time == gt_time:
                    stats['score_time']['correct'] += 1
                stats['score_time']['total'] += 1
                
                # Feed predicted time token
                time_token = torch.tensor([[pred_time]], dtype=torch.long).to(device)
                outputs = model(time_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                # Predict duration token
                logits = outputs.logits[0, -1]
                pred_dur = logits.argmax().item()
                gt_dur = tokens[dur_pos]
                
                error = abs(pred_dur - gt_dur)
                stats['score_dur']['errors'].append(error)
                if pred_dur == gt_dur:
                    stats['score_dur']['correct'] += 1
                stats['score_dur']['total'] += 1
                
                # Feed predicted duration token
                dur_token = torch.tensor([[pred_dur]], dtype=torch.long).to(device)
                outputs = model(dur_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                # Predict pitch token
                logits = outputs.logits[0, -1]
                pred_pitch = logits.argmax().item()
                gt_pitch = tokens[pitch_pos]
                
                error = abs(pred_pitch - gt_pitch)
                stats['score_pitch']['errors'].append(error)
                if pred_pitch == gt_pitch:
                    stats['score_pitch']['correct'] += 1
                stats['score_pitch']['total'] += 1
                
                # Feed predicted pitch token
                pitch_token = torch.tensor([[pred_pitch]], dtype=torch.long).to(device)
                outputs = model(pitch_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                last_pos = pitch_pos + 1

# Print results
print("\n" + "="*80)
print("RESULTS - AUTOREGRESSIVE SCORE PREDICTION")
print("="*80)
print("(Controls given as ground truth, scores predicted autoregressively)")
print()

for token_type in ['score_time', 'score_dur', 'score_pitch']:
    correct = stats[token_type]['correct']
    total = stats[token_type]['total']
    accuracy = (correct / total * 100) if total > 0 else 0
    mean_error = np.mean(stats[token_type]['errors']) if stats[token_type]['errors'] else 0
    median_error = np.median(stats[token_type]['errors']) if stats[token_type]['errors'] else 0
    print(f"  {token_type:20s}: {accuracy:6.2f}% ({correct:6d}/{total:6d}) | Mean err: {mean_error:7.2f} | Median err: {median_error:6.0f}")

print("\nOVERALL:")
total_correct = sum(s['correct'] for s in stats.values())
total_tokens = sum(s['total'] for s in stats.values())
overall_accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0
print(f"  All score tokens:    {overall_accuracy:6.2f}% ({total_correct:6d}/{total_tokens:6d})")

# Create error distribution plots
print("\n" + "="*80)
print("Generating error distribution plots...")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Autoregressive Score Token Error Distribution - Sliding Model', fontsize=16)

token_types = ['score_time', 'score_dur', 'score_pitch']
titles = ['Score Time', 'Score Duration', 'Score Pitch']

for idx, (token_type, title) in enumerate(zip(token_types, titles)):
    ax = axes[idx]
    
    errors = stats[token_type]['errors']
    if errors:
        # Create histogram
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.1f}')
        ax.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.0f}')
        
        accuracy = (stats[token_type]['correct'] / stats[token_type]['total'] * 100) if stats[token_type]['total'] > 0 else 0
        ax.set_title(f'{title}\nAccuracy: {accuracy:.2f}%', fontsize=12)
        ax.set_xlabel('Absolute Error (tokens)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)

plt.tight_layout()
plt.savefig('sliding_model_autoregressive_score_errors.png', dpi=150, bbox_inches='tight')
print(f"\nSaved error distribution plot to: sliding_model_autoregressive_score_errors.png")

# Create cumulative error plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.suptitle('Cumulative Error Distribution - Autoregressive Score Tokens', fontsize=16)

for token_type, label in [('score_time', 'Time'), ('score_dur', 'Duration'), ('score_pitch', 'Pitch')]:
    errors = sorted(stats[token_type]['errors'])
    if errors:
        cumulative = np.arange(1, len(errors) + 1) / len(errors) * 100
        ax.plot(errors, cumulative, label=label, linewidth=2)

ax.set_xlabel('Absolute Error (tokens)')
ax.set_ylabel('Cumulative Percentage (%)')
ax.legend()
ax.grid(True, alpha=0.3)
max_error = max([max(stats[t]['errors']) for t in token_types if stats[t]['errors']])
ax.set_xlim([0, min(100, max_error)])

plt.tight_layout()
plt.savefig('sliding_model_autoregressive_score_cumulative.png', dpi=150, bbox_inches='tight')
print(f"Saved cumulative error plot to: sliding_model_autoregressive_score_cumulative.png")

print("\n" + "="*80)
print("Note: This is TRUE AUTOREGRESSIVE evaluation:")
print("  - Control tokens fed as ground truth (performance given)")
print("  - Score tokens predicted and fed back (errors accumulate)")
print("="*80)
