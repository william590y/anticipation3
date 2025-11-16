"""
Evaluate sliding-model on test set with detailed statistics and error visualization.
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print("DETAILED EVALUATION OF SLIDING-MODEL ON TEST SET")
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
import random
random.seed(42)
sample_size = min(1000, len(lines))
lines = random.sample(lines, sample_size)
print(f"Sampling {sample_size} sequences for evaluation\n")

# Statistics trackers
stats = {
    'control_time': {'correct': 0, 'total': 0, 'errors': []},
    'control_dur': {'correct': 0, 'total': 0, 'errors': []},
    'control_pitch': {'correct': 0, 'total': 0, 'errors': []},
    'score_time': {'correct': 0, 'total': 0, 'errors': []},
    'score_dur': {'correct': 0, 'total': 0, 'errors': []},
    'score_pitch': {'correct': 0, 'total': 0, 'errors': []},
}

# Process sequences
print("Processing sequences...")
for line in tqdm(lines):
    # Parse tokens
    if '|' in line:
        token_part = line.split('|')[0].strip()
    else:
        token_part = line
    
    tokens = [int(t) for t in token_part.split()]
    
    # Find all triplet positions
    score_triplets = []
    control_triplets = []
    
    i = 1  # Skip mode token
    while i < len(tokens) - 2:
        if (tokens[i] < CONTROL_OFFSET and 
            tokens[i+1] < CONTROL_OFFSET and 
            tokens[i+2] < CONTROL_OFFSET):
            score_triplets.append((i, i+1, i+2))
            i += 3
        elif (tokens[i] >= CONTROL_OFFSET and 
              tokens[i+1] >= CONTROL_OFFSET and 
              tokens[i+2] >= CONTROL_OFFSET and
              tokens[i] < CONTROL_OFFSET + 27512):
            control_triplets.append((i, i+1, i+2))
            i += 3
        else:
            i += 1
    
    # Collect all positions to predict
    all_positions = []
    for time_pos, dur_pos, pitch_pos in score_triplets:
        all_positions.extend([
            ('score_time', time_pos),
            ('score_dur', dur_pos),
            ('score_pitch', pitch_pos)
        ])
    for time_pos, dur_pos, pitch_pos in control_triplets:
        all_positions.extend([
            ('control_time', time_pos),
            ('control_dur', dur_pos),
            ('control_pitch', pitch_pos)
        ])
    
    all_positions.sort(key=lambda x: x[1])
    
    if not all_positions:
        continue
    
    # Teacher forcing evaluation
    with torch.no_grad():
        # Feed all tokens in one go for efficiency
        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        
        # For each position, check if prediction at pos-1 matches ground truth at pos
        for token_type, pos in all_positions:
            if pos > 0 and pos < len(tokens):
                # Prediction for position pos comes from logits[pos-1]
                predicted_token = logits[pos - 1].argmax().item()
                ground_truth = tokens[pos]
                
                # Calculate error (absolute difference in token values)
                error = abs(predicted_token - ground_truth)
                stats[token_type]['errors'].append(error)
                
                if predicted_token == ground_truth:
                    stats[token_type]['correct'] += 1
                stats[token_type]['total'] += 1

# Print results
print("\n" + "="*80)
print("RESULTS - TEACHER FORCING ACCURACY (per token type)")
print("="*80)

print("\nCONTROL TOKENS (Performance):")
for token_type in ['control_time', 'control_dur', 'control_pitch']:
    correct = stats[token_type]['correct']
    total = stats[token_type]['total']
    accuracy = (correct / total * 100) if total > 0 else 0
    mean_error = np.mean(stats[token_type]['errors']) if stats[token_type]['errors'] else 0
    median_error = np.median(stats[token_type]['errors']) if stats[token_type]['errors'] else 0
    print(f"  {token_type:20s}: {accuracy:6.2f}% ({correct:6d}/{total:6d}) | Mean err: {mean_error:7.2f} | Median err: {median_error:6.0f}")

print("\nSCORE TOKENS:")
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
print(f"  All tokens:          {overall_accuracy:6.2f}% ({total_correct:6d}/{total_tokens:6d})")

# Create error distribution plots
print("\n" + "="*80)
print("Generating error distribution plots...")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Error Distribution by Token Type - Sliding Model', fontsize=16)

token_types = ['control_time', 'control_dur', 'control_pitch', 'score_time', 'score_dur', 'score_pitch']
titles = ['Control Time', 'Control Duration', 'Control Pitch', 'Score Time', 'Score Duration', 'Score Pitch']

for idx, (token_type, title) in enumerate(zip(token_types, titles)):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
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
plt.savefig('sliding_model_error_distribution.png', dpi=150, bbox_inches='tight')
print(f"\nSaved error distribution plot to: sliding_model_error_distribution.png")

# Create cumulative error plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Cumulative Error Distribution - Sliding Model', fontsize=16)

# Control tokens
ax = axes[0]
for token_type, label in [('control_time', 'Time'), ('control_dur', 'Duration'), ('control_pitch', 'Pitch')]:
    errors = sorted(stats[token_type]['errors'])
    if errors:
        cumulative = np.arange(1, len(errors) + 1) / len(errors) * 100
        ax.plot(errors, cumulative, label=label, linewidth=2)

ax.set_xlabel('Absolute Error (tokens)')
ax.set_ylabel('Cumulative Percentage (%)')
ax.set_title('Control Tokens')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([0, min(1000, max([max(stats[t]['errors']) for t in ['control_time', 'control_dur', 'control_pitch'] if stats[t]['errors']]))])

# Score tokens
ax = axes[1]
for token_type, label in [('score_time', 'Time'), ('score_dur', 'Duration'), ('score_pitch', 'Pitch')]:
    errors = sorted(stats[token_type]['errors'])
    if errors:
        cumulative = np.arange(1, len(errors) + 1) / len(errors) * 100
        ax.plot(errors, cumulative, label=label, linewidth=2)

ax.set_xlabel('Absolute Error (tokens)')
ax.set_ylabel('Cumulative Percentage (%)')
ax.set_title('Score Tokens')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim([0, min(100, max([max(stats[t]['errors']) for t in ['score_time', 'score_dur', 'score_pitch'] if stats[t]['errors']]))])

plt.tight_layout()
plt.savefig('sliding_model_cumulative_error.png', dpi=150, bbox_inches='tight')
print(f"Saved cumulative error plot to: sliding_model_cumulative_error.png")

print("\n" + "="*80)
print("Note: This uses TEACHER FORCING (each prediction sees ground truth context)")
print("="*80)
