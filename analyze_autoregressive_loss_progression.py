"""
Analyze loss progression for opening_model in autoregressive mode.
This shows how the model's prediction quality changes as it generates more tokens.
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

print("="*80)
print("LOSS PROGRESSION ANALYSIS - AUTOREGRESSIVE MODE")
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

# Sample 50 sequences for analysis
import random
random.seed(42)
sample_size = min(50, len(lines))
sampled_lines = random.sample(lines, sample_size)

print(f"Sampled {sample_size} sequences for analysis")

# Track loss per token position for different token types
# We'll track: REST time/dur/pitch, and NOTE time/dur/pitch
loss_by_position = {
    'rest_time': [],
    'rest_dur': [],
    'rest_pitch': [],
    'note_time': [],
    'note_dur': [],
    'note_pitch': [],
}

print(f"\nProcessing sequences...")

for line in tqdm(sampled_lines, desc="Analyzing", unit="seq"):
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
    
    # Autoregressive evaluation with loss tracking
    with torch.no_grad():
        first_score_time_pos = score_triplet_positions[0][0]
        init_context = torch.tensor([tokens[:first_score_time_pos]]).to(device)
        outputs = model(init_context, past_key_values=None, use_cache=True)
        past_key_values = outputs.past_key_values
        last_pos = first_score_time_pos
        
        position_in_sequence = 0  # Track position within generated sequence
        
        for triplet_idx, (time_pos, dur_pos, pitch_pos) in enumerate(score_triplet_positions):
            is_rest = (tokens[pitch_pos] == 27512)
            prefix = 'rest' if is_rest else 'note'
            
            # Process intermediate control tokens
            if time_pos > last_pos:
                intermediate = torch.tensor([tokens[last_pos:time_pos]]).to(device)
                outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
            
            # Predict TIME
            logits = outputs.logits[0, -1]
            gt_time = tokens[time_pos]
            
            # Calculate cross-entropy loss for this prediction
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), 
                                                     torch.tensor([gt_time]).to(device))
            loss_by_position[f'{prefix}_time'].append((position_in_sequence, loss.item()))
            
            pred_time = logits.argmax().item()
            next_token = torch.tensor([[pred_time]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Predict DURATION
            logits = outputs.logits[0, -1]
            gt_dur = tokens[dur_pos]
            
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0),
                                                     torch.tensor([gt_dur]).to(device))
            loss_by_position[f'{prefix}_dur'].append((position_in_sequence, loss.item()))
            
            pred_dur = logits.argmax().item()
            next_token = torch.tensor([[pred_dur]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Predict PITCH
            logits = outputs.logits[0, -1]
            gt_pitch = tokens[pitch_pos]
            
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0),
                                                     torch.tensor([gt_pitch]).to(device))
            loss_by_position[f'{prefix}_pitch'].append((position_in_sequence, loss.item()))
            
            pred_pitch = logits.argmax().item()
            next_token = torch.tensor([[pred_pitch]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            last_pos = pitch_pos + 1
            position_in_sequence += 1

# Plot loss progression
print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Loss Progression in Autoregressive Mode', fontsize=16)

token_types = [
    ('rest_time', 'REST Time', 0, 0),
    ('rest_dur', 'REST Duration', 0, 1),
    ('rest_pitch', 'REST Pitch', 0, 2),
    ('note_time', 'NOTE Time', 1, 0),
    ('note_dur', 'NOTE Duration', 1, 1),
    ('note_pitch', 'NOTE Pitch', 1, 2),
]

for key, title, row, col in token_types:
    ax = axes[row, col]
    
    if loss_by_position[key]:
        # Extract positions and losses
        positions = [p for p, l in loss_by_position[key]]
        losses = [l for p, l in loss_by_position[key]]
        
        # Create bins for averaging
        max_pos = max(positions) if positions else 0
        num_bins = min(50, max_pos + 1)
        
        if num_bins > 0:
            bin_edges = np.linspace(0, max_pos, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Calculate mean loss per bin
            binned_losses = []
            for i in range(num_bins):
                bin_start = bin_edges[i]
                bin_end = bin_edges[i + 1]
                bin_losses = [l for p, l in zip(positions, losses) 
                             if bin_start <= p < bin_end]
                if bin_losses:
                    binned_losses.append(np.mean(bin_losses))
                else:
                    binned_losses.append(np.nan)
            
            # Plot
            ax.plot(bin_centers, binned_losses, marker='o', markersize=3, alpha=0.7)
            ax.set_xlabel('Token Position in Sequence')
            ax.set_ylabel('Cross-Entropy Loss (bits)')
            ax.set_title(f'{title}\nTotal samples: {len(positions)}')
            ax.grid(True, alpha=0.3)
            
            # Add reference line for "good" performance (e.g., 1 bit)
            ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='1 bit (good)')
            ax.axhline(y=5.0, color='r', linestyle='--', alpha=0.5, label='5 bits (poor)')
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)

plt.tight_layout()
plt.savefig('autoregressive_loss_progression.png', dpi=150, bbox_inches='tight')
print(f"\nSaved plot to: autoregressive_loss_progression.png")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for key, title, row, col in token_types:
    if loss_by_position[key]:
        losses = [l for p, l in loss_by_position[key]]
        print(f"\n{title}:")
        print(f"  Mean loss: {np.mean(losses):.3f} bits")
        print(f"  Median loss: {np.median(losses):.3f} bits")
        print(f"  Min loss: {np.min(losses):.3f} bits")
        print(f"  Max loss: {np.max(losses):.3f} bits")
        print(f"  Total predictions: {len(losses)}")

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print("- Low loss (< 1 bit) = model is very confident and usually correct")
print("- Medium loss (1-5 bits) = model is uncertain or often wrong")
print("- High loss (> 5 bits) = model is very confused")
print()
print("If loss INCREASES with position, errors are compounding.")
print("If loss is FLAT, model maintains quality (or lack thereof).")
print("="*80)
