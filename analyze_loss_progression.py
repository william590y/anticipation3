"""
Analyze how loss (log p) changes throughout pieces for time, duration, and pitch tokens.
Plots loss progression for test sequences using the newest_model.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config
import matplotlib.pyplot as plt
from collections import defaultdict
import random

from anticipation.config import *
from anticipation.vocab import *

# Load the model
MODEL_PATH = 'opening_model'
print(f"Loading model from {MODEL_PATH}...")
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Model loaded on {device}")

# Load test sequences
TEST_FILE = 'data/test_openings.txt'
print(f"\nLoading test sequences from {TEST_FILE}...")
test_sequences = []
with open(TEST_FILE, 'r') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        
        # Split on | separator and take token part
        parts = line.split('|')
        token_str = parts[0].strip()
        tokens = [int(x) for x in token_str.split()]
        test_sequences.append((line_num, tokens))

print(f"Found {len(test_sequences)} test sequences")

# Use a subset of 100 sequences
print(f"\nSelecting 100 random test sequences...")
random.seed(42)
selected_sequences = random.sample(test_sequences, min(100, len(test_sequences)))

def compute_loss_per_token(tokens, max_len=1024):
    """Compute loss for each token prediction, processing in chunks if needed."""
    # Check vocab bounds
    vocab_size = model.config.vocab_size
    max_token = max(tokens)
    if max_token >= vocab_size:
        print(f"  WARNING: Token {max_token} exceeds vocab size {vocab_size}")
        # Filter invalid tokens
        tokens = [t if t < vocab_size else 0 for t in tokens]
    
    # If sequence is longer than max_len, only use first max_len tokens
    if len(tokens) > max_len:
        print(f"  Truncating from {len(tokens)} to {max_len} tokens")
        tokens = tokens[:max_len]
    
    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tokens_tensor, labels=tokens_tensor)
        logits = outputs.logits  # [1, seq_len, vocab_size]
    
    # Shift to align predictions with targets
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = tokens_tensor[:, 1:].contiguous()
    
    # Compute loss per position
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    losses = losses.cpu().numpy()
    
    return losses

# Collect data for plotting
all_data = defaultdict(lambda: {
    'time_control': [], 'time_score': [],
    'dur': [], 'pitch': [], 'position': []
})

for seq_idx, (line_num, tokens) in enumerate(selected_sequences):
    seq_name = f"Seq_{line_num}"
    
    if (seq_idx + 1) % 100 == 0:
        print(f"Processing sequence {seq_idx+1}/{len(selected_sequences)}...")
    
    # Don't print length for each sequence to avoid spam
    
    # Compute losses
    losses = compute_loss_per_token(tokens)
    
    # Categorize losses by token type
    # Start from position 4 (after mode + 3 SEPs)
    for i in range(4, min(len(tokens), len(losses) + 1)):
        token = tokens[i]
        loss_idx = i - 1  # Loss index is shifted by 1
        
        if loss_idx >= len(losses):
            break
        
        loss_val = losses[loss_idx]
        
        # Determine token type
        token_type = None
        if TIME_OFFSET <= token < TIME_OFFSET + MAX_TIME:
            token_type = 'time_score'  # Score time
        elif DUR_OFFSET <= token < DUR_OFFSET + MAX_DUR:
            token_type = 'dur'
        elif NOTE_OFFSET <= token < NOTE_OFFSET + MAX_NOTE:
            token_type = 'pitch'
        elif CONTROL_OFFSET <= token < CONTROL_OFFSET + MAX_TIME:
            token_type = 'time_control'  # Control time
        else:
            continue  # Skip other tokens (REST, SEPARATOR, etc.)
        
        all_data[seq_name][token_type].append(loss_val)
        all_data[seq_name]['position'].append(i - 4)  # Position relative to start

print(f"\n\nGenerating plots...")

# Create figure with subplots
fig, axes = plt.subplots(4, 1, figsize=(14, 16))
fig.suptitle(f'Loss Progression Throughout Sequences ({len(all_data)} Test Sequences)', fontsize=16, fontweight='bold')

token_types = ['time_control', 'time_score', 'dur', 'pitch']
titles = ['Control Time Token Loss (Performance)', 'Score Time Token Loss', 'Duration Token Loss', 'Pitch Token Loss']
colors = ['purple', 'blue', 'green', 'red']

for ax_idx, (token_type, title, color) in enumerate(zip(token_types, titles, colors)):
    ax = axes[ax_idx]
    
    # Collect all data for this token type
    all_losses_for_type = []
    all_positions_for_type = []
    for seq_name, data in all_data.items():
        if token_type in data:
            all_losses_for_type.extend(data[token_type])
            all_positions_for_type.extend(data['position'][:len(data[token_type])])
    
    if len(all_losses_for_type) > 0:
        # Scatter plot with moderate alpha to show variance
        ax.scatter(all_positions_for_type, all_losses_for_type, alpha=0.1, s=1, color=color, rasterized=True)
    
    # Don't plot individual sequences (too many), just compute statistics
    # for seq_name, data in all_data.items():
    #     ...
    
    ax.set_xlabel('Token Position in Sequence', fontsize=12)
    ax.set_ylabel('Loss (nats)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Compute binned statistics (already collected data above)
    if len(all_losses_for_type) > 0:
        # Sort by position and compute binned statistics
        sorted_pairs = sorted(zip(all_positions_for_type, all_losses_for_type))
        sorted_positions = np.array([p for p, l in sorted_pairs])
        sorted_losses = np.array([l for p, l in sorted_pairs])
        
        # Compute binned statistics (mean, std per position bin)
        bin_size = 10
        max_pos = max(sorted_positions)
        bins = np.arange(0, max_pos + bin_size, bin_size)
        bin_means = []
        bin_stds = []
        bin_positions = []
        
        for i in range(len(bins) - 1):
            mask = (sorted_positions >= bins[i]) & (sorted_positions < bins[i+1])
            if mask.sum() > 0:
                bin_means.append(sorted_losses[mask].mean())
                bin_stds.append(sorted_losses[mask].std())
                bin_positions.append((bins[i] + bins[i+1]) / 2)
        
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        bin_positions = np.array(bin_positions)
        
        # Plot mean line with error bars
        ax.errorbar(bin_positions, bin_means, yerr=bin_stds, 
                   color='black', linewidth=2, alpha=0.8,
                   capsize=3, capthick=1, elinewidth=1,
                   label=f'Mean ± Std (binned by {bin_size} tokens)', zorder=10)
        
        # Also plot median
        bin_medians = []
        for i in range(len(bins) - 1):
            mask = (sorted_positions >= bins[i]) & (sorted_positions < bins[i+1])
            if mask.sum() > 0:
                bin_medians.append(np.median(sorted_losses[mask]))
        
        ax.plot(bin_positions, bin_medians, color='red', linewidth=2, linestyle='--',
               label=f'Median (binned by {bin_size} tokens)', zorder=10)
        
        ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
output_path = 'loss_progression_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to: {output_path}")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for token_type, label in [('time_control', 'CONTROL TIME'), ('time_score', 'SCORE TIME'), ('dur', 'DUR'), ('pitch', 'PITCH')]:
    all_losses = []
    for seq_name, data in all_data.items():
        if token_type in data:
            all_losses.extend(data[token_type])
    
    if len(all_losses) > 0:
        print(f"\n{label} TOKENS:")
        print(f"  Mean loss: {np.mean(all_losses):.4f} nats ({np.mean(all_losses) / np.log(2):.4f} bits)")
        print(f"  Median loss: {np.median(all_losses):.4f} nats")
        print(f"  Std dev: {np.std(all_losses):.4f} nats")
        print(f"  Min loss: {np.min(all_losses):.4f} nats")
        print(f"  Max loss: {np.max(all_losses):.4f} nats")
        print(f"  Total tokens: {len(all_losses)}")

print("\n" + "="*80)
print(f"Analyzed {len(all_data)} sequences total")
print("="*80)
