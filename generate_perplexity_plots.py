"""
Generate perplexity plots for 50_model, 100_model, and 150_model.

For each model:
- Evaluate on 15 sequences
- Plot individual data points as scatter
- Plot average perplexity trend
- Show perplexity (not log probability) for better interpretability
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET, NOTE_OFFSET
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random

def calculate_perplexity_progression(model_path, model_name, test_file, output_dir, num_examples=15, device='cuda'):
    """
    Calculate perplexity progression for a model across sequences.
    
    Args:
        model_path: Path to the model directory
        model_name: Name for output (e.g., '50_model')
        test_file: Path to test data
        output_dir: Directory to save plots
        num_examples: Number of examples to evaluate
        device: Device to run on
    """
    print(f"\n{'='*80}")
    print(f"Processing: {model_name}")
    print(f"{'='*80}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Load test data
    print(f"\nLoading test data from {test_file}...")
    with open(test_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Sample sequences
    random.seed(42)
    sampled_lines = random.sample(lines, min(num_examples, len(lines)))
    print(f"Selected {len(sampled_lines)} sequences for evaluation")
    
    # Track perplexity per token position for different token types
    all_perplexities = {
        'time': [],      # (position, perplexity)
        'duration': [],
        'pitch': [],
    }
    
    print(f"\nCalculating perplexities...")
    
    with torch.no_grad():
        for line in tqdm(sampled_lines, desc="Evaluating", unit="seq"):
            # Parse tokens
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
            
            # Autoregressive evaluation with perplexity tracking
            first_score_time_pos = score_triplet_positions[0][0]
            init_context = torch.tensor([tokens[:first_score_time_pos]]).to(device)
            outputs = model(init_context, past_key_values=None, use_cache=True)
            past_key_values = outputs.past_key_values
            last_pos = first_score_time_pos
            
            position_in_sequence = 0
            
            for triplet_idx, (time_pos, dur_pos, pitch_pos) in enumerate(score_triplet_positions):
                # Process intermediate control tokens
                if time_pos > last_pos:
                    intermediate = torch.tensor([tokens[last_pos:time_pos]]).to(device)
                    outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                
                # Predict TIME
                logits = outputs.logits[0, -1]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                pred_time = logits.argmax().item()
                log_prob_time = log_probs[pred_time].item()
                perplexity_time = torch.exp(-torch.tensor(log_prob_time)).item()
                
                all_perplexities['time'].append((position_in_sequence, perplexity_time))
                
                next_token = torch.tensor([[pred_time]]).to(device)
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                # Predict DURATION
                logits = outputs.logits[0, -1]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                pred_dur = logits.argmax().item()
                log_prob_dur = log_probs[pred_dur].item()
                perplexity_dur = torch.exp(-torch.tensor(log_prob_dur)).item()
                
                all_perplexities['duration'].append((position_in_sequence, perplexity_dur))
                
                next_token = torch.tensor([[pred_dur]]).to(device)
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                # Predict PITCH
                logits = outputs.logits[0, -1]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                pred_pitch = logits.argmax().item()
                log_prob_pitch = log_probs[pred_pitch].item()
                perplexity_pitch = torch.exp(-torch.tensor(log_prob_pitch)).item()
                
                all_perplexities['pitch'].append((position_in_sequence, perplexity_pitch))
                
                next_token = torch.tensor([[pred_pitch]]).to(device)
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                last_pos = pitch_pos + 1
                position_in_sequence += 1
    
    # Create plot
    print(f"\nCreating perplexity plot...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'{model_name} - Token Perplexity Progression ({num_examples} sequences)', fontsize=14)
    
    token_types = [
        ('time', 'Time Token Perplexity', 0, 'blue'),
        ('duration', 'Duration Token Perplexity', 1, 'green'),
        ('pitch', 'Pitch Token Perplexity', 2, 'red'),
    ]
    
    for key, title, ax_idx, color in token_types:
        ax = axes[ax_idx]
        
        if all_perplexities[key]:
            positions = np.array([p for p, pp in all_perplexities[key]])
            perplexities = np.array([pp for p, pp in all_perplexities[key]])
            
            # Scatter plot of individual data points
            ax.scatter(positions, perplexities, alpha=0.3, s=10, color=color, label='Individual predictions')
            
            # Create bins for averaging
            max_pos = max(positions) if len(positions) > 0 else 0
            num_bins = min(50, max_pos + 1)
            
            if num_bins > 0:
                bin_edges = np.linspace(0, max_pos, num_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Calculate mean perplexity per bin
                binned_perplexities = []
                for i in range(num_bins):
                    bin_start = bin_edges[i]
                    bin_end = bin_edges[i + 1]
                    bin_vals = perplexities[(positions >= bin_start) & (positions < bin_end)]
                    if len(bin_vals) > 0:
                        binned_perplexities.append(np.mean(bin_vals))
                    else:
                        binned_perplexities.append(np.nan)
                
                # Plot average line
                ax.plot(bin_centers, binned_perplexities, color=color, linewidth=2, 
                       label='Average', alpha=0.9)
            
            ax.set_xlabel('Token Position in Sequence')
            ax.set_ylabel('Perplexity')
            ax.set_title(f'{title}\nTotal predictions: {len(positions)}')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # Set y-axis to log scale for better visualization
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    plt.tight_layout()
    plot_path = output_dir / f'{model_name}_perplexity_progression.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved perplexity plot: {plot_path}")
    
    # Print statistics
    print(f"\n{model_name} Perplexity Statistics:")
    for key, title, _, _ in token_types:
        if all_perplexities[key]:
            perps = [pp for p, pp in all_perplexities[key]]
            print(f"  {title}:")
            print(f"    Mean: {np.mean(perps):.2f}")
            print(f"    Median: {np.median(perps):.2f}")
            print(f"    Min: {np.min(perps):.2f}")
            print(f"    Max: {np.max(perps):.2f}")
    
    return all_perplexities

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_file = 'data/test_sliding.txt'
    output_dir = 'model_comparison_outputs'
    num_examples = 15
    
    models = [
        ('50_model', '50-model'),
        ('100_model', '100_model'),
        ('150_model', '150_model'),
    ]
    
    print("="*80)
    print("GENERATING PERPLEXITY PROGRESSION PLOTS")
    print("="*80)
    print(f"\nOutput directory: {output_dir}/")
    print(f"Number of examples per model: {num_examples}")
    print(f"Test data: {test_file}")
    print(f"Device: {device}")
    print("\nPerplexity interpretation:")
    print("  - Perplexity = 1: Perfect prediction (100% confident)")
    print("  - Perplexity = 10: Model narrows down to ~10 likely options")
    print("  - Perplexity = 100: Model is choosing from ~100 options")
    print("  - Perplexity > 1000: Model is very confused")
    
    for model_name, model_path in models:
        calculate_perplexity_progression(
            model_path=model_path,
            model_name=model_name,
            test_file=test_file,
            output_dir=output_dir,
            num_examples=num_examples,
            device=device
        )
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"\nPerplexity plots saved to: {output_dir}/")
    print("\nFiles generated:")
    print("  - 50_model_perplexity_progression.png")
    print("  - 100_model_perplexity_progression.png")
    print("  - 150_model_perplexity_progression.png")
    print("="*80)

if __name__ == "__main__":
    main()
