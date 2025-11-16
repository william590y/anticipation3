"""
Generate MIDI examples and log probability plots for 50_model, 100_model, and 150_model.

For each model:
1. Generate 5 example MIDI triplets (ground truth, generated score, performance)
2. Plot log probabilities of time/duration/pitch tokens as autoregression progresses
3. Save everything in organized folders
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET, NOTE_OFFSET
from anticipation.convert import events_to_midi
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random

def generate_examples_with_log_probs(model_path, model_name, test_file, output_base_dir, num_examples=5, device='cuda'):
    """
    Generate MIDI examples and log probability plots for a model.
    
    Args:
        model_path: Path to the model directory
        model_name: Name for output folder (e.g., '50_model')
        test_file: Path to test data
        output_base_dir: Base directory for outputs
        num_examples: Number of examples to generate
        device: Device to run on
    """
    print(f"\n{'='*80}")
    print(f"Processing: {model_name}")
    print(f"{'='*80}")
    
    # Create output directories
    output_dir = Path(output_base_dir) / model_name
    midi_dir = output_dir / 'midi_examples'
    midi_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Sample sequences for diversity
    random.seed(42)
    sampled_lines = random.sample(lines, min(num_examples, len(lines)))
    print(f"Selected {len(sampled_lines)} sequences for generation")
    
    # Track log probabilities for plotting
    all_log_probs = {
        'time': [],
        'duration': [],
        'pitch': [],
    }
    
    print(f"\nGenerating examples...")
    
    with torch.no_grad():
        for example_idx, line in enumerate(tqdm(sampled_lines, desc="Generating", unit="example")):
            # Parse tokens
            if '|' in line:
                parts = line.split('|')
                token_part = parts[0].strip()
                metadata = parts[1].strip() if len(parts) > 1 else ""
            else:
                token_part = line
                metadata = ""
            
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
            
            # Prepare for autoregressive generation
            first_score_time_pos = score_triplet_positions[0][0]
            init_context = torch.tensor([tokens[:first_score_time_pos]]).to(device)
            outputs = model(init_context, past_key_values=None, use_cache=True)
            past_key_values = outputs.past_key_values
            last_pos = first_score_time_pos
            
            # Generated tokens
            generated_tokens = tokens[:first_score_time_pos].copy()
            
            # Log probabilities for this example
            example_log_probs = {
                'time': [],
                'duration': [],
                'pitch': [],
            }
            
            position_in_sequence = 0
            
            for triplet_idx, (time_pos, dur_pos, pitch_pos) in enumerate(score_triplet_positions):
                # Process intermediate control tokens (ground truth)
                if time_pos > last_pos:
                    intermediate = tokens[last_pos:time_pos]
                    generated_tokens.extend(intermediate)
                    intermediate_tensor = torch.tensor([intermediate]).to(device)
                    outputs = model(intermediate_tensor, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                
                # Predict TIME
                logits = outputs.logits[0, -1]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                pred_time = logits.argmax().item()
                log_prob_time = log_probs[pred_time].item()
                
                example_log_probs['time'].append((position_in_sequence, log_prob_time))
                all_log_probs['time'].append((position_in_sequence, log_prob_time))
                
                generated_tokens.append(pred_time)
                next_token = torch.tensor([[pred_time]]).to(device)
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                # Predict DURATION
                logits = outputs.logits[0, -1]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                pred_dur = logits.argmax().item()
                log_prob_dur = log_probs[pred_dur].item()
                
                example_log_probs['duration'].append((position_in_sequence, log_prob_dur))
                all_log_probs['duration'].append((position_in_sequence, log_prob_dur))
                
                generated_tokens.append(pred_dur)
                next_token = torch.tensor([[pred_dur]]).to(device)
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                # Predict PITCH
                logits = outputs.logits[0, -1]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                pred_pitch = logits.argmax().item()
                log_prob_pitch = log_probs[pred_pitch].item()
                
                example_log_probs['pitch'].append((position_in_sequence, log_prob_pitch))
                all_log_probs['pitch'].append((position_in_sequence, log_prob_pitch))
                
                generated_tokens.append(pred_pitch)
                next_token = torch.tensor([[pred_pitch]]).to(device)
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                last_pos = pitch_pos + 1
                position_in_sequence += 1
            
            # Generate MIDI files
            example_dir = midi_dir / f'example_{example_idx + 1}'
            example_dir.mkdir(exist_ok=True)
            
            # Extract ground truth score (all score triplets)
            gt_score = []
            i = 0
            while i < len(tokens) - 2:
                if (tokens[i] < CONTROL_OFFSET and 
                    tokens[i+1] < CONTROL_OFFSET and 
                    tokens[i+2] < CONTROL_OFFSET):
                    gt_score.extend([tokens[i], tokens[i+1], tokens[i+2]])
                    i += 3
                else:
                    i += 1
            
            # Extract performance (all control triplets, subtract CONTROL_OFFSET)
            performance = []
            i = 0
            while i < len(tokens) - 2:
                if (tokens[i] >= CONTROL_OFFSET and tokens[i] < CONTROL_OFFSET + 27512 and
                    tokens[i+1] >= CONTROL_OFFSET and tokens[i+1] < CONTROL_OFFSET + 27512 and
                    tokens[i+2] >= CONTROL_OFFSET and tokens[i+2] < CONTROL_OFFSET + 27512):
                    performance.extend([tokens[i] - CONTROL_OFFSET, 
                                      tokens[i+1] - CONTROL_OFFSET, 
                                      tokens[i+2] - CONTROL_OFFSET])
                    i += 3
                else:
                    i += 1
            
            # 1. Ground truth score MIDI
            if gt_score:
                gt_midi_path = example_dir / f'{model_name}_example_{example_idx + 1}_ground_truth.mid'
                gt_midi = events_to_midi(gt_score)
                gt_midi.save(str(gt_midi_path))
            
            # 2. Generated score MIDI (extract from generated_tokens)
            gen_score = []
            i = 0
            while i < len(generated_tokens) - 2:
                if (generated_tokens[i] < CONTROL_OFFSET and 
                    generated_tokens[i+1] < CONTROL_OFFSET and 
                    generated_tokens[i+2] < CONTROL_OFFSET):
                    gen_score.extend([generated_tokens[i], generated_tokens[i+1], generated_tokens[i+2]])
                    i += 3
                else:
                    i += 1
            
            if gen_score:
                gen_midi_path = example_dir / f'{model_name}_example_{example_idx + 1}_generated_score.mid'
                gen_midi = events_to_midi(gen_score)
                gen_midi.save(str(gen_midi_path))
            
            # 3. Performance MIDI
            if performance:
                perf_midi_path = example_dir / f'{model_name}_example_{example_idx + 1}_performance.mid'
                perf_midi = events_to_midi(performance)
                perf_midi.save(str(perf_midi_path))
            
            # Plot log probabilities for this example
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle(f'{model_name} - Example {example_idx + 1} - Log Probabilities', fontsize=14)
            
            token_types = [
                ('time', 'Time Token Log Probability', 0),
                ('duration', 'Duration Token Log Probability', 1),
                ('pitch', 'Pitch Token Log Probability', 2),
            ]
            
            for key, title, ax_idx in token_types:
                ax = axes[ax_idx]
                if example_log_probs[key]:
                    positions = [p for p, lp in example_log_probs[key]]
                    log_probs_vals = [lp for p, lp in example_log_probs[key]]
                    
                    ax.plot(positions, log_probs_vals, marker='o', markersize=2, alpha=0.7)
                    ax.set_xlabel('Token Position in Sequence')
                    ax.set_ylabel('Log Probability (log p)')
                    ax.set_title(title)
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=-1.0, color='g', linestyle='--', alpha=0.5, label='p=0.37 (uncertain)')
                    ax.axhline(y=-5.0, color='r', linestyle='--', alpha=0.5, label='p=0.007 (very uncertain)')
                    ax.legend(fontsize=8)
            
            plt.tight_layout()
            plot_path = example_dir / f'{model_name}_example_{example_idx + 1}_log_probs.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    # Create aggregate plot across all examples
    print(f"\nCreating aggregate log probability plot...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'{model_name} - Aggregate Log Probabilities (All Examples)', fontsize=14)
    
    token_types = [
        ('time', 'Time Token Log Probability', 0),
        ('duration', 'Duration Token Log Probability', 1),
        ('pitch', 'Pitch Token Log Probability', 2),
    ]
    
    for key, title, ax_idx in token_types:
        ax = axes[ax_idx]
        if all_log_probs[key]:
            positions = [p for p, lp in all_log_probs[key]]
            log_probs_vals = [lp for p, lp in all_log_probs[key]]
            
            # Create bins for averaging
            max_pos = max(positions) if positions else 0
            num_bins = min(50, max_pos + 1)
            
            if num_bins > 0:
                bin_edges = np.linspace(0, max_pos, num_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Calculate mean log prob per bin
                binned_log_probs = []
                for i in range(num_bins):
                    bin_start = bin_edges[i]
                    bin_end = bin_edges[i + 1]
                    bin_vals = [lp for p, lp in zip(positions, log_probs_vals) 
                               if bin_start <= p < bin_end]
                    if bin_vals:
                        binned_log_probs.append(np.mean(bin_vals))
                    else:
                        binned_log_probs.append(np.nan)
                
                # Plot
                ax.plot(bin_centers, binned_log_probs, marker='o', markersize=3, alpha=0.7)
                ax.set_xlabel('Token Position in Sequence')
                ax.set_ylabel('Log Probability (log p)')
                ax.set_title(f'{title}\nTotal predictions: {len(positions)}')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=-1.0, color='g', linestyle='--', alpha=0.5, label='p=0.37 (uncertain)')
                ax.axhline(y=-5.0, color='r', linestyle='--', alpha=0.5, label='p=0.007 (very uncertain)')
                ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    plt.tight_layout()
    aggregate_plot_path = output_dir / f'{model_name}_aggregate_log_probs.png'
    plt.savefig(aggregate_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Generated {num_examples} MIDI triplets in: {midi_dir}")
    print(f"✓ Saved aggregate log probability plot: {aggregate_plot_path}")
    
    return all_log_probs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_file = 'data/test_sliding.txt'
    output_base_dir = 'model_comparison_outputs'
    num_examples = 5
    
    models = [
        ('50_model', '50-model'),
        ('100_model', '100_model'),
        ('150_model', '150_model'),
    ]
    
    print("="*80)
    print("GENERATING MIDI EXAMPLES AND LOG PROBABILITY PLOTS")
    print("="*80)
    print(f"\nOutput directory: {output_base_dir}/")
    print(f"Number of examples per model: {num_examples}")
    print(f"Test data: {test_file}")
    print(f"Device: {device}")
    
    for model_name, model_path in models:
        generate_examples_with_log_probs(
            model_path=model_path,
            model_name=model_name,
            test_file=test_file,
            output_base_dir=output_base_dir,
            num_examples=num_examples,
            device=device
        )
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_base_dir}/")
    print("\nDirectory structure:")
    print(f"  {output_base_dir}/")
    print(f"    50_model/")
    print(f"      50_model_aggregate_log_probs.png")
    print(f"      midi_examples/")
    print(f"        example_1/")
    print(f"          50_model_example_1_ground_truth.mid")
    print(f"          50_model_example_1_generated_score.mid")
    print(f"          50_model_example_1_performance.mid")
    print(f"          50_model_example_1_log_probs.png")
    print(f"        example_2/")
    print(f"        ...")
    print(f"    100_model/")
    print(f"      ...")
    print(f"    150_model/")
    print(f"      ...")
    print("="*80)

if __name__ == "__main__":
    main()
