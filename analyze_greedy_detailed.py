"""
Analyze greedy decoding with loss progression and error distribution.

For each sequence:
- Run greedy decoding with KV caching
- Track log probabilities for each token prediction
- Plot loss progression over generation
- Show error distribution by token type
- Save MIDI outputs
- Create per-piece folders with detailed analysis
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET, TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET, REST, ANTICIPATE, SEPARATOR
from anticipation.convert import events_to_midi
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_score_only(tokens):
    """Extract only score tokens (not performance/control tokens)."""
    # Skip ANTICIPATE token if present
    start_idx = 1 if (len(tokens) > 0 and tokens[0] == ANTICIPATE) else 0
    
    # Skip separator tokens (3 SEP tokens after ANTICIPATE)
    if start_idx == 1 and len(tokens) > 4:
        if tokens[1] == SEPARATOR:
            start_idx += 3
    
    events = []
    i = start_idx
    while i < len(tokens) - 2:
        time_tok, dur_tok, note_tok = tokens[i], tokens[i+1], tokens[i+2]
        
        # Score triplet only (not control) - include REST tokens for MIDI conversion
        if time_tok < CONTROL_OFFSET and dur_tok < CONTROL_OFFSET and note_tok < CONTROL_OFFSET:
            events.extend([time_tok, dur_tok, note_tok])
            i += 3
        else:
            i += 1
    
    return events

def extract_performance_only(tokens):
    """Extract only performance (control) tokens."""
    # Skip ANTICIPATE token if present
    start_idx = 1 if (len(tokens) > 0 and tokens[0] == ANTICIPATE) else 0
    
    # Skip separator tokens (3 SEP tokens after ANTICIPATE)
    if start_idx == 1 and len(tokens) > 4:
        if tokens[1] == SEPARATOR:
            start_idx += 3
    
    events = []
    i = start_idx
    while i < len(tokens) - 2:
        time_tok, dur_tok, note_tok = tokens[i], tokens[i+1], tokens[i+2]
        
        # Control triplet (performance) - remove CONTROL_OFFSET
        if time_tok >= CONTROL_OFFSET and dur_tok >= CONTROL_OFFSET and note_tok >= CONTROL_OFFSET:
            events.extend([time_tok - CONTROL_OFFSET, dur_tok - CONTROL_OFFSET, note_tok - CONTROL_OFFSET])
            i += 3
        else:
            i += 1
    
    return events

def save_midi_outputs(tokens, greedy_seq, output_dir, seq_idx):
    """Save MIDI files for input performance, ground truth, and greedy output."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract performance (input)
    perf_events = extract_performance_only(tokens)
    if perf_events:
        perf_midi_path = os.path.join(output_dir, f'seq{seq_idx}_input_performance.mid')
        perf_midi = events_to_midi(perf_events)
        perf_midi.save(perf_midi_path)
    
    # Convert tokens to events (score only)
    gt_events = extract_score_only(tokens)
    greedy_events = extract_score_only(greedy_seq)
    
    if gt_events:
        gt_midi_path = os.path.join(output_dir, f'seq{seq_idx}_ground_truth.mid')
        gt_midi = events_to_midi(gt_events)
        gt_midi.save(gt_midi_path)
    
    if greedy_events:
        greedy_midi_path = os.path.join(output_dir, f'seq{seq_idx}_greedy.mid')
        greedy_midi = events_to_midi(greedy_events)
        greedy_midi.save(greedy_midi_path)

def greedy_with_tracking(model, tokens, score_triplet_positions, device):
    """
    Greedy decoding with detailed loss tracking for each token type.
    Uses KV caching for efficiency.
    
    Returns:
        predictions: dict with 'time', 'duration', 'pitch' lists
        log_probs: dict with log probabilities for each token
        ground_truth: dict with ground truth tokens
        correct: dict with boolean correctness for each token
        generated_seq: complete generated sequence
    """
    first_score_time_pos = score_triplet_positions[0][0]
    init_context = torch.tensor([tokens[:first_score_time_pos]]).to(device)
    
    outputs = model(init_context, past_key_values=None, use_cache=True)
    past_key_values = outputs.past_key_values
    
    predictions = {'time': [], 'duration': [], 'pitch': []}
    log_probs = {'time': [], 'duration': [], 'pitch': []}
    ground_truth = {'time': [], 'duration': [], 'pitch': []}
    correct = {'time': [], 'duration': [], 'pitch': []}
    
    generated_seq = list(tokens[:first_score_time_pos])
    
    prev_pos = first_score_time_pos
    
    for time_pos, dur_pos, pitch_pos in score_triplet_positions:
        # Add intermediate control tokens if any
        if time_pos > prev_pos:
            intermediate = torch.tensor([tokens[prev_pos:time_pos]]).to(device)
            outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            generated_seq.extend(tokens[prev_pos:time_pos])
        
        # Predict TIME
        logits = outputs.logits[0, -1]
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
        pred_time = logits.argmax().item()
        gt_time = tokens[time_pos]
        
        predictions['time'].append(pred_time)
        log_probs['time'].append(-log_probs_all[gt_time].item())  # Store loss (negative log prob)
        ground_truth['time'].append(gt_time)
        correct['time'].append(pred_time == gt_time)
        generated_seq.append(pred_time)
        
        # Feed predicted TIME back
        next_token = torch.tensor([[pred_time]]).to(device)
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        
        # Predict DURATION
        logits = outputs.logits[0, -1]
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
        pred_dur = logits.argmax().item()
        gt_dur = tokens[dur_pos]
        
        predictions['duration'].append(pred_dur)
        log_probs['duration'].append(-log_probs_all[gt_dur].item())
        ground_truth['duration'].append(gt_dur)
        correct['duration'].append(pred_dur == gt_dur)
        generated_seq.append(pred_dur)
        
        # Feed predicted DURATION back
        next_token = torch.tensor([[pred_dur]]).to(device)
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        
        # Predict PITCH
        logits = outputs.logits[0, -1]
        log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
        pred_pitch = logits.argmax().item()
        gt_pitch = tokens[pitch_pos]
        
        predictions['pitch'].append(pred_pitch)
        log_probs['pitch'].append(-log_probs_all[gt_pitch].item())
        ground_truth['pitch'].append(gt_pitch)
        correct['pitch'].append(pred_pitch == gt_pitch)
        generated_seq.append(pred_pitch)
        
        # Feed predicted PITCH back
        next_token = torch.tensor([[pred_pitch]]).to(device)
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        
        prev_pos = pitch_pos + 1
    
    return predictions, log_probs, ground_truth, correct, generated_seq

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'model-experimental'
    results_dir = 'greedy_analysis_results_run_2'
    test_file = 'data/test_normalized.txt'
    num_sequences = 35
    
    print("="*80)
    print("GREEDY DECODING DETAILED ANALYSIS")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Test sequences: {num_sequences}")
    print(f"Results directory: {results_dir}")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # Load and sample test sequences
    print(f"\nLoading test sequences from {test_file}...")
    with open(test_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(lines)} test sequences")
    random.seed(42)
    sampled_lines = random.sample(lines, min(num_sequences, len(lines)))
    print(f"Selected {len(sampled_lines)} sequences for analysis")
    
    # Parse sequences
    all_sequences = []
    for line in sampled_lines:
        if '|' in line:
            token_part = line.split('|')[0].strip()
        else:
            token_part = line
        tokens = [int(t) for t in token_part.split()]
        
        # Find score triplet positions (skip special tokens)
        score_triplet_positions = []
        i = 4  # Skip [ANTICIPATE, SEP, SEP, SEP]
        while i < len(tokens) - 2:
            time_tok, dur_tok, note_tok = tokens[i], tokens[i+1], tokens[i+2]
            
            if (time_tok >= TIME_OFFSET and time_tok < CONTROL_OFFSET and 
                dur_tok >= DUR_OFFSET and dur_tok < CONTROL_OFFSET and 
                note_tok >= NOTE_OFFSET and note_tok < CONTROL_OFFSET and
                note_tok != REST):
                score_triplet_positions.append((i, i+1, i+2))
                i += 3
            else:
                i += 1
        
        if len(score_triplet_positions) > 0:
            all_sequences.append((tokens, score_triplet_positions))
    
    print(f"Valid sequences with score triplets: {len(all_sequences)}")
    
    # Track aggregate statistics
    all_greedy_losses = {'time': [], 'duration': [], 'pitch': []}
    all_greedy_correct = {'time': [], 'duration': [], 'pitch': []}
    
    token_types = ['time', 'duration', 'pitch']
    
    print(f"\n{'='*80}")
    print("Processing sequences...")
    print(f"{'='*80}\n")
    
    for seq_idx, (tokens, score_triplet_positions) in enumerate(tqdm(all_sequences, desc="Analyzing")):
        print(f"\nSequence {seq_idx + 1}/{len(all_sequences)}")
        print(f"Score triplets to generate: {len(score_triplet_positions)}")
        
        # Clear CUDA cache before processing each sequence
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Run greedy with tracking
        greedy_pred, greedy_losses, greedy_gt, greedy_corr, greedy_seq = greedy_with_tracking(
            model, tokens, score_triplet_positions, device
        )
        
        # Clear CUDA cache after processing
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Accumulate for aggregate statistics
        for tok_type in token_types:
            all_greedy_losses[tok_type].extend(greedy_losses[tok_type])
            all_greedy_correct[tok_type].extend(greedy_corr[tok_type])
        
        # Create detailed plot for this sequence
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Sequence {seq_idx + 1} - Greedy Decoding Analysis', fontsize=16)
        
        # Top row: Loss progression
        for col, tok_type in enumerate(token_types):
            ax = axes[0, col]
            losses = greedy_losses[tok_type]
            x = np.arange(len(losses))
            
            # Plot loss over time
            ax.plot(x, losses, 'o-', alpha=0.6, markersize=4)
            
            # Add mean line
            mean_loss = np.mean(losses)
            ax.axhline(mean_loss, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_loss:.3f}')
            
            ax.set_xlabel('Triplet Index')
            ax.set_ylabel('Loss (negative log prob)')
            ax.set_title(f'{tok_type.upper()} - Loss Progression')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Bottom row: Accuracy comparison
        for col, tok_type in enumerate(token_types):
            ax = axes[1, col]
            
            greedy_acc = sum(greedy_corr[tok_type]) / len(greedy_corr[tok_type]) * 100
            greedy_errors = len(greedy_corr[tok_type]) - sum(greedy_corr[tok_type])
            
            # Single bar showing greedy accuracy
            bars = ax.bar([0], [greedy_acc], width=0.5, alpha=0.8, color='steelblue')
            
            ax.set_ylim(0, 105)
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'{tok_type.upper()} - Accuracy')
            ax.set_xticks([0])
            ax.set_xticklabels(['Greedy'])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add accuracy value on top of bar
            ax.text(0, greedy_acc + 2, f'{greedy_acc:.1f}%', ha='center', fontsize=12, fontweight='bold')
            
            # Add error count below
            if greedy_errors > 0:
                ax.text(0, greedy_acc + 5, f'{greedy_errors} errors', ha='center', fontsize=9)
        
        # Save MIDI outputs first to create directory
        output_dir = os.path.join(results_dir, f'seq{seq_idx + 1}')
        save_midi_outputs(tokens, greedy_seq, output_dir, seq_idx + 1)
        
        # Save plot to same directory
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print summary
        print(f"Results:")
        print(f"{'Token Type':<12} {'Accuracy':<15} {'Mean Loss':<15}")
        print("-" * 45)
        for tok_type in token_types:
            greedy_acc = sum(greedy_corr[tok_type]) / len(greedy_corr[tok_type]) * 100
            mean_loss = np.mean(greedy_losses[tok_type])
            print(f"{tok_type.upper():<12} {greedy_acc:>6.2f}%{'':<8} {mean_loss:>6.3f}")
        
        print(f"Saved to: {output_dir}/")
    
    # Create aggregate loss progression plot
    print(f"\n{'='*80}")
    print("Creating aggregate loss progression plot...")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Aggregate Loss Progression - {len(all_sequences)} Sequences (Greedy)', fontsize=16)
    
    for col, tok_type in enumerate(token_types):
        ax = axes[col]
        
        losses = all_greedy_losses[tok_type]
        x = np.arange(len(losses))
        
        # Scatter plot of all data points
        ax.scatter(x, losses, alpha=0.3, s=10, label='Individual predictions')
        
        # Compute moving average for smoother trend line
        window_size = max(10, len(losses) // 100)
        if len(losses) >= window_size:
            moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            moving_avg_x = np.arange(window_size//2, window_size//2 + len(moving_avg))
            ax.plot(moving_avg_x, moving_avg, 'r-', linewidth=2, label=f'Moving avg (window={window_size})')
        
        # Overall mean line
        mean_loss = np.mean(losses)
        ax.axhline(mean_loss, color='green', linestyle='--', linewidth=2, label=f'Mean = {mean_loss:.3f}')
        
        ax.set_xlabel('Triplet Index (across all sequences)')
        ax.set_ylabel('Loss (negative log prob)')
        ax.set_title(f'{tok_type.upper()} - Loss Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    aggregate_loss_path = os.path.join(results_dir, 'aggregate_loss.png')
    plt.savefig(aggregate_loss_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {aggregate_loss_path}")
    plt.close()
    
    # Create aggregate accuracy plot
    print(f"\n{'='*80}")
    print("Creating aggregate accuracy plot...")
    print(f"{'='*80}")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f'Aggregate Accuracy - {len(all_sequences)} Sequences (Greedy)', fontsize=16)
    
    x = np.arange(len(token_types))
    width = 0.4
    
    greedy_accs = [sum(all_greedy_correct[tok]) / len(all_greedy_correct[tok]) * 100 for tok in token_types]
    
    bars = ax.bar(x, greedy_accs, width, alpha=0.8, color='steelblue', label='Greedy')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, greedy_accs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Token Type')
    ax.set_title('Greedy Decoding Accuracy by Token Type')
    ax.set_xticks(x)
    ax.set_xticklabels([tok.upper() for tok in token_types])
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    aggregate_acc_path = os.path.join(results_dir, 'aggregate_accuracy.png')
    plt.savefig(aggregate_acc_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {aggregate_acc_path}")
    plt.close()
    
    # Print aggregate statistics
    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS")
    print(f"{'='*80}")
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Total triplets: {len(all_greedy_losses['time'])}")
    print()
    
    print(f"{'Token Type':<12} {'Mean Loss':<15} {'Accuracy':<15} {'Total Predictions':<20}")
    print("-" * 65)
    for tok_type in token_types:
        mean_loss = np.mean(all_greedy_losses[tok_type])
        greedy_acc = sum(all_greedy_correct[tok_type]) / len(all_greedy_correct[tok_type]) * 100
        total_preds = len(all_greedy_correct[tok_type])
        
        print(f"{tok_type.upper():<12} {mean_loss:>6.3f}{'':<9} {greedy_acc:>6.2f}%{'':<8} {total_preds:>8}")
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to: {results_dir}/")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
