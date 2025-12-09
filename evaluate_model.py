"""
Evaluate model's greedy autoregressive pitch accuracy on test set.

Loads interleaved sequences from test_normalized.txt, uses full performance
context (positions 0-201: ANTICIPATE + SEP + control+rest pairs), generates
alternating section (positions 202+), and saves MIDI outputs.

Measures:
- Model pitch accuracy: how well model deduces score from performance
- Ground truth alignment: verifies score[i] == control[i] in test data
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET, REST, TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET, SEPARATOR, ANTICIPATE, CONTEXT_SIZE
from anticipation.convert import events_to_midi
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import numpy as np

def extract_score_only(tokens):
    """Extract only score tokens (not performance/control tokens)."""
    # Skip ANTICIPATE token and SEP tokens
    start_idx = 1 if (len(tokens) > 0 and tokens[0] == ANTICIPATE) else 0
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
    # Skip ANTICIPATE token and SEP tokens
    start_idx = 1 if (len(tokens) > 0 and tokens[0] == ANTICIPATE) else 0
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

def greedy_pitch_accuracy(model, tokens, device):
    """
    Run greedy autoregressive generation and measure pitch accuracy.
    Uses positions 0-201 as context, generates alternating section from 202+.
    
    Returns:
        (stats_dict, generated_sequence, gt_mismatches)
        - stats_dict: dict with 'time', 'dur', 'pitch' accuracy and error tracking, plus loss progression
        - generated_sequence: full generated token sequence
        - gt_mismatches: list of alignment mismatches in ground truth
    """
    # Extract score positions from alternating section (positions 202+)
    # and control positions from control+rest pairs (positions 4-201)
    score_positions = []
    control_positions = []
    
    # Control+rest section: positions 4-201 (33 pairs × 6 tokens = 198 tokens)
    k = 33
    for i in range(k):
        base = 4 + i * 6  # Each pair is 6 tokens (control triplet + rest triplet)
        if base + 2 < len(tokens):
            time_tok = tokens[base]
            dur_tok = tokens[base + 1]
            note_tok = tokens[base + 2]
            
            # Control triplet (not a REST)
            if (time_tok >= CONTROL_OFFSET and 
                dur_tok >= CONTROL_OFFSET and 
                note_tok >= CONTROL_OFFSET and
                note_tok - CONTROL_OFFSET != REST):
                control_positions.append((base, base+1, base+2))
    
    # Alternating section: positions 202+ (score, control, score, control, ...)
    pos = 202
    while pos + 2 < len(tokens):
        time_tok = tokens[pos]
        dur_tok = tokens[pos + 1]
        note_tok = tokens[pos + 2]
        
        # Score triplet (not a REST)
        if (time_tok >= TIME_OFFSET and time_tok < CONTROL_OFFSET and
            dur_tok >= DUR_OFFSET and dur_tok < CONTROL_OFFSET and
            note_tok >= NOTE_OFFSET and note_tok < CONTROL_OFFSET and
            note_tok != REST):
            score_positions.append((pos, pos+1, pos+2))
            pos += 3
            
            # Skip the following control triplet
            if pos + 2 < len(tokens):
                ctrl_time = tokens[pos]
                ctrl_dur = tokens[pos + 1]
                ctrl_pitch = tokens[pos + 2]
                
                if (ctrl_time >= CONTROL_OFFSET and 
                    ctrl_dur >= CONTROL_OFFSET and 
                    ctrl_pitch >= CONTROL_OFFSET and
                    ctrl_pitch - CONTROL_OFFSET != REST):
                    control_positions.append((pos, pos+1, pos+2))
                pos += 3
        else:
            break
    
    if len(score_positions) == 0:
        empty_stats = {
            'time': {'correct': 0, 'total': 0, 'errors': []},
            'dur': {'correct': 0, 'total': 0, 'errors': []},
            'pitch': {'correct': 0, 'total': 0, 'errors': []},
            'gt_aligned': 0,
            'gt_total': 0,
            'losses': []
        }
        return empty_stats, tokens, []
    
    # Validate ground truth alignment (score[i] should match control[i])
    # First 33 controls from control+rest section, rest from alternating section
    gt_aligned = 0
    gt_total = len(score_positions)
    gt_mismatches = []
    
    for score_idx in range(len(score_positions)):
        if score_idx >= len(control_positions):
            break
            
        score_pitch_tok = tokens[score_positions[score_idx][2]]
        control_pitch_tok = tokens[control_positions[score_idx][2]]
        
        # Remove offsets to get actual pitch values
        score_pitch = score_pitch_tok - NOTE_OFFSET if score_pitch_tok >= NOTE_OFFSET else score_pitch_tok
        control_pitch = control_pitch_tok - CONTROL_OFFSET - NOTE_OFFSET if control_pitch_tok >= CONTROL_OFFSET + NOTE_OFFSET else control_pitch_tok - CONTROL_OFFSET
        
        if score_pitch == control_pitch:
            gt_aligned += 1
        else:
            gt_mismatches.append({
                'idx': score_idx,
                'score_pos': score_positions[score_idx][2],
                'control_pos': control_positions[score_idx][2],
                'score_pitch': score_pitch,
                'control_pitch': control_pitch,
                'score_tok': score_pitch_tok,
                'control_tok': control_pitch_tok
            })
    
    # Use positions 0-201 as context (ANTICIPATE + SEP SEP SEP + all control+rest pairs)
    context_end = 202
    context = tokens[:context_end]
    
    # Statistics tracking
    stats = {
        'time': {'correct': 0, 'total': 0, 'errors': []},
        'dur': {'correct': 0, 'total': 0, 'errors': []},
        'pitch': {'correct': 0, 'total': 0, 'errors': []},
        'gt_aligned': gt_aligned,
        'gt_total': gt_total,
        'losses': []  # Track loss for each predicted token
    }
    
    model.eval()
    with torch.no_grad():
        # Initialize with full context (with KV caching)
        init_context = torch.tensor([context]).to(device)
        outputs = model(init_context, past_key_values=None, use_cache=True)
        past_key_values = outputs.past_key_values
        
        generated = context.copy()
        
        # Generate alternating section: score, control, score, control, ...
        for score_idx, (time_pos, dur_pos, pitch_pos) in enumerate(score_positions):
            # === Generate SCORE triplet ===
            
            # Predict TIME
            logits = outputs.logits[0, -1]
            pred_time = logits.argmax().item()
            generated.append(pred_time)
            
            # Calculate loss and track accuracy for TIME
            true_time = tokens[time_pos]
            time_loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0), 
                torch.tensor([true_time]).to(device)
            ).item()
            stats['losses'].append(time_loss)
            
            time_error = abs(pred_time - true_time)
            stats['time']['errors'].append(time_error)
            if pred_time == true_time:
                stats['time']['correct'] += 1
            stats['time']['total'] += 1
            
            # Feed predicted TIME back
            next_token = torch.tensor([[pred_time]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Predict DURATION
            logits = outputs.logits[0, -1]
            pred_dur = logits.argmax().item()
            generated.append(pred_dur)
            
            # Calculate loss and track accuracy for DURATION
            true_dur = tokens[dur_pos]
            dur_loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0), 
                torch.tensor([true_dur]).to(device)
            ).item()
            stats['losses'].append(dur_loss)
            
            dur_error = abs(pred_dur - true_dur)
            stats['dur']['errors'].append(dur_error)
            if pred_dur == true_dur:
                stats['dur']['correct'] += 1
            stats['dur']['total'] += 1
            
            # Feed predicted DURATION back
            next_token = torch.tensor([[pred_dur]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Predict PITCH
            logits = outputs.logits[0, -1]
            pred_pitch = logits.argmax().item()
            generated.append(pred_pitch)
            
            # Calculate loss and track accuracy for PITCH
            true_pitch = tokens[pitch_pos]
            pitch_loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0), 
                torch.tensor([true_pitch]).to(device)
            ).item()
            stats['losses'].append(pitch_loss)
            
            pitch_error = abs(pred_pitch - true_pitch)
            stats['pitch']['errors'].append(pitch_error)
            if pred_pitch == true_pitch:
                stats['pitch']['correct'] += 1
            stats['pitch']['total'] += 1
            
            # Feed predicted PITCH back
            next_token = torch.tensor([[pred_pitch]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # === Add ground truth CONTROL triplet ===
            # Find corresponding control position in alternating section
            control_idx_in_alt = score_idx + k  # Offset by k (33) to get into alternating section
            if control_idx_in_alt < len(control_positions):
                ctrl_time_pos, ctrl_dur_pos, ctrl_pitch_pos = control_positions[control_idx_in_alt]
                
                # Add ground truth control triplet
                for pos in [ctrl_time_pos, ctrl_dur_pos, ctrl_pitch_pos]:
                    gt_token = tokens[pos]
                    generated.append(gt_token)
                    
                    # Feed back
                    next_token = torch.tensor([[gt_token]]).to(device)
                    outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
    
    return stats, generated, gt_mismatches

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'newest_model'
    
    print(f"Loading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    
    # Load test sequences
    test_file = 'data/test_normalized.txt'
    print(f"Loading test sequences from {test_file}...")
    
    with open(test_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(lines)} test sequences")
    
    # Sample random sequences
    num_sequences = 20
    random.seed(42)
    sampled_lines = random.sample(lines, min(num_sequences, len(lines)))
    
    # Create output directory for MIDI files
    output_dir = 'inference_outputs'
    os.makedirs(output_dir, exist_ok=True)
    print(f"MIDI outputs will be saved to: {output_dir}/")
    
    # Statistics tracking
    total_stats = {
        'time': {'correct': 0, 'total': 0, 'errors': []},
        'dur': {'correct': 0, 'total': 0, 'errors': []},
        'pitch': {'correct': 0, 'total': 0, 'errors': []},
        'gt_aligned': 0,
        'gt_total': 0,
        'losses': []
    }
    all_mismatches = []
    
    # Track per-sequence metrics for plotting
    sequence_accuracies = []
    sequence_num_notes = []
    
    # Save first 10 sequences as MIDI
    num_midi_saves = 10
    
    print(f"\nEvaluating on {len(sampled_lines)} sequences...")
    for seq_idx, line in enumerate(tqdm(sampled_lines, desc="Testing")):
        if '|' in line:
            token_part = line.split('|')[0].strip()
        else:
            token_part = line
        
        tokens = [int(t) for t in token_part.split()]
        
        seq_stats, generated_seq, mismatches = greedy_pitch_accuracy(model, tokens, device)
        
        # Aggregate statistics
        for key in ['time', 'dur', 'pitch']:
            total_stats[key]['correct'] += seq_stats[key]['correct']
            total_stats[key]['total'] += seq_stats[key]['total']
            total_stats[key]['errors'].extend(seq_stats[key]['errors'])
        
        total_stats['gt_aligned'] += seq_stats['gt_aligned']
        total_stats['gt_total'] += seq_stats['gt_total']
        total_stats['losses'].extend(seq_stats['losses'])
        
        # Track per-sequence accuracy (based on pitch)
        seq_accuracy = (seq_stats['pitch']['correct'] / seq_stats['pitch']['total'] * 100) if seq_stats['pitch']['total'] > 0 else 0
        sequence_accuracies.append(seq_accuracy)
        sequence_num_notes.append(seq_stats['pitch']['total'])
        
        if mismatches:
            all_mismatches.append({
                'seq_idx': seq_idx,
                'mismatches': mismatches
            })
        
        # Save MIDI files for first few sequences
        if seq_idx < num_midi_saves:
            # Extract and save performance (input control)
            perf_events = extract_performance_only(tokens)
            if perf_events:
                perf_midi_path = os.path.join(output_dir, f'seq{seq_idx:03d}_input_performance.mid')
                perf_midi = events_to_midi(perf_events)
                perf_midi.save(perf_midi_path)
            
            # Extract and save ground truth score
            gt_events = extract_score_only(tokens)
            if gt_events:
                gt_midi_path = os.path.join(output_dir, f'seq{seq_idx:03d}_ground_truth.mid')
                gt_midi = events_to_midi(gt_events)
                gt_midi.save(gt_midi_path)
            
            # Extract and save greedy generated score
            greedy_events = extract_score_only(generated_seq)
            if greedy_events:
                greedy_midi_path = os.path.join(output_dir, f'seq{seq_idx:03d}_greedy.mid')
                greedy_midi = events_to_midi(greedy_events)
                greedy_midi.save(greedy_midi_path)
    
    # Calculate accuracies
    time_accuracy = (total_stats['time']['correct'] / total_stats['time']['total'] * 100) if total_stats['time']['total'] > 0 else 0
    dur_accuracy = (total_stats['dur']['correct'] / total_stats['dur']['total'] * 100) if total_stats['dur']['total'] > 0 else 0
    pitch_accuracy = (total_stats['pitch']['correct'] / total_stats['pitch']['total'] * 100) if total_stats['pitch']['total'] > 0 else 0
    gt_alignment_acc = (total_stats['gt_aligned'] / total_stats['gt_total'] * 100) if total_stats['gt_total'] > 0 else 0
    
    # Overall accuracy (all three elements)
    total_correct = total_stats['time']['correct'] + total_stats['dur']['correct'] + total_stats['pitch']['correct']
    total_tokens = total_stats['time']['total'] + total_stats['dur']['total'] + total_stats['pitch']['total']
    overall_accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Test file: {test_file}")
    print(f"Sequences evaluated: {len(sampled_lines)}")
    print(f"\n--- MODEL PERFORMANCE ---")
    print(f"Score Time Accuracy:     {time_accuracy:.2f}%")
    print(f"  Correct predictions: {total_stats['time']['correct']}/{total_stats['time']['total']}")
    print(f"  Mean error: {np.mean(total_stats['time']['errors']):.2f}" if total_stats['time']['errors'] else "  Mean error: N/A")
    print(f"\nScore Duration Accuracy: {dur_accuracy:.2f}%")
    print(f"  Correct predictions: {total_stats['dur']['correct']}/{total_stats['dur']['total']}")
    print(f"  Mean error: {np.mean(total_stats['dur']['errors']):.2f}" if total_stats['dur']['errors'] else "  Mean error: N/A")
    print(f"\nScore Pitch Accuracy:    {pitch_accuracy:.2f}%")
    print(f"  Correct predictions: {total_stats['pitch']['correct']}/{total_stats['pitch']['total']}")
    print(f"  Mean error: {np.mean(total_stats['pitch']['errors']):.2f}" if total_stats['pitch']['errors'] else "  Mean error: N/A")
    print(f"\nOverall Token Accuracy:  {overall_accuracy:.2f}%")
    print(f"  Correct predictions: {total_correct}/{total_tokens}")
    print(f"  Mean loss: {np.mean(total_stats['losses']):.4f}" if total_stats['losses'] else "  Mean loss: N/A")
    print(f"\n--- DATA QUALITY CHECK ---")
    print(f"Ground Truth Alignment: {gt_alignment_acc:.2f}%")
    print(f"  GT scores matching controls: {total_stats['gt_aligned']}/{total_stats['gt_total']}")
    print(f"  (Verifies score[i] == control[i] in test data)")
    if gt_alignment_acc >= 99.99:
        print(f"  ✓ Data alignment preserved correctly")
    else:
        print(f"  ⚠ Warning: Data may have alignment issues")
        
        # Show detailed mismatch information
        if all_mismatches:
            print(f"\n  Alignment mismatches detected in {len(all_mismatches)} sequences:")
            for seq_info in all_mismatches[:3]:  # Show first 3 sequences with errors
                seq_idx = seq_info['seq_idx']
                mismatches = seq_info['mismatches']
                print(f"\n  Sequence {seq_idx}: {len(mismatches)} mismatches")
                for err in mismatches[:3]:  # Show first 3 errors per sequence
                    print(f"    Position {err['idx']}: score_pitch={err['score_pitch']} (tok={err['score_tok']}) "
                          f"vs control_pitch={err['control_pitch']} (tok={err['control_tok']})")
                    print(f"      Score position in tokens: {err['score_pos']}, Control position: {err['control_pos']}")
    
    print(f"\nMIDI outputs saved to: {output_dir}/")
    print(f"  - First {num_midi_saves} sequences saved")
    print(f"  - Files: input_performance.mid, ground_truth.mid, greedy.mid")
    print(f"{'='*60}")
    
    # Generate plots
    print(f"\n{'='*60}")
    print(f"Generating visualizations...")
    print(f"{'='*60}")
    
    # Plot 1: Token-level accuracy breakdown (Time, Duration, Pitch)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Token-Level Evaluation Results - {model_path}', fontsize=16)
    
    # Subplot 1: Accuracy comparison
    ax1 = axes[0, 0]
    token_types = ['Time', 'Duration', 'Pitch', 'Overall']
    accuracies = [time_accuracy, dur_accuracy, pitch_accuracy, overall_accuracy]
    colors = ['steelblue', 'orange', 'green', 'purple']
    bars = ax1.bar(token_types, accuracies, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Token-Level Accuracy Breakdown')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Error distributions
    ax2 = axes[0, 1]
    error_data = [total_stats['time']['errors'], total_stats['dur']['errors'], total_stats['pitch']['errors']]
    bp = ax2.boxplot(error_data, labels=['Time', 'Duration', 'Pitch'], patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors[:3]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Absolute Error (tokens)')
    ax2.set_title('Error Distribution by Token Type')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')
    
    # Subplot 3: Loss progression
    ax3 = axes[1, 0]
    if total_stats['losses']:
        # Smooth loss curve with moving average
        window = max(1, len(total_stats['losses']) // 50)
        if len(total_stats['losses']) > window:
            smoothed = np.convolve(total_stats['losses'], np.ones(window)/window, mode='valid')
            ax3.plot(smoothed, linewidth=2, color='red', label=f'Smoothed (window={window})')
        ax3.plot(total_stats['losses'], alpha=0.3, color='gray', label='Raw')
        ax3.axhline(np.mean(total_stats['losses']), color='blue', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(total_stats['losses']):.4f}')
        ax3.set_xlabel('Token Position')
        ax3.set_ylabel('Cross-Entropy Loss')
        ax3.set_title('Loss Progression During Generation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax3.transAxes)
    
    # Subplot 4: Cumulative error distribution
    ax4 = axes[1, 1]
    for token_type, label, color in [('time', 'Time', 'steelblue'), 
                                      ('dur', 'Duration', 'orange'), 
                                      ('pitch', 'Pitch', 'green')]:
        errors = sorted(total_stats[token_type]['errors'])
        if errors:
            cumulative = np.arange(1, len(errors) + 1) / len(errors) * 100
            ax4.plot(errors, cumulative, label=label, linewidth=2, color=color)
    
    ax4.set_xlabel('Absolute Error (tokens)')
    ax4.set_ylabel('Cumulative Percentage (%)')
    ax4.set_title('Cumulative Error Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'token_level_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved token-level analysis to: {plot_path}")
    plt.close()
    
    # Plot 2: Per-sequence pitch accuracy
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Per-Sequence Pitch Accuracy - {model_path}', fontsize=16)
    
    # Subplot 1: Accuracy per sequence
    ax1 = axes[0]
    seq_indices = list(range(len(sequence_accuracies)))
    bars = ax1.bar(seq_indices, sequence_accuracies, alpha=0.7, edgecolor='black')
    
    # Color bars by accuracy (red=poor, yellow=medium, green=good)
    for bar, acc in zip(bars, sequence_accuracies):
        if acc >= 80:
            bar.set_color('green')
        elif acc >= 50:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax1.axhline(pitch_accuracy, color='blue', linestyle='--', linewidth=2, 
                label=f'Overall: {pitch_accuracy:.2f}%')
    ax1.set_xlabel('Sequence Index')
    ax1.set_ylabel('Pitch Accuracy (%)')
    ax1.set_title('Pitch Accuracy per Sequence')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 105])
    
    # Subplot 2: Accuracy distribution histogram
    ax2 = axes[1]
    ax2.hist(sequence_accuracies, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
    ax2.axvline(pitch_accuracy, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {pitch_accuracy:.2f}%')
    ax2.axvline(np.median(sequence_accuracies), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(sequence_accuracies):.2f}%')
    ax2.set_xlabel('Pitch Accuracy (%)')
    ax2.set_ylabel('Number of Sequences')
    ax2.set_title('Distribution of Pitch Accuracies')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'pitch_accuracy.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved pitch accuracy plot to: {plot_path}")
    plt.close()
    
    # Plot 3: Notes per sequence vs accuracy
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    scatter = ax.scatter(sequence_num_notes, sequence_accuracies, 
                        alpha=0.6, s=100, c=sequence_accuracies, 
                        cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    ax.axhline(pitch_accuracy, color='blue', linestyle='--', linewidth=2, 
               label=f'Overall Accuracy: {pitch_accuracy:.2f}%')
    ax.set_xlabel('Number of Notes in Sequence')
    ax.set_ylabel('Pitch Accuracy (%)')
    ax.set_title('Pitch Accuracy vs Sequence Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Accuracy (%)')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'accuracy_vs_length.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved accuracy vs length plot to: {plot_path}")
    plt.close()
    
    # Plot 4: Summary statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Evaluation Summary - {model_path}', fontsize=16)
    
    # Subplot 1: Accuracy breakdown
    ax1 = axes[0, 0]
    categories = ['Time', 'Duration', 'Pitch', 'Overall', 'GT Align']
    values = [time_accuracy, dur_accuracy, pitch_accuracy, overall_accuracy, gt_alignment_acc]
    colors_bar = ['steelblue', 'orange', 'green', 'purple', 'red']
    bars = ax1.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Metrics')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Subplot 2: Cumulative accuracy
    ax2 = axes[0, 1]
    sorted_accs = sorted(sequence_accuracies)
    cumulative = np.arange(1, len(sorted_accs) + 1) / len(sorted_accs) * 100
    ax2.plot(sorted_accs, cumulative, linewidth=2, color='steelblue')
    ax2.axvline(pitch_accuracy, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {pitch_accuracy:.2f}%')
    ax2.set_xlabel('Pitch Accuracy (%)')
    ax2.set_ylabel('Cumulative Percentage of Sequences (%)')
    ax2.set_title('Cumulative Distribution of Accuracies')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 100])
    
    # Subplot 3: Token counts pie chart
    ax3 = axes[1, 0]
    labels = ['Correct\nPredictions', 'Incorrect\nPredictions']
    sizes = [total_correct, total_tokens - total_correct]
    colors_pie = ['green', 'red']
    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie, 
                                         autopct='%1.1f%%', startangle=90,
                                         textprops={'fontsize': 10})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax3.set_title(f'Overall Token Prediction\n({total_correct}/{total_tokens} correct)')
    
    # Subplot 4: Summary stats table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    summary_data = [
        ['Metric', 'Value'],
        ['Sequences Evaluated', f'{len(sampled_lines)}'],
        ['Total Score Tokens', f'{total_tokens}'],
        ['Time Accuracy', f'{time_accuracy:.2f}%'],
        ['Duration Accuracy', f'{dur_accuracy:.2f}%'],
        ['Pitch Accuracy', f'{pitch_accuracy:.2f}%'],
        ['Overall Accuracy', f'{overall_accuracy:.2f}%'],
        ['Mean Loss', f'{np.mean(total_stats["losses"]):.4f}' if total_stats['losses'] else 'N/A'],
        ['GT Alignment', f'{gt_alignment_acc:.2f}%'],
    ]
    
    table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax4.set_title('Summary Statistics', fontweight='bold', fontsize=12, pad=20)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'evaluation_summary.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved evaluation summary to: {plot_path}")
    plt.close()
    
    print(f"\nAll visualizations saved to: {output_dir}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
