"""
Analyze triplet-level beam search with loss progression and error distribution.

This version performs beam search at the triplet level, exploring complete
(TIME, DURATION, PITCH) combinations before pruning, rather than pruning
after each individual token.

For each sequence:
- Run greedy and triplet-level beam search
- Track log probabilities for each token prediction
- Plot loss progression over generation
- Show error distribution by token type
- Save MIDI outputs for comparison
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET, EVENT_OFFSET, TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET, REST, ANTICIPATE, SEPARATOR
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
        start_idx += 3
    
    events = []
    for i in range(start_idx, len(tokens), 3):
        if i+2 >= len(tokens):
            break
        
        time_tok, dur_tok, note_tok = tokens[i], tokens[i+1], tokens[i+2]
        
        # Score triplet only (not control) - include REST tokens for MIDI conversion
        if time_tok < CONTROL_OFFSET and dur_tok < CONTROL_OFFSET and note_tok < CONTROL_OFFSET:
            events.extend([time_tok, dur_tok, note_tok])
    
    return events

def extract_performance_only(tokens):
    """Extract only performance (control) tokens."""
    # Skip ANTICIPATE token if present
    start_idx = 1 if (len(tokens) > 0 and tokens[0] == ANTICIPATE) else 0
    
    # Skip separator tokens (3 SEP tokens after ANTICIPATE)
    if start_idx == 1 and len(tokens) > 4:
        start_idx += 3
    
    events = []
    for i in range(start_idx, len(tokens), 3):
        if i+2 >= len(tokens):
            break
        
        time_tok, dur_tok, note_tok = tokens[i], tokens[i+1], tokens[i+2]
        
        # Control triplet (performance) - remove CONTROL_OFFSET
        if time_tok >= CONTROL_OFFSET and dur_tok >= CONTROL_OFFSET and note_tok >= CONTROL_OFFSET:
            events.extend([time_tok - CONTROL_OFFSET, dur_tok - CONTROL_OFFSET, note_tok - CONTROL_OFFSET])
    
    return events

def save_midi_outputs(tokens, greedy_seq, beam_seq, output_dir, seq_idx):
    """Save MIDI files for input performance, ground truth, greedy, and beam outputs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract performance (input)
    perf_events = extract_performance_only(tokens)
    perf_midi_path = os.path.join(output_dir, f'seq{seq_idx}_input_performance.mid')
    perf_midi = events_to_midi(perf_events)
    perf_midi.save(perf_midi_path)
    
    # Convert tokens to events (score only - no control/performance tokens)
    gt_events = extract_score_only(tokens)
    greedy_events = extract_score_only(greedy_seq)
    beam_events = extract_score_only(beam_seq)
    
    # Save as MIDI
    gt_midi_path = os.path.join(output_dir, f'seq{seq_idx}_ground_truth.mid')
    gt_midi = events_to_midi(gt_events)
    gt_midi.save(gt_midi_path)
    
    greedy_midi_path = os.path.join(output_dir, f'seq{seq_idx}_greedy.mid')
    greedy_midi = events_to_midi(greedy_events)
    greedy_midi.save(greedy_midi_path)
    
    beam_midi_path = os.path.join(output_dir, f'seq{seq_idx}_beam.mid')
    beam_midi = events_to_midi(beam_events)
    beam_midi.save(beam_midi_path)
    
    return perf_midi_path, gt_midi_path, greedy_midi_path, beam_midi_path

def greedy_with_tracking(model, tokens, score_triplet_positions, device):
    """Greedy decoding with detailed tracking."""
    first_score_time_pos = score_triplet_positions[0][0]
    init_context = torch.tensor([tokens[:first_score_time_pos]]).to(device)
    
    outputs = model(init_context, past_key_values=None, use_cache=True)
    past_key_values = outputs.past_key_values
    last_pos = first_score_time_pos
    
    # Track predictions and log probs
    predictions = {'time': [], 'duration': [], 'pitch': []}
    log_probs = {'time': [], 'duration': [], 'pitch': []}
    ground_truth = {'time': [], 'duration': [], 'pitch': []}
    correct = {'time': [], 'duration': [], 'pitch': []}
    
    # Build generated sequence
    generated_seq = list(tokens[:first_score_time_pos])
    
    for time_pos, dur_pos, pitch_pos in score_triplet_positions:
        # Process intermediate control tokens
        if time_pos > last_pos:
            intermediate = torch.tensor([tokens[last_pos:time_pos]]).to(device)
            generated_seq.extend(tokens[last_pos:time_pos])
            outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
        
        # Predict TIME
        logits = outputs.logits[0, -1]
        log_probs_tensor = torch.nn.functional.log_softmax(logits, dim=-1)
        pred_time = logits.argmax().item()
        gt_time = tokens[time_pos]
        
        predictions['time'].append(pred_time)
        log_probs['time'].append(log_probs_tensor[pred_time].item())
        ground_truth['time'].append(gt_time)
        correct['time'].append(pred_time == gt_time)
        generated_seq.append(pred_time)
        
        next_token = torch.tensor([[pred_time]]).to(device)
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        
        # Predict DURATION
        logits = outputs.logits[0, -1]
        log_probs_tensor = torch.nn.functional.log_softmax(logits, dim=-1)
        pred_dur = logits.argmax().item()
        gt_dur = tokens[dur_pos]
        
        predictions['duration'].append(pred_dur)
        log_probs['duration'].append(log_probs_tensor[pred_dur].item())
        ground_truth['duration'].append(gt_dur)
        correct['duration'].append(pred_dur == gt_dur)
        generated_seq.append(pred_dur)
        
        next_token = torch.tensor([[pred_dur]]).to(device)
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        
        # Predict PITCH
        logits = outputs.logits[0, -1]
        log_probs_tensor = torch.nn.functional.log_softmax(logits, dim=-1)
        pred_pitch = logits.argmax().item()
        gt_pitch = tokens[pitch_pos]
        
        predictions['pitch'].append(pred_pitch)
        log_probs['pitch'].append(log_probs_tensor[pred_pitch].item())
        ground_truth['pitch'].append(gt_pitch)
        correct['pitch'].append(pred_pitch == gt_pitch)
        generated_seq.append(pred_pitch)
        
        next_token = torch.tensor([[pred_pitch]]).to(device)
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        
        last_pos = pitch_pos + 1
    
    return predictions, log_probs, ground_truth, correct, generated_seq

def triplet_beam_search_with_tracking(model, tokens, score_triplet_positions, num_beams, device):
    """
    Triplet-level beam search with detailed tracking.
    
    Explores complete (TIME, DURATION, PITCH) triplets before pruning,
    rather than pruning after each individual token.
    
    Args:
        num_beams: Number of hypotheses to maintain (prunes to this after each triplet)
    """
    first_score_time_pos = score_triplet_positions[0][0]
    init_context = tokens[:first_score_time_pos]
    beams = [(0.0, init_context)]
    
    last_pos = first_score_time_pos
    
    # Track best beam's predictions and log probs at each step
    predictions = {'time': [], 'duration': [], 'pitch': []}
    log_probs = {'time': [], 'duration': [], 'pitch': []}
    ground_truth = {'time': [], 'duration': [], 'pitch': []}
    correct = {'time': [], 'duration': [], 'pitch': []}
    
    # Track generated sequence (from best beam at end)
    generated_seq = None
    
    for triplet_idx, (time_pos, dur_pos, pitch_pos) in enumerate(tqdm(score_triplet_positions, desc="  Triplets", leave=False)):
        # Add intermediate control tokens
        if time_pos > last_pos:
            intermediate = tokens[last_pos:time_pos]
            beams = [(score, seq + intermediate) for score, seq in beams]
        
        # Generate all complete triplets for all beams
        # This explores num_beams^3 complete triplets before pruning
        all_triplet_candidates = []
        
        for beam_score, beam_seq in beams:
            # STEP 1: Predict TIME tokens
            with torch.no_grad():
                input_ids = torch.tensor([beam_seq], device=device)
                outputs = model(input_ids)
                time_logits = outputs.logits[0, -1, :]
                time_log_probs = torch.nn.functional.log_softmax(time_logits, dim=-1)
                
                # Get top num_beams TIME candidates
                top_time_log_probs, top_time_indices = torch.topk(time_log_probs, num_beams)
            
            # STEP 2: For each TIME, predict DURATION tokens
            for time_idx in range(num_beams):
                time_token = top_time_indices[time_idx].item()
                time_log_prob = top_time_log_probs[time_idx].item()
                
                time_seq = beam_seq + [time_token]
                
                with torch.no_grad():
                    input_ids = torch.tensor([time_seq], device=device)
                    outputs = model(input_ids)
                    dur_logits = outputs.logits[0, -1, :]
                    dur_log_probs = torch.nn.functional.log_softmax(dur_logits, dim=-1)
                    
                    # Get top num_beams DURATION candidates
                    top_dur_log_probs, top_dur_indices = torch.topk(dur_log_probs, num_beams)
                
                # STEP 3: For each (TIME, DURATION), predict PITCH tokens
                for dur_idx in range(num_beams):
                    dur_token = top_dur_indices[dur_idx].item()
                    dur_log_prob = top_dur_log_probs[dur_idx].item()
                    
                    dur_seq = time_seq + [dur_token]
                    
                    with torch.no_grad():
                        input_ids = torch.tensor([dur_seq], device=device)
                        outputs = model(input_ids)
                        pitch_logits = outputs.logits[0, -1, :]
                        pitch_log_probs = torch.nn.functional.log_softmax(pitch_logits, dim=-1)
                        
                        # Get top num_beams PITCH candidates
                        top_pitch_log_probs, top_pitch_indices = torch.topk(pitch_log_probs, num_beams)
                    
                    # STEP 4: Create complete triplet candidates
                    for pitch_idx in range(num_beams):
                        pitch_token = top_pitch_indices[pitch_idx].item()
                        pitch_log_prob = top_pitch_log_probs[pitch_idx].item()
                        
                        # Complete triplet
                        triplet_seq = dur_seq + [pitch_token]
                        triplet_score = beam_score + time_log_prob + dur_log_prob + pitch_log_prob
                        
                        all_triplet_candidates.append((triplet_score, triplet_seq))
        
        # Prune to top num_beams complete triplets
        all_triplet_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_triplet_candidates[:num_beams]
        
        last_pos = pitch_pos + 1
        
        if triplet_idx % 20 == 0:
            torch.cuda.empty_cache()
    
    # Extract predictions from best beam
    best_score, best_seq = beams[0]
    
    pred_idx = first_score_time_pos
    prev_pos = first_score_time_pos
    
    for time_pos, dur_pos, pitch_pos in score_triplet_positions:
        if time_pos > prev_pos:
            num_intermediate = time_pos - prev_pos
            pred_idx += num_intermediate
        
        # Extract TIME
        if pred_idx < len(best_seq):
            pred_time = best_seq[pred_idx]
            gt_time = tokens[time_pos]
            predictions['time'].append(pred_time)
            ground_truth['time'].append(gt_time)
            correct['time'].append(pred_time == gt_time)
            log_probs['time'].append(0.0)
            pred_idx += 1
        
        # Extract DURATION
        if pred_idx < len(best_seq):
            pred_dur = best_seq[pred_idx]
            gt_dur = tokens[dur_pos]
            predictions['duration'].append(pred_dur)
            ground_truth['duration'].append(gt_dur)
            correct['duration'].append(pred_dur == gt_dur)
            log_probs['duration'].append(0.0)
            pred_idx += 1
        
        # Extract PITCH
        if pred_idx < len(best_seq):
            pred_pitch = best_seq[pred_idx]
            gt_pitch = tokens[pitch_pos]
            predictions['pitch'].append(pred_pitch)
            ground_truth['pitch'].append(gt_pitch)
            correct['pitch'].append(pred_pitch == gt_pitch)
            log_probs['pitch'].append(0.0)
            pred_idx += 1
        
        prev_pos = pitch_pos + 1
    
    # best_seq is the generated sequence
    generated_seq = best_seq
    
    return predictions, log_probs, ground_truth, correct, generated_seq

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run analysis on multiple models
    models_to_test = [
        ('50_model', 'results_50_triplet_beam'),
        ('100_model', 'results_100_triplet_beam'),
        ('150_model', 'results_150_triplet_beam')
    ]
    
    test_file = 'data/test_sliding.txt'
    num_sequences = 20
    num_beams = 5
    
    # Load test data once and sample the same sequences for all models
    print("\nLoading test data and sampling sequences...")
    with open(test_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    random.seed(42)
    sampled_lines = random.sample(lines, min(num_sequences, len(lines)))
    print(f"Selected {len(sampled_lines)} sequences (same for all models)")
    
    # Store aggregate results across all models for comparison
    model_comparison = {}
    
    for model_path, results_dir in models_to_test:
        print("\n" + "="*80)
        print(f"ANALYZING MODEL: {model_path}")
        print("="*80)
        
        # Check if model directory exists
        if not os.path.exists(model_path):
            print(f"ERROR: Model directory '{model_path}' not found. Skipping...")
            continue
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        print("="*80)
        print("TRIPLET-LEVEL BEAM SEARCH ANALYSIS")
        print("="*80)
        print(f"Model: {model_path}")
        print(f"Test sequences: {num_sequences}")
        print(f"Beam width: {num_beams}")
        print(f"Triplet candidates per beam: {num_beams}^3 = {num_beams**3}")
        print(f"Results directory: {results_dir}")
        
        # Load model (local_files_only to prevent HuggingFace downloads)
        print(f"\nLoading model from local directory...")
        try:
            model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
        except Exception as e:
            print(f"ERROR loading model: {e}")
            print(f"Skipping {model_path}...")
            continue
        model = model.to(device)
        model.eval()
        
        # Parse sequences
        all_sequences = []
        for line in sampled_lines:
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
            
            if score_triplet_positions:
                all_sequences.append((tokens, score_triplet_positions))
        
        print(f"Valid sequences: {len(all_sequences)}")
        
        # Store results across all sequences for aggregate plot
        all_greedy_losses = {'time': [], 'duration': [], 'pitch': []}
        all_greedy_correct = {'time': [], 'duration': [], 'pitch': []}
        all_beam_correct = {'time': [], 'duration': [], 'pitch': []}
        
        # Run analysis on each sequence
        with torch.no_grad():
            for seq_idx, (tokens, score_triplet_positions) in enumerate(tqdm(all_sequences, desc="Sequences")):
                print(f"\n{'='*80}")
                print(f"Sequence {seq_idx + 1}/{len(all_sequences)}")
                print(f"{'='*80}")
                print(f"Total triplets: {len(score_triplet_positions)}")
                
                # Run greedy
                print("Running greedy decoding...")
                greedy_preds, greedy_lp, greedy_gt, greedy_corr, greedy_seq = greedy_with_tracking(
                    model, tokens, score_triplet_positions, device
                )
                
                # Run triplet-level beam search
                print(f"Running triplet-level beam search (num_beams={num_beams})...")
                beam_preds, beam_lp, beam_gt, beam_corr, beam_seq = triplet_beam_search_with_tracking(
                    model, tokens, score_triplet_positions, num_beams, device
                )
                
                # Store for aggregate plot
                for tok_type in ['time', 'duration', 'pitch']:
                    greedy_loss = [-lp for lp in greedy_lp[tok_type]]
                    all_greedy_losses[tok_type].extend(greedy_loss)
                    all_greedy_correct[tok_type].extend(greedy_corr[tok_type])
                    all_beam_correct[tok_type].extend(beam_corr[tok_type])
                
                # Create plots
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle(f'Sequence {seq_idx + 1} - Greedy vs Triplet-Level Beam Search (beams={num_beams})', fontsize=16)
                
                token_types = ['time', 'duration', 'pitch']
                
                # Row 1: Loss progression (negative log prob)
                for col, tok_type in enumerate(token_types):
                    ax = axes[0, col]
                    
                    # Greedy loss
                    greedy_loss = [-lp for lp in greedy_lp[tok_type]]
                    x = np.arange(len(greedy_loss))
                    
                    ax.plot(x, greedy_loss, 'o-', label='Greedy', alpha=0.7, markersize=3)
                    
                    # Mark errors
                    greedy_errors = [i for i, c in enumerate(greedy_corr[tok_type]) if not c]
                    if greedy_errors:
                        ax.scatter([greedy_errors], [greedy_loss[i] for i in greedy_errors], 
                                  color='red', s=50, marker='x', label='Greedy errors', zorder=5)
                    
                    beam_errors = [i for i, c in enumerate(beam_corr[tok_type]) if not c]
                    if beam_errors:
                        ax.scatter([beam_errors], [greedy_loss[i] for i in beam_errors], 
                                  color='orange', s=50, marker='^', label='Beam errors', zorder=5)
                    
                    ax.set_xlabel('Triplet Index')
                    ax.set_ylabel('Loss (negative log prob)')
                    ax.set_title(f'{tok_type.upper()} - Loss Progression')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # Row 2: Error distribution
                for col, tok_type in enumerate(token_types):
                    ax = axes[1, col]
                    
                    greedy_acc = sum(greedy_corr[tok_type]) / len(greedy_corr[tok_type]) * 100
                    beam_acc = sum(beam_corr[tok_type]) / len(beam_corr[tok_type]) * 100
                    
                    methods = ['Greedy', 'Triplet Beam']
                    accuracies = [greedy_acc, beam_acc]
                    colors = ['skyblue', 'lightcoral']
                    
                    bars = ax.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black')
                    
                    # Add percentage labels on bars
                    for bar, acc in zip(bars, accuracies):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{acc:.1f}%',
                               ha='center', va='bottom', fontsize=12, fontweight='bold')
                    
                    ax.set_ylabel('Accuracy (%)')
                    ax.set_title(f'{tok_type.upper()} - Accuracy')
                    ax.set_ylim([0, 105])
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add error count
                    greedy_errors = len([c for c in greedy_corr[tok_type] if not c])
                    beam_errors = len([c for c in beam_corr[tok_type] if not c])
                    ax.text(0, greedy_acc + 5, f'{greedy_errors} errors', ha='center', fontsize=9)
                    ax.text(1, beam_acc + 5, f'{beam_errors} errors', ha='center', fontsize=9)
                
                # Save MIDI outputs first to create directory
                print("Saving MIDI outputs...")
                output_dir = os.path.join(results_dir, f'seq{seq_idx + 1}')
                save_midi_outputs(tokens, greedy_seq, beam_seq, output_dir, seq_idx + 1)
                
                # Save plot to same directory
                plt.tight_layout()
                plot_path = os.path.join(output_dir, f'analysis.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {plot_path}")
                plt.close()
                
                # Print summary
                print(f"Results:")
                print(f"{'Token Type':<12} {'Greedy Acc':<15} {'Beam Acc':<15} {'Improvement':<15}")
                print("-" * 60)
                for tok_type in token_types:
                    greedy_acc = sum(greedy_corr[tok_type]) / len(greedy_corr[tok_type]) * 100
                    beam_acc = sum(beam_corr[tok_type]) / len(beam_corr[tok_type]) * 100
                    improvement = beam_acc - greedy_acc
                    print(f"{tok_type.upper():<12} {greedy_acc:>6.2f}%{'':<8} {beam_acc:>6.2f}%{'':<8} {improvement:>+6.2f}%")
        
        # Create aggregate plot across all sequences
        print(f"\n{'='*80}")
        print("Creating aggregate loss progression plot...")
        print(f"{'='*80}")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Aggregate Loss Progression - {len(all_sequences)} Sequences (Greedy)', fontsize=16)
        
        token_types = ['time', 'duration', 'pitch']
        
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
        
        # Print aggregate statistics
        print(f"\n{'='*80}")
        print("AGGREGATE RESULTS")
        print(f"{'='*80}")
        print(f"Total sequences: {len(all_sequences)}")
        print(f"Total triplets: {len(all_greedy_losses['time'])}")
        print()
        
        print(f"{'Token Type':<12} {'Mean Loss':<15} {'Greedy Acc':<15} {'Beam Acc':<15} {'Improvement':<15}")
        print("-" * 75)
        for tok_type in token_types:
            mean_loss = np.mean(all_greedy_losses[tok_type])
            greedy_acc = sum(all_greedy_correct[tok_type]) / len(all_greedy_correct[tok_type]) * 100
            beam_acc = sum(all_beam_correct[tok_type]) / len(all_beam_correct[tok_type]) * 100
            improvement = beam_acc - greedy_acc
            
            print(f"{tok_type.upper():<12} {mean_loss:>6.3f}{'':<9} {greedy_acc:>6.2f}%{'':<8} {beam_acc:>6.2f}%{'':<8} {improvement:>+6.2f}%")
        
        # Create aggregate accuracy comparison plot
        print(f"\n{'='*80}")
        print("Creating aggregate accuracy comparison plot...")
        print(f"{'='*80}")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(f'Aggregate Accuracy Comparison - {len(all_sequences)} Sequences', fontsize=16)
        
        x = np.arange(len(token_types))
        width = 0.35
        
        greedy_accs = [sum(all_greedy_correct[tok]) / len(all_greedy_correct[tok]) * 100 for tok in token_types]
        beam_accs = [sum(all_beam_correct[tok]) / len(all_beam_correct[tok]) * 100 for tok in token_types]
        
        bars1 = ax.bar(x - width/2, greedy_accs, width, label='Greedy', alpha=0.8)
        bars2 = ax.bar(x + width/2, beam_accs, width, label='Triplet Beam', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Token Type Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([tok.upper() for tok in token_types])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        aggregate_acc_path = os.path.join(results_dir, 'aggregate_accuracy.png')
        plt.savefig(aggregate_acc_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {aggregate_acc_path}")
        plt.close()
        
        print(f"\n{'='*80}")
        print(f"Analysis complete for {model_path}!")
        print(f"{'='*80}")
        print(f"All results saved to: {results_dir}/")
        
        # Store results for cross-model comparison
        model_comparison[model_path] = {
            'greedy': {tok: sum(all_greedy_correct[tok]) / len(all_greedy_correct[tok]) * 100 
                      for tok in ['time', 'duration', 'pitch']},
            'beam': {tok: sum(all_beam_correct[tok]) / len(all_beam_correct[tok]) * 100 
                    for tok in ['time', 'duration', 'pitch']}
        }
        
        # Clean up model from GPU
        del model
        torch.cuda.empty_cache()
    
    # Create cross-model comparison plot
    print("\n" + "="*80)
    print("CREATING CROSS-MODEL COMPARISON")
    print("="*80)
    
    if len(model_comparison) == 0:
        print("ERROR: No models were successfully analyzed. Exiting.")
        return
    
    token_types = ['time', 'duration', 'pitch']
    model_names = list(model_comparison.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Comparison: Greedy vs Triplet-Level Beam Search', fontsize=16)
    
    for col, tok_type in enumerate(token_types):
        ax = axes[col]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        greedy_accs = [model_comparison[model]['greedy'][tok_type] for model in model_names]
        beam_accs = [model_comparison[model]['beam'][tok_type] for model in model_names]
        
        bars1 = ax.bar(x - width/2, greedy_accs, width, label='Greedy', alpha=0.8)
        bars2 = ax.bar(x + width/2, beam_accs, width, label='Triplet Beam', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{tok_type.upper()} Token')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 105)
    
    plt.tight_layout()
    comparison_path = 'model_comparison_greedy_vs_triplet_beam.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {comparison_path}")
    plt.close()
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Model':<15} {'Method':<12} {'TIME Acc':<12} {'DUR Acc':<12} {'PITCH Acc':<12}")
    print("-" * 63)
    for model in model_names:
        greedy_data = model_comparison[model]['greedy']
        beam_data = model_comparison[model]['beam']
        
        print(f"{model:<15} {'Greedy':<12} {greedy_data['time']:>6.2f}%{'':<5} {greedy_data['duration']:>6.2f}%{'':<5} {greedy_data['pitch']:>6.2f}%")
        print(f"{'':<15} {'Triplet Beam':<12} {beam_data['time']:>6.2f}%{'':<5} {beam_data['duration']:>6.2f}%{'':<5} {beam_data['pitch']:>6.2f}%")
        
        # Calculate improvements
        time_imp = beam_data['time'] - greedy_data['time']
        dur_imp = beam_data['duration'] - greedy_data['duration']
        pitch_imp = beam_data['pitch'] - greedy_data['pitch']
        print(f"{'':<15} {'Improvement':<12} {time_imp:>+6.2f}%{'':<5} {dur_imp:>+6.2f}%{'':<5} {pitch_imp:>+6.2f}%")
        print("-" * 63)
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
