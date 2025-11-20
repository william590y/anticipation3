"""
Analyze triplet-aware Monte Carlo Tree Search (MCTS) with loss progression and error distribution.

For each sequence:
- Run greedy and MCTS
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
import math

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

def save_midi_outputs(tokens, greedy_seq, mcts_seq, output_dir, seq_idx):
    """Save MIDI files for input performance, ground truth, greedy, and MCTS outputs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract performance (input)
    perf_events = extract_performance_only(tokens)
    perf_midi_path = os.path.join(output_dir, f'seq{seq_idx}_input_performance.mid')
    perf_midi = events_to_midi(perf_events)
    perf_midi.save(perf_midi_path)
    
    # Convert tokens to events (score only - no control/performance tokens)
    gt_events = extract_score_only(tokens)
    greedy_events = extract_score_only(greedy_seq)
    mcts_events = extract_score_only(mcts_seq)
    
    # Save as MIDI
    gt_midi_path = os.path.join(output_dir, f'seq{seq_idx}_ground_truth.mid')
    gt_midi = events_to_midi(gt_events)
    gt_midi.save(gt_midi_path)
    
    greedy_midi_path = os.path.join(output_dir, f'seq{seq_idx}_greedy.mid')
    greedy_midi = events_to_midi(greedy_events)
    greedy_midi.save(greedy_midi_path)
    
    mcts_midi_path = os.path.join(output_dir, f'seq{seq_idx}_mcts.mid')
    mcts_midi = events_to_midi(mcts_events)
    mcts_midi.save(mcts_midi_path)
    
    return perf_midi_path, gt_midi_path, greedy_midi_path, mcts_midi_path

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

class MCTSNode:
    """Node in the MCTS tree for triplet generation."""
    
    def __init__(self, sequence, parent=None, token=None, log_prob=0.0):
        self.sequence = sequence  # Current token sequence
        self.parent = parent
        self.token = token  # Token that led to this node
        self.log_prob = log_prob  # Log probability of the token
        self.children = {}  # token -> MCTSNode
        self.visits = 0
        self.total_value = 0.0
        self.untried_tokens = None  # Will be set during expansion
        
    def is_fully_expanded(self):
        return self.untried_tokens is not None and len(self.untried_tokens) == 0
    
    def best_child(self, c_param=1.414):
        """Select best child using UCB1 formula."""
        choices_weights = []
        for child in self.children.values():
            if child.visits == 0:
                return child
            # UCB1: exploitation + exploration
            exploit = child.total_value / child.visits
            explore = c_param * math.sqrt(math.log(self.visits) / child.visits)
            choices_weights.append(exploit + explore)
        
        return list(self.children.values())[choices_weights.index(max(choices_weights))]
    
    def best_final_child(self):
        """Select best child for final decision (max visits)."""
        return max(self.children.values(), key=lambda c: c.visits)

def mcts_with_tracking(model, tokens, score_triplet_positions, num_simulations, device):
    """
    Triplet-aware MCTS with detailed tracking.
    
    Args:
        num_simulations: Number of MCTS simulations per triplet position
    """
    first_score_time_pos = score_triplet_positions[0][0]
    init_context = tokens[:first_score_time_pos]
    
    last_pos = first_score_time_pos
    
    # Track predictions and log probs at each step
    predictions = {'time': [], 'duration': [], 'pitch': []}
    log_probs = {'time': [], 'duration': [], 'pitch': []}
    ground_truth = {'time': [], 'duration': [], 'pitch': []}
    correct = {'time': [], 'duration': [], 'pitch': []}
    
    # Build generated sequence
    generated_seq = list(init_context)
    
    for triplet_idx, (time_pos, dur_pos, pitch_pos) in enumerate(tqdm(score_triplet_positions, desc="  Triplets", leave=False)):
        # Add intermediate control tokens
        if time_pos > last_pos:
            intermediate = tokens[last_pos:time_pos]
            generated_seq.extend(intermediate)
        
        # MCTS for each token in the triplet
        for token_idx, (gt_pos, token_type) in enumerate([(time_pos, 'time'), (dur_pos, 'duration'), (pitch_pos, 'pitch')]):
            # Root node with current sequence
            root = MCTSNode(generated_seq.copy())
            
            # Run MCTS simulations
            for _ in range(num_simulations):
                node = root
                
                # 1. Selection: traverse tree using UCB1
                while node.is_fully_expanded() and len(node.children) > 0:
                    node = node.best_child()
                
                # 2. Expansion: add new child if not fully expanded
                if not node.is_fully_expanded():
                    # Get model predictions for this sequence
                    with torch.no_grad():
                        input_ids = torch.tensor([node.sequence], device=device)
                        outputs = model(input_ids)
                        logits = outputs.logits[0, -1, :]
                        log_probs_tensor = torch.nn.functional.log_softmax(logits, dim=-1)
                        
                        # Get top-k candidates for expansion
                        top_k = min(50, logits.size(0))  # Expand top 50 tokens
                        top_log_probs, top_indices = torch.topk(log_probs_tensor, top_k)
                        
                        if node.untried_tokens is None:
                            node.untried_tokens = [(top_indices[i].item(), top_log_probs[i].item()) 
                                                   for i in range(top_k)]
                    
                    # Expand one untried token
                    if len(node.untried_tokens) > 0:
                        token, token_log_prob = node.untried_tokens.pop(0)
                        new_seq = node.sequence + [token]
                        child = MCTSNode(new_seq, parent=node, token=token, log_prob=token_log_prob)
                        node.children[token] = child
                        node = child
                
                # 3. Simulation: rollout from this node
                # For language models, we use the log probability as the value
                value = node.log_prob
                
                # 4. Backpropagation: update all ancestors
                while node is not None:
                    node.visits += 1
                    node.total_value += value
                    node = node.parent
            
            # Select best token based on visit counts
            if len(root.children) > 0:
                best_child = root.best_final_child()
                pred_token = best_child.token
                pred_log_prob = best_child.log_prob
            else:
                # Fallback to greedy if no children (shouldn't happen)
                with torch.no_grad():
                    input_ids = torch.tensor([generated_seq], device=device)
                    outputs = model(input_ids)
                    logits = outputs.logits[0, -1, :]
                    log_probs_tensor = torch.nn.functional.log_softmax(logits, dim=-1)
                    pred_token = logits.argmax().item()
                    pred_log_prob = log_probs_tensor[pred_token].item()
            
            gt_token = tokens[gt_pos]
            
            predictions[token_type].append(pred_token)
            log_probs[token_type].append(pred_log_prob)
            ground_truth[token_type].append(gt_token)
            correct[token_type].append(pred_token == gt_token)
            generated_seq.append(pred_token)
        
        last_pos = pitch_pos + 1
    
    return predictions, log_probs, ground_truth, correct, generated_seq

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run analysis on multiple models
    models_to_test = [
        ('50_model', 'results_50_mcts'),
        ('100_model', 'results_100_mcts'),
        ('150_model', 'results_150_mcts')
    ]
    
    test_file = 'data/test_sliding.txt'
    num_sequences = 20
    num_simulations = 100  # Number of MCTS simulations per token
    
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
        print("TRIPLET MCTS ANALYSIS")
        print("="*80)
        print(f"Model: {model_path}")
        print(f"Test sequences: {num_sequences}")
        print(f"Simulations per token: {num_simulations}")
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
        all_mcts_correct = {'time': [], 'duration': [], 'pitch': []}
        
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
                
                # Run MCTS
                print(f"Running MCTS (simulations={num_simulations})...")
                mcts_preds, mcts_lp, mcts_gt, mcts_corr, mcts_seq = mcts_with_tracking(
                    model, tokens, score_triplet_positions, num_simulations, device
                )
                
                # Store for aggregate plot
                for tok_type in ['time', 'duration', 'pitch']:
                    greedy_loss = [-lp for lp in greedy_lp[tok_type]]
                    all_greedy_losses[tok_type].extend(greedy_loss)
                    all_greedy_correct[tok_type].extend(greedy_corr[tok_type])
                    all_mcts_correct[tok_type].extend(mcts_corr[tok_type])
                
                # Create plots
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle(f'Sequence {seq_idx + 1} - Greedy vs MCTS (simulations={num_simulations})', fontsize=16)
                
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
                    
                    mcts_errors = [i for i, c in enumerate(mcts_corr[tok_type]) if not c]
                    if mcts_errors:
                        ax.scatter([mcts_errors], [greedy_loss[i] for i in mcts_errors], 
                                  color='orange', s=50, marker='^', label='MCTS errors', zorder=5)
                    
                    ax.set_xlabel('Triplet Index')
                    ax.set_ylabel('Loss (negative log prob)')
                    ax.set_title(f'{tok_type.upper()} - Loss Progression')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                # Row 2: Error distribution
                for col, tok_type in enumerate(token_types):
                    ax = axes[1, col]
                    
                    greedy_acc = sum(greedy_corr[tok_type]) / len(greedy_corr[tok_type]) * 100
                    mcts_acc = sum(mcts_corr[tok_type]) / len(mcts_corr[tok_type]) * 100
                    
                    methods = ['Greedy', 'MCTS']
                    accuracies = [greedy_acc, mcts_acc]
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
                    mcts_errors = len([c for c in mcts_corr[tok_type] if not c])
                    ax.text(0, greedy_acc + 5, f'{greedy_errors} errors', ha='center', fontsize=9)
                    ax.text(1, mcts_acc + 5, f'{mcts_errors} errors', ha='center', fontsize=9)
                
                # Save MIDI outputs first to create directory
                print("Saving MIDI outputs...")
                output_dir = os.path.join(results_dir, f'seq{seq_idx + 1}')
                save_midi_outputs(tokens, greedy_seq, mcts_seq, output_dir, seq_idx + 1)
                
                # Save plot to same directory
                plt.tight_layout()
                plot_path = os.path.join(output_dir, f'analysis.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"Saved: {plot_path}")
                plt.close()
                
                # Print summary
                print(f"Results:")
                print(f"{'Token Type':<12} {'Greedy Acc':<15} {'MCTS Acc':<15} {'Improvement':<15}")
                print("-" * 60)
                for tok_type in token_types:
                    greedy_acc = sum(greedy_corr[tok_type]) / len(greedy_corr[tok_type]) * 100
                    mcts_acc = sum(mcts_corr[tok_type]) / len(mcts_corr[tok_type]) * 100
                    improvement = mcts_acc - greedy_acc
                    print(f"{tok_type.upper():<12} {greedy_acc:>6.2f}%{'':<8} {mcts_acc:>6.2f}%{'':<8} {improvement:>+6.2f}%")
        
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
        
        print(f"{'Token Type':<12} {'Mean Loss':<15} {'Greedy Acc':<15} {'MCTS Acc':<15} {'Improvement':<15}")
        print("-" * 75)
        for tok_type in token_types:
            mean_loss = np.mean(all_greedy_losses[tok_type])
            greedy_acc = sum(all_greedy_correct[tok_type]) / len(all_greedy_correct[tok_type]) * 100
            mcts_acc = sum(all_mcts_correct[tok_type]) / len(all_mcts_correct[tok_type]) * 100
            improvement = mcts_acc - greedy_acc
            
            print(f"{tok_type.upper():<12} {mean_loss:>6.3f}{'':<9} {greedy_acc:>6.2f}%{'':<8} {mcts_acc:>6.2f}%{'':<8} {improvement:>+6.2f}%")
        
        # Create aggregate accuracy comparison plot
        print(f"\n{'='*80}")
        print("Creating aggregate accuracy comparison plot...")
        print(f"{'='*80}")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(f'Aggregate Accuracy Comparison - {len(all_sequences)} Sequences', fontsize=16)
        
        x = np.arange(len(token_types))
        width = 0.35
        
        greedy_accs = [sum(all_greedy_correct[tok]) / len(all_greedy_correct[tok]) * 100 for tok in token_types]
        mcts_accs = [sum(all_mcts_correct[tok]) / len(all_mcts_correct[tok]) * 100 for tok in token_types]
        
        bars1 = ax.bar(x - width/2, greedy_accs, width, label='Greedy', alpha=0.8)
        bars2 = ax.bar(x + width/2, mcts_accs, width, label='MCTS', alpha=0.8)
        
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
            'mcts': {tok: sum(all_mcts_correct[tok]) / len(all_mcts_correct[tok]) * 100 
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
    fig.suptitle('Model Comparison: Greedy vs MCTS Across Training Checkpoints', fontsize=16)
    
    for col, tok_type in enumerate(token_types):
        ax = axes[col]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        greedy_accs = [model_comparison[model]['greedy'][tok_type] for model in model_names]
        mcts_accs = [model_comparison[model]['mcts'][tok_type] for model in model_names]
        
        bars1 = ax.bar(x - width/2, greedy_accs, width, label='Greedy', alpha=0.8)
        bars2 = ax.bar(x + width/2, mcts_accs, width, label='MCTS', alpha=0.8)
        
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
    comparison_path = 'model_comparison_greedy_vs_mcts.png'
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
        mcts_data = model_comparison[model]['mcts']
        
        print(f"{model:<15} {'Greedy':<12} {greedy_data['time']:>6.2f}%{'':<5} {greedy_data['duration']:>6.2f}%{'':<5} {greedy_data['pitch']:>6.2f}%")
        print(f"{'':<15} {'MCTS':<12} {mcts_data['time']:>6.2f}%{'':<5} {mcts_data['duration']:>6.2f}%{'':<5} {mcts_data['pitch']:>6.2f}%")
        
        # Calculate improvements
        time_imp = mcts_data['time'] - greedy_data['time']
        dur_imp = mcts_data['duration'] - greedy_data['duration']
        pitch_imp = mcts_data['pitch'] - greedy_data['pitch']
        print(f"{'':<15} {'Improvement':<12} {time_imp:>+6.2f}%{'':<5} {dur_imp:>+6.2f}%{'':<5} {pitch_imp:>+6.2f}%")
        print("-" * 63)
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
