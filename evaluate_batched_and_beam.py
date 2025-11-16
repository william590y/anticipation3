"""
Evaluate all three models with batched inference and beam search.

Two evaluation modes:
1. Greedy decoding (batched for speed)
2. Beam search decoding (num_beams=100)

Both use batched inference for parallelization.
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET, NOTE_OFFSET
from tqdm import tqdm
import random
import numpy as np

def evaluate_greedy_batched(model_path, model_name, test_file, num_sequences=100, batch_size=16, device='cuda'):
    """
    Evaluate with greedy decoding using batched KV caching for speed.
    
    Args:
        model_path: Path to the model
        model_name: Name for display
        test_file: Test data file
        num_sequences: Number of sequences to evaluate
        batch_size: Batch size for parallel processing
        device: Device to use
    """
    print(f"\n{'='*80}")
    print(f"GREEDY DECODING - {model_name}")
    print(f"{'='*80}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Load test data
    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Sample sequences
    random.seed(42)
    sampled_lines = random.sample(lines, min(num_sequences, len(lines)))
    print(f"Sampled {len(sampled_lines)} sequences")
    
    # Parse all sequences
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
    
    # Accuracy statistics
    stats = {
        'time': {'correct': 0, 'total': 0},
        'duration': {'correct': 0, 'total': 0},
        'pitch': {'correct': 0, 'total': 0},
    }
    
    print(f"\nEvaluating with greedy decoding (sequential with KV caching)...")
    
    with torch.no_grad():
        # Process sequences one by one (simpler and faster than complex batching)
        for tokens, score_triplet_positions in tqdm(all_sequences, desc="Sequences"):
            first_score_time_pos = score_triplet_positions[0][0]
            init_context = torch.tensor([tokens[:first_score_time_pos]]).to(device)
            outputs = model(init_context, past_key_values=None, use_cache=True)
            past_key_values = outputs.past_key_values
            last_pos = first_score_time_pos
            
            for time_pos, dur_pos, pitch_pos in score_triplet_positions:
                # Process intermediate control tokens
                if time_pos > last_pos:
                    intermediate = torch.tensor([tokens[last_pos:time_pos]]).to(device)
                    outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
                    past_key_values = outputs.past_key_values
                
                # Predict TIME
                logits = outputs.logits[0, -1]
                pred_time = logits.argmax().item()
                gt_time = tokens[time_pos]
                
                stats['time']['total'] += 1
                if pred_time == gt_time:
                    stats['time']['correct'] += 1
                
                next_token = torch.tensor([[pred_time]]).to(device)
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                # Predict DURATION
                logits = outputs.logits[0, -1]
                pred_dur = logits.argmax().item()
                gt_dur = tokens[dur_pos]
                
                stats['duration']['total'] += 1
                if pred_dur == gt_dur:
                    stats['duration']['correct'] += 1
                
                next_token = torch.tensor([[pred_dur]]).to(device)
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                # Predict PITCH
                logits = outputs.logits[0, -1]
                pred_pitch = logits.argmax().item()
                gt_pitch = tokens[pitch_pos]
                
                stats['pitch']['total'] += 1
                if pred_pitch == gt_pitch:
                    stats['pitch']['correct'] += 1
                
                next_token = torch.tensor([[pred_pitch]]).to(device)
                outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
                
                last_pos = pitch_pos + 1
    
    # Calculate percentages
    results = {
        'time': stats['time']['correct'] / stats['time']['total'] * 100 if stats['time']['total'] > 0 else 0,
        'duration': stats['duration']['correct'] / stats['duration']['total'] * 100 if stats['duration']['total'] > 0 else 0,
        'pitch': stats['pitch']['correct'] / stats['pitch']['total'] * 100 if stats['pitch']['total'] > 0 else 0,
        'counts': stats
    }
    
    return results

def evaluate_beam_search(model_path, model_name, test_file, num_sequences=100, num_beams=100, device='cuda'):
    """
    Evaluate with PROPER beam search decoding.
    
    Maintains num_beams hypotheses and scores complete triplet sequences.
    Uses batched inference to process all beams in parallel.
    
    Args:
        model_path: Path to the model
        model_name: Name for display
        test_file: Test data file
        num_sequences: Number of sequences to evaluate
        num_beams: Number of beams for beam search
        device: Device to use
    """
    print(f"\n{'='*80}")
    print(f"BEAM SEARCH (num_beams={num_beams}) - {model_name}")
    print(f"{'='*80}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Load test data
    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Sample sequences
    random.seed(42)
    sampled_lines = random.sample(lines, min(num_sequences, len(lines)))
    print(f"Sampled {len(sampled_lines)} sequences")
    
    # Parse all sequences
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
    
    # Accuracy statistics
    stats = {
        'time': {'correct': 0, 'total': 0},
        'duration': {'correct': 0, 'total': 0},
        'pitch': {'correct': 0, 'total': 0},
    }
    
    print(f"\nEvaluating with PROPER beam search (num_beams={num_beams}, batched)...")
    print("Beam search maintains multiple hypotheses and scores complete triplet sequences")
    
    with torch.no_grad():
        for tokens, score_triplet_positions in tqdm(all_sequences, desc="Sequences"):
            first_score_time_pos = score_triplet_positions[0][0]
            
            # Initialize beams - each beam is (score, sequence_tokens)
            # Start with single beam containing the context
            init_context = tokens[:first_score_time_pos]
            beams = [(0.0, init_context)]  # (cumulative_log_prob, tokens)
            
            last_pos = first_score_time_pos
            
            # Process each triplet
            for triplet_idx, (time_pos, dur_pos, pitch_pos) in enumerate(score_triplet_positions):
                # Add intermediate control tokens to all beams
                if time_pos > last_pos:
                    intermediate = tokens[last_pos:time_pos]
                    beams = [(score, seq + intermediate) for score, seq in beams]
                
                # Expand beams for TIME token
                new_beams = []
                
                # Batch process all current beams
                if len(beams) > 0:
                    batch_seqs = [seq for _, seq in beams]
                    batch_scores = [score for score, _ in beams]
                    
                    # Pad sequences to same length
                    max_len = max(len(seq) for seq in batch_seqs)
                    padded_seqs = []
                    attention_masks = []
                    
                    for seq in batch_seqs:
                        padding_len = max_len - len(seq)
                        padded_seq = [0] * padding_len + seq  # Left padding
                        padded_seqs.append(padded_seq)
                        attention_masks.append([0] * padding_len + [1] * len(seq))
                    
                    # Forward pass for all beams in batch
                    input_ids = torch.tensor(padded_seqs).to(device)
                    attention_mask = torch.tensor(attention_masks).to(device)
                    
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    
                    # For each beam, get top-k candidates
                    for beam_idx, (beam_score, beam_seq) in enumerate(beams):
                        beam_log_probs = log_probs[beam_idx]
                        top_k_log_probs, top_k_indices = torch.topk(beam_log_probs, min(num_beams, len(beam_log_probs)))
                        
                        for k in range(len(top_k_indices)):
                            token_id = top_k_indices[k].item()
                            token_log_prob = top_k_log_probs[k].item()
                            new_score = beam_score + token_log_prob
                            new_seq = beam_seq + [token_id]
                            new_beams.append((new_score, new_seq))
                    
                    # Keep top num_beams
                    new_beams.sort(key=lambda x: x[0], reverse=True)
                    beams = new_beams[:num_beams]
                
                # Expand beams for DURATION token
                new_beams = []
                
                if len(beams) > 0:
                    batch_seqs = [seq for _, seq in beams]
                    batch_scores = [score for score, _ in beams]
                    
                    max_len = max(len(seq) for seq in batch_seqs)
                    padded_seqs = []
                    attention_masks = []
                    
                    for seq in batch_seqs:
                        padding_len = max_len - len(seq)
                        padded_seq = [0] * padding_len + seq
                        padded_seqs.append(padded_seq)
                        attention_masks.append([0] * padding_len + [1] * len(seq))
                    
                    input_ids = torch.tensor(padded_seqs).to(device)
                    attention_mask = torch.tensor(attention_masks).to(device)
                    
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[:, -1, :]
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    
                    for beam_idx, (beam_score, beam_seq) in enumerate(beams):
                        beam_log_probs = log_probs[beam_idx]
                        top_k_log_probs, top_k_indices = torch.topk(beam_log_probs, min(num_beams, len(beam_log_probs)))
                        
                        for k in range(len(top_k_indices)):
                            token_id = top_k_indices[k].item()
                            token_log_prob = top_k_log_probs[k].item()
                            new_score = beam_score + token_log_prob
                            new_seq = beam_seq + [token_id]
                            new_beams.append((new_score, new_seq))
                    
                    new_beams.sort(key=lambda x: x[0], reverse=True)
                    beams = new_beams[:num_beams]
                
                # Expand beams for PITCH token
                new_beams = []
                
                if len(beams) > 0:
                    batch_seqs = [seq for _, seq in beams]
                    batch_scores = [score for score, _ in beams]
                    
                    max_len = max(len(seq) for seq in batch_seqs)
                    padded_seqs = []
                    attention_masks = []
                    
                    for seq in batch_seqs:
                        padding_len = max_len - len(seq)
                        padded_seq = [0] * padding_len + seq
                        padded_seqs.append(padded_seq)
                        attention_masks.append([0] * padding_len + [1] * len(seq))
                    
                    input_ids = torch.tensor(padded_seqs).to(device)
                    attention_mask = torch.tensor(attention_masks).to(device)
                    
                    outputs = model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[:, -1, :]
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    
                    for beam_idx, (beam_score, beam_seq) in enumerate(beams):
                        beam_log_probs = log_probs[beam_idx]
                        top_k_log_probs, top_k_indices = torch.topk(beam_log_probs, min(num_beams, len(beam_log_probs)))
                        
                        for k in range(len(top_k_indices)):
                            token_id = top_k_indices[k].item()
                            token_log_prob = top_k_log_probs[k].item()
                            new_score = beam_score + token_log_prob
                            new_seq = beam_seq + [token_id]
                            new_beams.append((new_score, new_seq))
                    
                    new_beams.sort(key=lambda x: x[0], reverse=True)
                    beams = new_beams[:num_beams]
                
                last_pos = pitch_pos + 1
            
            # Select best beam
            best_score, best_seq = beams[0]
            
            # Extract predictions from best sequence and compare to ground truth
            # Find the score triplets in the generated sequence
            pred_idx = first_score_time_pos
            for time_pos, dur_pos, pitch_pos in score_triplet_positions:
                # Account for intermediate control tokens
                if time_pos > first_score_time_pos:
                    # Skip intermediate controls in both
                    num_intermediate = time_pos - last_pos
                    pred_idx += num_intermediate
                
                # Compare TIME
                if pred_idx < len(best_seq):
                    pred_time = best_seq[pred_idx]
                    gt_time = tokens[time_pos]
                    stats['time']['total'] += 1
                    if pred_time == gt_time:
                        stats['time']['correct'] += 1
                    pred_idx += 1
                
                # Compare DURATION
                if pred_idx < len(best_seq):
                    pred_dur = best_seq[pred_idx]
                    gt_dur = tokens[dur_pos]
                    stats['duration']['total'] += 1
                    if pred_dur == gt_dur:
                        stats['duration']['correct'] += 1
                    pred_idx += 1
                
                # Compare PITCH
                if pred_idx < len(best_seq):
                    pred_pitch = best_seq[pred_idx]
                    gt_pitch = tokens[pitch_pos]
                    stats['pitch']['total'] += 1
                    if pred_pitch == gt_pitch:
                        stats['pitch']['correct'] += 1
                    pred_idx += 1
    
    # Calculate percentages
    results = {
        'time': stats['time']['correct'] / stats['time']['total'] * 100 if stats['time']['total'] > 0 else 0,
        'duration': stats['duration']['correct'] / stats['duration']['total'] * 100 if stats['duration']['total'] > 0 else 0,
        'pitch': stats['pitch']['correct'] / stats['pitch']['total'] * 100 if stats['pitch']['total'] > 0 else 0,
        'counts': stats
    }
    
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_file = 'data/test_sliding.txt'
    num_sequences = 5
    batch_size = 16
    num_beams = 5
    
    models = [
        ('50_model', '50-model'),
        ('100_model', '100_model'),
        ('150_model', '150_model'),
    ]
    
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)
    print(f"\nTest set: {test_file}")
    print(f"Sequences: {num_sequences}")
    print(f"Device: {device}")
    print(f"\nEvaluation modes:")
    print(f"  1. Greedy decoding (sequential with KV caching)")
    print(f"  2. Beam search (num_beams={num_beams})")
    
    # Store all results
    greedy_results = {}
    beam_results = {}
    
    # Evaluate all models - Greedy
    print("\n" + "="*80)
    print("PHASE 1: GREEDY DECODING")
    print("="*80)
    
    for model_name, model_path in models:
        results = evaluate_greedy_batched(model_path, model_name, test_file, 
                                         num_sequences, batch_size, device)
        greedy_results[model_name] = results
    
    # Evaluate all models - Beam Search
    print("\n" + "="*80)
    print("PHASE 2: BEAM SEARCH DECODING")
    print("="*80)
    
    for model_name, model_path in models:
        results = evaluate_beam_search(model_path, model_name, test_file,
                                      num_sequences, num_beams, device)
        beam_results[model_name] = results
    
    # Print comparison tables
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print("\n" + "-"*80)
    print("GREEDY DECODING RESULTS")
    print("-"*80)
    print(f"\n{'Model':<15} {'Time':<20} {'Duration':<20} {'Pitch':<20} {'Overall':<15}")
    print("-" * 80)
    
    for model_name in ['50_model', '100_model', '150_model']:
        if model_name in greedy_results:
            r = greedy_results[model_name]
            overall = (r['time'] + r['duration'] + r['pitch']) / 3
            print(f"{model_name:<15} "
                  f"{r['time']:>6.2f}% ({r['counts']['time']['correct']:>4}/{r['counts']['time']['total']:<5}) "
                  f"{r['duration']:>6.2f}% ({r['counts']['duration']['correct']:>4}/{r['counts']['duration']['total']:<5}) "
                  f"{r['pitch']:>6.2f}% ({r['counts']['pitch']['correct']:>4}/{r['counts']['pitch']['total']:<5}) "
                  f"{overall:>6.2f}%")
    
    print("\n" + "-"*80)
    print(f"BEAM SEARCH RESULTS (num_beams={num_beams})")
    print("-"*80)
    print(f"\n{'Model':<15} {'Time':<20} {'Duration':<20} {'Pitch':<20} {'Overall':<15}")
    print("-" * 80)
    
    for model_name in ['50_model', '100_model', '150_model']:
        if model_name in beam_results:
            r = beam_results[model_name]
            overall = (r['time'] + r['duration'] + r['pitch']) / 3
            print(f"{model_name:<15} "
                  f"{r['time']:>6.2f}% ({r['counts']['time']['correct']:>4}/{r['counts']['time']['total']:<5}) "
                  f"{r['duration']:>6.2f}% ({r['counts']['duration']['correct']:>4}/{r['counts']['duration']['total']:<5}) "
                  f"{r['pitch']:>6.2f}% ({r['counts']['pitch']['correct']:>4}/{r['counts']['pitch']['total']:<5}) "
                  f"{overall:>6.2f}%")
    
    print("\n" + "-"*80)
    print("IMPROVEMENT: Beam Search vs Greedy")
    print("-"*80)
    print(f"\n{'Model':<15} {'Time Δ':<15} {'Duration Δ':<15} {'Pitch Δ':<15} {'Overall Δ':<15}")
    print("-" * 80)
    
    for model_name in ['50_model', '100_model', '150_model']:
        if model_name in greedy_results and model_name in beam_results:
            g = greedy_results[model_name]
            b = beam_results[model_name]
            time_delta = b['time'] - g['time']
            dur_delta = b['duration'] - g['duration']
            pitch_delta = b['pitch'] - g['pitch']
            overall_delta = (time_delta + dur_delta + pitch_delta) / 3
            
            print(f"{model_name:<15} "
                  f"{time_delta:>+6.2f}%{'':<8} "
                  f"{dur_delta:>+6.2f}%{'':<8} "
                  f"{pitch_delta:>+6.2f}%{'':<8} "
                  f"{overall_delta:>+6.2f}%")
    
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    for model_name in ['50_model', '100_model', '150_model']:
        print(f"\n{model_name}:")
        
        if model_name in greedy_results:
            g = greedy_results[model_name]
            print(f"  Greedy Decoding:")
            print(f"    Time:     {g['time']:>6.2f}% ({g['counts']['time']['correct']}/{g['counts']['time']['total']})")
            print(f"    Duration: {g['duration']:>6.2f}% ({g['counts']['duration']['correct']}/{g['counts']['duration']['total']})")
            print(f"    Pitch:    {g['pitch']:>6.2f}% ({g['counts']['pitch']['correct']}/{g['counts']['pitch']['total']})")
            overall_g = (g['time'] + g['duration'] + g['pitch']) / 3
            print(f"    Overall:  {overall_g:>6.2f}%")
        
        if model_name in beam_results:
            b = beam_results[model_name]
            print(f"  Beam Search (num_beams={num_beams}):")
            print(f"    Time:     {b['time']:>6.2f}% ({b['counts']['time']['correct']}/{b['counts']['time']['total']})")
            print(f"    Duration: {b['duration']:>6.2f}% ({b['counts']['duration']['correct']}/{b['counts']['duration']['total']})")
            print(f"    Pitch:    {b['pitch']:>6.2f}% ({b['counts']['pitch']['correct']}/{b['counts']['pitch']['total']})")
            overall_b = (b['time'] + b['duration'] + b['pitch']) / 3
            print(f"    Overall:  {overall_b:>6.2f}%")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
