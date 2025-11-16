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
            
            # Process each triplet with BATCHED JOINT TRIPLET BEAM SEARCH
            # This scores complete (TIME, DURATION, PITCH) triplets instead of individual tokens
            for triplet_idx, (time_pos, dur_pos, pitch_pos) in enumerate(tqdm(score_triplet_positions, desc="Triplets", leave=False)):
                # Add intermediate control tokens to all beams
                if time_pos > last_pos:
                    intermediate = tokens[last_pos:time_pos]
                    beams = [(score, seq + intermediate) for score, seq in beams]
                
                # BATCHED JOINT TRIPLET EXPANSION
                # Process all beams in parallel with batched inference
                # Memory-conscious: limit expansion to avoid OOM
                new_beams = []
                
                if len(beams) > 0:
                    # Memory-aware exploration budget
                    # The real memory issue is sequence length × batch size × hidden_dim
                    # For long sequences, we must reduce batch size drastically
                    seq_len = len(beams[0][1])
                    
                    # Adjust k values based on how expensive batching will be
                    if seq_len > 800:
                        k_time, k_dur, k_pitch = 1, 1, 1  # Greedy for very long sequences
                        chunk_size = 1  # Process one at a time
                    elif seq_len > 500:
                        k_time, k_dur, k_pitch = 2, 1, 1
                        chunk_size = 5  # Very small batches
                    elif seq_len > 300:
                        k_time, k_dur, k_pitch = 2, 2, 1
                        chunk_size = 10
                    else:
                        # Short sequences can handle more exploration
                        if num_beams <= 5:
                            k_time, k_dur, k_pitch = 3, 2, 1
                        elif num_beams <= 10:
                            k_time, k_dur, k_pitch = 2, 2, 1
                        else:
                            k_time, k_dur, k_pitch = 2, 1, 1
                        chunk_size = 20
                    
                    # === STEP 1: Get TIME candidates (batched) ===
                    batch_seqs = [seq for _, seq in beams]
                    
                    # Pad sequences to same length
                    max_len = max(len(seq) for seq in batch_seqs)
                    padded_seqs = []
                    attention_masks = []
                    
                    for seq in batch_seqs:
                        padding_len = max_len - len(seq)
                        padded_seq = [0] * padding_len + seq  # Left padding
                        padded_seqs.append(padded_seq)
                        attention_masks.append([0] * padding_len + [1] * len(seq))
                    
                    # Batched forward pass for all beams
                    input_ids = torch.tensor(padded_seqs, device=device)
                    attention_mask = torch.tensor(attention_masks, device=device)
                    
                    outputs = model(input_ids, attention_mask=attention_mask)
                    time_logits = outputs.logits[:, -1, :]  # (num_beams, vocab_size)
                    time_log_probs = torch.nn.functional.log_softmax(time_logits, dim=-1)
                    
                    # Get top-k TIME candidates for each beam
                    top_k_time_log_probs, top_k_time_indices = torch.topk(time_log_probs, k_time, dim=-1)
                    
                    # Clean up
                    del outputs, time_logits, time_log_probs, input_ids, attention_mask
                    
                    # === STEP 2: Expand each beam with TIME candidates, get DURATION (batched) ===
                    time_expanded_seqs = []
                    time_expanded_scores = []
                    
                    for beam_idx, (beam_score, beam_seq) in enumerate(beams):
                        for time_idx in range(k_time):
                            time_token = top_k_time_indices[beam_idx, time_idx].item()
                            time_log_prob = top_k_time_log_probs[beam_idx, time_idx].item()
                            
                            time_expanded_seqs.append(beam_seq + [time_token])
                            time_expanded_scores.append(beam_score + time_log_prob)
                    
                    del top_k_time_log_probs, top_k_time_indices
                    
                    # === STEP 2: Get DURATION candidates (process in chunks to save memory) ===
                    top_k_dur_log_probs_list = []
                    top_k_dur_indices_list = []
                    
                    for chunk_start in range(0, len(time_expanded_seqs), chunk_size):
                        chunk_end = min(chunk_start + chunk_size, len(time_expanded_seqs))
                        chunk_seqs = time_expanded_seqs[chunk_start:chunk_end]
                        
                        # Batch this chunk
                        max_len = max(len(seq) for seq in chunk_seqs)
                        padded_seqs = []
                        attention_masks = []
                        
                        for seq in chunk_seqs:
                            padding_len = max_len - len(seq)
                            padded_seq = [0] * padding_len + seq
                            padded_seqs.append(padded_seq)
                            attention_masks.append([0] * padding_len + [1] * len(seq))
                        
                        input_ids = torch.tensor(padded_seqs, device=device)
                        attention_mask = torch.tensor(attention_masks, device=device)
                        
                        outputs = model(input_ids, attention_mask=attention_mask)
                        dur_logits = outputs.logits[:, -1, :]
                        dur_log_probs = torch.nn.functional.log_softmax(dur_logits, dim=-1)
                        
                        # Get top-k DURATION candidates for this chunk
                        top_k_dur_log_probs, top_k_dur_indices = torch.topk(dur_log_probs, k_dur, dim=-1)
                        
                        top_k_dur_log_probs_list.append(top_k_dur_log_probs.cpu())
                        top_k_dur_indices_list.append(top_k_dur_indices.cpu())
                        
                        # Clean up
                        del outputs, dur_logits, dur_log_probs, input_ids, attention_mask
                    
                    # Concatenate results from all chunks
                    top_k_dur_log_probs = torch.cat(top_k_dur_log_probs_list, dim=0)
                    top_k_dur_indices = torch.cat(top_k_dur_indices_list, dim=0)
                    
                    # === STEP 3: Expand with DURATION, get PITCH (batched) ===
                    dur_expanded_seqs = []
                    dur_expanded_scores = []
                    
                    for idx, (seq, score) in enumerate(zip(time_expanded_seqs, time_expanded_scores)):
                        for dur_idx in range(k_dur):
                            dur_token = top_k_dur_indices[idx, dur_idx].item()
                            dur_log_prob = top_k_dur_log_probs[idx, dur_idx].item()
                            
                            dur_expanded_seqs.append(seq + [dur_token])
                            dur_expanded_scores.append(score + dur_log_prob)
                    
                    del top_k_dur_log_probs, top_k_dur_indices, time_expanded_seqs, time_expanded_scores
                    
                    # === STEP 3: Get PITCH candidates (process in chunks to save memory) ===
                    chunk_size = 12
                    top_k_pitch_log_probs_list = []
                    top_k_pitch_indices_list = []
                    
                    for chunk_start in range(0, len(dur_expanded_seqs), chunk_size):
                        chunk_end = min(chunk_start + chunk_size, len(dur_expanded_seqs))
                        chunk_seqs = dur_expanded_seqs[chunk_start:chunk_end]
                        
                        # Batch this chunk
                        max_len = max(len(seq) for seq in chunk_seqs)
                        padded_seqs = []
                        attention_masks = []
                        
                        for seq in chunk_seqs:
                            padding_len = max_len - len(seq)
                            padded_seq = [0] * padding_len + seq
                            padded_seqs.append(padded_seq)
                            attention_masks.append([0] * padding_len + [1] * len(seq))
                        
                        input_ids = torch.tensor(padded_seqs, device=device)
                        attention_mask = torch.tensor(attention_masks, device=device)
                        
                        outputs = model(input_ids, attention_mask=attention_mask)
                        pitch_logits = outputs.logits[:, -1, :]
                        pitch_log_probs = torch.nn.functional.log_softmax(pitch_logits, dim=-1)
                        
                        # Get top-k PITCH candidates
                        top_k_pitch_log_probs, top_k_pitch_indices = torch.topk(pitch_log_probs, k_pitch, dim=-1)
                        
                        top_k_pitch_log_probs_list.append(top_k_pitch_log_probs.cpu())
                        top_k_pitch_indices_list.append(top_k_pitch_indices.cpu())
                        
                        # Clean up
                        del outputs, pitch_logits, pitch_log_probs, input_ids, attention_mask
                    
                    # Concatenate results from all chunks
                    top_k_pitch_log_probs = torch.cat(top_k_pitch_log_probs_list, dim=0)
                    top_k_pitch_indices = torch.cat(top_k_pitch_indices_list, dim=0)
                    
                    # === STEP 4: Create complete triplet candidates ===
                    for idx, (seq, score) in enumerate(zip(dur_expanded_seqs, dur_expanded_scores)):
                        for pitch_idx in range(k_pitch):
                            pitch_token = top_k_pitch_indices[idx, pitch_idx].item()
                            pitch_log_prob = top_k_pitch_log_probs[idx, pitch_idx].item()
                            
                            final_seq = seq + [pitch_token]
                            final_score = score + pitch_log_prob
                            
                            new_beams.append((final_score, final_seq))
                    
                    del top_k_pitch_log_probs, top_k_pitch_indices, dur_expanded_seqs, dur_expanded_scores
                    
                    # Keep top num_beams across all triplet candidates
                    new_beams.sort(key=lambda x: x[0], reverse=True)
                    beams = new_beams[:num_beams]
                    
                    # Periodic cache clearing
                    if triplet_idx % 20 == 0:
                        torch.cuda.empty_cache()
                
                last_pos = pitch_pos + 1
            
            # Select best beam
            best_score, best_seq = beams[0]

            # Extract predictions from best sequence and compare to ground truth
            # Find the score triplets in the generated sequence
            pred_idx = first_score_time_pos
            # Use prev_pos to track the last processed ground-truth position
            prev_pos = first_score_time_pos
            for time_pos, dur_pos, pitch_pos in score_triplet_positions:
                # Account for intermediate control tokens between prev_pos and time_pos
                if time_pos > prev_pos:
                    num_intermediate = time_pos - prev_pos
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

                # Advance prev_pos to after the pitch token
                prev_pos = pitch_pos + 1
    
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
    num_sequences = 1  # Start small to test memory usage
    batch_size = 16
    
    # Test different beam widths on 150_model
    # Note: Joint triplet beam search with chunking to fit in 16GB VRAM
    beam_widths = [5]  # Start with just 5 to test
    
    models = [
        ('150_model', '150_model'),
    ]
    
    print("="*80)
    print("COMPREHENSIVE BEAM WIDTH COMPARISON - 150_model")
    print("="*80)
    print(f"\nTest set: {test_file}")
    print(f"Sequences: {num_sequences}")
    print(f"Device: {device}")
    print(f"\nEvaluation modes:")
    print(f"  1. Greedy decoding (sequential with KV caching)")
    print(f"  2. Beam search with varying beam widths: {beam_widths}")
    
    # Store all results
    greedy_results = {}
    beam_results = {width: {} for width in beam_widths}
    
    # Evaluate 150_model - Greedy (baseline)
    print("\n" + "="*80)
    print("PHASE 1: GREEDY DECODING (baseline)")
    print("="*80)
    
    for model_name, model_path in models:
        results = evaluate_greedy_batched(model_path, model_name, test_file, 
                                         num_sequences, batch_size, device)
        greedy_results[model_name] = results
    
    # Evaluate 150_model with different beam widths
    print("\n" + "="*80)
    print("PHASE 2: BEAM SEARCH WITH VARYING BEAM WIDTHS")
    print("="*80)
    
    for num_beams in beam_widths:
        print(f"\n{'='*80}")
        print(f"Testing beam_width = {num_beams}")
        print(f"{'='*80}")
        for model_name, model_path in models:
            results = evaluate_beam_search(model_path, model_name, test_file,
                                          num_sequences, num_beams, device)
            beam_results[num_beams][model_name] = results
    
    # Print comparison tables
    print("\n" + "="*80)
    print("RESULTS SUMMARY - BEAM WIDTH COMPARISON")
    print("="*80)
    
    print("\n" + "-"*80)
    print("GREEDY DECODING (baseline)")
    print("-"*80)
    print(f"\n{'Model':<15} {'Time':<20} {'Duration':<20} {'Pitch':<20} {'Overall':<15}")
    print("-" * 80)
    
    for model_name in ['150_model']:
        if model_name in greedy_results:
            r = greedy_results[model_name]
            overall = (r['time'] + r['duration'] + r['pitch']) / 3
            print(f"{model_name:<15} "
                  f"{r['time']:>6.2f}% ({r['counts']['time']['correct']:>4}/{r['counts']['time']['total']:<5}) "
                  f"{r['duration']:>6.2f}% ({r['counts']['duration']['correct']:>4}/{r['counts']['duration']['total']:<5}) "
                  f"{r['pitch']:>6.2f}% ({r['counts']['pitch']['correct']:>4}/{r['counts']['pitch']['total']:<5}) "
                  f"{overall:>6.2f}%")
    
    # Print results for each beam width
    for num_beams in beam_widths:
        print("\n" + "-"*80)
        print(f"BEAM SEARCH (num_beams={num_beams})")
        print("-"*80)
        print(f"\n{'Model':<15} {'Time':<20} {'Duration':<20} {'Pitch':<20} {'Overall':<15}")
        print("-" * 80)
        
        for model_name in ['150_model']:
            if model_name in beam_results[num_beams]:
                r = beam_results[num_beams][model_name]
                overall = (r['time'] + r['duration'] + r['pitch']) / 3
                print(f"{model_name:<15} "
                      f"{r['time']:>6.2f}% ({r['counts']['time']['correct']:>4}/{r['counts']['time']['total']:<5}) "
                      f"{r['duration']:>6.2f}% ({r['counts']['duration']['correct']:>4}/{r['counts']['duration']['total']:<5}) "
                      f"{r['pitch']:>6.2f}% ({r['counts']['pitch']['correct']:>4}/{r['counts']['pitch']['total']:<5}) "
                      f"{overall:>6.2f}%")
    
    print("\n" + "-"*80)
    print("IMPROVEMENT vs GREEDY (Δ%)")
    print("-"*80)
    print(f"\n{'Beam Width':<15} {'Time Δ':<15} {'Duration Δ':<15} {'Pitch Δ':<15} {'Overall Δ':<15}")
    print("-" * 80)
    
    model_name = '150_model'
    if model_name in greedy_results:
        g = greedy_results[model_name]
        for num_beams in beam_widths:
            if model_name in beam_results[num_beams]:
                b = beam_results[num_beams][model_name]
                time_delta = b['time'] - g['time']
                dur_delta = b['duration'] - g['duration']
                pitch_delta = b['pitch'] - g['pitch']
                overall_delta = (time_delta + dur_delta + pitch_delta) / 3
                
                print(f"{num_beams:<15} "
                      f"{time_delta:>+6.2f}%{'':<8} "
                      f"{dur_delta:>+6.2f}%{'':<8} "
                      f"{pitch_delta:>+6.2f}%{'':<8} "
                      f"{overall_delta:>+6.2f}%")
    
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    model_name = '150_model'
    print(f"\n{model_name}:")
    
    if model_name in greedy_results:
        g = greedy_results[model_name]
        print(f"  Greedy Decoding:")
        print(f"    Time:     {g['time']:>6.2f}% ({g['counts']['time']['correct']}/{g['counts']['time']['total']})")
        print(f"    Duration: {g['duration']:>6.2f}% ({g['counts']['duration']['correct']}/{g['counts']['duration']['total']})")
        print(f"    Pitch:    {g['pitch']:>6.2f}% ({g['counts']['pitch']['correct']}/{g['counts']['pitch']['total']})")
        overall_g = (g['time'] + g['duration'] + g['pitch']) / 3
        print(f"    Overall:  {overall_g:>6.2f}%")
    
    for num_beams in beam_widths:
        if model_name in beam_results[num_beams]:
            b = beam_results[num_beams][model_name]
            print(f"  Beam Search (num_beams={num_beams}):")
            print(f"    Time:     {b['time']:>6.2f}% ({b['counts']['time']['correct']}/{b['counts']['time']['total']})")
            print(f"    Duration: {b['duration']:>6.2f}% ({b['counts']['duration']['correct']}/{b['counts']['duration']['total']})")
            print(f"    Pitch:    {b['pitch']:>6.2f}% ({b['counts']['pitch']['correct']}/{b['counts']['pitch']['total']})")
            overall_b = (b['time'] + b['duration'] + b['pitch']) / 3
            print(f"    Overall:  {overall_b:>6.2f}%")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
