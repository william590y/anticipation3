"""
Evaluate autoregressive accuracy for newly trained models (100_model and 150_model).
Measures time, duration, and pitch accuracy on SCORE tokens only (not controls).
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET, NOTE_OFFSET
from tqdm import tqdm
import random

def evaluate_autoregressive(model_path, test_file, num_sequences=50, device='cuda'):
    """
    Evaluate a model's autoregressive accuracy on score tokens.
    
    Args:
        model_path: Path to the model directory
        test_file: Path to test data file
        num_sequences: Number of sequences to evaluate
        device: Device to run on
    
    Returns:
        dict: Accuracy statistics for time, duration, and pitch
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*80}")
    
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
    sample_size = min(num_sequences, len(lines))
    sampled_lines = random.sample(lines, sample_size)
    print(f"Sampled {sample_size} sequences for evaluation")
    
    # Track accuracy
    stats = {
        'time': {'correct': 0, 'total': 0},
        'duration': {'correct': 0, 'total': 0},
        'pitch': {'correct': 0, 'total': 0},
    }
    
    print(f"\nProcessing sequences...")
    
    with torch.no_grad():
        for line in tqdm(sampled_lines, desc="Evaluating", unit="seq"):
            # Parse tokens
            if '|' in line:
                token_part = line.split('|')[0].strip()
            else:
                token_part = line
            
            tokens = [int(t) for t in token_part.split()]
            
            # Find score triplet positions (time, dur, pitch)
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
            
            # Autoregressive evaluation
            first_score_time_pos = score_triplet_positions[0][0]
            init_context = torch.tensor([tokens[:first_score_time_pos]]).to(device)
            outputs = model(init_context, past_key_values=None, use_cache=True)
            past_key_values = outputs.past_key_values
            last_pos = first_score_time_pos
            
            for triplet_idx, (time_pos, dur_pos, pitch_pos) in enumerate(score_triplet_positions):
                # Process intermediate control tokens (if any)
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
                
                # Feed predicted time back
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
                
                # Feed predicted duration back
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
                
                # Feed predicted pitch back
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_file = 'data/test_sliding.txt'
    num_sequences = 50  # Evaluate on 50 sequences for good statistics
    
    models = [
        ('100_model', '100_model'),
        ('150_model', '150_model'),
    ]
    
    all_results = {}
    
    for model_name, model_path in models:
        results = evaluate_autoregressive(model_path, test_file, num_sequences, device)
        all_results[model_name] = results
    
    # Print comparison
    print(f"\n{'='*80}")
    print("AUTOREGRESSIVE VALIDATION ACCURACY - COMPARISON")
    print(f"{'='*80}")
    print(f"\nTest set: {test_file}")
    print(f"Sequences evaluated: {num_sequences}")
    print(f"\n{'Model':<15} {'Time':<15} {'Duration':<15} {'Pitch':<15} {'Overall':<15}")
    print('-' * 80)
    
    for model_name in ['100_model', '150_model']:
        if model_name in all_results:
            r = all_results[model_name]
            overall = (r['time'] + r['duration'] + r['pitch']) / 3
            print(f"{model_name:<15} {r['time']:>6.2f}% ({r['counts']['time']['correct']:>4}/{r['counts']['time']['total']:<4}) "
                  f"{r['duration']:>6.2f}% ({r['counts']['duration']['correct']:>4}/{r['counts']['duration']['total']:<4}) "
                  f"{r['pitch']:>6.2f}% ({r['counts']['pitch']['correct']:>4}/{r['counts']['pitch']['total']:<4}) "
                  f"{overall:>6.2f}%")
    
    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}")
    
    for model_name in ['100_model', '150_model']:
        if model_name in all_results:
            r = all_results[model_name]
            overall = (r['time'] + r['duration'] + r['pitch']) / 3
            print(f"\n{model_name}:")
            print(f"  Score TIME:     {r['time']:>6.2f}% ({r['counts']['time']['correct']}/{r['counts']['time']['total']})")
            print(f"  Score DURATION: {r['duration']:>6.2f}% ({r['counts']['duration']['correct']}/{r['counts']['duration']['total']})")
            print(f"  Score PITCH:    {r['pitch']:>6.2f}% ({r['counts']['pitch']['correct']}/{r['counts']['pitch']['total']})")
            print(f"  Overall:        {overall:>6.2f}%")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
