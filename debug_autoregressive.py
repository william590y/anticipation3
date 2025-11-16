"""
Debug autoregressive evaluation to understand why sequence 5 performs so much better.
Check if the high accuracy is coming from the REST prefix or actual notes.
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET

print("="*80)
print("DEBUGGING AUTOREGRESSIVE EVALUATION - SEQUENCE 5")
print("="*80)

# Load model
print("\nLoading model from opening_model/...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained('opening_model/')
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Load ALL 5 sequences
print("\nLoading all 5 sequences from data/test_openings.txt...")
with open('data/test_openings.txt', 'r') as f:
    lines = [f.readline().strip() for _ in range(5)]

all_results = []

for seq_num, line in enumerate(lines, 1):
    print(f"\n{'='*80}")
    print(f"PROCESSING SEQUENCE {seq_num}")
    print('='*80)
    
    if '|' in line:
        token_part = line.split('|')[0].strip()
    else:
        token_part = line
    
    tokens = [int(t) for t in token_part.split()]
    
    # Find score triplet positions
    score_triplet_positions = []
    i = 1  # Skip mode token
    while i < len(tokens) - 2:
        if (tokens[i] < CONTROL_OFFSET and 
            tokens[i+1] < CONTROL_OFFSET and 
            tokens[i+2] < CONTROL_OFFSET):
            score_triplet_positions.append((i, i+1, i+2))
            i += 3
        else:
            i += 1
    
    print(f"Total score triplets: {len(score_triplet_positions)}")
    
    # Separate REST vs actual notes
    rest_indices = []
    note_indices = []
    for idx, (time_pos, dur_pos, pitch_pos) in enumerate(score_triplet_positions):
        if tokens[pitch_pos] == 27512:  # REST
            rest_indices.append(idx)
        else:
            note_indices.append(idx)
    
    print(f"  REST triplets: {len(rest_indices)}")
    print(f"  Actual note triplets: {len(note_indices)}")
    
    # Generate predictions autoregressively
    with torch.no_grad():
        first_score_time_pos = score_triplet_positions[0][0]
        
        # Initialize context
        init_context = torch.tensor([tokens[:first_score_time_pos]]).to(device)
        outputs = model(init_context, past_key_values=None, use_cache=True)
        past_key_values = outputs.past_key_values
        
        # Statistics: separate REST vs NOTE
        rest_stats = {
            'time': {'correct': 0, 'total': 0},
            'dur': {'correct': 0, 'total': 0},
            'pitch': {'correct': 0, 'total': 0}
        }
        note_stats = {
            'time': {'correct': 0, 'total': 0},
            'dur': {'correct': 0, 'total': 0},
            'pitch': {'correct': 0, 'total': 0}
        }
        
        last_pos = first_score_time_pos
        
        for triplet_idx, (time_pos, dur_pos, pitch_pos) in enumerate(score_triplet_positions):
            is_rest = (tokens[pitch_pos] == 27512)
            stats = rest_stats if is_rest else note_stats
            
            # Process intermediate control tokens
            if time_pos > last_pos:
                intermediate = torch.tensor([tokens[last_pos:time_pos]]).to(device)
                outputs = model(intermediate, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values
            
            # Predict TIME
            logits = outputs.logits[0, -1]
            pred_time = logits.argmax().item()
            if pred_time == tokens[time_pos]:
                stats['time']['correct'] += 1
            stats['time']['total'] += 1
            
            # Feed predicted time back
            next_token = torch.tensor([[pred_time]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Predict DURATION
            logits = outputs.logits[0, -1]
            pred_dur = logits.argmax().item()
            if pred_dur == tokens[dur_pos]:
                stats['dur']['correct'] += 1
            stats['dur']['total'] += 1
            
            # Feed predicted duration back
            next_token = torch.tensor([[pred_dur]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # Predict PITCH
            logits = outputs.logits[0, -1]
            pred_pitch = logits.argmax().item()
            if pred_pitch == tokens[pitch_pos]:
                stats['pitch']['correct'] += 1
            stats['pitch']['total'] += 1
            
            # Feed predicted pitch back
            next_token = torch.tensor([[pred_pitch]]).to(device)
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            last_pos = pitch_pos + 1
    
    # Calculate accuracies
    rest_total_correct = sum(s['correct'] for s in rest_stats.values())
    rest_total_tokens = sum(s['total'] for s in rest_stats.values())
    rest_overall = (rest_total_correct / rest_total_tokens * 100) if rest_total_tokens > 0 else 0
    
    note_total_correct = sum(s['correct'] for s in note_stats.values())
    note_total_tokens = sum(s['total'] for s in note_stats.values())
    note_overall = (note_total_correct / note_total_tokens * 100) if note_total_tokens > 0 else 0
    
    combined_correct = rest_total_correct + note_total_correct
    combined_total = rest_total_tokens + note_total_tokens
    combined_overall = (combined_correct / combined_total * 100) if combined_total > 0 else 0
    
    all_results.append({
        'seq_num': seq_num,
        'rest_time': (rest_stats['time']['correct'] / rest_stats['time']['total'] * 100) if rest_stats['time']['total'] > 0 else 0,
        'rest_dur': (rest_stats['dur']['correct'] / rest_stats['dur']['total'] * 100) if rest_stats['dur']['total'] > 0 else 0,
        'rest_pitch': (rest_stats['pitch']['correct'] / rest_stats['pitch']['total'] * 100) if rest_stats['pitch']['total'] > 0 else 0,
        'rest_overall': rest_overall,
        'note_time': (note_stats['time']['correct'] / note_stats['time']['total'] * 100) if note_stats['time']['total'] > 0 else 0,
        'note_dur': (note_stats['dur']['correct'] / note_stats['dur']['total'] * 100) if note_stats['dur']['total'] > 0 else 0,
        'note_pitch': (note_stats['pitch']['correct'] / note_stats['pitch']['total'] * 100) if note_stats['pitch']['total'] > 0 else 0,
        'note_overall': note_overall,
        'combined_overall': combined_overall
    })
    
    print(f"\nRESULTS:")
    print(f"  REST prefix:   {rest_overall:6.2f}% (time={rest_stats['time']['correct']}/{rest_stats['time']['total']}, dur={rest_stats['dur']['correct']}/{rest_stats['dur']['total']}, pitch={rest_stats['pitch']['correct']}/{rest_stats['pitch']['total']})")
    print(f"  Actual notes:  {note_overall:6.2f}% (time={note_stats['time']['correct']}/{note_stats['time']['total']}, dur={note_stats['dur']['correct']}/{note_stats['dur']['total']}, pitch={note_stats['pitch']['correct']}/{note_stats['pitch']['total']})")
    print(f"  Combined:      {combined_overall:6.2f}%")

# Print summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print("\nREST PREFIX (33 triplets):")
print("Seq | Time   | Dur    | Pitch  | Overall")
print("----|--------|--------|--------|--------")
for r in all_results:
    print(f" {r['seq_num']}  | {r['rest_time']:5.1f}% | {r['rest_dur']:5.1f}% | {r['rest_pitch']:5.1f}% | {r['rest_overall']:5.1f}%")

print("\nACTUAL NOTES (137 triplets):")
print("Seq | Time   | Dur    | Pitch  | Overall")
print("----|--------|--------|--------|--------")
for r in all_results:
    print(f" {r['seq_num']}  | {r['note_time']:5.1f}% | {r['note_dur']:5.1f}% | {r['note_pitch']:5.1f}% | {r['note_overall']:5.1f}%")

print("\nCOMBINED (170 triplets):")
print("Seq | Overall")
print("----|--------")
for r in all_results:
    print(f" {r['seq_num']}  | {r['combined_overall']:5.1f}%")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print("Compare sequence 5 (high accuracy) vs sequences 1-4 (low accuracy)")
print("to understand what makes a sequence easy vs hard to predict.")
print("="*80)
