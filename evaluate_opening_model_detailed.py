"""
Evaluate opening_model on test set with detailed statistics for time, duration, and pitch tokens.
"""
import torch
from transformers import GPT2LMHeadModel
from anticipation.vocab import CONTROL_OFFSET
from tqdm import tqdm

print("="*80)
print("DETAILED EVALUATION OF OPENING_MODEL ON TEST SET")
print("="*80)

# Load model once
print("\nLoading model from opening_model/...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained('opening_model/')
model = model.to(device)
model.eval()
print(f"Model loaded on {device}\n")

# Load test data
print("Loading test data from data/test_openings.txt...")
with open('data/test_openings.txt', 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Total test sequences: {len(lines)}\n")

# Statistics trackers
stats = {
    'control_time': {'correct': 0, 'total': 0},
    'control_dur': {'correct': 0, 'total': 0},
    'control_pitch': {'correct': 0, 'total': 0},
    'score_time': {'correct': 0, 'total': 0},
    'score_dur': {'correct': 0, 'total': 0},
    'score_pitch': {'correct': 0, 'total': 0},
}

# Process sequences
print("Processing sequences...")
for line in tqdm(lines):
    # Parse tokens
    if '|' in line:
        token_part = line.split('|')[0].strip()
    else:
        token_part = line
    
    tokens = [int(t) for t in token_part.split()]
    
    # Find all triplet positions
    score_triplets = []
    control_triplets = []
    
    i = 1  # Skip mode token
    while i < len(tokens) - 2:
        if (tokens[i] < CONTROL_OFFSET and 
            tokens[i+1] < CONTROL_OFFSET and 
            tokens[i+2] < CONTROL_OFFSET):
            score_triplets.append((i, i+1, i+2))
            i += 3
        elif (tokens[i] >= CONTROL_OFFSET and 
              tokens[i+1] >= CONTROL_OFFSET and 
              tokens[i+2] >= CONTROL_OFFSET and
              tokens[i] < CONTROL_OFFSET + 27512):
            control_triplets.append((i, i+1, i+2))
            i += 3
        else:
            i += 1
    
    # Collect all positions to predict
    all_positions = []
    for time_pos, dur_pos, pitch_pos in score_triplets:
        all_positions.extend([
            ('score_time', time_pos),
            ('score_dur', dur_pos),
            ('score_pitch', pitch_pos)
        ])
    for time_pos, dur_pos, pitch_pos in control_triplets:
        all_positions.extend([
            ('control_time', time_pos),
            ('control_dur', dur_pos),
            ('control_pitch', pitch_pos)
        ])
    
    all_positions.sort(key=lambda x: x[1])
    
    if not all_positions:
        continue
    
    # Teacher forcing evaluation
    with torch.no_grad():
        # Feed all tokens in one go for efficiency
        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        
        # For each position, check if prediction at pos-1 matches ground truth at pos
        for token_type, pos in all_positions:
            if pos > 0 and pos < len(tokens):
                # Prediction for position pos comes from logits[pos-1]
                predicted_token = logits[pos - 1].argmax().item()
                ground_truth = tokens[pos]
                
                if predicted_token == ground_truth:
                    stats[token_type]['correct'] += 1
                stats[token_type]['total'] += 1

# Print results
print("\n" + "="*80)
print("RESULTS - TEACHER FORCING ACCURACY (per token type)")
print("="*80)

print("\nCONTROL TOKENS (Performance):")
for token_type in ['control_time', 'control_dur', 'control_pitch']:
    correct = stats[token_type]['correct']
    total = stats[token_type]['total']
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"  {token_type:20s}: {accuracy:6.2f}% ({correct:6d}/{total:6d})")

print("\nSCORE TOKENS:")
for token_type in ['score_time', 'score_dur', 'score_pitch']:
    correct = stats[token_type]['correct']
    total = stats[token_type]['total']
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"  {token_type:20s}: {accuracy:6.2f}% ({correct:6d}/{total:6d})")

print("\nOVERALL:")
total_correct = sum(s['correct'] for s in stats.values())
total_tokens = sum(s['total'] for s in stats.values())
overall_accuracy = (total_correct / total_tokens * 100) if total_tokens > 0 else 0
print(f"  All tokens:          {overall_accuracy:6.2f}% ({total_correct:6d}/{total_tokens:6d})")

print("\n" + "="*80)
print("Note: This uses TEACHER FORCING (each prediction sees ground truth context)")
print("="*80)
