"""
Debug: Why can't the transformer learn the simple lookup pattern?

The task is trivial:
- Look back 66 triplet positions (198 tokens) 
- Copy the pitch from the control note to the score note
- But model only gets 12.5% pitch accuracy autoregressively

Let's check:
1. Is the offset consistent?
2. Are the pitches actually matching in training data?
3. Is attention able to reach that far back?
4. What's the model actually predicting?
"""
import torch
from transformers import AutoModelForCausalLM
from anticipation.vocab import CONTROL_OFFSET, NOTE_OFFSET, REST
from anticipation.config import MAX_PITCH, CONTEXT_SIZE

print("="*80)
print("DEBUGGING: Why can't the transformer learn the simple lookup?")
print("="*80)
print()

# Load model
model = AutoModelForCausalLM.from_pretrained('newest_model/')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()
print(f"Model context size: {CONTEXT_SIZE}")
print(f"Model loaded on {device}")
print()

# Load one sequence
with open('data/test_clean.txt', 'r') as f:
    line = f.readline()

tokens = [int(x) for x in line.strip().split() if x != '|']

print(f"Sequence length: {len(tokens)} tokens")
print()

# Find the pattern
print("="*80)
print("CHECKING THE LOOKUP PATTERN")
print("="*80)
print()

# Skip mode token and SEP tokens
i = 1  # Skip ANTICIPATE
i += 3  # Skip SEP tokens

# Find first score note
score_start = None
bootstrap_notes = 0

while i < len(tokens) - 2:
    if (tokens[i] >= CONTROL_OFFSET and 
        tokens[i+1] >= CONTROL_OFFSET and 
        tokens[i+2] >= CONTROL_OFFSET):
        bootstrap_notes += 1
        i += 3
    else:
        score_start = i
        break

print(f"Bootstrap control notes: {bootstrap_notes}")
print(f"First score note starts at position: {score_start}")
print(f"Tokens in bootstrap: {score_start - 4} (including mode + 3 SEP)")
print()

# Check the offset for first few score notes
print("Checking lookup offset for first 10 score notes:")
print("-"*80)
print("The pattern should be: score_note[i] has same pitch as control_note[i-bootstrap]")
print()

for note_idx in range(10):
    score_pos = score_start + note_idx * 3 + 2  # Note token position
    
    if score_pos >= len(tokens):
        break
    
    # The corresponding control note should be note_idx positions back in the bootstrap
    # Bootstrap starts at position 4 (after mode + 3 SEP)
    # Each control triplet is 3 tokens
    if note_idx < bootstrap_notes:
        control_pos = 4 + note_idx * 3 + 2  # Bootstrap control note
    else:
        # Once we've used up bootstrap, we wrap around or there's no match
        print(f"Note {note_idx}: No bootstrap control (only {bootstrap_notes} bootstrap notes)")
        continue
    
    score_note = tokens[score_pos]
    control_note = tokens[control_pos]
    
    # Extract pitches
    if score_note >= NOTE_OFFSET and control_note >= CONTROL_OFFSET:
        score_pitch = (score_note - NOTE_OFFSET) % MAX_PITCH
        control_pitch = (control_note - CONTROL_OFFSET - NOTE_OFFSET) % MAX_PITCH
        
        match = "✓" if score_pitch == control_pitch else "✗"
        offset = score_pos - control_pos
        print(f"Note {note_idx}: score_pos={score_pos}, control_pos={control_pos}, "
              f"offset={offset}, "
              f"score_pitch={score_pitch}, control_pitch={control_pitch} {match}")

print()

# Now check what the model predicts
print("="*80)
print("WHAT DOES THE MODEL PREDICT?")
print("="*80)
print()

with torch.no_grad():
    input_ids = torch.tensor([tokens]).to(device)
    outputs = model(input_ids)
    logits = outputs.logits[0]  # [seq_len, vocab_size]

print("For the same 5 score notes:")
print("-"*80)

for note_idx in range(5):
    score_pos = score_start + note_idx * 3 + 2
    
    if score_pos >= len(tokens):
        break
    
    control_pos = score_pos - 198
    
    if control_pos < 0:
        continue
    
    # Ground truth
    true_note = tokens[score_pos]
    true_pitch = (true_note - NOTE_OFFSET) % MAX_PITCH if true_note >= NOTE_OFFSET else -1
    
    # Model's top prediction
    pred_note = logits[score_pos - 1].argmax().item()
    pred_pitch = (pred_note - NOTE_OFFSET) % MAX_PITCH if pred_note >= NOTE_OFFSET else -1
    
    # Model's top 5 predictions
    top5_logits, top5_tokens = torch.topk(logits[score_pos - 1], 5)
    top5_pitches = []
    for tok in top5_tokens:
        tok_val = tok.item()
        if tok_val >= NOTE_OFFSET:
            pitch = (tok_val - NOTE_OFFSET) % MAX_PITCH
            top5_pitches.append(pitch)
        else:
            top5_pitches.append(-1)
    
    # Control note pitch
    control_note = tokens[control_pos]
    control_pitch = (control_note - CONTROL_OFFSET - NOTE_OFFSET) % MAX_PITCH
    
    match = "✓" if pred_pitch == true_pitch else "✗"
    in_top5 = "✓" if true_pitch in top5_pitches else "✗"
    
    print(f"\nNote {note_idx}:")
    print(f"  True pitch: {true_pitch} (should match control pitch: {control_pitch})")
    print(f"  Predicted pitch: {pred_pitch} {match}")
    print(f"  Top 5 predictions: {top5_pitches}")
    print(f"  True pitch in top 5? {in_top5}")
    print(f"  Prediction confidence: {torch.softmax(logits[score_pos-1], dim=0)[pred_note]:.4f}")

print()

# Check attention distance
print("="*80)
print("CAN THE MODEL ATTEND 198 TOKENS BACK?")
print("="*80)
print()
print(f"Model context size: {CONTEXT_SIZE} tokens")
print(f"Required lookback: 198 tokens (66 triplets × 3)")
print(f"Can it reach? {'YES' if 198 < CONTEXT_SIZE else 'NO'}")
print()

# Check if there's a positional pattern
print("="*80)
print("IS THE OFFSET ALWAYS 198?")
print("="*80)
print()

# Check 20 notes
offsets = []
for note_idx in range(20):
    score_pos = score_start + note_idx * 3 + 2
    
    if score_pos >= len(tokens):
        break
    
    control_pos = score_pos - 198
    if control_pos < 0:
        continue
    
    offsets.append(score_pos - control_pos)

print(f"Checked {len(offsets)} notes")
print(f"All offsets the same? {len(set(offsets)) == 1}")
print(f"Offsets: {set(offsets)}")
print()

print("="*80)
print("CONCLUSION")
print("="*80)
print("The task is:")
print("  1. Look back exactly 198 tokens")
print("  2. Copy the pitch value")
print("  3. Combine with predicted time/duration")
print()
print("This should be TRIVIAL for a transformer with:")
print(f"  - Context size: {CONTEXT_SIZE} (can easily reach 198 tokens back)")
print("  - Self-attention: Can attend to any position")
print("  - Pattern: Fixed offset, no variation")
print()
print("Yet the model only achieves 12.5% pitch accuracy autoregressively!")
print()
print("Possible reasons:")
print("  1. Model never learned the pattern (training issue)")
print("  2. Exposure bias: Wrong timing → wrong context position → wrong pitch")
print("  3. Position embeddings confuse absolute vs relative position")
print("  4. Model learned spurious correlations instead of the simple rule")
