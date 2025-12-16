"""
Test checkpoint-750 using the SAME autoregressive evaluation as train.py:
- Provide all control+rest pairs as context (positions 0-201)
- Generate score triplets one at a time
- After each score triplet, feed ground truth control triplet back to model
- This matches the autoregressive_accuracy calculation in train.py
"""
import torch
from transformers import AutoModelForCausalLM
from anticipation.vocab import *
from anticipation.config import *
from tqdm import tqdm

print("="*80)
print("TESTING checkpoint-750 WITH TRAIN.PY AUTOREGRESSIVE METHOD")
print("="*80)
print()

# Load model
model_path = 'checkpoint-750'
print(f"Loading model from {model_path}/...")
model = AutoModelForCausalLM.from_pretrained(model_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")
print()

# Load test data
print("Loading test data...")
with open('data/test_normalized.txt', 'r') as f:
    lines = f.readlines()

num_examples = 3
print(f"Testing on {num_examples} examples (matches train.py slow method, no KV caching)")
print(f"Method: Generate score triplets one at a time, feed back ground truth control")
print()

autoregressive_correct = 0
autoregressive_total = 0
examples_processed = 0

for example_idx in tqdm(range(min(num_examples, len(lines))), desc="Evaluating"):
    line = lines[example_idx]
    
    # Parse sequence
    if '|' in line:
        token_str, _ = line.split('|')
        tokens = [int(t) for t in token_str.strip().split()]
    else:
        tokens = [int(t) for t in line.strip().split()]
    
    seq = torch.tensor(tokens)
    
    # Find where alternating section starts (position 202)
    alternating_start = 202
    if len(seq) <= alternating_start:
        continue
    
    # Start context with: ANTICIPATE + SEP SEP SEP + all control+rest pairs (positions 0-201)
    # This gives the model all the performance information
    context = seq[:alternating_start].tolist()
    
    # Now autoregressively generate the alternating score/control section
    # Pattern: score_triplet, control_triplet, score_triplet, control_triplet, ...
    pos = alternating_start
    while pos + 5 < len(seq):
        # Check if this is a score triplet (all 3 tokens < CONTROL_OFFSET)
        if (seq[pos] < CONTROL_OFFSET and 
            seq[pos+1] < CONTROL_OFFSET and 
            seq[pos+2] < CONTROL_OFFSET and
            seq[pos+2] != REST):
            
            # This is a score triplet - generate it autoregressively
            # Generate TIME token
            input_tensor = torch.tensor([context]).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits[0, -1, :]
                pred_time = logits.argmax().item()
            context.append(pred_time)
            
            # Generate DURATION token
            input_tensor = torch.tensor([context]).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits[0, -1, :]
                pred_dur = logits.argmax().item()
            context.append(pred_dur)
            
            # Generate PITCH token
            input_tensor = torch.tensor([context]).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits[0, -1, :]
                pred_pitch = logits.argmax().item()
            context.append(pred_pitch)
            
            # Check if predicted pitch matches ground truth
            true_pitch = seq[pos + 2].item()
            if pred_pitch == true_pitch:
                autoregressive_correct += 1
            autoregressive_total += 1
            
            pos += 3
            
            # After score triplet, add ground truth control triplet to context
            # (We're only testing score generation, not control generation)
            if pos + 2 < len(seq):
                context.extend([seq[pos].item(), seq[pos+1].item(), seq[pos+2].item()])
                pos += 3
        else:
            # Not a score triplet, add to context and continue
            context.append(seq[pos].item())
            pos += 1
    
    examples_processed += 1

print()
print("="*80)
print("RESULTS")
print("="*80)
print(f"Examples processed: {examples_processed}")
print(f"Total pitches compared: {autoregressive_total}")
print(f"Total correct: {autoregressive_correct}")
print()
if autoregressive_total > 0:
    autoregressive_accuracy = 100.0 * autoregressive_correct / autoregressive_total
    print(f"Autoregressive pitch accuracy: {autoregressive_accuracy:.2f}%")
else:
    print("No pitches to evaluate")
print()
print("="*80)
print()
print("This should match the 'autoregressive_accuracy' reported during training.")
print("If this is ~80% and your previous test was ~35%, the difference is because:")
print("  - This test: Generate one triplet, get ground truth control back")
print("  - Previous test: Generate everything purely autoregressively")
