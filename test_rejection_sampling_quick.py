"""
Quick test of rejection sampling with limited attempts.
"""

import torch
from transformers import AutoModelForCausalLM
from anticipation.sample import generate4, generate4_forced
from anticipation.config import *
from anticipation.vocab import *


def extract_from_sequence(sequence_tokens, prefix_controls=33):
    """Extract controls and scores from test sequence."""
    if len(sequence_tokens) < 4:
        return [], []
    
    tokens = sequence_tokens[4:]
    all_controls = []
    all_scores = []
    i = 0
    k = prefix_controls
    
    for _ in range(k):
        if i + 6 <= len(tokens):
            control_triplet = tokens[i:i+3]
            if control_triplet[0] >= CONTROL_OFFSET:
                all_controls.extend(control_triplet)
                i += 6
            else:
                break
        else:
            break
    
    while i + 3 <= len(tokens):
        triplet = tokens[i:i+3]
        if triplet[0] >= CONTROL_OFFSET:
            all_controls.extend(triplet)
        elif triplet[2] != REST:
            all_scores.extend(triplet)
        i += 3
    
    num_scores = len(all_scores) // 3
    controls_trimmed = all_controls[:num_scores * 3]
    
    return controls_trimmed, all_scores


print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "hf-ckpt-3500/checkpoint-3500",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print(f"✓ Model loaded on {next(model.parameters()).device}")

# Load one test sequence
with open('data/test_output.txt', 'r') as f:
    line = f.readline()

sequence_tokens = [int(tok) for tok in line.strip().split()]
controls, ground_truth = extract_from_sequence(sequence_tokens)

print(f"\nTest sequence: {len(controls)//3} controls, {len(ground_truth)//3} scores")

# Normal generation
print("\n1. Normal generation...")
events_normal, _ = generate4(model, controls=controls, top_p=0.95)

# Forced generation with rejection sampling (max 20 attempts)
print("\n2. Rejection sampling (max 20 attempts per note)...")
events_forced, _ = generate4_forced(model, controls=controls, ground_truth_scores=ground_truth, 
                                     top_p=0.95, max_attempts=20)

# Compare
def count_matches(gen, gt):
    timing = sum(1 for i in range(min(len(gen)//3, len(gt)//3)) 
                 if gen[i*3] == gt[i*3])
    duration = sum(1 for i in range(min(len(gen)//3, len(gt)//3)) 
                   if gen[i*3+1] == gt[i*3+1])
    pitch = sum(1 for i in range(min(len(gen)//3, len(gt)//3)) 
                if gen[i*3+2] == gt[i*3+2])
    total = min(len(gen)//3, len(gt)//3)
    return timing, duration, pitch, total

t_norm, d_norm, p_norm, total = count_matches(events_normal, ground_truth)
t_forced, d_forced, p_forced, _ = count_matches(events_forced, ground_truth)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"\nNormal generation:")
print(f"  Timing:   {t_norm}/{total} = {t_norm/total*100:.1f}%")
print(f"  Duration: {d_norm}/{total} = {d_norm/total*100:.1f}%")
print(f"  Pitch:    {p_norm}/{total} = {p_norm/total*100:.1f}%")

print(f"\nRejection sampling (max 20 attempts):")
print(f"  Timing:   {t_forced}/{total} = {t_forced/total*100:.1f}%")
print(f"  Duration: {d_forced}/{total} = {d_forced/total*100:.1f}%")
print(f"  Pitch:    {p_forced}/{total} = {p_forced/total*100:.1f}%")
