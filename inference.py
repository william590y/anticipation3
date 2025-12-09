"""
Extract test examples using newest_model with greedy decoding (top_k=1).
Creates MIDI files for: ground_truth_score, performance, and model_predictions.
"""
import os
import torch
from transformers import AutoModelForCausalLM
from anticipation.vocab import *
from anticipation.config import *
from anticipation.convert import events_to_midi

def greedy_decode_sequence(model, input_ids, max_new_tokens=1024):
    """Greedy decoding with KV caching."""
    device = model.device
    input_ids = input_ids.to(device)
    
    generated = input_ids.clone()
    past_key_values = None
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if past_key_values is None:
                model_inputs = generated
            else:
                model_inputs = generated[:, -1:]
            
            outputs = model(model_inputs, past_key_values=past_key_values, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # Greedy: argmax
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if generated.shape[1] >= CONTEXT_SIZE:
                break
    
    return generated

def extract_aligned_pairs(tokens):
    """
    Extract aligned performance-score pairs from the interleaved sequence.
    
    Sequence format:
    - Position 0: ANTICIPATE
    - Positions 1-3: SEP SEP SEP
    - Positions 4-201: 33 control+rest pairs (k=33)
    - Positions 202+: Alternating score/control triplets
    
    Alignment relationship:
    - Performance note i from control+rest pairs → Score note i in alternating section
    - Performance note 33+j from alternating → Score note 33+j in alternating
    
    Returns:
        performance_triplets: List of [time, dur, pitch] (no CONTROL_OFFSET)
        score_triplets: List of [time, dur, pitch] (with offsets)
        Both lists have same length and are aligned by index
    """
    # Skip ANTICIPATE (position 0) and SEP SEP SEP (positions 1-3)
    # Start at position 4
    body = tokens[4:]
    
    k = 33  # Number of control+rest pairs
    control_rest_section_length = k * 6  # 33 pairs × 6 tokens = 198 tokens
    
    # Extract performance from control+rest pairs (positions 4-201)
    perf_from_pairs = []
    for i in range(k):
        base = i * 6
        # Control triplet (first 3 tokens of each pair)
        ctrl_time = body[base] - CONTROL_OFFSET
        ctrl_dur = body[base + 1] - CONTROL_OFFSET
        ctrl_pitch = body[base + 2] - CONTROL_OFFSET
        perf_from_pairs.append([ctrl_time, ctrl_dur, ctrl_pitch])
        # Rest triplet is ignored (positions base+3 to base+5)
    
    # Extract from alternating section (positions 202+, which is index 198 in body)
    alternating = body[control_rest_section_length:]
    
    score_from_alternating = []
    perf_from_alternating = []
    
    pos = 0
    while pos + 5 < len(alternating):
        # Score triplet (first 3 tokens)
        score_time = alternating[pos]
        score_dur = alternating[pos + 1]
        score_pitch = alternating[pos + 2]
        
        # Verify it's a score triplet (all < CONTROL_OFFSET, not REST)
        if (score_time < CONTROL_OFFSET and 
            score_dur < CONTROL_OFFSET and 
            score_pitch < CONTROL_OFFSET and 
            score_pitch != REST):
            score_from_alternating.append([score_time, score_dur, score_pitch])
        else:
            # Not a valid score triplet, stop extraction
            break
        
        pos += 3
        
        # Control triplet (next 3 tokens)
        if pos + 2 < len(alternating):
            ctrl_time = alternating[pos] - CONTROL_OFFSET
            ctrl_dur = alternating[pos + 1] - CONTROL_OFFSET
            ctrl_pitch = alternating[pos + 2] - CONTROL_OFFSET
            
            # Verify it's a control triplet
            if (alternating[pos] >= CONTROL_OFFSET and 
                alternating[pos + 1] >= CONTROL_OFFSET and 
                alternating[pos + 2] >= CONTROL_OFFSET):
                perf_from_alternating.append([ctrl_time, ctrl_dur, ctrl_pitch])
            else:
                # Not a valid control triplet, stop
                break
        
        pos += 3
    
    # Combine performance: all from control+rest pairs + those from alternating
    all_performance = perf_from_pairs + perf_from_alternating
    
    # Score only comes from alternating section
    all_score = score_from_alternating
    
    # The alignment is:
    # - Performance note 0-32 (from control+rest) → Score note 0-32 (from alternating)
    # - Performance note 33+ (from alternating) → Score note 33+ (from alternating)
    # So both lists should have the same length
    
    return all_performance, all_score

print("="*80)
print("EXTRACTING GREEDY EXAMPLES FROM newest_model")
print("="*80)
print()

# Load model
print("Loading model from newest_model/...")
model = AutoModelForCausalLM.from_pretrained('newest_model/')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")
print()

# Load test data
print("Loading test data...")
with open('data/test_clean.txt', 'r') as f:
    lines = f.readlines()

num_examples = 5
print(f"Extracting {num_examples} examples with greedy decoding (top_k=1)")
print()

total_correct = 0
total_notes = 0

for example_idx in range(num_examples):
    print(f"Processing example {example_idx + 1}/{num_examples}...")
    
    line = lines[example_idx]
    
    # Parse sequence
    if '|' in line:
        token_str, _ = line.split('|')
        tokens = [int(t) for t in token_str.strip().split()]
    else:
        tokens = [int(t) for t in line.strip().split()]
    
    # Find where score notes start
    score_start_idx = None
    for i in range(1, len(tokens), 3):
        if i+2 < len(tokens):
            if (tokens[i] < CONTROL_OFFSET and 
                tokens[i+1] < CONTROL_OFFSET and 
                tokens[i+2] < CONTROL_OFFSET and
                tokens[i+2] != REST):
                score_start_idx = i
                break
    
    if score_start_idx is None:
        print(f"  Skipping - no score notes found")
        continue
    
    # Use bootstrap prefix as context
    context_tokens = tokens[:score_start_idx]
    
    # Convert to tensor and generate
    input_ids = torch.tensor([context_tokens])
    generated = greedy_decode_sequence(model, input_ids, max_new_tokens=len(tokens) - score_start_idx)
    
    # Get predicted tokens (remove context)
    predicted_tokens = generated[0, len(context_tokens):].cpu().tolist()
    
    # Reconstruct full sequences for saving
    full_ground_truth = tokens
    full_predicted = tokens[:score_start_idx] + predicted_tokens[:len(tokens) - score_start_idx]
    
    # Extract performance, ground truth score, and predicted score with proper alignment
    gt_perf, gt_score = extract_aligned_pairs(full_ground_truth)
    pred_perf, pred_score = extract_aligned_pairs(full_predicted)
    
    # Verify alignment: performance pitches should match (100% by design)
    gt_perf_pitches = [p[2] for p in gt_perf]
    gt_score_pitches = [s[2] - NOTE_OFFSET for s in gt_score]
    
    alignment_correct = sum(1 for i in range(min(len(gt_perf_pitches), len(gt_score_pitches))) 
                           if gt_perf_pitches[i] == gt_score_pitches[i])
    alignment_total = min(len(gt_perf_pitches), len(gt_score_pitches))
    
    # Calculate pitch accuracy: compare predicted score to ground truth score
    pred_score_pitches = [s[2] - NOTE_OFFSET for s in pred_score]
    
    min_len = min(len(gt_score_pitches), len(pred_score_pitches))
    if min_len > 0:
        correct = sum(1 for i in range(min_len) if gt_score_pitches[i] == pred_score_pitches[i])
        accuracy = 100.0 * correct / min_len
        total_correct += correct
        total_notes += min_len
    else:
        correct = 0
        accuracy = 0.0
    
    # Convert to flat lists for MIDI conversion
    performance_events = []
    for p in gt_perf:
        performance_events.extend([p[0] + TIME_OFFSET, p[1] + DUR_OFFSET, p[2] + NOTE_OFFSET])
    
    gt_score_events = []
    for s in gt_score:
        gt_score_events.extend(s)  # Already has offsets
    
    pred_score_events = []
    for s in pred_score:
        pred_score_events.extend(s)  # Already has offsets
    
    # Create output directory
    output_dir = f'test_examples/example_{example_idx + 1}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as MIDI files
    print(f"  Saving MIDI files to {output_dir}/")
    
    # Performance MIDI
    perf_midi_path = os.path.join(output_dir, 'performance.mid')
    perf_midi = events_to_midi(performance_events)
    perf_midi.save(perf_midi_path)
    
    # Ground truth score MIDI
    gt_midi_path = os.path.join(output_dir, 'ground_truth_score.mid')
    gt_midi = events_to_midi(gt_score_events)
    gt_midi.save(gt_midi_path)
    
    # Predicted score MIDI
    pred_midi_path = os.path.join(output_dir, 'model_predictions.mid')
    pred_midi = events_to_midi(pred_score_events)
    pred_midi.save(pred_midi_path)
    
    print(f"  Performance notes: {len(gt_perf)}")
    print(f"  Ground truth score notes: {len(gt_score)}")
    print(f"  Predicted score notes: {len(pred_score)}")
    print(f"  Alignment accuracy: {100.0 * alignment_correct / alignment_total:.2f}% ({alignment_correct}/{alignment_total})")
    print(f"  Pitch accuracy: {accuracy:.2f}% ({correct}/{min_len})")
    print()

print("="*80)
print("DONE!")
print("="*80)
print(f"Saved {num_examples} examples to test_examples/")
print()
print(f"Overall pitch accuracy: {100.0 * total_correct / total_notes:.2f}% ({total_correct}/{total_notes})")
print()
print("Each example contains:")
print("  • performance.mid - The input performance")
print("  • ground_truth_score.mid - The actual score")
print("  • model_predictions.mid - Model's greedy predictions (top_k=1)")