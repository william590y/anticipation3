"""
Detailed comparison of ground truth vs predictions from test_examples.
Shows exact token matching, pitch matching, timing differences, etc.
"""
import os
from anticipation.vocab import NOTE_OFFSET, CONTROL_OFFSET, REST, TIME_OFFSET, DUR_OFFSET
from anticipation.config import MAX_INSTR, MAX_PITCH

def analyze_example(example_dir):
    """Read back the original tokens and analyze differences."""
    # We need to re-read the original data to get the exact tokens
    # Since MIDI conversion loses some precision
    pass

def compare_sequences_detailed(gt_tokens, pred_tokens):
    """Detailed comparison of two token sequences."""
    # Skip ANTICIPATE token and SEP tokens
    gt_tokens = gt_tokens[4:]
    pred_tokens = pred_tokens[4:]
    
    # Extract score triplets from both
    gt_score_triplets = []
    pred_score_triplets = []
    
    for i in range(0, len(gt_tokens), 3):
        if i+2 < len(gt_tokens):
            time, dur, note = gt_tokens[i], gt_tokens[i+1], gt_tokens[i+2]
            if time < CONTROL_OFFSET and dur < CONTROL_OFFSET and note < CONTROL_OFFSET and note != REST:
                gt_score_triplets.append((time, dur, note))
    
    for i in range(0, len(pred_tokens), 3):
        if i+2 < len(pred_tokens):
            time, dur, note = pred_tokens[i], pred_tokens[i+1], pred_tokens[i+2]
            if time < CONTROL_OFFSET and dur < CONTROL_OFFSET and note < CONTROL_OFFSET and note != REST:
                pred_score_triplets.append((time, dur, note))
    
    print(f"  Ground truth score notes: {len(gt_score_triplets)}")
    print(f"  Predicted score notes: {len(pred_score_triplets)}")
    
    # Compare triplets
    min_len = min(len(gt_score_triplets), len(pred_score_triplets))
    
    exact_matches = 0
    pitch_matches = 0
    time_matches = 0
    duration_matches = 0
    
    time_diffs = []
    dur_diffs = []
    
    for i in range(min_len):
        gt_time, gt_dur, gt_note = gt_score_triplets[i]
        pred_time, pred_dur, pred_note = pred_score_triplets[i]
        
        # Extract pitch and instrument
        gt_note_val = gt_note - NOTE_OFFSET
        pred_note_val = pred_note - NOTE_OFFSET
        
        gt_pitch = gt_note_val % MAX_PITCH
        pred_pitch = pred_note_val % MAX_PITCH
        
        gt_instr = gt_note_val // MAX_PITCH
        pred_instr = pred_note_val // MAX_PITCH
        
        # Check matches
        if gt_time == pred_time and gt_dur == pred_dur and gt_note == pred_note:
            exact_matches += 1
        
        if gt_pitch == pred_pitch:
            pitch_matches += 1
        
        if gt_time == pred_time:
            time_matches += 1
        
        if gt_dur == pred_dur:
            duration_matches += 1
        
        # Track differences
        time_diffs.append(abs(gt_time - pred_time))
        dur_diffs.append(abs(gt_dur - pred_dur))
        
        # Show first few mismatches
        if i < 5 and (gt_time != pred_time or gt_dur != pred_dur or gt_note != pred_note):
            print(f"\n  Mismatch at position {i}:")
            print(f"    GT:   time={gt_time}, dur={gt_dur}, pitch={gt_pitch}, instr={gt_instr}, note_token={gt_note}")
            print(f"    Pred: time={pred_time}, dur={pred_dur}, pitch={pred_pitch}, instr={pred_instr}, note_token={pred_note}")
            print(f"    Diff: time={pred_time-gt_time}, dur={pred_dur-gt_dur}, pitch_match={gt_pitch==pred_pitch}")
    
    print(f"\n  Accuracy metrics:")
    print(f"    Exact token match: {100.0 * exact_matches / min_len:.2f}% ({exact_matches}/{min_len})")
    print(f"    Pitch match: {100.0 * pitch_matches / min_len:.2f}% ({pitch_matches}/{min_len})")
    print(f"    Time match: {100.0 * time_matches / min_len:.2f}% ({time_matches}/{min_len})")
    print(f"    Duration match: {100.0 * duration_matches / min_len:.2f}% ({duration_matches}/{min_len})")
    
    if time_diffs:
        avg_time_diff = sum(time_diffs) / len(time_diffs)
        max_time_diff = max(time_diffs)
        print(f"    Avg time difference: {avg_time_diff:.2f} units ({avg_time_diff*10:.1f}ms)")
        print(f"    Max time difference: {max_time_diff} units ({max_time_diff*10:.0f}ms)")
    
    if dur_diffs:
        avg_dur_diff = sum(dur_diffs) / len(dur_diffs)
        max_dur_diff = max(dur_diffs)
        print(f"    Avg duration difference: {avg_dur_diff:.2f} units ({avg_dur_diff*10:.1f}ms)")
        print(f"    Max duration difference: {max_dur_diff} units ({max_dur_diff*10:.0f}ms)")
    
    return exact_matches, pitch_matches, time_matches, duration_matches, min_len


print("="*80)
print("DETAILED COMPARISON OF PREDICTIONS VS GROUND TRUTH")
print("="*80)
print()

# Load the test data to get original tokens
with open('data/test_clean.txt', 'r') as f:
    lines = f.readlines()

total_exact = 0
total_pitch = 0
total_time = 0
total_dur = 0
total_notes = 0

for example_idx in range(5):
    print(f"Example {example_idx + 1}:")
    print("-" * 80)
    
    line = lines[example_idx]
    
    # Parse sequence
    if '|' in line:
        token_str, _ = line.split('|')
        gt_tokens = [int(t) for t in token_str.strip().split()]
    else:
        gt_tokens = [int(t) for t in line.strip().split()]
    
    # Load prediction file - need to regenerate or load from saved data
    # For now, let's re-run the prediction
    import torch
    from transformers import AutoModelForCausalLM
    
    if example_idx == 0:
        print("\n  Loading model...")
        model = AutoModelForCausalLM.from_pretrained('newest_model/')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
    
    # Find where score notes start
    score_start_idx = None
    for i in range(1, len(gt_tokens), 3):
        if i+2 < len(gt_tokens):
            if (gt_tokens[i] < CONTROL_OFFSET and 
                gt_tokens[i+1] < CONTROL_OFFSET and 
                gt_tokens[i+2] < CONTROL_OFFSET and
                gt_tokens[i+2] != REST):
                score_start_idx = i
                break
    
    if score_start_idx is None:
        print("  No score notes found, skipping")
        continue
    
    # Generate predictions
    from extract_greedy_examples import greedy_decode_sequence
    context_tokens = gt_tokens[:score_start_idx]
    input_ids = torch.tensor([context_tokens])
    generated = greedy_decode_sequence(model, input_ids, max_new_tokens=len(gt_tokens) - score_start_idx)
    predicted_tokens = generated[0, len(context_tokens):].cpu().tolist()
    
    # Reconstruct full predicted sequence
    full_predicted = gt_tokens[:score_start_idx] + predicted_tokens[:len(gt_tokens) - score_start_idx]
    
    # Compare
    exact, pitch, time, dur, count = compare_sequences_detailed(gt_tokens, full_predicted)
    
    total_exact += exact
    total_pitch += pitch
    total_time += time
    total_dur += dur
    total_notes += count
    
    print()

print("="*80)
print("OVERALL STATISTICS")
print("="*80)
print(f"Total notes compared: {total_notes}")
print(f"Exact token match: {100.0 * total_exact / total_notes:.2f}% ({total_exact}/{total_notes})")
print(f"Pitch-only match: {100.0 * total_pitch / total_notes:.2f}% ({total_pitch}/{total_notes})")
print(f"Time-only match: {100.0 * total_time / total_notes:.2f}% ({total_time}/{total_notes})")
print(f"Duration-only match: {100.0 * total_dur / total_notes:.2f}% ({total_dur}/{total_notes})")
print()
print("NOTE: The 100% pitch accuracy reported earlier only checked PITCH matching,")
print("not exact token matching (which includes timing and duration).")
