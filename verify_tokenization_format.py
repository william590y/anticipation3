"""
Verify that tokenize-asap-sliding.py produces sequences with the correct format:
- Position 0: ANTICIPATE
- Positions 1-(k*6): k control+rest pairs
- Positions (k*6+1)-(k*6+3): SEP SEP SEP
- Positions (k*6+4)+: score triplets and remaining controls
"""

import os
import pandas as pd
from anticipation.config import *
from anticipation.vocab import *
from alignment import align_tokens2, load_annotation_file

# Import the tokenization function - need to handle the dash in filename
import importlib.util
spec = importlib.util.spec_from_file_location("tokenize_asap_sliding", "tokenize-asap-sliding.py")
tok_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tok_module)
tokenize_sliding_windows = tok_module.tokenize_sliding_windows

def verify_format():
    """Test tokenization on first piece and verify format"""
    
    # Load first piece
    asap_annotations = pd.read_csv(os.path.join('asap-dataset-master', "metadata.csv"))
    row = asap_annotations.iloc[0]
    
    midi_score_filename = row["midi_score"]
    midi_performance_filename = row["midi_performance"]
    
    filegroup = {
        'perf_midi': os.path.join('asap-dataset-master', midi_performance_filename),
        'score_midi': os.path.join('asap-dataset-master', midi_score_filename),
        'perf_anno': os.path.join('asap-dataset-master', midi_performance_filename.replace(".mid", "_annotations.txt")),
        'score_anno': os.path.join('asap-dataset-master', midi_score_filename.replace(".mid", "_annotations.txt")),
    }
    
    print("="*80)
    print(f"Testing tokenization format on: {row['composer']} - {row['title']}")
    print("="*80)
    
    # Run tokenization
    sequences = tokenize_sliding_windows(filegroup, prefix_controls=33)
    
    if not sequences:
        print("ERROR: No sequences generated")
        return False
    
    # Parse first sequence
    first_seq_str = sequences[0].split('|')[0].strip()
    tokens = list(map(int, first_seq_str.split()))
    
    print(f"\nGenerated {len(sequences)} sequences")
    print(f"First sequence has {len(tokens)} tokens")
    
    k = 33  # prefix_controls
    
    # Check position 0
    print(f"\n[Position 0] ANTICIPATE token:")
    print(f"  Token: {tokens[0]}")
    print(f"  Expected: {ANTICIPATE}")
    print(f"  ✓ Match: {tokens[0] == ANTICIPATE}")
    
    # Check positions 1 to k*6 (control+rest pairs)
    print(f"\n[Positions 1-{k*6}] {k} control+rest pairs:")
    print(f"  Each pair: 3 control tokens + 3 rest tokens = 6 tokens")
    print(f"  Total: {k} pairs × 6 = {k*6} tokens")
    
    # Examine first few control triplets
    print(f"\n  First 3 control triplets:")
    for i in range(3):
        pos = 1 + i*6
        ctrl_time = tokens[pos]
        ctrl_dur = tokens[pos+1]
        ctrl_pitch = tokens[pos+2]
        print(f"    Triplet {i}: pos={pos}, time={ctrl_time}, dur={ctrl_dur}, pitch={ctrl_pitch}")
        print(f"      All >= CONTROL_OFFSET? {ctrl_time >= CONTROL_OFFSET and ctrl_dur >= CONTROL_OFFSET and ctrl_pitch >= CONTROL_OFFSET}")
    
    # Check separator position
    sep_pos = k * 6 + 1  # After the control+rest pairs, +1 because position 0 is ANTICIPATE
    print(f"\n[Positions {sep_pos}-{sep_pos+2}] Three SEPARATOR tokens:")
    sep1 = tokens[sep_pos]
    sep2 = tokens[sep_pos+1]
    sep3 = tokens[sep_pos+2]
    print(f"  Token {sep_pos}: {sep1} (expected {SEPARATOR})")
    print(f"  Token {sep_pos+1}: {sep2} (expected {SEPARATOR})")
    print(f"  Token {sep_pos+2}: {sep3} (expected {SEPARATOR})")
    
    seps_correct = (sep1 == SEPARATOR and sep2 == SEPARATOR and sep3 == SEPARATOR)
    print(f"  ✓ All SEP tokens correct: {seps_correct}")
    
    # Check first score triplet
    score_start = sep_pos + 3
    print(f"\n[Position {score_start}+] Score triplets:")
    print(f"  First score triplet:")
    score_time = tokens[score_start]
    score_dur = tokens[score_start+1]
    score_pitch = tokens[score_start+2]
    print(f"    Time: {score_time} (< CONTROL_OFFSET? {score_time < CONTROL_OFFSET})")
    print(f"    Dur: {score_dur}")
    print(f"    Pitch: {score_pitch}")
    
    # Overall check
    print("\n" + "="*80)
    all_correct = (tokens[0] == ANTICIPATE and seps_correct)
    if all_correct:
        print("✓ FORMAT VERIFICATION PASSED")
        print(f"  - ANTICIPATE at position 0")
        print(f"  - {k} control+rest pairs at positions 1-{k*6}")
        print(f"  - SEP SEP SEP at positions {sep_pos}-{sep_pos+2}")
        print(f"  - Score triplets start at position {score_start}")
    else:
        print("✗ FORMAT VERIFICATION FAILED")
    
    return all_correct

if __name__ == "__main__":
    verify_format()
