"""
Check how often score pitches and performance pitches match in the raw data
"""
from anticipation.vocab import CONTROL_OFFSET, TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET

print("="*80)
print("CHECKING SCORE vs PERFORMANCE PITCH MATCHING IN DATA")
print("="*80)

# Load sample sequences
print("\nLoading 100 sequences from train_clean.txt...")
with open('data/train_clean.txt', 'r') as f:
    lines = f.readlines()[:100]

total_notes = 0
matching_pitches = 0
different_pitches = 0

for line in lines:
    line = line.strip()
    if '|' in line:
        token_str, _ = line.split('|')
        tokens = [int(t) for t in token_str.strip().split()]
    else:
        tokens = [int(t) for t in line.split()]
    
    # Find all score and control triplets
    score_triplets = []
    control_triplets = []
    
    i = 1  # Skip ANTICIPATE token
    while i < len(tokens) - 2:
        if (tokens[i] < CONTROL_OFFSET and 
            tokens[i+1] < CONTROL_OFFSET and 
            tokens[i+2] < CONTROL_OFFSET):
            # Score triplet
            score_triplets.append({
                'time': tokens[i] - TIME_OFFSET,
                'dur': tokens[i+1] - DUR_OFFSET,
                'note': tokens[i+2] - NOTE_OFFSET
            })
            i += 3
        elif (tokens[i] >= CONTROL_OFFSET and 
              tokens[i+1] >= CONTROL_OFFSET and 
              tokens[i+2] >= CONTROL_OFFSET):
            # Control triplet
            control_triplets.append({
                'time': tokens[i] - CONTROL_OFFSET,
                'dur': tokens[i+1] - CONTROL_OFFSET,
                'note': tokens[i+2] - CONTROL_OFFSET
            })
            i += 3
        else:
            i += 1
    
    # Match based on position (they should be aligned)
    # The tokenization creates alternating score/control pairs
    num_pairs = min(len(score_triplets), len(control_triplets))
    
    for i in range(num_pairs):
        # NOTE: We already subtracted offsets above, so these are raw MIDI note values
        score_note = score_triplets[i]['note']
        control_note = control_triplets[i]['note']
        
        total_notes += 1
        if score_note == control_note:
            matching_pitches += 1
        else:
            different_pitches += 1
            # Debug first few mismatches
            if different_pitches <= 5:
                print(f"  Mismatch {different_pitches}: Score={score_note}, Perf={control_note}")

match_rate = matching_pitches / total_notes * 100 if total_notes > 0 else 0

print(f"\nResults from {len(lines)} sequences:")
print(f"Total note pairs checked: {total_notes:,}")
print(f"Matching pitches: {matching_pitches:,} ({match_rate:.2f}%)")
print(f"Different pitches: {different_pitches:,} ({100-match_rate:.2f}%)")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print(f"\nIn the TRAINING DATA:")
print(f"  - Score and performance pitches match {match_rate:.2f}% of the time")
print(f"  - This is the ground truth from the ASAP dataset")
print(f"")
print(f"MODEL PERFORMANCE (Teacher Forcing):")
print(f"  - Training set: 95.70% accuracy")
print(f"  - Validation set: 91.29% accuracy")
print(f"")
print(f"What this means:")
print(f"  - If pitches match {match_rate:.2f}% in data, the task ceiling is ~{match_rate:.2f}%")
print(f"  - Model achieving 95.70% suggests it's learning the task well")
print(f"  - The ~4% error could be:")
print(f"    * Cases where score/performance pitches differ (legitimate variations)")
print(f"    * Model errors on ambiguous timing")
print(f"    * Inherent difficulty in the task")
