"""
Analyze how many notes are filtered out during the alignment process.
This will show us what percentage of performance notes are omitted due to
pitch mismatches or timing issues.
"""
import os
from alignment import align_tokens2
from anticipation.convert import midi_to_events
from anticipation.vocab import *

# Get a sample of ASAP files
asap_dir = 'asap-dataset-master'
piece_types = [
    ('Bach', 'Fugue'),
    ('Bach', 'Prelude'),
    ('Beethoven', 'Sonata'),
    ('Mozart', 'Sonata'),
]

print("="*80)
print("ANALYZING ALIGNMENT FILTERING")
print("="*80)

total_perf_notes = 0
total_score_notes = 0
total_matched_pairs = 0
samples_analyzed = 0
max_samples = 10  # Analyze 10 pieces

for composer, piece_type in piece_types:
    type_dir = os.path.join(asap_dir, composer, piece_type)
    if not os.path.exists(type_dir):
        continue
    
    for piece_dir in os.listdir(type_dir):
        if samples_analyzed >= max_samples:
            break
            
        piece_path = os.path.join(type_dir, piece_dir)
        if not os.path.isdir(piece_path):
            continue
        
        # Find the performance and score files
        perf_midi = None
        score_midi = None
        perf_annot = None
        score_annot = None
        
        for fname in os.listdir(piece_path):
            if fname == 'midi_score.mid':
                score_midi = os.path.join(piece_path, fname)
            elif fname == 'midi_score_annotations.txt':
                score_annot = os.path.join(piece_path, fname)
            elif fname.endswith('.mid') and fname != 'midi_score.mid':
                perf_midi = os.path.join(piece_path, fname)
            elif fname.endswith('_annotations.txt') and fname != 'midi_score_annotations.txt':
                perf_annot = os.path.join(piece_path, fname)
        
        if not all([perf_midi, score_midi, perf_annot, score_annot]):
            continue
        
        try:
            # Get raw note counts
            perf_events = midi_to_events(perf_midi, quantize=False)
            score_events = midi_to_events(score_midi, quantize=False)
            
            num_perf_notes = len(perf_events) // 3
            num_score_notes = len(score_events) // 3
            
            # Get aligned pairs
            matched_tuples = align_tokens2(perf_midi, score_midi, perf_annot, score_annot, 
                                          skip_Nones=True, thres=0.1)
            num_matched = len(matched_tuples)
            
            total_perf_notes += num_perf_notes
            total_score_notes += num_score_notes
            total_matched_pairs += num_matched
            
            perf_retention = 100.0 * num_matched / num_perf_notes if num_perf_notes > 0 else 0
            score_retention = 100.0 * num_matched / num_score_notes if num_score_notes > 0 else 0
            
            print(f"\n{piece_dir}:")
            print(f"  Performance notes: {num_perf_notes}")
            print(f"  Score notes: {num_score_notes}")
            print(f"  Matched pairs: {num_matched}")
            print(f"  Performance retention: {perf_retention:.1f}%")
            print(f"  Score retention: {score_retention:.1f}%")
            
            samples_analyzed += 1
            
        except Exception as e:
            print(f"Error processing {piece_dir}: {e}")
            continue

print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)
print(f"Samples analyzed: {samples_analyzed}")
print(f"Total performance notes: {total_perf_notes:,}")
print(f"Total score notes: {total_score_notes:,}")
print(f"Total matched pairs: {total_matched_pairs:,}")

if total_perf_notes > 0 and total_score_notes > 0:
    print()
    print(f"Performance note retention: {100.0 * total_matched_pairs / total_perf_notes:.2f}%")
    print(f"Score note retention: {100.0 * total_matched_pairs / total_score_notes:.2f}%")
    print()
    print(f"Performance notes FILTERED OUT: {total_perf_notes - total_matched_pairs:,} "
          f"({100.0 * (total_perf_notes - total_matched_pairs) / total_perf_notes:.2f}%)")
    print(f"Score notes FILTERED OUT: {total_score_notes - total_matched_pairs:,} "
          f"({100.0 * (total_score_notes - total_matched_pairs) / total_score_notes:.2f}%)")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print()
    print("Filtered performance notes could be:")
    print("  • Wrong notes (pitch errors)")
    print("  • Ornaments/embellishments not in score")
    print("  • Repeated notes")
    print("  • Notes outside timing threshold")
    print()
    print("Filtered score notes could be:")
    print("  • Omitted notes (performer skipped them)")
    print("  • Notes outside the performance time range")
    print("  • Alignment annotation errors")
    print()
    print("This filtering ensures 100% pitch accuracy in training data,")
    print("but means the model never learns to handle pitch errors!")
else:
    print("\nNo samples were successfully analyzed.")
