"""
Check if the MIDI files actually contain the same notes or if notes are missing/different.
"""
import mido

def get_note_list(midi_file):
    """Extract all note_on events with their pitches."""
    notes = []
    for track in mido.MidiFile(midi_file).tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)
    return notes

for example_idx in range(1, 6):
    print(f"\n{'='*80}")
    print(f"Example {example_idx}")
    print('='*80)
    
    gt_file = f'test_examples/example_{example_idx}/ground_truth_score.mid'
    pred_file = f'test_examples/example_{example_idx}/model_predictions.mid'
    
    gt_notes = get_note_list(gt_file)
    pred_notes = get_note_list(pred_file)
    
    print(f"Ground truth: {len(gt_notes)} notes")
    print(f"Predictions:  {len(pred_notes)} notes")
    
    if len(gt_notes) != len(pred_notes):
        print(f"❌ NOTE COUNT MISMATCH: {len(gt_notes)} vs {len(pred_notes)}")
    else:
        print(f"✓ Note counts match")
    
    # Check if the notes are the same
    matches = sum(1 for g, p in zip(gt_notes, pred_notes) if g == p)
    print(f"Exact pitch matches: {matches}/{len(gt_notes)} ({100*matches/len(gt_notes):.1f}%)")
    
    # Show first few differences
    differences = []
    for i, (g, p) in enumerate(zip(gt_notes, pred_notes)):
        if g != p:
            differences.append((i, g, p))
            if len(differences) >= 5:
                break
    
    if differences:
        print(f"\nFirst {len(differences)} pitch differences:")
        for i, gt_pitch, pred_pitch in differences:
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            gt_name = f"{note_names[gt_pitch % 12]}{gt_pitch // 12 - 1}" if 0 <= gt_pitch < 128 else f"Invalid({gt_pitch})"
            pred_name = f"{note_names[pred_pitch % 12]}{pred_pitch // 12 - 1}" if 0 <= pred_pitch < 128 else f"Invalid({pred_pitch})"
            print(f"  Position {i}: GT={gt_name} (MIDI {gt_pitch}), Pred={pred_name} (MIDI {pred_pitch}), Diff={pred_pitch-gt_pitch} semitones")
    
    # Check pitch range
    if gt_notes:
        print(f"\nPitch ranges:")
        print(f"  GT:   {min(gt_notes)} to {max(gt_notes)} (MIDI note numbers)")
        print(f"  Pred: {min(pred_notes)} to {max(pred_notes)} (MIDI note numbers)")

print(f"\n{'='*80}")
print("SUMMARY")
print('='*80)
print("If you see missing bass notes, it could be:")
print("1. Notes at wrong pitches (different MIDI note numbers)")
print("2. Notes at correct pitches but wrong timing (so they don't sound right)")
print("3. Notes at correct pitches but wrong velocities (volume)")
print("4. Notes with wrong durations (too short to hear)")
