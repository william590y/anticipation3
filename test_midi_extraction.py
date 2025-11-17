from anticipation.vocab import CONTROL_OFFSET, ANTICIPATE, REST
from anticipation.convert import events_to_midi

def extract_score_only(tokens):
    """Extract only score tokens (not performance/control tokens)."""
    start_idx = 1 if (len(tokens) > 0 and tokens[0] == ANTICIPATE) else 0
    if start_idx == 1 and len(tokens) > 4:
        start_idx += 3
    
    events = []
    for i in range(start_idx, len(tokens), 3):
        if i+2 >= len(tokens):
            break
        time_tok, dur_tok, note_tok = tokens[i], tokens[i+1], tokens[i+2]
        if time_tok < CONTROL_OFFSET and dur_tok < CONTROL_OFFSET and note_tok < CONTROL_OFFSET:
            if note_tok != REST:
                events.extend([time_tok, dur_tok, note_tok])
    return events

def extract_performance_only(tokens):
    """Extract only performance (control) tokens."""
    start_idx = 1 if (len(tokens) > 0 and tokens[0] == ANTICIPATE) else 0
    if start_idx == 1 and len(tokens) > 4:
        start_idx += 3
    
    events = []
    for i in range(start_idx, len(tokens), 3):
        if i+2 >= len(tokens):
            break
        time_tok, dur_tok, note_tok = tokens[i], tokens[i+1], tokens[i+2]
        if time_tok >= CONTROL_OFFSET and dur_tok >= CONTROL_OFFSET and note_tok >= CONTROL_OFFSET:
            events.extend([time_tok - CONTROL_OFFSET, dur_tok - CONTROL_OFFSET, note_tok - CONTROL_OFFSET])
    return events

# Test on first sequence
with open('data/test_sliding.txt') as f:
    line = f.readline().strip()
    tokens = [int(t) for t in line.split('|')[0].split()]

print("Testing MIDI extraction...")
print(f"Total tokens: {len(tokens)}")

perf_events = extract_performance_only(tokens)
print(f"\nPerformance events: {len(perf_events)} ({len(perf_events)//3} notes)")
print(f"First 15 perf events: {perf_events[:15]}")

score_events = extract_score_only(tokens)
print(f"\nScore events: {len(score_events)} ({len(score_events)//3} notes)")
print(f"First 15 score events: {score_events[:15]}")

# Try to create MIDI
try:
    print("\nTesting performance MIDI creation...")
    perf_midi = events_to_midi(perf_events)
    print(f"✓ Performance MIDI created successfully")
    
    print("\nTesting score MIDI creation...")
    score_midi = events_to_midi(score_events)
    print(f"✓ Score MIDI created successfully")
    
    print("\n✓ All MIDI extractions working correctly!")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
