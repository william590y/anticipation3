"""
Test the correct pitch extraction formula.
"""
from anticipation.vocab import NOTE_OFFSET
from anticipation.config import MAX_PITCH, MAX_INSTR

# Example note token
note_token = 11000 + (128 * 0 + 60)  # C4 (MIDI 60) on instrument 0

print(f"Test note token: {note_token}")
print(f"NOTE_OFFSET: {NOTE_OFFSET}")
print(f"MAX_PITCH: {MAX_PITCH}")
print(f"MAX_INSTR: {MAX_INSTR}")
print()

# Wrong way (what the script was doing)
wrong_pitch = (note_token - NOTE_OFFSET) // MAX_INSTR
print(f"WRONG pitch extraction (// MAX_INSTR): {wrong_pitch}")

# Correct way
correct_pitch = (note_token - NOTE_OFFSET) % MAX_PITCH
correct_instr = (note_token - NOTE_OFFSET) // MAX_PITCH
print(f"CORRECT pitch extraction (% MAX_PITCH): {correct_pitch}")
print(f"CORRECT instrument extraction (// MAX_PITCH): {correct_instr}")
print()

# Test another example: A3 (MIDI 57) on instrument 1
note_token2 = 11000 + (128 * 1 + 57)
print(f"Test note token 2: {note_token2}")
wrong_pitch2 = (note_token2 - NOTE_OFFSET) // MAX_INSTR
correct_pitch2 = (note_token2 - NOTE_OFFSET) % MAX_PITCH
correct_instr2 = (note_token2 - NOTE_OFFSET) // MAX_PITCH
print(f"WRONG pitch: {wrong_pitch2}, CORRECT pitch: {correct_pitch2}, CORRECT instr: {correct_instr2}")
