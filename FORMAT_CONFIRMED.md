# Sequence Format Confirmation

## Status: ✅ CONFIRMED

All components now use the format documented in `CONSISTENCY_VERIFIED.md`:

```
[ANTICIPATE,           # Position 0
 SEP, SEP, SEP,        # Positions 1-3
 
 # Prefix: k=33 control+rest pairs (positions 4-201)
 ctrl0_time, ctrl0_dur, ctrl0_pitch,  # Control triplet
 rest0_time, rest0_dur, REST,          # Rest triplet
 ...
 ctrl32_time, ctrl32_dur, ctrl32_pitch,
 rest32_time, rest32_dur, REST,
 
 # Body: alternating score/control (positions 202+)
 score0_time, score0_dur, score0_pitch,
 ctrl33_time, ctrl33_dur, ctrl33_pitch,
 score1_time, score1_dur, score1_pitch,
 ctrl34_time, ctrl34_dur, ctrl34_pitch,
 ...]
```

## Token Offsets

### Control Triplets
**All 3 elements** have `CONTROL_OFFSET` (27513) added:
```python
[CONTROL_OFFSET + time,
 CONTROL_OFFSET + duration,  # Also has DUR_OFFSET underneath
 CONTROL_OFFSET + pitch]     # Also has NOTE_OFFSET underneath
```

Example from output:
- Position 4: `27513, 37586, 38575`
- Decodes to: time=0.000s, dur=0.730s, pitch=62

### Score Triplets  
Regular offsets without CONTROL_OFFSET:
```python
[TIME_OFFSET + time,      # 0 to 9999
 DUR_OFFSET + duration,   # 10000 to 10999
 NOTE_OFFSET + pitch]     # 11000 to 27511
```

Example from output:
- Position 202: `0, 10025, 11062`  
- Decodes to: time=0.000s, dur=0.250s, pitch=62

### Rest Triplets
Used for padding in prefix:
```python
[TIME_OFFSET + time,
 DUR_OFFSET + 0,    # Always 10000 
 REST]              # 27512
```

## Files Updated

### ✅ tokenize-asap-sliding.py
- Line 237: Subtracts CONTROL_OFFSET from all 3 elements when extracting
- Lines 254-258: Adds CONTROL_OFFSET to all 3 elements when inserting
- Line 284: Prepends SEP SEP SEP at beginning (position 0 of interleaved_tokens)
- Line 300: Prepends ANTICIPATE to get final format

### ✅ check_interleaving.py
- Lines 121-123: Subtracts CONTROL_OFFSET from all 3 elements
- Lines 135-146: Builds as ANTICIPATE + SEP SEP SEP + control+rest pairs + alternating
- Lines 196-203: Decodes control triplets correctly
- Lines 233-242: Validates SEP SEP SEP at positions 1-3

### ✅ alignment.py
- Line 165: Adds CONTROL_OFFSET to all 3 elements - **ALREADY CORRECT**

## Verification Results

Ran `check_interleaving.py` on Bach Fugue BWV 846:

```
[Position 0] ANTICIPATE: 55027 ✓
[Positions 1-3] SEP SEP SEP: 55025, 55025, 55025 ✓ 
[Positions 4-201] 33 control+rest pairs:
  - Control durations: 0.730s, 0.750s, 1.010s... ✓ (realistic values)
  - Times monotonic from 0.000s ✓
[Positions 202+] Score triplets:
  - Times: 0.000s, 0.250s, 0.500s... ✓
  - Beat-aligned at 0.5 sec intervals ✓
```

## Next Steps

1. ✅ Format verified and consistent across all files
2. ⏭️ Ready to run full tokenization with `tokenize-asap-sliding.py`
3. ⏭️ Train model with confirmed format
4. ⏭ Verify `test_triplet_beam_search.py` handles this format correctly

## Notes

- Control triplets have CONTROL_OFFSET on **ALL 3 tokens**, not just time
- This matches the documented format in `CONSISTENCY_VERIFIED.md`
- Separators come immediately after ANTICIPATE (positions 1-3)
- First control+rest pair starts at position 4
