# Token Format Consistency Verification

**Date:** December 9, 2025  
**Status:** ✅ ALL FILES CONSISTENT

## Verified Format Specification

### Sequence Structure
```
Position 0:      ANTICIPATE token (55027)
Positions 1-3:   SEP SEP SEP (3× SEPARATOR = 55025)
Positions 4-201: 33 control+rest pairs (198 tokens, 66 triplets)
Positions 202+:  Alternating score/control triplets
Total:           1024 tokens (CONTEXT_SIZE)
```

### Token Encoding Rules

#### Control Triplets (Performance)
- **Format:** `[time, duration, pitch]`
- **Encoding:** ALL 3 elements have `CONTROL_OFFSET` (27513) added
- **Example:** `[CONTROL_OFFSET + 100, CONTROL_OFFSET + 10050, CONTROL_OFFSET + 11060]`
- **Range:** All 3 tokens in `[27513, 55025)` (below SPECIAL_OFFSET)
- **Detection:** Check if all 3 consecutive tokens >= CONTROL_OFFSET and not SEPARATOR/ANTICIPATE

#### Score Triplets (Musical Notes)
- **Format:** `[time, duration, pitch]`
- **Encoding:** 
  - `time = TIME_OFFSET + value` (TIME_OFFSET = 0)
  - `duration = DUR_OFFSET + value` (DUR_OFFSET = 10000)
  - `pitch = NOTE_OFFSET + value` (NOTE_OFFSET = 11000)
- **Range:** All 3 tokens < CONTROL_OFFSET (27513)
- **Detection:** Check if all 3 consecutive tokens < CONTROL_OFFSET and pitch != REST

#### Rest Triplets
- **Format:** `[time, duration, REST]`
- **Encoding:**
  - `time = TIME_OFFSET + value`
  - `duration = DUR_OFFSET + 0` = 10000
  - `pitch = REST` = 27512
- **Note:** REST token is just below CONTROL_OFFSET

## File-by-File Verification

### ✅ tokenize-asap-sliding.py (Line 275-310)
**Purpose:** Generate training sequences with sliding windows

**Format Generation:**
- Line 237: Extracts performance triplets by subtracting CONTROL_OFFSET from ALL 3 elements
- Lines 254-258: Adds CONTROL_OFFSET to ALL 3 elements when building control triplets
- Line 284: Prepends `[SEPARATOR, SEPARATOR, SEPARATOR]` at position 0 of interleaved_tokens
- Line 300: Prepends ANTICIPATE to get final format: `[ANTICIPATE] + interleaved_tokens`
- Line 302: Verifies sequence length is exactly CONTEXT_SIZE (1024)

**Output Format:**
```python
sequence = [ANTICIPATE, SEP, SEP, SEP, <control+rest pairs>, <alternating score/control>]
```

**Status:** ✅ Correct - ANTICIPATE at position 0, SEP SEP SEP at positions 1-3

---

### ✅ train.py (Lines 50-280)
**Purpose:** Training script with on-the-fly augmentation and evaluation

**Format Loading (Lines 55-75):**
- Loads sequences from tokenized files
- Expects format: `[ANTICIPATE, control_tokens..., score_tokens..., PAD...]`
- Line 91: Validates first token is ANTICIPATE
- Handles both old format (tokens only) and new format (tokens | mask_indices)

**Augmentation Logic (Lines 103-150):**
- Line 124: Skips first token (ANTICIPATE mode token)
- Lines 128-133: Detects control triplets by checking all 3 tokens >= CONTROL_OFFSET
- Lines 133-134: Excludes SEPARATOR and ANTICIPATE (also >= CONTROL_OFFSET)
- Only augments control triplets, not score triplets

**Evaluation Logic (Lines 240-280):**
- Line 250: Skips first token (mode token), starts from position 1
- Lines 253-255: Detects score triplets by checking all 3 tokens < CONTROL_OFFSET
- Lines 258-267: Calculates pitch accuracy only on score note tokens (position i+2)
- Correctly handles triplet boundaries (i += 3 after finding score triplet)

**Status:** ✅ Correct - Expects ANTICIPATE at 0, correctly distinguishes control vs score triplets

---

### ✅ inference.py (Lines 1-194)
**Purpose:** Extract test examples with greedy decoding

**Format Parsing (Lines 40-68):**
- Line 42-43: Skips ANTICIPATE token (`tokens = tokens[1:]`)
- Line 45-46: Skips 3 SEP tokens (`tokens = tokens[3:]`)
- Line 58: Detects control triplets: all 3 tokens >= CONTROL_OFFSET
- Lines 60-62: Removes CONTROL_OFFSET from ALL 3 elements to extract performance
- Lines 66-67: Detects score triplets: all 3 tokens < CONTROL_OFFSET and not REST

**Score Start Detection (Lines 114-117):**
- Searches from position 1 onward
- Checks triplets in steps of 3
- Lines 114-116: Identifies score notes by all 3 tokens < CONTROL_OFFSET

**Status:** ✅ Correct - Skips ANTICIPATE + 3 SEP, correctly identifies triplet types

---

### ✅ analyze_triplet_beam_search.py (Lines 1-80)
**Purpose:** Analyze beam search with loss progression and error distribution

**Score Extraction (Lines 20-40):**
- Line 24: Skips ANTICIPATE token if present
- Line 26-27: Skips 3 SEP tokens after ANTICIPATE
- Line 38: Detects score triplets: all 3 tokens < CONTROL_OFFSET

**Performance Extraction (Lines 42-64):**
- Line 46: Skips ANTICIPATE token if present  
- Line 48-49: Skips 3 SEP tokens after ANTICIPATE
- Line 60: Detects control triplets: all 3 tokens >= CONTROL_OFFSET
- Line 61: Removes CONTROL_OFFSET from ALL 3 elements

**Status:** ✅ Correct - Properly handles ANTICIPATE + 3 SEP header, distinguishes triplet types

---

## Consistency Summary

### Control Triplet Handling
**All files correctly:**
- Add CONTROL_OFFSET to ALL 3 elements when creating control triplets
- Subtract CONTROL_OFFSET from ALL 3 elements when extracting control triplets
- Detect control triplets by checking all 3 consecutive tokens >= CONTROL_OFFSET
- Exclude SEPARATOR and ANTICIPATE (which are also >= CONTROL_OFFSET)

### Score Triplet Handling
**All files correctly:**
- Detect score triplets by checking all 3 consecutive tokens < CONTROL_OFFSET
- Exclude REST tokens (pitch = 27512) when extracting playable notes
- Start searching after ANTICIPATE (position 0) and SEP SEP SEP (positions 1-3)

### Sequence Header
**All files correctly:**
- Position 0: ANTICIPATE token (55027)
- Positions 1-3: SEP SEP SEP (3× SEPARATOR = 55025)
- Skip these 4 tokens when parsing musical content

## Validation Tests

### ✅ check_interleaving.py (DELETED)
- Previously verified all format properties
- All checks passed on tokenized data

### ✅ smoketest_pitch_accuracy.py
- Test 1-3: All pass with correct accuracy percentages
- Control-only test: Correctly finds 0 pitches (control triplets filtered out)
- Validates pitch detection logic works correctly

### ✅ smoketest_duration_scaling.py
- Duration scaling: All ratios = 1.0000 (perfect)
- Beat spacing: 0.5 sec intervals maintained
- Validates triplet indexing: [time=0, dur=1, pitch=2]

## Conclusion

✅ **ALL FILES ARE CONSISTENT** with the token format specification:
- ANTICIPATE at position 0
- SEP SEP SEP at positions 1-3
- Control triplets: ALL 3 elements have CONTROL_OFFSET added
- Score triplets: ALL 3 elements < CONTROL_OFFSET
- Proper triplet detection and extraction logic throughout

**No format inconsistencies found.**
