# Consistency Verification: Tokenization, Training, and Generation

## Status: ✅ VERIFIED CONSISTENT

All three components (tokenization, training, generation) use the same format and assumptions.

---

## Token Structure

### Vocabulary Ranges
```
TIME_OFFSET = 0
DUR_OFFSET = 10000
NOTE_OFFSET = 11000
REST = 27512
CONTROL_OFFSET = 27513
SPECIAL_OFFSET = 55025
SEPARATOR = 55025
ANTICIPATE = 55027
VOCAB_SIZE = 55028
```

### Token Classification
- **Score tokens**: 0 to 27512 (time, duration, note, REST)
- **Control tokens**: 27513 to 55024 (anticipated performance with CONTROL_OFFSET)
- **Special tokens**: 55025 to 55027 (SEPARATOR, ANTICIPATE)

---

## Sequence Format

All three components use this **exact** format:

```
[ANTICIPATE,
 SEP, SEP, SEP,
 
 # Prefix: k=33 control+rest pairs
 ctrl0_time, ctrl0_dur, ctrl0_pitch, rest0_time, rest0_dur, REST,
 ctrl1_time, ctrl1_dur, ctrl1_pitch, rest1_time, rest1_dur, REST,
 ...
 ctrl32_time, ctrl32_dur, ctrl32_pitch, rest32_time, rest32_dur, REST,
 
 # Body: alternating score/control
 score0_time, score0_dur, score0_pitch,
 ctrl33_time, ctrl33_dur, ctrl33_pitch,
 score1_time, score1_dur, score1_pitch,
 ctrl34_time, ctrl34_dur, ctrl34_pitch,
 ...]
```

### Triplet Structure

**Control Triplet** (performance tokens):
```python
[CONTROL_OFFSET + time,
 CONTROL_OFFSET + duration,
 CONTROL_OFFSET + pitch]

# All 3 tokens >= 27513 and < 55025
```

**Score Triplet** (score tokens):
```python
[TIME_OFFSET + time,      # 0 to 9999
 DUR_OFFSET + duration,   # 10000 to 10999
 NOTE_OFFSET + pitch]     # 11000 to 27511

# All 3 tokens < 27513
```

**Rest Triplet** (padding in prefix):
```python
[TIME_OFFSET + time,      # 0 to 9999
 DUR_OFFSET + 0,          # 10000
 REST]                    # 27512

# All 3 tokens < 27513
```

---

## Component Consistency

### 1. Tokenization (tokenize-asap.py)

**Creates:**
- Clean sequences without augmentation
- Format: `"token1 token2 ... | "` (mask indices empty for clean data)
- Control triplets: all tokens have CONTROL_OFFSET
- Score triplets: all tokens use TIME/DUR/NOTE offsets

**Output:**
- `data/train_clean.txt`
- `data/test_clean.txt`

### 2. Training (train.py TokenizedDataset)

**Augmentation Detection:**
```python
if (token[i] >= CONTROL_OFFSET and
    token[i+1] >= CONTROL_OFFSET and
    token[i+2] >= CONTROL_OFFSET and
    token[i] != SEPARATOR and
    token[i] != ANTICIPATE):
    # This is a control triplet → apply augmentation
```

**Augmentation Applied:**
1. **Time Perturbation** (control triplets only):
   ```python
   base_time = token[i] - CONTROL_OFFSET
   perturbation = N(0, perturb_std_ms)
   perturbed = CONTROL_OFFSET + max(0, base_time + perturbation)
   ```

2. **Masking** (control triplets only):
   ```python
   if rand() < mask_prob:
       mask_indices.extend([i, i+1, i+2])
       labels[mask_indices] = -100  # Ignored in loss
   ```

**NOT Augmented:**
- Score triplets (all tokens < CONTROL_OFFSET)
- Rest triplets (all tokens < CONTROL_OFFSET)
- Special tokens (ANTICIPATE, SEPARATOR)

**Verified:**
- ✅ Only control triplets augmented
- ✅ Score/rest/special tokens unchanged
- ✅ Each epoch sees different random augmentation

### 3. Generation (sample.py generate4)

**Creates Same Format:**
1. **Prefix**: k=33 control+rest pairs
   ```python
   for i in range(k):
       tokens.extend(controls[i*3:i*3+3])  # Control triplet
       cc_time = controls[i*3] - CONTROL_OFFSET
       tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])
   ```

2. **Body**: Alternating score/control
   ```python
   for i in range(num_controls):
       new_token = add_token(...)  # Generate score triplet
       tokens.extend(new_token)     # [TIME+t, DUR+d, NOTE+n]
       if i+k < num_controls:
           tokens.extend(controls[(i+k)*3:(i+k)*3+3])  # Add future control
   ```

**Safe Logits:**
```python
logits[CONTROL_OFFSET:SPECIAL_OFFSET] = -inf  # Can't generate controls
logits[SPECIAL_OFFSET:] = -inf                 # Can't generate special tokens
```

**Verified:**
- ✅ Generates same format as tokenization
- ✅ Never generates control or special tokens
- ✅ Score tokens use TIME/DUR/NOTE offsets

---

## Test Results

### Consistency Test (verify_consistency.py)
```
✓ Control triplet detection: PASS
✓ Score triplet detection: PASS
✓ Rest triplet detection: PASS
✓ Separator filtering: PASS
✓ Anticipate filtering: PASS
✓ Format matching: PASS
```

### Augmentation Test (test_augmentation_training.py)
```
Sample 1: 9 masked positions (3 control triplets)
  ctrl0: perturbed (Δ=-1)
  ctrl1: perturbed (Δ=10), masked
  ctrl2: perturbed (Δ=-4), masked
  ctrl3: perturbed (Δ=2), masked
  ✓ score0: unchanged
  ✓ score1: unchanged
  ✓ ANTICIPATE, SEP, REST: unchanged
```

**Results:**
- ✅ Control triplets perturbed and/or masked
- ✅ Score triplets never modified
- ✅ Protected tokens (ANTICIPATE, SEP, REST) never modified
- ✅ Different augmentation each call (randomization working)

---

## Critical Assumptions

### All Components Agree On:

1. **Control Triplet Identification**:
   - All 3 tokens >= CONTROL_OFFSET
   - All 3 tokens < SPECIAL_OFFSET
   - First token != SEPARATOR and != ANTICIPATE

2. **Score Triplet Identification**:
   - All 3 tokens < CONTROL_OFFSET

3. **Sequence Structure**:
   - Starts with ANTICIPATE token
   - Followed by 3 SEPARATOR tokens
   - Then k=33 control+rest pairs (prefix)
   - Then alternating score/control (body)

4. **Token Offsets**:
   - Control: CONTROL_OFFSET + value
   - Score time: TIME_OFFSET + value
   - Score dur: DUR_OFFSET + value
   - Score note: NOTE_OFFSET + value
   - Rest: [TIME_OFFSET+t, DUR_OFFSET+0, REST]

5. **Augmentation**:
   - Applied ONLY to control triplets
   - Time perturbation: adds N(0, std) to time component
   - Masking: sets labels=-100 for entire triplet
   - Score triplets NEVER augmented

---

## Workflow

### 1. Tokenization
```bash
python tokenize-asap.py
```
- Outputs: `data/train_clean.txt`, `data/test_clean.txt`
- Format: Clean sequences, no augmentation
- Control triplets have CONTROL_OFFSET
- Score triplets use TIME/DUR/NOTE offsets

### 2. Training
```bash
python train.py --perturb_std_ms 50.0 --mask_prob 0.5
```
- Reads clean sequences
- Applies on-the-fly augmentation to control triplets
- Ignores masked positions in loss (labels=-100)
- Score triplets used for prediction targets

### 3. Generation
```python
from anticipation.sample import generate4
events, tokens = generate4(model, controls, top_p=0.95)
```
- Creates same format as training
- Model predicts score triplets
- Controls provided as input (with CONTROL_OFFSET)
- Output events are score triplets (without offsets for MIDI conversion)

---

## Summary

✅ **VERIFIED CONSISTENT**

All three components (tokenization, training, generation) are consistent:

- Same sequence format
- Same token offset conventions
- Same control vs score identification logic
- Same assumptions about what can/cannot be augmented
- Same assumptions about what model generates

**Ready for training and generation!**
