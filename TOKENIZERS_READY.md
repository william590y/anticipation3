# Tokenizers Ready for Production

Both tokenization scripts have been cleaned and are ready to run.

## Changes Made

### Removed from Both Files:
- ✅ All augmentation logic (time perturbation)
- ✅ All masking logic (mask_prob, mask indices)
- ✅ Misleading DELTA constant references
- ✅ Complex augmentation parameters and loops

### Result:
Both tokenizers now perform **clean tokenization only**:
1. Load MIDI performance and score files
2. Load annotation files
3. Align performance notes with score notes (filters wrong notes)
4. Create interleaved sequences (control, score, control, score...)
5. Output clean token sequences

## File Overview

### tokenize-asap.py (Original with Sequence Packing)
**Purpose:** Tokenize full pieces with sequence packing

**Configuration:**
- Workers: Set via command line or default
- Sequence packing: YES (concatenates pieces, chunks into 1024-token sequences)
- Output: `data/train_clean.txt`, `data/test_clean.txt`

**Key Details:**
- Multiple sequences per piece (as many as fit)
- May have control/score pairs split across sequence boundaries
- More total training sequences

**Function Signature:**
```python
def _interleave_tokenize4_single(filegroup, skip_Nones=True, prefix_controls=33)
```

**Output Format:**
```
token1 token2 token3 ... tokenN | 
```
(Empty mask indices after `|`)

---

### tokenize-asap-openings.py (NEW - Openings Only)
**Purpose:** Tokenize only the opening 1024 tokens from each piece

**Configuration:**
- Workers: 128 (hardcoded for better machine)
- Sequence packing: NO (one sequence per piece)
- Output: `data/train_openings.txt`, `data/test_openings.txt`

**Key Details:**
- One sequence per piece (opening only)
- Guarantees control note is always in context window
- Fewer total sequences (~851 train, ~216 test)
- No cross-piece boundaries

**Function:**
```python
def tokenize_opening(filegroup, skip_Nones=True, prefix_controls=33, context_size=CONTEXT_SIZE)
```

**Output Format:**
```
token1 token2 token3 ... tokenN | 
```
(Same format as original - empty mask indices after `|`)

---

## Sequence Structure (Both Files)

Both tokenizers produce the same interleaved structure:

**Total:** 1024 tokens = Mode (1) + Bootstrap SEPs (3) + Interleaved (1020)

**Interleaving Pattern:**
1. **Prefix:** 33 control+REST pairs (66 triplets = 198 tokens)
   - Control triplet: (time, duration, control_note)
   - REST triplet: (time, duration=0, REST_token)

2. **Main sequence:** Alternating score/control pairs
   - Score triplet: (time, duration, note) or (time, duration=0, REST)
   - Control triplet: (time, duration, control_note)
   - Pattern: CSCSCSCS... (822 tokens = 137 score+control pairs)

**Pattern:** `CRCRCR...CSCSCS`
- C = Control triplet
- R = REST triplet  
- S = Score triplet

## Model Task

Given a control triplet at position i, predict the score triplet at position i+3.

**Prediction includes:**
1. Should it be REST or NOTE?
2. If NOTE, what pitch? (should match control pitch due to alignment filtering)
3. What timing? (may differ from control - expressive timing)
4. What duration? (may differ from control - expressive duration)

## Train/Test Split

Both tokenizers use the same split strategy:

**Split by score:** 80% train / 20% test (by unique MIDI scores)
- Prevents data leakage (same piece in train and test)
- Different performances of same piece stay in same split
- Ensures model generalizes to new pieces

**Expected Counts:**
- Original (sequence packing): ~several thousand sequences
- Openings only: ~851 train sequences, ~216 test sequences

## Alignment Filtering

Both tokenizers use `align_tokens2()` which filters performer mistakes:

```python
# From alignment.py line 211
if p_note != s_note: continue
```

**Result:**
- Only matches performance notes with SAME PITCH as score
- Wrong notes from performer are discarded
- Training data shows 100% pitch matching
- ~98.45% performance notes retained
- ~96.05% score notes retained

## Next Steps

### Run Tokenization:

**Option 1: Openings Only (Recommended for testing)**
```powershell
python tokenize-asap-openings.py
```
- Uses 128 workers
- Generates `data/train_openings.txt` and `data/test_openings.txt`
- ~5-10 minutes on good machine

**Option 2: Full with Sequence Packing**
```powershell
python tokenize-asap.py --workers 128
```
- Generates `data/train_clean.txt` and `data/test_clean.txt`
- More training data but potential context issues

### Implement Augmentation Elsewhere

Both files now produce clean data. Implement augmentation:

1. **During training** (in `train.py`):
   - Apply perturbation in dataset `__getitem__`
   - Apply masking in dataset `__getitem__`
   - Already implemented in current `train.py`

2. **Separate preprocessing** (if needed):
   - Read clean tokenized files
   - Apply augmentation
   - Write augmented files
   - Keep original clean files

### Training

After tokenization, train using existing `train.py`:

```powershell
# Example: Train with time augmentation
python train.py --data_path data/train_openings.txt --val_data_path data/test_openings.txt --perturb_std_ms 50.0 --mask_prob 0.05
```

The training script already handles augmentation on-the-fly.

## Verification

Both tokenizers confirmed clean:
```bash
# No matches found for augmentation/masking keywords:
grep -i "perturb\|mask_prob\|augment" tokenize-asap.py
grep -i "perturb\|mask_prob\|augment" tokenize-asap-openings.py
```

✅ Ready for production use!
