# Masking Migration: From Vocabulary Token to Attention Masking

## Overview
This document describes the migration from using a MASK vocabulary token to proper attention-based masking during training.

## Problem with Previous Approach
- **Issue**: MASK was added as a vocabulary token (position 55028), increasing VOCAB_SIZE to 55029
- **Why it's wrong**: 
  - Unnecessarily increases model vocabulary size
  - Model has to learn embeddings for a token that only appears during training
  - Not the standard approach for masking in transformers
  - Complicates tokenization and detokenization

## New Approach: Attention-Based Masking
- **Solution**: Use HuggingFace's built-in masking mechanism via labels tensor
- **How it works**:
  1. Tokenization outputs full sequences (no token replacement)
  2. Mask indices are stored separately in the output format: `"tokens... | mask_indices..."`
  3. During training, masked positions have their labels set to -100
  4. CrossEntropyLoss automatically ignores positions with label=-100

## Changes Made

### 1. anticipation/vocab.py
**Before:**
```python
MASK = SPECIAL_OFFSET + 3  # 55028
VOCAB_SIZE = MASK + 1      # 55029
```

**After:**
```python
# MASK token removed - masking now handled via attention mechanism
ANTICIPATE = SPECIAL_OFFSET + 2  # 55027
VOCAB_SIZE = ANTICIPATE + 1      # 55028
```

**Impact**: VOCAB_SIZE reduced from 55029 to 55028

### 2. tokenize-asap.py
**Before:**
- Generated mask_decisions boolean array
- Replaced control triplets with `[MASK, MASK, MASK]` when mask_decisions[i]=True
- Output format: `"token1 token2 token3 ..."`

**After:**
- Generates mask_decisions boolean array (same)
- Keeps original control triplets (NO replacement)
- Tracks mask indices throughout sequence construction
- Output format: `"token1 token2 ... | mask_idx1 mask_idx2 ..."`

**Key changes:**
- Line 65-71: Time perturbation applied WITHOUT masking check
- Line 75: Added `mask_indices = []` to track mask positions
- Line 81-87: If should_mask, add positions to mask_indices instead of replacing tokens
- Line 140-143: Output format includes mask indices after `|` separator

### 3. alignment.py
**Before:**
```python
if should_mask:
    from anticipation.vocab import MASK
    l[0] = [MASK, MASK, MASK]
else:
    # Apply perturbation and convert to tokens
```

**After:**
```python
should_mask = mask_prob > 0 and np.random.random() < mask_prob
# Always apply perturbation and convert to tokens (no masking here)
# Mask decision not returned - tokenization handles masking internally
```

**Impact**: alignment.py no longer replaces tokens, always returns valid control tokens

### 4. train.py TokenizedDataset
**Before:**
```python
def __init__(self, file_path):
    self.sequences = []
    tokens = list(map(int, line.strip().split()))
    self.sequences.append(torch.tensor(tokens, dtype=torch.long))

def __getitem__(self, idx):
    tokens = self.sequences[idx]
    return {"input_ids": tokens, "labels": tokens}
```

**After:**
```python
def __init__(self, file_path):
    self.sequences = []
    self.mask_indices = []
    if '|' in line:
        token_str, mask_str = line.split('|')
        tokens = list(map(int, token_str.strip().split()))
        mask_idxs = list(map(int, mask_str.strip().split())) if mask_str.strip() else []
    else:
        # Old format compatibility
        tokens = list(map(int, line.split()))
        mask_idxs = []
    self.sequences.append(torch.tensor(tokens, dtype=torch.long))
    self.mask_indices.append(mask_idxs)

def __getitem__(self, idx):
    tokens = self.sequences[idx]
    labels = tokens.clone()
    mask_idxs = self.mask_indices[idx]
    if mask_idxs:
        labels[mask_idxs] = -100  # Ignore in loss calculation
    return {"input_ids": tokens, "labels": labels}
```

**Impact**: Training now properly ignores masked positions in loss calculation

## Migration Steps

### For Fresh Training
1. ✅ Code updated (vocab.py, tokenize-asap.py, alignment.py, train.py)
2. ⏳ Re-tokenize dataset: `python tokenize-asap.py`
3. ⏳ Train model with new approach: `python train.py`

### For Existing Checkpoints
- Existing checkpoints have VOCAB_SIZE=55029 (with MASK token)
- New code uses VOCAB_SIZE=55028 (without MASK token)
- **Incompatible**: Must train from scratch with new tokenization

## Verification
After re-tokenizing, verify the new format:
```python
# Check first line of train_perturbed.txt
with open('data/train_perturbed.txt') as f:
    line = f.readline().strip()
    if '|' in line:
        tokens, masks = line.split('|')
        print(f"✓ New format detected")
        print(f"  Tokens: {len(tokens.split())} tokens")
        print(f"  Masks: {len(masks.split())} indices")
    else:
        print(f"⚠ Old format - need to re-tokenize")
```

## Benefits
1. **Standard approach**: Aligns with HuggingFace transformers best practices
2. **Smaller vocabulary**: VOCAB_SIZE reduced by 1 (55029 → 55028)
3. **Cleaner semantics**: Model doesn't need to learn embeddings for a special masking token
4. **Better training**: CrossEntropyLoss natively handles -100 labels
5. **Simpler generation**: No need to handle MASK tokens during inference

## Next Steps
1. Run `python tokenize-asap.py` to generate new tokenized files with mask indices
2. Verify output format contains `|` separator with mask indices
3. Train model from scratch using `python train.py`
4. Model will now properly ignore masked positions during training
