# Data Augmentation for Anticipation Model

## Overview
The tokenization now includes **two augmentation strategies** applied to control/performance tokens to improve model robustness and generalization:

1. **Time Perturbation**: Adds random noise to timing of control tokens
2. **Token Masking**: Randomly masks control tokens to encourage model robustness

## Augmentation Details

### 1. Time Perturbation (50ms std dev)
- Applies Gaussian noise with standard deviation of 50ms to the timing of each control/performance token
- Implemented as: `time_perturbed = time_original + N(0, 50ms)`
- Only affects the TIME component of control triplets (time, duration, note)
- Simulates natural timing variations in human performance

### 2. Token Masking (50% probability)
- Each control/performance token triplet has a 50% chance of being replaced with `[MASK, MASK, MASK]`
- MASK token value: `55028` (new special token added to vocabulary)
- Encourages model to generate performances even with incomplete control information
- Similar to BERT-style masking for robustness

### 3. Multiple Augmentations (20x per piece)
- Each piece in the ASAP dataset generates **20 augmented versions**
- Each version gets different random perturbations and masks
- Dramatically increases training data diversity: ~230 pieces × 20 augmentations = ~4,600 unique training examples
- Each augmentation is independent (different random seed)

## Vocabulary Changes

### New Token
- **MASK**: Token `55028` - used to replace masked control tokens
- **VOCAB_SIZE**: Updated from `55028` to `55029`

### Token Distribution in Augmented Data
For a typical sequence with augmentation:
- ~50% of control tokens are MASK triplets `[55028, 55028, 55028]`
- ~50% of control tokens are perturbed by ±50ms timing noise
- Score tokens remain unchanged (ground truth)

## Implementation

### Command Line Arguments (defaults set)
```bash
python tokenize-asap.py
# Uses these defaults:
#   --perturb-std-ms 50.0          # 50ms time perturbation
#   --mask-prob 0.5                # 50% masking probability
#   --num-augmentations 20         # 20 versions per piece
#   --workers 128                  # 128 parallel workers
#   --skip-nones (enabled)         # Drop unmatched performance notes
```

### Modified Functions
1. **alignment.py::align_tokens2()**: Added `perturb_std_ms` and `mask_prob` parameters
2. **tokenize-asap.py::_interleave_tokenize4_single()**: Added augmentation loop
3. **anticipation/vocab.py**: Added MASK token

## Benefits

### 1. Timing Robustness
- Model learns to handle timing variations in control inputs
- Reduces overfitting to exact timing values
- Better generalization to new performances

### 2. Missing Information Robustness  
- Model learns to generate reasonable performances even with incomplete controls
- Encourages reliance on musical context rather than just immediate controls
- Similar to dropout but at the token level

### 3. Data Efficiency
- 20x data augmentation without additional annotation cost
- Each augmentation provides genuinely different training signal
- Helps prevent overfitting on small dataset (~230 pieces)

## Verification

Test results from `test_augmentation.py`:
```
✓ Time perturbation: 91.8% of tokens perturbed
✓ Masking: 50.0% of tokens masked (target: 50%)
✓ Uniqueness: 72-75% difference between augmentations
✓ Both augmentations work independently and together
```

## Training Implications

### Dataset Size
- **Original**: ~230 pieces → ~14k training sequences
- **Augmented**: ~230 pieces × 20 augmentations → ~280k training sequences (estimated)

### Model Requirements
- Must learn to handle MASK tokens in input
- Must learn robust representations despite timing noise
- Validation should use non-augmented data for fair comparison

### Expected Improvements
- Better generalization to new performances
- More robust to timing variations
- Reduced overfitting (more diverse training data)
- Potential for better note accuracy and timing accuracy

## Next Steps

1. **Retokenize dataset**: Run `python tokenize-asap.py` to generate augmented data
2. **Update training script**: Ensure MASK token is in vocabulary (already done - VOCAB_SIZE=55029)
3. **Train model**: Use masked loss training on augmented data
4. **Evaluate**: Compare model trained on augmented vs non-augmented data
