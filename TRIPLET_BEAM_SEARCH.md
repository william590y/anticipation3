# Triplet-Aware Beam Search Implementation

## What Changed

**Previous implementation:** Standard token-by-token beam search
- Expand TIME → prune → Expand DURATION → prune → Expand PITCH → prune
- Myopic: only looks one token ahead
- Doesn't consider joint probability of (TIME, DURATION, PITCH) triplets

**New implementation:** Joint triplet beam search
- For each beam, explores complete (TIME, DURATION, PITCH) triplets
- Scores triplets jointly: P(TIME) × P(DURATION|TIME) × P(PITCH|TIME,DURATION)
- Prunes after evaluating complete triplets

## Why This Matters

Musical tokens have strong internal dependencies:
- TIME influences likely DURATION (longer notes have longer durations)
- TIME influences likely PITCH (timing affects which notes appear)
- DURATION and PITCH are correlated (note length and pitch relationship)

Standard beam search might pick a TIME token that scores well individually but pairs poorly with likely DURATION/PITCH combinations. Joint triplet search considers these dependencies.

## Implementation Details

For computational efficiency, uses a two-stage exploration strategy:

1. **Stage 1:** Get top-k TIME candidates (k = min(num_beams, 10))
2. **Stage 2:** For each TIME, get top-k (DURATION, PITCH) pairs (k = num_beams / k_time)

This reduces complexity from O(k³) to O(k²) candidates per beam while still looking ahead to complete triplets.

### Example with num_beams=10:
- Explore top-10 TIME tokens
- For each TIME, explore 1 (DURATION, PITCH) pair
- Total: 10 triplet candidates per beam

### Example with num_beams=50:
- Explore top-10 TIME tokens  
- For each TIME, explore 5 (DURATION, PITCH) pairs
- Total: 50 triplet candidates per beam

## Verification

Created `test_triplet_beam_search.py` with three tests:

1. **Beam Diversity Test:** Verifies beams have different scores (exploring alternatives)
2. **Joint vs Sequential Test:** Shows joint search can find different triplets than greedy
3. **Score Computation Test:** Validates P(TIME) × P(DUR|TIME) × P(PITCH|TIME,DUR) calculation

All tests pass ✓

## Expected Impact

Based on preliminary results showing +18.59% pitch improvement with standard beam search (num_beams=5), joint triplet beam search should show:

- **Greater improvement** for pitch (better captures pitch dependencies)
- **Potential improvement** for time/duration (exploits joint structure)
- **Diminishing returns** at higher beam widths (model may still be overconfident)

## Running the Evaluation

```powershell
python .\evaluate_batched_and_beam.py
```

This will test beam widths [5, 10, 20, 50] on 150_model with 20 sequences and compare:
- Greedy baseline
- Joint triplet beam search at each beam width
- Improvement deltas

Expected runtime: ~10-15 minutes (joint triplet search is more expensive than token-by-token)
