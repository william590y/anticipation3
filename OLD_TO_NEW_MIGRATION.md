# What To Do If Old Tokenization Job Is Interrupted

## The Problem

**You can't reliably resume from the old script** because:
- Old script uses `imap_unordered` (processes pieces in random order)
- No checkpoint file to track which pieces were processed
- Can't determine which pieces to skip

## Your Options

### Option 1: Let Old Job Finish (RECOMMENDED)
**Best approach:** Just let the current job complete if possible.
- No risk of duplicates or missed pieces
- Clean, complete dataset
- Use new script with checkpointing for next tokenization

### Option 2: Start Fresh with New Script
If old job gets interrupted:
```bash
# Start completely fresh with new script (has checkpoint support)
rm ./data/train_output.txt
rm ./data/test_output.txt
python tokenize-asap.py
```

**Pros:**
- ✓ Full checkpoint support going forward
- ✓ Can resume if interrupted again
- ✓ No risk of duplicates

**Cons:**
- ✗ Lose progress from old job
- ✗ Have to start over

### Option 3: Continue and Accept Duplicates
If old job processed ~50% and got interrupted:
```bash
# Just run new script - it will reprocess everything since no checkpoint exists
python tokenize-asap.py
```

This will reprocess all pieces (including already-done ones), resulting in:
- ~2x data for already-processed pieces
- Normal amount for remaining pieces

**Is this bad?** Actually, **NO!** 
- More data = better training
- Duplicates with different augmentations (random masks/perturbations)
- Worse case: ~1.5x total data (not harmful for training)

### Option 4: Manual Tracking (Advanced)
If you know from the progress bar that X pieces were processed:
1. Note the exact number of completed pieces
2. Modify the new script to skip first X pieces
3. This is error-prone and not recommended

## Recommended Action

**What I suggest:**

1. **Check your SLURM time remaining:**
   ```bash
   squeue -u $USER
   ```

2. **If >12 hours remaining:** Let it finish

3. **If <12 hours remaining:** 
   - Cancel job: `scancel <job_id>`
   - Start fresh with new script: `python tokenize-asap.py`
   - New script has checkpoint support for future interruptions

## Why New Script Can't Resume Old Script

The old script:
```python
# Processes pieces in random order - no way to know which ones
for split, lines, stats in pool.imap_unordered(_worker_split, payloads):
    # Write output
```

The new script:
```python
# Tracks each piece in checkpoint file
for split, lines, stats, piece_id in pool.imap_unordered(_worker_split, payloads):
    f_checkpoint.write(piece_id + '\n')  # Old script doesn't do this!
```

**Without the checkpoint file, we can't know which pieces to skip.**

## Summary

Since you're running the old script:
- ✅ Let it finish if you have time
- ✅ Start fresh with new script if interrupted
- ❌ Don't try to resume old → new (not reliably possible)
- ✅ Use new script for all future tokenizations (has checkpoint support)

The checkpoint system only helps **going forward**, not retroactively.
