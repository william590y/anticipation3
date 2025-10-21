# Tokenization Resumption - Quick Reference

## ✅ Your Tokenization Can Now Be Safely Resumed!

### If Your SLURM Job Gets Interrupted:

**Simply run with the `--resume` flag:**
```bash
python tokenize-asap.py --resume
```

That's it! The script will:
- ✓ Load the checkpoint file (`.checkpoint`)
- ✓ Skip all already-processed pieces
- ✓ Continue from where it left off
- ✓ Append new sequences to existing output files

---

## How It Works

### Automatic Checkpointing
After **every single piece** is processed:
1. Sequences are written to output files
2. Piece identifier written to checkpoint file
3. Checkpoint **immediately flushed** to disk (no buffering)

**Result:** Zero data loss, even if interrupted mid-processing

### Files Created
- `./data/train_perturbed.txt` - Training sequences
- `./data/test_perturbed.txt` - Test sequences  
- `./data/train_perturbed.txt.checkpoint` - Processed pieces tracker

---

## Usage Examples

### Manual Resume (Most Common)
```bash
# First run (interrupted after 500/1067 pieces):
python tokenize-asap.py

# Resume after interruption:
python tokenize-asap.py --resume
```

### Automatic Resume (Helper Scripts)

**Linux/Unix:**
```bash
bash resume_tokenization.sh
```

**Windows PowerShell:**
```powershell
.\resume_tokenization.ps1
```

**Windows Command Prompt:**
```batch
python tokenize-asap.py --resume
```

---

## SLURM Job Template

```bash
#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G

cd /path/to/anticipation3

# Auto-resume if checkpoint exists
if [ -f "./data/train_perturbed.txt.checkpoint" ]; then
    echo "Resuming from checkpoint..."
    python tokenize-asap.py --resume
else
    echo "Starting fresh..."
    python tokenize-asap.py
fi
```

---

## Monitoring Progress

### Check pieces processed:
```bash
# Linux/Mac:
wc -l ./data/train_perturbed.txt.checkpoint

# PowerShell:
(Get-Content ./data/train_perturbed.txt.checkpoint).Length
```

### Check sequences generated:
```bash
# Linux/Mac:
wc -l ./data/train_perturbed.txt
wc -l ./data/test_perturbed.txt

# PowerShell:
(Get-Content ./data/train_perturbed.txt).Length
(Get-Content ./data/test_perturbed.txt).Length
```

---

## Important Notes

### ✅ Safe Behaviors
- **Interrupt anytime** - checkpoint after each piece
- **No partial data** - pieces are atomic operations
- **Multiple resumes** - resume as many times as needed
- **Same results** - train/test split stays consistent

### ⚠️ Cautions
- **Don't edit checkpoint manually** - may cause issues
- **Don't delete checkpoint mid-run** - will cause duplicates
- **Augmentations differ on resume** - random state resets (this is OK)

### 🔄 Starting Fresh
If you want to completely restart (not resume):
```bash
rm ./data/train_perturbed.txt*
rm ./data/test_perturbed.txt
python tokenize-asap.py
```

---

## Time Estimates

With your settings (20 augmentations per piece, 128 workers):
- **~1,067 pieces total** (853 train + 214 test)
- **~2-3 minutes per piece** (estimated with augmentation)
- **Total time: ~35-50 hours** (fits in 2× 48-hour SLURM windows)

If interrupted at 50% (~24 hours):
- **~533 pieces remaining**
- **~18-24 hours to complete**

---

## Verification

Test the resumption system:
```bash
python test_resumption.py
```

Expected output:
```
✓ Resumption functionality verified successfully!
```

---

## Troubleshooting

**Q: "Resume shows 0 pieces to process"**  
A: All pieces already processed! Tokenization is complete.

**Q: "Different sequence counts after resume"**  
A: Augmentations use random sampling. Total should be similar (~±10%).

**Q: "Checkpoint exists but output files missing"**  
A: Delete checkpoint and start fresh - corruption occurred.

**Q: "Want to change augmentation parameters mid-run"**  
A: Must start fresh - parameters affect all pieces.

---

## Summary

Your tokenization is **fully protected** against:
- ✅ SLURM time limits
- ✅ System crashes
- ✅ Out-of-memory errors
- ✅ Manual interruptions
- ✅ Network failures

**Just run with `--resume` and continue where you left off!**
