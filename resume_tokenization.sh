#!/bin/bash
# Resume tokenization after interruption
# This script automatically detects if tokenization was interrupted and resumes

echo "Checking for incomplete tokenization..."

if [ -f "./data/train_perturbed.txt.checkpoint" ]; then
    CHECKPOINT_COUNT=$(wc -l < "./data/train_perturbed.txt.checkpoint")
    echo "Found checkpoint file with $CHECKPOINT_COUNT processed pieces"
    echo "Resuming tokenization..."
    python tokenize-asap.py --resume
else
    echo "No checkpoint found. Starting fresh tokenization..."
    python tokenize-asap.py
fi
