# Resume tokenization after interruption (PowerShell version)
# This script automatically detects if tokenization was interrupted and resumes

Write-Host "Checking for incomplete tokenization..." -ForegroundColor Cyan

$checkpointFile = ".\data\train_perturbed.txt.checkpoint"

if (Test-Path $checkpointFile) {
    $checkpointCount = (Get-Content $checkpointFile).Length
    Write-Host "Found checkpoint file with $checkpointCount processed pieces" -ForegroundColor Yellow
    Write-Host "Resuming tokenization..." -ForegroundColor Green
    python tokenize-asap.py --resume
} else {
    Write-Host "No checkpoint found. Starting fresh tokenization..." -ForegroundColor Green
    python tokenize-asap.py
}
