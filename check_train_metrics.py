"""
Check what metrics were actually reported during training.
"""
import numpy as np

# Load the training metrics
data = np.load('newest_model/losses.npz')

print("Available keys in losses.npz:")
for key in data.files:
    print(f"  - {key}")

print("\nTraining metrics:")
if 'train_losses' in data:
    train_losses = data['train_losses']
    print(f"  Training steps: {len(train_losses)}")
    print(f"  Final training loss: {train_losses[-1]:.4f}")

if 'val_losses' in data:
    val_losses = data['val_losses']
    print(f"  Validation checks: {len(val_losses)}")
    print(f"  Final validation loss: {val_losses[-1]:.4f}")

if 'val_accuracies' in data:
    val_accuracies = data['val_accuracies']
    print(f"  Validation accuracy checks: {len(val_accuracies)}")
    print(f"  Validation accuracies: {val_accuracies}")
    print(f"  Final validation 'pitch accuracy': {val_accuracies[-1]:.2f}%")
    print(f"  Max validation 'pitch accuracy': {max(val_accuracies):.2f}%")
    print(f"  Min validation 'pitch accuracy': {min(val_accuracies):.2f}%")

if 'validation_steps' in data:
    val_steps = data['validation_steps']
    print(f"  Validation steps: {val_steps}")
