"""
Training script with MASKED LOSS for conditional score generation.

This script trains the model to predict ONLY score tokens, while masking:
- Header tokens (ANTICIPATE, SEPARATORs)
- Control tokens (performance with CONTROL_OFFSET)
- REST tokens

This forces the model to learn: "Given controls, generate scores"
"""

import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
from anticipation.vocab import ANTICIPATE, SEPARATOR, CONTROL_OFFSET, REST, NOTE_OFFSET
from anticipation.config import MAX_NOTE

def print_gpu_memory_stats():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

def check_model_for_nans(model):
    """Check for NaN values in model parameters"""
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in parameter {name}")
            return True
    return False

# Print device info
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_count = torch.cuda.device_count()
    print(f"✓ CUDA is available with {device_count} device(s)")
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"  Device {i}: {device_name}")
        props = torch.cuda.get_device_properties(i)
        print(f"    - Total memory: {props.total_memory / 1024**3:.2f} GB")
else:
    device = torch.device("cpu")
    print("✗ CUDA is not available! Training will be slower on CPU.")

print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")


class MaskedTokenDataset(Dataset):
    """
    Dataset that applies masked loss to train only on score tokens.
    
    Tokenization format:
    - [ANTICIPATE, SEP, SEP, SEP] - header (positions 0-3)
    - [ctrl, rest] × 33 - prefix (positions 4-201)
    - [ctrl, score, ctrl, score, ...] - body (positions 202+)
    
    We mask (set label to -100):
    - ANTICIPATE token
    - SEPARATOR tokens  
    - All tokens with value >= CONTROL_OFFSET (controls)
    - REST tokens
    
    We predict:
    - Score tokens (everything else)
    """
    
    def __init__(self, file_path, mask_controls=True):
        self.sequences = []
        self.mask_controls = mask_controls
        
        with open(file_path, 'r') as f:
            for line in f:
                tokens = list(map(int, line.strip().split()))
                self.sequences.append(torch.tensor(tokens, dtype=torch.long))
        
        self.sequence_length = len(self.sequences[0]) if self.sequences else 0
        print(f"Loaded {len(self.sequences)} sequences with length {self.sequence_length}")
        
        # Analyze first sequence to show masking strategy
        if self.sequences and mask_controls:
            self._analyze_masking(self.sequences[0])
    
    def _analyze_masking(self, tokens):
        """Analyze and print masking statistics for a sample sequence"""
        labels = self._create_labels(tokens)
        
        total = len(tokens)
        masked = (labels == -100).sum().item()
        predicted = total - masked
        
        print(f"\nMasking analysis:")
        print(f"  Total tokens: {total}")
        print(f"  Masked tokens: {masked} ({masked/total*100:.1f}%)")
        print(f"  Predicted tokens: {predicted} ({predicted/total*100:.1f}%)")
        print(f"  Predicted triplets: {predicted//3}")
        
        # Show first few predictions
        pred_positions = (labels != -100).nonzero(as_tuple=True)[0][:15]
        print(f"\n  First predicted token positions: {pred_positions.tolist()}")
        
        # Verify they are scores (not controls)
        sample_tokens = tokens[pred_positions[:9]]
        print(f"  Sample predicted tokens: {sample_tokens.tolist()}")
        is_ctrl = (sample_tokens >= CONTROL_OFFSET)
        print(f"  Are these controls? {is_ctrl.tolist()} (should all be False)")
    
    def _create_labels(self, input_ids):
        """
        Create labels with masking:
        - Set -100 for tokens that should NOT be predicted (controls, rests, seps)
        - Keep original token ID for tokens that SHOULD be predicted (scores)
        
        Process triplets (groups of 3 tokens):
        - If any token in triplet is control/rest/sep, mask the entire triplet
        - Otherwise, predict all 3 tokens in the triplet
        """
        labels = input_ids.clone()
        
        # Process triplets (after header)
        # First 4 tokens are header, rest are triplets
        for i in range(4):
            labels[i] = -100  # Mask header
        
        # Process body as triplets
        for i in range(4, len(labels), 3):
            if i + 3 > len(labels):
                break  # Incomplete triplet at end
            
            triplet = input_ids[i:i+3]
            time_tok, dur_tok, note_tok = triplet[0].item(), triplet[1].item(), triplet[2].item()
            
            # Check if this is a control, rest, or separator triplet
            is_control = (time_tok >= CONTROL_OFFSET or 
                         dur_tok >= CONTROL_OFFSET or 
                         note_tok >= CONTROL_OFFSET)
            is_rest = (note_tok == REST)
            is_separator = (time_tok == SEPARATOR or dur_tok == SEPARATOR or note_tok == SEPARATOR)
            
            if is_control or is_rest or is_separator:
                # Mask entire triplet
                labels[i] = -100
                labels[i+1] = -100
                labels[i+2] = -100
            # Otherwise, keep all 3 tokens for prediction (it's a score triplet)
        
        return labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_ids = self.sequences[idx]
        
        if self.mask_controls:
            labels = self._create_labels(input_ids)
        else:
            labels = input_ids.clone()
        
        return {"input_ids": input_ids, "labels": labels}


def evaluate_model(model, dataloader, accelerator, max_batches=50):
    """Calculate validation loss and note accuracy on a subset of data"""
    model.eval()
    total_loss = 0
    total_samples = 0
    total_note_correct = 0
    total_note_predictions = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False, total=min(max_batches, len(dataloader)))):
            if batch_idx >= max_batches:
                break
                
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            
            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Calculate note accuracy
            # Note tokens are at positions where i % 3 == 2 (third token in each triplet)
            # We check ALL triplet positions and see if label is not masked
            labels = batch["labels"]
            predictions = torch.argmax(logits, dim=-1)
            
            # Find positions where we predict note tokens
            # Triplets start at position 4 (after header), so notes are at 6, 9, 12, ...
            for seq_idx in range(batch_size):
                for triplet_start in range(4, labels.size(1), 3):  # Start of each triplet
                    if triplet_start + 2 >= labels.size(1):
                        break
                    note_pos = triplet_start + 2  # Third position in triplet is the note
                    label = labels[seq_idx, note_pos]
                    if label != -100:  # Not masked (it's a score note)
                        # Check if label is a note token (in valid range)
                        if NOTE_OFFSET <= label < NOTE_OFFSET + MAX_NOTE:
                            pred = predictions[seq_idx, note_pos]
                            if pred == label:
                                total_note_correct += 1
                            total_note_predictions += 1
    
    model.train()
    avg_loss = total_loss / total_samples
    note_accuracy = total_note_correct / total_note_predictions if total_note_predictions > 0 else 0.0
    return avg_loss, note_accuracy


def plot_losses(train_losses, val_losses, validation_steps, output_dir):
    """Plot training and validation losses"""
    steps = list(range(1, len(train_losses) + 1))
    
    # Linear plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label='Training Loss', alpha=0.7, color='blue')
    plt.plot(validation_steps, val_losses, label='Validation Loss', 
             linestyle='--', marker='o', markersize=5, color='red')
    plt.xlabel('Steps (x10)')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss (Masked Loss on Scores Only)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = output_dir / "loss_plot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Linear loss plot saved to {plot_path}")
    
    # Log-log plot
    plt.figure(figsize=(10, 6))
    plt.loglog(steps, train_losses, label='Training Loss', alpha=0.7, color='blue')
    plt.loglog(validation_steps, val_losses, label='Validation Loss', 
               linestyle='--', marker='o', markersize=5, color='red')
    plt.xlabel('Steps (x10) [log scale]')
    plt.ylabel('Loss [log scale]')
    plt.title('Training and Validation Loss (Log-Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    loglog_path = output_dir / "loss_plot_loglog.png"
    plt.savefig(loglog_path)
    plt.close()
    print(f"Log-log loss plot saved to {loglog_path}")


def main():
    parser = argparse.ArgumentParser(description="Train with masked loss on score tokens only")
    parser.add_argument('--data_file', type=Path, default=Path('./data/train_output.txt'))
    parser.add_argument('--val_file', type=Path, default=Path('./data/test_output.txt'))
    parser.add_argument('--model_name', type=str, default='stanford-crfm/music-medium-800k')
    parser.add_argument('--output_dir', type=Path, default=Path('./masked_loss_training'))
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--max_steps', type=int, default=3500)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--no_masking', action='store_true', help='Disable masking (for comparison)')
    args = parser.parse_args()
    
    global device
    if args.force_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as requested")
    
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max steps: {args.max_steps}")
    print(f"Masked loss: {not args.no_masking}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    try:
        # Initialize accelerator
        mixed_precision = 'bf16' if torch.cuda.is_available() and not args.force_cpu else 'no'
        print(f"Mixed precision mode: {mixed_precision}")
        
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            cpu=args.force_cpu,
            mixed_precision=mixed_precision,
        )
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load datasets
        print(f"Loading training dataset from {args.data_file}...")
        train_dataset = MaskedTokenDataset(args.data_file, mask_controls=not args.no_masking)
        
        print(f"\nLoading validation dataset from {args.val_file}...")
        val_dataset = MaskedTokenDataset(args.val_file, mask_controls=not args.no_masking)
        
        def collate_fn(batch):
            input_ids = torch.stack([item["input_ids"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            return {"input_ids": input_ids, "labels": labels}
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Load model
        print(f"\nLoading model: {args.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True
        )
        print(f"✓ Model loaded")
        
        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        num_training_steps = args.max_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Prepare for distributed training
        model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, scheduler
        )
        
        print(f"✓ Model prepared on device: {accelerator.device}")
        print_gpu_memory_stats()
        
        # Training loop
        print(f"\n{'='*60}")
        print(f"STARTING TRAINING")
        print(f"{'='*60}\n")
        
        model.train()
        global_step = 0
        train_losses = []
        val_losses = []
        validation_steps = []
        
        progress_bar = tqdm(total=args.max_steps, desc="Training")
        train_iterator = iter(train_dataloader)
        
        while global_step < args.max_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)
            
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            if global_step % 10 == 0:
                train_losses.append(loss.item())
            
            # Evaluation
            if global_step % args.eval_steps == 0 and global_step > 0:
                val_loss, note_acc = evaluate_model(model, val_dataloader, accelerator)
                val_losses.append(val_loss)
                validation_steps.append(len(train_losses))
                
                print(f"\nStep {global_step}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}, note_acc={note_acc:.4f}")
                
                # Check for NaNs
                if check_model_for_nans(model):
                    print("ERROR: NaN detected in model! Stopping training.")
                    break
            
            # Checkpointing
            if global_step % args.save_steps == 0 and global_step > 0:
                checkpoint_dir = args.output_dir / f"checkpoint-{global_step}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(checkpoint_dir)
                
                # Save losses
                np.savez(
                    checkpoint_dir / "losses.npz",
                    train_losses=np.array(train_losses),
                    val_losses=np.array(val_losses),
                    validation_steps=np.array(validation_steps)
                )
                
                # Plot losses
                plot_losses(train_losses, val_losses, validation_steps, checkpoint_dir)
                
                print(f"✓ Checkpoint saved to {checkpoint_dir}")
            
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        progress_bar.close()
        
        # Save final model
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}\n")
        
        final_dir = args.output_dir / "final"
        os.makedirs(final_dir, exist_ok=True)
        
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(final_dir)
        
        # Final evaluation
        final_val_loss, final_note_acc = evaluate_model(model, val_dataloader, accelerator)
        print(f"Final validation loss: {final_val_loss:.4f}")
        print(f"Final note accuracy: {final_note_acc:.4f}")
        
        # Save all losses
        np.savez(
            final_dir / "losses.npz",
            train_losses=np.array(train_losses),
            val_losses=np.array(val_losses),
            validation_steps=np.array(validation_steps)
        )
        
        plot_losses(train_losses, val_losses, validation_steps, final_dir)
        
        print(f"✓ Final model saved to {final_dir}")
        print(f"✓ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
