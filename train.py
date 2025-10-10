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
import gc
import traceback
import matplotlib.pyplot as plt
from anticipation.vocab import ANTICIPATE, AUTOREGRESS  # Import the flag token constants

# Helper function to monitor GPU memory usage
def print_gpu_memory_stats():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
            print(f"GPU {i} max memory allocated: {torch.cuda.max_memory_allocated(i) / 1024**2:.2f} MB")

# Check for NaN values in model parameters
def check_model_for_nans(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in parameter {name}")
            return True
    return False

# Force CUDA if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_count = torch.cuda.device_count()
    print(f"✓ CUDA is available with {device_count} device(s)")
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"  Device {i}: {device_name}")
        props = torch.cuda.get_device_properties(i)
        print(f"    - Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    - CUDA capability: {props.major}.{props.minor}")
else:
    device = torch.device("cpu")
    print("✗ CUDA is not available! Training will be much slower on CPU.")

# Explicitly print which device we're using
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

class TokenizedDataset(Dataset):
    """Simple dataset that loads pre-tokenized sequences.
    
    Sequences are already packed and formatted by tokenize-asap.py:
    - Each sequence is exactly 1024 tokens
    - Format: [SEP, SEP, SEP, control_flag, ...tokens...]
    """
    def __init__(self, file_path):
        self.sequences = []
        with open(file_path, 'r') as f:
            for line in f:
                tokens = list(map(int, line.strip().split()))
                self.sequences.append(torch.tensor(tokens, dtype=torch.long))
        
        self.sequence_length = len(self.sequences[0]) if self.sequences else 0
        print(f"Loaded {len(self.sequences)} sequences with length {self.sequence_length}")
        
        # Validate format
        if self.sequences:
            from anticipation.vocab import SEPARATOR, AUTOREGRESS, ANTICIPATE
            sample = self.sequences[0].tolist()
            if len(sample) >= 4:
                if sample[0] == SEPARATOR and sample[1] == SEPARATOR and sample[2] == SEPARATOR:
                    if sample[3] in [AUTOREGRESS, ANTICIPATE]:
                        print(f"✓ Tokenization format validated (3 SEPARATORs + control flag)")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        tokens = self.sequences[idx]
        return {"input_ids": tokens, "labels": tokens}

def evaluate_model(model, dataloader, accelerator):
    """Calculate validation loss on a dataset"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            outputs = model(**batch)
            loss = outputs.loss
            
            # Get batch size from the input shape
            batch_size = batch["input_ids"].size(0)
            
            # Accumulate loss (weighted by batch size)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    # Return average loss
    return total_loss / total_samples

def plot_losses(train_losses, val_losses, validation_steps, output_dir):
    """
    Plot training and validation losses and save the figures (both linear and log-log)
    
    Args:
        train_losses (list): Training loss history
        val_losses (list): Validation loss history
        validation_steps (list): Steps at which validation was performed
        output_dir (Path): Directory to save the plots
    """
    steps = list(range(1, len(train_losses) + 1))
    
    # Linear plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label='Training Loss', alpha=0.7, color='blue')
    plt.plot(validation_steps, val_losses, label='Validation Loss', 
             linestyle='--', marker='o', markersize=5, color='red')
    plt.xlabel('Steps (x10)')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
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
    plt.grid(True, alpha=0.1, which='minor')
    loglog_path = output_dir / "loss_plot_loglog.png"
    plt.savefig(loglog_path)
    plt.close()
    print(f"Log-log loss plot saved to {loglog_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=Path, default=Path('./data/train_output.txt'))
    parser.add_argument('--val_file', type=Path, default=Path('./data/test_output.txt'))
    parser.add_argument('--model_name', type=str, default='stanford-crfm/music-medium-800k')
    parser.add_argument('--output_dir', type=Path, default=Path('./fine_tuned_full_new'))
    parser.add_argument('--batch_size', type=int, default=32) 
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)  # For effective batch size 128
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--max_steps', type=int, default=3500)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--reduce_memory', action='store_true', help='Use memory-saving techniques')
    args = parser.parse_args()
    
    # Override device if requested
    global device
    if args.force_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as requested")
    
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Final device confirmation: {device}")
    
    try:
        # Initialize accelerator with memory optimization if requested
        # Use bf16 instead of fp16 for better numerical stability
        mixed_precision = 'bf16' if torch.cuda.is_available() and not args.force_cpu else 'no'
        print(f"Mixed precision mode: {mixed_precision}")
        
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            cpu=args.force_cpu,
            mixed_precision=mixed_precision,
        )
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Monitor initial GPU memory
        print("Initial GPU memory stats:")
        print_gpu_memory_stats()
        
        # Load training dataset
        print(f"Loading training dataset from {args.data_file}...")
        train_dataset = TokenizedDataset(args.data_file)
        
        def collate_fn(batch):
            input_ids = torch.stack([item["input_ids"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            return {"input_ids": input_ids, "labels": labels}
            
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available() and not args.force_cpu,
            num_workers=0,  # Avoid multiprocessing issues
        )
        
        # Load validation dataset
        print(f"Loading validation dataset from {args.val_file}...")
        val_dataset = TokenizedDataset(args.val_file)
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.val_batch_size,
            shuffle=False,  # No need to shuffle validation data
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available() and not args.force_cpu,
            num_workers=0,
        )
        
        # Load model with memory optimizations
        print(f"Loading model {args.model_name}...")
        model_kwargs = {
            "trust_remote_code": True,
            "use_cache": False,  # Important for training
        }
        
        if args.reduce_memory and torch.cuda.is_available():
            print("Using memory reduction techniques...")
            # BF16 is more stable than FP16
            model_kwargs.update({
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                "low_cpu_mem_usage": True,
            })
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                **model_kwargs
            )
        except Exception as e:
            print(f"Error loading model with advanced options: {e}")
            print("Trying with basic options...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                trust_remote_code=True,
                use_cache=False
            )
        
        # Check memory after loading model
        print("GPU memory after loading model:")
        print_gpu_memory_stats()
        
        # Explicitly move model to our device before creating optimizer
        model = model.to(device)
        print(f"Model moved to: {next(model.parameters()).device}")
        
        # Setup optimizer with gradient clipping to prevent exploding gradients
        # Using a lower learning rate and better epsilon value for numerical stability
        optimizer = AdamW(
            model.parameters(), 
            lr=args.learning_rate,
            eps=1e-6,  # More stable epsilon
            weight_decay=0.01,
            betas=(0.9, 0.999),  # Stable default betas
        )
        
        # Prepare for training with accelerate
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        val_dataloader = accelerator.prepare_data_loader(val_dataloader)
        print(f"After accelerator preparation, model device: {next(model.parameters()).device}")
        
        # Learning rate scheduler - cosine decay from 3e-5 to 3e-6
        # num_cycles=0.5 gives one half of a cosine curve (decay from max to min)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_steps,
            num_cycles=0.5,  # Half cosine for smooth decay
        )
        
        # Manually adjust to decay to 3e-6 instead of 0
        # We'll modify the learning rate calculation
        initial_lr = args.learning_rate  # 3e-5
        final_lr = 3e-6
        
        # Override scheduler with custom lambda that decays to final_lr
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(current_step):
            if current_step < args.warmup_steps:
                # Warmup phase
                return float(current_step) / float(max(1, args.warmup_steps))
            # Cosine decay phase
            progress = float(current_step - args.warmup_steps) / float(max(1, args.max_steps - args.warmup_steps))
            # Cosine annealing from 1.0 to (final_lr / initial_lr)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (final_lr / initial_lr) + (1.0 - final_lr / initial_lr) * cosine_decay
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        # Check memory before training
        print("GPU memory before training:")
        print_gpu_memory_stats()
        
        # Disable anomaly detection which can cause overhead
        torch.autograd.set_detect_anomaly(False)
        
        # Set deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = False  # Better performance
        torch.backends.cudnn.benchmark = True  # Better performance
        
        if torch.cuda.is_available():
            print("Clearing CUDA cache before training")
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)
        
        # Training loop
        print("Starting training...")
        model.train()
        completed_steps = 0
        step = 0
        
        # Lists to track losses
        train_losses = []
        val_losses = []
        validation_steps = []
        
        # Use standard tqdm with disable=False to ensure it always displays
        progress_bar = tqdm(total=args.max_steps, desc="Training", disable=False)
        
        try:
            while completed_steps < args.max_steps:
                for batch in train_dataloader:
                    try:
                        with accelerator.accumulate(model):
                            # Forward pass with gradient scaling
                            outputs = model(**batch)
                            loss = outputs.loss
                            
                            # Check for NaN loss
                            if torch.isnan(loss).any() or torch.isinf(loss).any():
                                print(f"WARNING: NaN or Inf loss detected: {loss.item()}")
                                # Skip this backward pass
                                optimizer.zero_grad()
                                continue
                                
                            # Backward pass
                            accelerator.backward(loss)
                            
                            # Only update optimizer and scheduler when gradients are synchronized
                            if accelerator.sync_gradients:
                                # Gradient clipping
                                accelerator.clip_grad_norm_(model.parameters(), max_norm=0.5)
                                
                                # Check for NaN in gradients
                                has_nan_grads = False
                                for name, param in model.named_parameters():
                                    if param.grad is not None and torch.isnan(param.grad).any():
                                        print(f"NaN gradient detected in {name}")
                                        has_nan_grads = True
                                        break
                                        
                                if has_nan_grads:
                                    print("Skipping update due to NaN gradients")
                                    optimizer.zero_grad()
                                    continue
                                
                                # Only update optimizer and scheduler here
                                optimizer.step()
                                scheduler.step()
                                optimizer.zero_grad()
                                
                                # Only update step counters when we actually update weights
                                completed_steps += 1
                                progress_bar.update(1)
                                
                                # Log progress
                                if completed_steps % 10 == 0:
                                    # Store the training loss every 10 steps
                                    train_losses.append(loss.item())
                                    
                                    # Print more precise learning rate
                                    print(f"Step: {completed_steps}/{args.max_steps}, Loss: {loss.item():.4f}, "
                                          f"LR: {scheduler.get_last_lr()[0]:.8e}")
                                    
                                    # Check for NaN parameters periodically
                                    if check_model_for_nans(model):
                                        print("NaN parameters detected in model! Training may be unstable.")
                                    
                                    # Check memory periodically
                                    if completed_steps % 100 == 0:
                                        print_gpu_memory_stats()
                                
                                # Run validation periodically
                                if completed_steps % args.eval_steps == 0:
                                    print(f"\nRunning validation at step {completed_steps}...")
                                    val_loss = evaluate_model(model, val_dataloader, accelerator)
                                    validation_steps.append(completed_steps // 10)  # Store step number (divided by 10 for plotting)
                                    val_losses.append(val_loss)
                                    print(f"Validation Loss: {val_loss:.4f}")
                                    
                                    # Return to training mode
                                    model.train()
                                    
                                    # Free up memory after validation
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                        gc.collect()
                                
                                # Save checkpoint
                                if completed_steps % args.save_steps == 0:
                                    checkpoint_dir = args.output_dir / f"checkpoint-{completed_steps}"
                                    os.makedirs(checkpoint_dir, exist_ok=True)
                                    
                                    # Unwrap model before saving
                                    unwrapped_model = accelerator.unwrap_model(model)
                                    unwrapped_model.save_pretrained(
                                        checkpoint_dir,
                                        is_main_process=accelerator.is_main_process,
                                        save_function=accelerator.save,
                                    )
                                    print(f"Saved checkpoint to {checkpoint_dir}")
                                    
                                    # Save the losses so far
                                    np.savez(
                                        checkpoint_dir / "losses.npz",
                                        train_losses=np.array(train_losses),
                                        val_losses=np.array(val_losses),
                                        validation_steps=np.array(validation_steps)
                                    )
                                    
                                    # Create and save loss plot
                                    plot_losses(train_losses, val_losses, validation_steps, checkpoint_dir)
                                    
                                    # Free up memory
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                        gc.collect()
                            
                            # Zero gradients even if we don't sync (needed for some accelerator configurations)
                            if not accelerator.sync_gradients:
                                optimizer.zero_grad()
                                
                            # Check if we've reached max steps
                            if completed_steps >= args.max_steps:
                                break
                            
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            print(f"CUDA OOM error! Current batch size: {args.batch_size}")
                            print(f"Current memory usage:")
                            print_gpu_memory_stats()
                            print("Consider reducing batch size or model size.")
                            print(f"Error details: {str(e)}")
                            raise
                        elif "nan" in str(e).lower() or "inf" in str(e).lower():
                            print(f"NaN/Inf error: {str(e)}")
                            print("Trying to recover by skipping this batch...")
                            optimizer.zero_grad()
                            continue
                        else:
                            print(f"Runtime error: {str(e)}")
                            print(traceback.format_exc())
                            raise
            
        except Exception as e:
            print(f"Error during training: {e}")
            print(traceback.format_exc())
            raise
        finally:
            # Make sure we always close the progress bar
            progress_bar.close()
            
            # Always try to save whatever we have and generate the final plot
            try:
                # Final validation run
                print("\nRunning final validation...")
                final_val_loss = evaluate_model(model, val_dataloader, accelerator)
                validation_steps.append(completed_steps // 10)
                val_losses.append(final_val_loss)
                print(f"Final validation Loss: {final_val_loss:.4f}")
                
                # Final save
                final_dir = args.output_dir / "final"
                os.makedirs(final_dir, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    final_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
                print(f"Saved final model to {final_dir}")
                
                # Save the final losses
                np.savez(
                    final_dir / "losses.npz",
                    train_losses=np.array(train_losses),
                    val_losses=np.array(val_losses),
                    validation_steps=np.array(validation_steps)
                )
                
                # Create and save final loss plot
                plot_losses(train_losses, val_losses, validation_steps, final_dir)
                
            except Exception as save_error:
                print(f"Error saving final model or generating plot: {save_error}")
            
    except Exception as setup_error:
        print(f"Error in setup: {setup_error}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()