import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import gc
import traceback
import matplotlib.pyplot as plt
from anticipation.vocab import ANTICIPATE, AUTOREGRESS  # Import the flag token constants

# Set CUDA memory allocator configuration for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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

class SequencePackedDataset(Dataset):
    def __init__(self, file_path, context_length=1024, max_packed_sequences=4):
        """Load data from tokenized file and implement sequence packing
        
        Args:
            file_path: Path to the tokenized data file
            context_length: Maximum context length (default 1024)
            max_packed_sequences: Maximum number of sequences to pack together (default 4)
        
        Note: The new tokenization format already includes:
        - 3 SEPARATOR tokens at the start of each sequence
        - Control flag (ANTICIPATE) as the 4th token
        - The rest of the sequence content
        """
        from anticipation.vocab import SEPARATOR, AUTOREGRESS, ANTICIPATE
        
        # Read all individual sequences
        individual_sequences = []
        with open(file_path, 'r') as f:
            for line in f:
                tokens = list(map(int, line.strip().split()))
                individual_sequences.append(tokens)
        
        print(f"Loaded {len(individual_sequences)} individual sequences")
        
        # Validate new format: each sequence should start with 3 SEPARATORs + control flag
        if individual_sequences:
            sample = individual_sequences[0]
            if len(sample) >= 4:
                if sample[0] == SEPARATOR and sample[1] == SEPARATOR and sample[2] == SEPARATOR:
                    if sample[3] in [AUTOREGRESS, ANTICIPATE]:
                        print(f"✓ New tokenization format detected (3 SEPARATORs + control flag)")
                    else:
                        print(f"⚠ Warning: Expected control flag at position 3, got {sample[3]}")
                else:
                    print(f"⚠ Warning: Expected 3 SEPARATORs at start, got {sample[:3]}")
        
        # Create packed sequences
        self.packed_sequences = []
        self.attention_masks = []
        
        # Keep track of statistics
        self.total_packed = 0
        self.avg_sequences_per_pack = 0
        sequences_per_pack = []
        
        # Process sequences in random order for better mixing
        import random
        random.shuffle(individual_sequences)
        
        # Pack sequences
        current_packed = []
        current_positions = []  # Track positions for creating attention masks
        
        for sequence in individual_sequences:
            # New format: sequences already have 3 SEPARATORs at the start
            # Just use them as-is
            
            # If adding this sequence would exceed context length, start a new packed sequence
            if len(current_packed) > 0 and (len(current_packed) + len(sequence) > context_length or 
                                           len(current_positions) >= max_packed_sequences):
                # Finalize current packed sequence
                if len(current_packed) > 0:
                    # Create attention mask (1 for tokens to attend to, 0 for tokens to ignore)
                    attention_mask = torch.zeros(context_length, dtype=torch.long)
                    for start, end in current_positions:
                        attention_mask[start:end] = 1
                    
                    # Pad to context length if needed
                    if len(current_packed) < context_length:
                        padding_length = context_length - len(current_packed)
                        current_packed.extend([SEPARATOR] * padding_length)
                    
                    # Convert to tensor and store
                    self.packed_sequences.append(torch.tensor(current_packed[:context_length], dtype=torch.long))
                    self.attention_masks.append(attention_mask)
                    sequences_per_pack.append(len(current_positions))
                    self.total_packed += 1
                
                # Start a new packed sequence
                current_packed = []
                current_positions = []
            
            # Record start position
            start_pos = len(current_packed)
            
            # Add the entire sequence (which already has separators and control flag)
            current_packed.extend(sequence)
            end_pos = len(current_packed)
            
            # Record the position of this sequence for attention masking
            current_positions.append((start_pos, end_pos))
        
        # Add the final packed sequence if not empty
        if len(current_packed) > 0:
            attention_mask = torch.zeros(context_length, dtype=torch.long)
            for start, end in current_positions:
                attention_mask[start:end] = 1
                
            # Pad to context length if needed
            if len(current_packed) < context_length:
                padding_length = context_length - len(current_packed)
                current_packed.extend([SEPARATOR] * padding_length)
            
            # Convert to tensor and store
            self.packed_sequences.append(torch.tensor(current_packed[:context_length], dtype=torch.long))
            self.attention_masks.append(attention_mask)
            sequences_per_pack.append(len(current_positions))
            self.total_packed += 1
        
        # Calculate statistics
        if sequences_per_pack:
            self.avg_sequences_per_pack = sum(sequences_per_pack) / len(sequences_per_pack)
        
        print(f"Created {len(self.packed_sequences)} packed sequences")
        print(f"Average sequences per pack: {self.avg_sequences_per_pack:.2f}")
        
    def __len__(self):
        return len(self.packed_sequences)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.packed_sequences[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.packed_sequences[idx],
        }

def collate_packed_sequences(batch):
    """Collate function for packed sequences that includes attention masks"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }

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
    Plot training and validation losses and save the figure
    
    Args:
        train_losses (list): Training loss history
        val_losses (list): Validation loss history
        validation_steps (list): Steps at which validation was performed
        output_dir (Path): Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot all training losses
    steps = list(range(1, len(train_losses) + 1))
    plt.plot(steps, train_losses, label='Training Loss', alpha=0.7, color='blue')
    
    # Plot validation losses at specific steps
    plt.plot(validation_steps, val_losses, label='Validation Loss', 
             linestyle='--', marker='o', markersize=5, color='red')
    
    plt.xlabel('Steps (x10)')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plot_path = output_dir / "loss_plot.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Loss plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=Path, default=Path('./data/train_output.txt'))
    parser.add_argument('--val_file', type=Path, default=Path('./data/test_output.txt'))
    parser.add_argument('--model_name', type=str, default='stanford-crfm/music-small-800k')
    parser.add_argument('--output_dir', type=Path, default=Path('./fine_tuned_full'))
    parser.add_argument('--batch_size', type=int, default=8) 
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)  # For effective batch size 256
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--max_steps', type=int, default=3500)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--reduce_memory', action='store_true', help='Use memory-saving techniques')
    parser.add_argument('--context_length', type=int, default=1024, help='Maximum context length')
    parser.add_argument('--max_packed_sequences', type=int, default=4, 
                       help='Maximum number of sequences to pack together (set to 1 to disable packing)')
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
        if args.max_packed_sequences > 1:
            print(f"Using sequence packing with max {args.max_packed_sequences} sequences per pack")
            train_dataset = SequencePackedDataset(
                args.data_file, 
                context_length=args.context_length,
                max_packed_sequences=args.max_packed_sequences
            )
            collate_fn_train = collate_packed_sequences
        else:
            print("Sequence packing disabled - using single sequences")
            # Original dataset class for backward compatibility
            from anticipation.vocab import SEPARATOR
            individual_sequences = []
            with open(args.data_file, 'r') as f:
                for line in f:
                    tokens = list(map(int, line.strip().split()))
                    individual_sequences.append(torch.tensor(tokens, dtype=torch.long))
            
            class TokenizedDataset(Dataset):
                def __init__(self, sequences):
                    self.sequences = sequences
                    self.sequence_length = len(self.sequences[0]) if self.sequences else 0
                    print(f"Loaded {len(self.sequences)} sequences with length {self.sequence_length}")
                
                def __len__(self):
                    return len(self.sequences)
                
                def __getitem__(self, idx):
                    tokens = self.sequences[idx]
                    return {"input_ids": tokens, "labels": tokens}
            
            train_dataset = TokenizedDataset(individual_sequences)
            
            def collate_fn_train(batch):
                input_ids = torch.stack([item["input_ids"] for item in batch])
                labels = torch.stack([item["labels"] for item in batch])
                return {"input_ids": input_ids, "labels": labels}
            
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            collate_fn=collate_fn_train,
            pin_memory=torch.cuda.is_available() and not args.force_cpu,
            num_workers=0,  # Avoid multiprocessing issues
        )
        
        # Load validation dataset
        print(f"Loading validation dataset from {args.val_file}...")
        if args.max_packed_sequences > 1:
            val_dataset = SequencePackedDataset(
                args.val_file, 
                context_length=args.context_length,
                max_packed_sequences=args.max_packed_sequences
            )
            collate_fn_val = collate_packed_sequences
        else:
            # Load validation sequences
            val_sequences = []
            with open(args.val_file, 'r') as f:
                for line in f:
                    tokens = list(map(int, line.strip().split()))
                    val_sequences.append(torch.tensor(tokens, dtype=torch.long))
            
            val_dataset = TokenizedDataset(val_sequences)
            collate_fn_val = collate_fn_train
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.val_batch_size,
            shuffle=False,  # No need to shuffle validation data
            collate_fn=collate_fn_val,
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
        
        # Learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_steps,
        )
        
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