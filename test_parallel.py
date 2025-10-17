"""
Parallelized test script for evaluating the model on the test dataset using generate4.
Uses producer-consumer pattern with multiple model instances for faster processing.
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm
import traceback
import multiprocessing as mp
from queue import Empty
import time

# Enable more detailed CUDA error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Path to your checkpoint model
CHECKPOINT_PATH = r'hf-ckpt-3500\checkpoint-3500'
TEST_DATA_PATH = r'data\test_output.txt'
OUTPUT_DIR = r'test_outputs'

# Parallelization settings
NUM_WORKERS = 2  # Number of parallel model instances (reduced to avoid memory issues)
QUEUE_SIZE = 20  # Size of work queue


# Import these at module level so they can be pickled
def extract_controls_from_sequence(sequence_tokens, prefix_controls=33):
    """
    Extract control tokens from a tokenized sequence.
    
    The tokenization format is:
    [ANTICIPATE, SEP, SEP, SEP, 
     perf_ctrl_0, rest_0, ..., perf_ctrl_32, rest_32,  # PREFIX (33 performance controls)
     score_0, perf_ctrl_33, score_1, perf_ctrl_34, ...]  # ALTERNATING (score, future performance)
    
    Where:
    - perf_ctrl = performance tokens already WITH CONTROL_OFFSET
    - score = score tokens without offset
    - rest = [TIME, DUR=0, REST]
    
    Args:
        sequence_tokens: List of tokens from the test dataset
        prefix_controls: Number of control tokens in the prefix (default 33)
    
    Returns:
        controls: List of performance tokens (already with CONTROL_OFFSET)
    """
    from anticipation.vocab import CONTROL_OFFSET
    
    # First 4 tokens should be: ANTICIPATE/AUTOREGRESS, SEP, SEP, SEP
    if len(sequence_tokens) < 4:
        return []
    
    # Skip the control flag and 3 SEPARATORs
    tokens = sequence_tokens[4:]
    
    controls = []
    i = 0
    
    # Extract the prefix controls (first k performance triplets, each followed by rest triplet)
    # Performance tokens already have CONTROL_OFFSET added during alignment
    for _ in range(prefix_controls):
        if i + 6 <= len(tokens):
            # Extract performance control triplet (already has CONTROL_OFFSET)
            control_triplet = tokens[i:i+3]
            rest_triplet = tokens[i+3:i+6]
            
            # Validate this is actually a control token (should be >= CONTROL_OFFSET)
            if control_triplet[0] >= CONTROL_OFFSET:
                controls.extend(control_triplet)
            else:
                # Prefix ended early
                break
            
            i += 6
        else:
            break
    
    # After the prefix, extract future performance controls from alternating pattern
    # Pattern is: [future_perf_ctrl, score, future_perf_ctrl, score, ...]
    # Future performance tokens (odd positions) already have CONTROL_OFFSET - extract as-is!
    while i + 3 <= len(tokens):
        # tokens[i:i+3] is future performance triplet (extract if has CONTROL_OFFSET)
        triplet = tokens[i:i+3]
        
        # Check if this has CONTROL_OFFSET (future performance)
        if triplet[0] >= CONTROL_OFFSET:
            controls.extend(triplet)
        
        i += 3
    
    return controls


def process_test_sequence(model, sequence_idx, sequence_tokens, top_p=0.95, save_midi=True):
    """
    Process a single test sequence: extract controls, generate, and optionally save MIDI.
    
    Args:
        model: The loaded model instance
        sequence_idx: Index of the sequence in the test set
        sequence_tokens: List of tokens for this sequence
        top_p: Nucleus sampling parameter
        save_midi: Whether to save the generated MIDI file
    
    Returns:
        dict with results: {'success': bool, 'error': str or None}
    """
    from anticipation.sample import generate4
    from anticipation.convert import events_to_midi
    from anticipation.vocab import VOCAB_SIZE
    
    try:
        # Extract control tokens from the sequence
        controls = extract_controls_from_sequence(sequence_tokens)
        
        if len(controls) == 0:
            return {'success': False, 'error': 'No control tokens found', 'sequence_idx': sequence_idx}
        
        if len(controls) % 3 != 0:
            return {'success': False, 'error': f'Invalid control tokens length: {len(controls)} (not divisible by 3)', 'sequence_idx': sequence_idx}
        
        # Validate all control tokens are in valid range
        for i, tok in enumerate(controls):
            if tok < 0 or tok >= VOCAB_SIZE:
                return {'success': False, 'error': f'Invalid control token {tok} at index {i} (vocab_size={VOCAB_SIZE})', 'sequence_idx': sequence_idx}
        
        # Generate using the new scheme
        events, tokens = generate4(model, controls=controls, top_p=top_p, prefix_controls=33)
        
        # Save MIDI if requested
        if save_midi:
            midi_path = os.path.join(OUTPUT_DIR, f'test_seq_{sequence_idx:04d}.mid')
            mid = events_to_midi(events)
            mid.save(midi_path)
        
        return {
            'success': True,
            'error': None,
            'sequence_idx': sequence_idx,
            'num_control_events': len(controls) // 3,
            'num_generated_events': len(events) // 3
        }
        
    except Exception as e:
        error_msg = f"Error processing sequence {sequence_idx}: {str(e)}"
        traceback.print_exc()
        return {'success': False, 'error': error_msg, 'sequence_idx': sequence_idx}


def worker_process(worker_id, work_queue, result_queue, checkpoint_path, log_queue):
    """
    Worker process that loads a model and processes sequences from the work queue.
    
    Args:
        worker_id: ID of this worker (0 to NUM_WORKERS-1)
        work_queue: Queue to receive work items (sequence_idx, sequence_tokens)
        result_queue: Queue to send results
        checkpoint_path: Path to the model checkpoint
        log_queue: Queue for sending log messages to main process
    """
    # Import heavy dependencies only in worker process to avoid memory issues during spawn
    import torch
    from transformers import AutoModelForCausalLM
    from anticipation.sample import generate4
    from anticipation.convert import events_to_midi
    from anticipation.vocab import VOCAB_SIZE, CONTROL_OFFSET
    
    try:
        # Set CUDA device for this worker (if multiple GPUs available)
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 1:
                device_id = worker_id % device_count
                torch.cuda.set_device(device_id)
                log_queue.put(f"Worker {worker_id}: Using GPU {device_id}")
            else:
                log_queue.put(f"Worker {worker_id}: Using GPU 0 (only 1 GPU available)")
        
        # Load model for this worker
        log_queue.put(f"Worker {worker_id}: Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map={"": f"cuda:{worker_id % torch.cuda.device_count()}"} if torch.cuda.is_available() else "cpu"
        )
        model.eval()
        log_queue.put(f"Worker {worker_id}: Model loaded successfully")
        
        # Process sequences from the queue
        while True:
            try:
                # Get work item with timeout
                work_item = work_queue.get(timeout=5)
                
                # Check for sentinel value (None means shutdown)
                if work_item is None:
                    log_queue.put(f"Worker {worker_id}: Received shutdown signal")
                    break
                
                sequence_idx, sequence_tokens = work_item
                
                # Process the sequence (suppress output to avoid interfering with progress bar)
                result = process_test_sequence(model, sequence_idx, sequence_tokens, top_p=0.95, save_midi=True)
                
                # Send result back
                result_queue.put(result)
                
            except Empty:
                # No work available, continue waiting
                continue
            except Exception as e:
                log_queue.put(f"Worker {worker_id}: Error in main loop: {e}")
                # Send error result
                result_queue.put({
                    'success': False,
                    'error': f'Worker {worker_id} error: {str(e)}',
                    'sequence_idx': -1
                })
                
    except Exception as e:
        log_queue.put(f"Worker {worker_id}: Fatal error during initialization: {e}")
    finally:
        log_queue.put(f"Worker {worker_id}: Shutting down")


def load_test_sequences(test_file_path):
    """Load all test sequences from the test output file."""
    sequences = []
    with open(test_file_path, 'r') as f:
        for line in f:
            tokens = list(map(int, line.strip().split()))
            sequences.append(tokens)
    return sequences


def main():
    print(f"\n{'='*60}")
    print(f"PARALLEL Testing with {NUM_WORKERS} workers")
    print(f"Testing model on dataset: {TEST_DATA_PATH}")
    print(f"Model checkpoint: {CHECKPOINT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load test sequences
    print("Loading test sequences...")
    test_sequences = load_test_sequences(TEST_DATA_PATH)
    print(f"Loaded {len(test_sequences)} test sequences\n")
    
    # Create queues
    work_queue = mp.Queue(maxsize=QUEUE_SIZE)
    result_queue = mp.Queue()
    log_queue = mp.Queue()
    
    # Start worker processes
    workers = []
    print(f"Starting {NUM_WORKERS} worker processes...")
    for i in range(NUM_WORKERS):
        worker = mp.Process(
            target=worker_process,
            args=(i, work_queue, result_queue, CHECKPOINT_PATH, log_queue)
        )
        worker.start()
        workers.append(worker)
    
    # Give workers time to initialize and print their log messages
    print("Initializing workers...\n")
    time.sleep(2)
    
    # Drain the log queue to show initialization messages
    while not log_queue.empty():
        try:
            msg = log_queue.get_nowait()
            print(msg)
        except Empty:
            break
    
    print(f"\nAll workers initialized\n")
    
    # Producer: Add all sequences to the work queue
    print("Adding sequences to work queue...")
    for idx, sequence in enumerate(test_sequences):
        work_queue.put((idx, sequence))
    
    # Add sentinel values to signal workers to shutdown
    for _ in range(NUM_WORKERS):
        work_queue.put(None)
    
    print(f"All {len(test_sequences)} sequences queued\n")
    
    # Consumer: Collect results with a single progress bar
    results = [None] * len(test_sequences)
    successful = 0
    failed = 0
    
    print("Processing sequences...")
    # Use tqdm with position=0 and leave=True to ensure single, clean progress bar
    with tqdm(total=len(test_sequences), desc="Progress", position=0, leave=True, ncols=100) as pbar:
        results_collected = 0
        while results_collected < len(test_sequences):
            try:
                # Get result with timeout
                result = result_queue.get(timeout=1)
                
                # Store result in correct position
                seq_idx = result['sequence_idx']
                if seq_idx >= 0 and seq_idx < len(test_sequences):
                    results[seq_idx] = result
                
                # Update statistics
                if result['success']:
                    successful += 1
                else:
                    failed += 1
                
                results_collected += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix_str(f"Success: {successful}/{results_collected} ({100*successful/results_collected:.1f}%)")
                
            except Empty:
                # Check for log messages without blocking
                while not log_queue.empty():
                    try:
                        msg = log_queue.get_nowait()
                        tqdm.write(msg)  # Use tqdm.write to avoid interfering with progress bar
                    except Empty:
                        break
                continue
    
    # Final log messages
    print("\n\nDraining final log messages...")
    while not log_queue.empty():
        try:
            msg = log_queue.get_nowait()
            print(msg)
        except Empty:
            break
    
    # Wait for all workers to finish
    print("\nWaiting for workers to finish...")
    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            print(f"Warning: Worker still alive, terminating...")
            worker.terminate()
    
    # Final statistics
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total sequences: {len(test_sequences)}")
    print(f"Successful: {successful} ({100*successful/len(test_sequences):.1f}%)")
    print(f"Failed: {failed} ({100*failed/len(test_sequences):.1f}%)")
    
    # Save results to file
    results_file = os.path.join(OUTPUT_DIR, 'results_summary.txt')
    with open(results_file, 'w') as f:
        f.write(f"Test Results Summary (Parallel)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Model: {CHECKPOINT_PATH}\n")
        f.write(f"Test Data: {TEST_DATA_PATH}\n")
        f.write(f"Workers: {NUM_WORKERS}\n")
        f.write(f"Total sequences: {len(test_sequences)}\n")
        f.write(f"Successful: {successful} ({100*successful/len(test_sequences):.1f}%)\n")
        f.write(f"Failed: {failed} ({100*failed/len(test_sequences):.1f}%)\n\n")
        
        f.write(f"\nDetailed Results:\n")
        f.write(f"{'-'*60}\n")
        for idx, result in enumerate(results):
            if result is None:
                f.write(f"Seq {idx:04d}: MISSING RESULT\n")
            elif result['success']:
                f.write(f"Seq {idx:04d}: SUCCESS - "
                       f"Controls: {result['num_control_events']}, "
                       f"Generated: {result['num_generated_events']}\n")
            else:
                f.write(f"Seq {idx:04d}: FAILED - {result['error']}\n")
    
    print(f"\nResults saved to: {results_file}")
    print(f"MIDI files saved to: {OUTPUT_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.set_start_method('spawn', force=True)
    main()
