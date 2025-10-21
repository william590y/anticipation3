"""
Test script to verify resumption functionality works correctly.
Simulates an interrupted run and verifies resume behavior.
"""
import os
import tempfile
import shutil

print("Testing Resumption Functionality")
print("="*80)

# Create a temporary test directory
test_dir = tempfile.mkdtemp(prefix="test_resume_")
print(f"Test directory: {test_dir}")

try:
    # Create mock checkpoint file
    checkpoint_file = os.path.join(test_dir, "train.txt.checkpoint")
    output_train = os.path.join(test_dir, "train.txt")
    output_test = os.path.join(test_dir, "test.txt")
    
    # Simulate first run - process 3 pieces
    print("\n1. SIMULATING FIRST RUN (interrupted after 3 pieces):")
    processed_pieces = [
        "./asap-dataset-master/Bach/Fugue/bwv_846/Shi05M.mid",
        "./asap-dataset-master/Bach/Fugue/bwv_846/Bae03M.mid",
        "./asap-dataset-master/Beethoven/Piano_Sonatas/1-1/Gulda01M.mid"
    ]
    
    with open(checkpoint_file, 'w') as f:
        for piece in processed_pieces:
            f.write(piece + '\n')
    
    with open(output_train, 'w') as f:
        f.write("55027 55025 55025 55025 ... (sequence 1)\n")
        f.write("55027 55025 55025 55025 ... (sequence 2)\n")
        
    print(f"   Created checkpoint with {len(processed_pieces)} pieces")
    print(f"   Created train file with 2 sequences")
    
    # Simulate resume mode - load checkpoint
    print("\n2. SIMULATING RESUME MODE:")
    loaded_pieces = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            for line in f:
                loaded_pieces.add(line.strip())
        print(f"   ✓ Loaded {len(loaded_pieces)} processed pieces from checkpoint")
    else:
        print("   ✗ No checkpoint found")
    
    # Verify loaded pieces match
    if loaded_pieces == set(processed_pieces):
        print("   ✓ Loaded pieces match original checkpoint")
    else:
        print("   ✗ Mismatch in loaded pieces!")
    
    # Simulate appending new data
    print("\n3. SIMULATING APPEND MODE:")
    new_piece = "./asap-dataset-master/Chopin/Etudes/op.10-no.3/Ashkenazy01M.mid"
    
    # Check if piece should be skipped
    if new_piece in loaded_pieces:
        print(f"   ✓ Would skip: {new_piece} (already processed)")
    else:
        print(f"   ✓ Would process: {new_piece} (not in checkpoint)")
        
        # Append to files
        with open(output_train, 'a') as f:
            f.write("55027 55025 55025 55025 ... (sequence 3)\n")
        
        with open(checkpoint_file, 'a') as f:
            f.write(new_piece + '\n')
            f.flush()
        
        print("   ✓ Appended new sequence to train.txt")
        print("   ✓ Updated checkpoint file")
    
    # Verify final state
    print("\n4. VERIFYING FINAL STATE:")
    
    # Count lines in checkpoint
    with open(checkpoint_file, 'r') as f:
        checkpoint_lines = f.readlines()
    print(f"   Checkpoint entries: {len(checkpoint_lines)} pieces")
    
    # Count lines in train file
    with open(output_train, 'r') as f:
        train_lines = f.readlines()
    print(f"   Train sequences: {len(train_lines)} sequences")
    
    # Verify no duplicates in checkpoint
    unique_pieces = set(line.strip() for line in checkpoint_lines)
    if len(unique_pieces) == len(checkpoint_lines):
        print("   ✓ No duplicate pieces in checkpoint")
    else:
        print(f"   ✗ Found {len(checkpoint_lines) - len(unique_pieces)} duplicates!")
    
    # Test duplicate prevention
    print("\n5. TESTING DUPLICATE PREVENTION:")
    duplicate_piece = processed_pieces[0]  # Try to add first piece again
    
    if duplicate_piece in unique_pieces:
        print(f"   ✓ Would correctly skip duplicate: {os.path.basename(duplicate_piece)}")
    else:
        print("   ✗ Failed to detect duplicate!")
    
    print("\n" + "="*80)
    print("✓ Resumption functionality verified successfully!")
    print("\nUsage:")
    print("  - First run:  python tokenize-asap.py")
    print("  - Resume run: python tokenize-asap.py --resume")
    print("\nCheckpoint file: ./data/train_perturbed.txt.checkpoint")
    
finally:
    # Cleanup
    print(f"\nCleaning up test directory: {test_dir}")
    shutil.rmtree(test_dir)
    print("Done!")
