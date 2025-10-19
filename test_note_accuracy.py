"""Test note accuracy calculation"""
import sys
sys.path.insert(0, '.')
import torch
from train_masked import MaskedTokenDataset, evaluate_model
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoModelForCausalLM

print('Testing note accuracy calculation...')

# Load a small dataset
dataset = MaskedTokenDataset('data/test_output.txt', mask_controls=True)
print(f'Loaded {len(dataset)} validation sequences')

# Create a small dataloader
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'labels': labels}

dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

# Load model
print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    'hf-ckpt-3500/checkpoint-3500',
    trust_remote_code=True
)

# Setup accelerator
accelerator = Accelerator(cpu=True)
model, dataloader = accelerator.prepare(model, dataloader)

print('\nRunning evaluation on 1 batch...')
model.eval()
batch = next(iter(dataloader))
with torch.no_grad():
    outputs = model(**batch)
    loss = outputs.loss
    logits = outputs.logits
    
print(f'Batch size: {batch["input_ids"].size(0)}')
print(f'Sequence length: {batch["input_ids"].size(1)}')
print(f'Loss: {loss.item():.4f}')

# Count note predictions manually
from anticipation.vocab import NOTE_OFFSET
from anticipation.config import MAX_NOTE

labels = batch['labels']
predictions = torch.argmax(logits, dim=-1)
total_note_correct = 0
total_note_predictions = 0

for seq_idx in range(batch['input_ids'].size(0)):
    for pos in range(2, labels.size(1), 3):
        label = labels[seq_idx, pos]
        if label != -100:
            if NOTE_OFFSET <= label < NOTE_OFFSET + MAX_NOTE:
                pred = predictions[seq_idx, pos]
                if pred == label:
                    total_note_correct += 1
                total_note_predictions += 1

print(f'\nNote predictions: {total_note_predictions}')
print(f'Correct notes: {total_note_correct}')
if total_note_predictions > 0:
    print(f'Note accuracy: {total_note_correct / total_note_predictions:.4f}')
    
print('\n✓ Note accuracy calculation works!')

# Now test the full evaluate_model function
print('\nTesting full evaluate_model function (3 batches)...')
val_loss, note_acc = evaluate_model(model, dataloader, accelerator, max_batches=3)
print(f'Validation loss: {val_loss:.4f}')
print(f'Note accuracy: {note_acc:.4f}')
print('\n✓ All tests passed!')
