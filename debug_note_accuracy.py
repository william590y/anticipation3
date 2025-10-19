"""Debug note accuracy calculation"""
import sys
sys.path.insert(0, '.')
import torch
from train_masked import MaskedTokenDataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from anticipation.vocab import NOTE_OFFSET
from anticipation.config import MAX_NOTE

print('Debugging note accuracy calculation...')

# Load a small dataset
dataset = MaskedTokenDataset('data/test_output.txt', mask_controls=True)
print(f'Loaded {len(dataset)} validation sequences\n')

# Create a small dataloader
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'labels': labels}

dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Load model
print('Loading model...')
model = AutoModelForCausalLM.from_pretrained(
    'hf-ckpt-3500/checkpoint-3500',
    trust_remote_code=True
)
model.eval()
model = model.cuda()

print('Running evaluation on 1 batch...\n')
batch = next(iter(dataloader))
batch = {k: v.cuda() for k, v in batch.items()}

with torch.no_grad():
    outputs = model(**batch)
    logits = outputs.logits

labels = batch['labels']
predictions = torch.argmax(logits, dim=-1)

print(f"NOTE_OFFSET: {NOTE_OFFSET}")
print(f"MAX_NOTE: {MAX_NOTE}")
print(f"Valid note range: [{NOTE_OFFSET}, {NOTE_OFFSET + MAX_NOTE})\n")

# Analyze what notes we're looking at
seq_idx = 0
print(f"Analyzing sequence {seq_idx}:")

note_positions = []
note_labels = []
note_preds = []
note_in_range = []

for triplet_start in range(4, labels.size(1), 3):
    if triplet_start + 2 >= labels.size(1):
        break
    note_pos = triplet_start + 2
    label = labels[seq_idx, note_pos]
    if label != -100:  # Not masked
        note_positions.append(note_pos)
        note_labels.append(label.item())
        note_preds.append(predictions[seq_idx, note_pos].item())
        is_in_range = NOTE_OFFSET <= label < NOTE_OFFSET + MAX_NOTE
        note_in_range.append(is_in_range)

print(f"Found {len(note_positions)} unmasked notes")
print(f"First 10 note positions: {note_positions[:10]}")
print(f"First 10 note labels: {note_labels[:10]}")
print(f"First 10 note predictions: {note_preds[:10]}")
print(f"First 10 in valid range: {note_in_range[:10]}")
print(f"How many in valid range: {sum(note_in_range)} / {len(note_in_range)}")

if len(note_labels) > 0:
    matches = sum(1 for l, p in zip(note_labels, note_preds) if l == p)
    print(f"\nTotal matches: {matches} / {len(note_labels)} = {matches/len(note_labels):.2%}")
