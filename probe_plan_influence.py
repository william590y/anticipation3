import torch
from transformers import GPT2Config, GPT2LMHeadModel

from anticipation.packed_sequence import ALTERNATING_START, dummy_rest_triplet
from anticipation.vocab import (
    ADUR_OFFSET, ANOTE_OFFSET, ATIME_OFFSET, DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET, VOCAB_SIZE,
)
from plan_lm import PACKED_LENGTH, PlanLayout, assemble_inputs, prepare_plan_model
from plan_vq import BODY_SLOTS, NUM_PERF_NOTES


def make_packed_window(seed):
    rng = torch.Generator().manual_seed(seed)
    r = lambda h: int(torch.randint(0, h, (1,), generator=rng).item())
    perf = [(80 * i + r(20), 20 + r(40), 21 + r(80)) for i in range(NUM_PERF_NOTES)]
    score = [(50 * i + r(3), 25 + r(50), 21 + r(80)) for i in range(BODY_SLOTS)]
    tokens = []
    for i in range(ALTERNATING_START // 6):
        o, d, p = perf[i]
        tokens += [ATIME_OFFSET + o, ADUR_OFFSET + d, ANOTE_OFFSET + p] + dummy_rest_triplet(0)
    for s in range(BODY_SLOTS):
        o, d, p = score[s]
        tokens += [TIME_OFFSET + o, DUR_OFFSET + d, NOTE_OFFSET + p]
        o, d, p = perf[ALTERNATING_START // 6 + s]
        tokens += [ATIME_OFFSET + o, ADUR_OFFSET + d, ANOTE_OFFSET + p]
    return torch.tensor(tokens, dtype=torch.long)


batch = torch.stack([make_packed_window(1), make_packed_window(2)])
layout = PlanLayout(num_codes=6, codebook_size=32, placement="front")

config = GPT2Config(vocab_size=VOCAB_SIZE, n_positions=1024, n_embd=64, n_layer=2, n_head=2,
                    resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False)
torch.manual_seed(0)
model = GPT2LMHeadModel(config).eval()
prepare_plan_model(model, layout, verbose=False)

lo = torch.zeros(2, 6, dtype=torch.long)
hi = torch.full((2, 6), 31, dtype=torch.long)


def run(codes, mask):
    tokens, positions = assemble_inputs(batch, codes, layout)
    kwargs = {"position_ids": positions}
    if mask:
        kwargs["attention_mask"] = torch.ones_like(tokens)
    with torch.no_grad():
        return model(input_ids=tokens, **kwargs).logits


for mask in (True, False):
    a, b = run(lo, mask), run(hi, mask)
    print(f"attention_mask={mask}")
    print(f"  logit scale (std)                     {a.std().item():.4e}")
    for idx, label in (
        (layout.plan_length - 1, "last code -> first packed token"),
        (layout.plan_length + 10, "10 tokens into the window"),
        (layout.plan_length + 100, "100 tokens in"),
        (layout.score_start_idx - 1, "last prefix token -> first score onset"),
        (layout.total_length - 2, "end of the window"),
    ):
        diff = (a[:, idx] - b[:, idx]).abs().max().item()
        rel = diff / a[:, idx].std().item()
        print(f"  idx {idx:5d} {label:40s} absdiff {diff:.4e}  rel {rel:.4e}")

# baseline: how much does changing a real CONTROL token move the same logits?
alt = batch.clone()
alt[:, 0] = ATIME_OFFSET + 999
tok_a, pos_a = assemble_inputs(batch, lo, layout)
tok_b, pos_b = assemble_inputs(alt, lo, layout)
with torch.no_grad():
    la = model(input_ids=tok_a, attention_mask=torch.ones_like(tok_a), position_ids=pos_a).logits
    lb = model(input_ids=tok_b, attention_mask=torch.ones_like(tok_b), position_ids=pos_b).logits
idx = layout.score_start_idx - 1
print(f"\nbaseline: changing packed token 0 moves idx {idx} by "
      f"{(la[:, idx] - lb[:, idx]).abs().max().item():.4e}")
