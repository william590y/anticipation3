import torch
from transformers import GPT2Config, GPT2LMHeadModel

from anticipation.packed_sequence import ALTERNATING_START, dummy_rest_triplet
from anticipation.vocab import (
    ADUR_OFFSET, ANOTE_OFFSET, ATIME_OFFSET, DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET, VOCAB_SIZE,
)
from plan_lm import PACKED_LENGTH, PLAN_CODE_OFFSET, PlanLayout, assemble_inputs, prepare_plan_model
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

wte = model.transformer.wte.weight.detach()
plan_delta = (wte[PLAN_CODE_OFFSET + 0] - wte[PLAN_CODE_OFFSET + 31]).norm().item()
packed_delta = (wte[ATIME_OFFSET + 5] - wte[ATIME_OFFSET + 999]).norm().item()
print(f"||emb(code 0) - emb(code 31)||          = {plan_delta:.5f}")
print(f"||emb(atime 5) - emb(atime 999)||       = {packed_delta:.5f}")
print(f"mean row norm, base vocab               = {wte[:VOCAB_SIZE].norm(dim=1).mean():.5f}")
print(f"mean row norm, plan vocab               = {wte[VOCAB_SIZE:].norm(dim=1).mean():.5f}")

lo = torch.zeros(2, 6, dtype=torch.long)
hi = torch.full((2, 6), 31, dtype=torch.long)


def logits_for(packed, codes):
    tokens, positions = assemble_inputs(packed, codes, layout)
    with torch.no_grad():
        return model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                     position_ids=positions).logits


base = logits_for(batch, lo)
plan_changed = logits_for(batch, hi)

# A comparable single-token control change, placed at the SAME assembled index
# as the last plan code so the two decay curves start from the same distance.
alt = batch.clone()
alt[:, 0] = ATIME_OFFSET + 999
token_changed = logits_for(alt, lo)

print(f"\n{'assembled idx':>14}  {'plan change':>13}  {'1-token change':>15}")
for idx in (7, 10, 20, 50, 100, 198, 400, 800, 1025):
    a = (base[:, idx] - plan_changed[:, idx]).abs().max().item()
    b = (base[:, idx] - token_changed[:, idx]).abs().max().item()
    print(f"{idx:>14}  {a:>13.3e}  {b:>15.3e}")
