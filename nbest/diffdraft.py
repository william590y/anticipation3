"""A masked (absorbing-state) discrete diffusion drafter for the packed score format.

WHY A DIFFUSION DRAFTER AT ALL
------------------------------
`onpolicy_rollout.rollout_score_slots` spends **one target forward per generated
score token**: 414 strictly sequential KV-cached forwards for a 1020-token /
138-slot window. Measured on a 3090 that is ~14 ms/step at batch 8 and ~35 ms at
batch 32 against ~0.2-0.7 ms of actual matmul -- the decode is *kernel-launch
bound by one to two orders of magnitude*. An autoregressive drafter does not
touch that: it still runs one forward per drafted token, it just runs a cheaper
one. A masked-diffusion drafter proposes a whole block of tokens in K forwards
with K << block length, which attacks the launch count directly.

The format makes this unusually natural. From `ALTERNATING_START = 192` the body
strictly alternates one SCORE triplet and one CONTROL triplet, and the control
triplets are the *conditioning performance stream*: they are known for the whole
window in advance and are teacher-forced into the target during AR decode. So a
bidirectional denoiser can be handed the true window with only the score slots
replaced by [MASK] and asked to fill all of them at once, with every control it
is allowed to see already in place.

WHY MASKED / ABSORBING RATHER THAN SEDD OR UNIFORM-KERNEL D3PM
--------------------------------------------------------------
The candidates were absorbing-state D3PM (Austin et al. 2021) and its
continuous-time simplification MDLM (Sahoo et al. 2024) / LLaDA, versus
score-entropy discrete diffusion (SEDD, Lou et al. 2024), versus a uniform-noise
kernel. We take masked/absorbing, for four reasons that are specific to the
drafter role rather than to generative quality:

1. The reverse posterior is trivial: an unmasked token is frozen, so "unmask the
   top-n most confident positions and never revisit them" (MaskGIT, Chang et al.
   2022) is a *valid* sampler, not a heuristic. With a uniform kernel every
   position stays live at every step, so a K-step sampler has to keep re-writing
   tokens the verifier has already accepted -- work we would throw away.
2. It emits a per-position categorical directly, so "confidence" is just the max
   probability. SEDD parameterises concrete score *ratios*; turning those into
   the calibrated per-token proposal probability that speculative sampling's
   accept test needs costs an extra normalisation step per position.
3. A drafter is never used as a likelihood model, so MDLM's main technical draw
   (a tight, variance-reduced NELBO) buys us nothing, while its parameterisation
   ("zero masking probabilities", carry-over unmasking) is exactly what we do use.
4. Under a *data-independent, left-to-right-contiguous* unmask schedule the
   masked model's block proposal factorises exactly as an autoregressive
   q(x_i | x_<i) -- see `nbest/diffdraft_decode.py`. That is what makes an exact
   sampled speculative-decoding path possible at all, and it is a property of the
   absorbing kernel (frozen prefix) that the uniform kernel does not have.

ARCHITECTURE: BLOCK-CAUSAL, NOT FULLY BIDIRECTIONAL
---------------------------------------------------
The naive design -- a fully bidirectional encoder re-run over all 1020 tokens at
every denoising step -- is a trap. A full-window forward of even a 6-layer,
1024-wide model at batch 32 is ~0.5 TFLOP *per denoising step*, several times the
cost of the target forward it is supposed to replace. So this model is
**block-causal** (Arriola et al. 2025, "Block Diffusion" / BD3-LM):

    prefix tokens  : attend causally among themselves        -> KV-cacheable
    block tokens   : attend to the whole prefix AND fully
                     bidirectionally within the block        -> recomputed each step

The committed prefix is encoded once and its KV cache is *extended*, never
recomputed, so a denoising step costs one forward over 6*B tokens, not 1020.

The thing we give up is sight of the controls *beyond* the current block. That is
a smaller loss than it looks: score slot k pairs with control k, and control
32+j sits after score slot j, so the target itself decodes slot k having seen
controls up to k+31 -- exactly 32 notes of lookahead. A block-causal drafter
drafting slots [c, c+B) has controls up to c+B+30 inside its own input, i.e.
**more** lookahead than the target has, for free, for every slot but the last.
Widening the field of view to the whole window's controls is the obvious next
experiment; it costs the prefix cache.

VOCABULARY
----------
One token is appended: `MASK_ID = VOCAB_SIZE` (55028), so the embedding table is
55029 rows. The repo lesson from the plan-LM arm applies -- HF's default
`resize_token_embeddings` mean-resizing initialises new rows at ~1/265 of the
right scale -- so this module never uses it: it allocates the table itself and
initialises the MASK row from `N(0, sigma)` with sigma matched to the *empirical*
per-element std of the pretrained `wte`, checked in `init_from_target`.
MASK is an input-only token: `constrain_score_token_logits` masks
`[CONTROL_OFFSET:VOCAB_SIZE]` and would happily leave column 55028 alive, so
every logit read here slices `[..., :VOCAB_SIZE]` *before* constraining.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import NewGELUActivation
from transformers.pytorch_utils import Conv1D

from anticipation.packed_sequence import ALTERNATING_START
from anticipation.score_constraints import constrain_score_token_logits
from anticipation.vocab import VOCAB_SIZE

MASK_ID = VOCAB_SIZE
DIFF_VOCAB_SIZE = VOCAB_SIZE + 1
PACKED_LENGTH = 1020
N_BODY_SLOTS = (PACKED_LENGTH - ALTERNATING_START) // 6  # 138


# ---------------------------------------------------------------------------
# Slot / position geometry
# ---------------------------------------------------------------------------


def slot_token_start(slot: int) -> int:
    """Absolute token index of body score slot `slot` (slot 0 == ALTERNATING_START)."""
    return ALTERNATING_START + 6 * slot


def block_token_end(slot_end: int) -> int:
    """One past the last token of the control triplet that follows slot `slot_end - 1`."""
    return ALTERNATING_START + 6 * slot_end


def score_positions(slot_start: int, slot_end: int, device=None) -> torch.Tensor:
    """Absolute positions of the 3*(slot_end-slot_start) score tokens in a slot range."""
    return torch.tensor(
        [slot_token_start(s) + r for s in range(slot_start, slot_end) for r in range(3)],
        dtype=torch.long,
        device=device,
    )


_ROLE_MASK_CACHE: dict = {}


def role_constraint_mask(device) -> torch.Tensor:
    """(3, VOCAB_SIZE) bool, True where the token is illegal for that triplet role.

    Built by interrogating `constrain_score_token_logits` itself, exactly as
    `onpolicy_rollout._role_constraint_mask` does, so the drafter's support can
    never drift from the decoder's. Note the width is VOCAB_SIZE, not
    DIFF_VOCAB_SIZE: MASK is stripped off before this is applied.
    """
    key = str(device)
    cached = _ROLE_MASK_CACHE.get(key)
    if cached is None:
        probe = torch.zeros(VOCAB_SIZE, device=device)
        cached = torch.stack(
            [torch.isinf(constrain_score_token_logits(probe, role)) for role in range(3)], dim=0
        )
        _ROLE_MASK_CACHE[key] = cached
    return cached


def constrain_by_role(logits: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
    """Mask a (..., n, DIFF_VOCAB_SIZE or VOCAB_SIZE) logit tensor by per-position role."""
    logits = logits[..., :VOCAB_SIZE]
    mask = role_constraint_mask(logits.device)[roles]
    return logits.masked_fill(mask, -float("inf"))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class DiffDraftConfig:
    n_layer: int = 6
    n_head: int = 16
    n_embd: int = 1024
    n_positions: int = 1024
    vocab_size: int = DIFF_VOCAB_SIZE
    dropout: float = 0.0
    # Which target block each drafter block was seeded from. Informational only --
    # see the note on attention scaling in DiffDraftAttention.
    source_layers: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.source_layers:
            self.source_layers = list(range(self.n_layer))

    def to_dict(self):
        return {
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "n_positions": self.n_positions,
            "vocab_size": self.vocab_size,
            "dropout": self.dropout,
            "source_layers": list(self.source_layers),
        }


class DiffDraftAttention(nn.Module):
    """GPT-2 attention, but the causal structure comes from an explicit mask.

    Deliberately not `transformers`' GPT2Attention: this module is called with a
    *block-causal* mask and with a KV cache that is extended for the prefix but
    thrown away for the block, and the repo's documented cached-decode trap (a
    chunk-shaped all-ones `attention_mask` being read as a padding mask over
    past+current keys, silently truncating the cache to one token) is exactly the
    kind of plumbing this avoids by owning the mask outright.
    """

    def __init__(self, config: DiffDraftConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = Conv1D(3 * config.n_embd, config.n_embd)
        self.c_proj = Conv1D(config.n_embd, config.n_embd)
        # 1/sqrt(head_dim) and NOTHING ELSE. The target's config sets
        # scale_attn_by_inverse_layer_idx=True, but transformers only honours that
        # in `eager_attention_forward`; the sdpa path reads `module.scaling`, which
        # GPT2Attention never sets, so it silently uses the plain 1/sqrt(d) scale.
        # The target is loaded (and was trained) under sdpa -- see the CLAUDE.md
        # note that sdpa and eager agree on only 24% of greedy tokens on this
        # checkpoint -- so the weights we copy in `init_from_target` were fitted
        # WITHOUT the per-layer divisor. Reinstating it here (dividing layer 23's
        # attention logits by 24) would have made the initialisation worthless.
        self.attn_scale = 1.0 / math.sqrt(self.head_dim)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.attn_dropout_p = config.dropout

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, hidden, attn_mask, past=None, append_to_cache=False):
        query, key, value = self.c_attn(hidden).split(hidden.shape[-1], dim=2)
        query = self._split(query)
        key = self._split(key)
        value = self._split(value)

        new_kv = (key, value) if append_to_cache else None
        if past is not None:
            key = torch.cat([past[0], key], dim=2)
            value = torch.cat([past[1], value], dim=2)
            if append_to_cache:
                new_kv = (key, value)

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            scale=self.attn_scale,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(hidden.shape)
        return self.resid_dropout(self.c_proj(out)), new_kv


class DiffDraftMLP(nn.Module):
    def __init__(self, config: DiffDraftConfig):
        super().__init__()
        self.c_fc = Conv1D(4 * config.n_embd, config.n_embd)
        self.c_proj = Conv1D(config.n_embd, 4 * config.n_embd)
        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden):
        return self.dropout(self.c_proj(self.act(self.c_fc(hidden))))


class DiffDraftBlock(nn.Module):
    def __init__(self, config: DiffDraftConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = DiffDraftAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = DiffDraftMLP(config)

    def forward(self, hidden, attn_mask, past=None, append_to_cache=False):
        attn_out, new_kv = self.attn(self.ln_1(hidden), attn_mask, past, append_to_cache)
        hidden = hidden + attn_out
        hidden = hidden + self.mlp(self.ln_2(hidden))
        return hidden, new_kv


class DiffDraftModel(nn.Module):
    """Block-causal masked-diffusion denoiser over packed score windows."""

    def __init__(self, config: DiffDraftConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList(DiffDraftBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)
        # Head tied to the input embedding, as in the target. MASK owns a row it
        # can never legally win: every read strips [..., :VOCAB_SIZE] first.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # -- shared trunk ------------------------------------------------------

    def _trunk(self, input_ids, positions, attn_mask, past=None, append_to_cache=False):
        hidden = self.drop(self.wte(input_ids) + self.wpe(positions))
        new_cache = [] if append_to_cache else None
        for i, block in enumerate(self.h):
            layer_past = None if past is None else past[i]
            hidden, kv = block(hidden, attn_mask, layer_past, append_to_cache)
            if append_to_cache:
                new_cache.append(kv)
        return self.ln_f(hidden), new_cache

    def head(self, hidden):
        return self.lm_head(hidden)

    # -- training / one-shot forward ---------------------------------------

    def forward(self, input_ids, prefix_len):
        """One forward over positions [0, N) with the block-causal mask.

        `prefix_len` tokens are the committed prefix (causal among themselves);
        everything from `prefix_len` on is the block and sees all of the prefix
        plus all of the block. Returns hidden states; the caller applies `head`
        only where it needs logits (the head is 56M params over a 55029 vocab and
        is comparable in cost to the whole 6-layer trunk if applied everywhere).
        """
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        mask = block_causal_mask(length, prefix_len, input_ids.device)
        hidden, _ = self._trunk(input_ids, positions, mask)
        return hidden

    # -- cached inference ---------------------------------------------------

    @torch.no_grad()
    def encode_prefix(self, input_ids, offset: int, past=None):
        """Extend the causal prefix cache by `input_ids` (absolute positions from `offset`).

        Valid to call incrementally because the prefix is *causal*: adding tokens
        never changes the keys/values of the ones already cached.
        """
        length = input_ids.shape[1]
        device = input_ids.device
        positions = torch.arange(offset, offset + length, device=device).unsqueeze(0)
        past_len = 0 if past is None else past[0][0].shape[2]
        mask = causal_extension_mask(length, past_len, device)
        _, new_cache = self._trunk(input_ids, positions, mask, past, append_to_cache=True)
        return new_cache

    @torch.no_grad()
    def forward_block(self, block_ids, offset: int, past):
        """Denoise one block: full attention to the cached prefix and within the block."""
        length = block_ids.shape[1]
        device = block_ids.device
        positions = torch.arange(offset, offset + length, device=device).unsqueeze(0)
        past_len = 0 if past is None else past[0][0].shape[2]
        mask = torch.ones(1, 1, length, past_len + length, dtype=torch.bool, device=device)
        hidden, _ = self._trunk(block_ids, positions, mask, past, append_to_cache=False)
        return hidden


def block_causal_mask(length: int, prefix_len: int, device) -> torch.Tensor:
    """(1,1,L,L) bool: prefix rows causal, block rows unrestricted."""
    idx = torch.arange(length, device=device)
    causal = idx.unsqueeze(1) >= idx.unsqueeze(0)  # (query, key)
    is_block = (idx >= prefix_len).unsqueeze(1)
    return (causal | is_block).view(1, 1, length, length)


def causal_extension_mask(length: int, past_len: int, device) -> torch.Tensor:
    """(1,1,L,past+L) bool: attend to all of the past, causally within the new segment."""
    q = torch.arange(length, device=device).unsqueeze(1)
    k = torch.arange(past_len + length, device=device).unsqueeze(0) - past_len
    return (k <= q).view(1, 1, length, past_len + length)


def crop_cache(past, length: int):
    """Trim every layer's KV to the first `length` positions (in place on the list)."""
    return [(k[:, :, :length], v[:, :, :length]) for k, v in past]


# ---------------------------------------------------------------------------
# Initialisation from the target checkpoint
# ---------------------------------------------------------------------------


def select_layer_indices(n_target_layers: int, n_draft_layers: int) -> list[int]:
    """Evenly spaced target blocks, always including the first and the last."""
    if n_draft_layers >= n_target_layers:
        return list(range(n_target_layers))
    if n_draft_layers == 1:
        return [n_target_layers - 1]
    step = (n_target_layers - 1) / (n_draft_layers - 1)
    return [int(round(i * step)) for i in range(n_draft_layers)]


def init_from_target(model: DiffDraftModel, target_state: dict, layer_indices: list[int]):
    """Copy the target's embeddings, ln_f and the selected blocks into the drafter.

    Everything but the extra MASK row is a straight copy: the drafter is
    GPT-2-shaped on purpose so `transformer.h.{i}.*` maps one-to-one. Attention is
    bidirectional within the block, which the copied heads were never trained for,
    but the 56M-parameter tied embedding/head is the bulk of the model and
    transfers exactly.
    """
    with torch.no_grad():
        wte = target_state["transformer.wte.weight"]
        std = float(wte.float().std())
        model.wte.weight[:VOCAB_SIZE].copy_(wte)
        # The plan-LM lesson: HF's default mean-resizing would seed this row at
        # the mean of the existing rows, whose per-element magnitude is ~265x
        # smaller than a real row's, and the new token then contributes nothing.
        # Match the pretrained table's own std instead.
        model.wte.weight[MASK_ID].normal_(mean=0.0, std=std)
        model.wpe.weight.copy_(target_state["transformer.wpe.weight"])
        model.ln_f.weight.copy_(target_state["transformer.ln_f.weight"])
        model.ln_f.bias.copy_(target_state["transformer.ln_f.bias"])
        for dst, src in enumerate(layer_indices):
            block = model.h[dst]
            prefix = f"transformer.h.{src}."
            block.ln_1.weight.copy_(target_state[prefix + "ln_1.weight"])
            block.ln_1.bias.copy_(target_state[prefix + "ln_1.bias"])
            block.ln_2.weight.copy_(target_state[prefix + "ln_2.weight"])
            block.ln_2.bias.copy_(target_state[prefix + "ln_2.bias"])
            block.attn.c_attn.weight.copy_(target_state[prefix + "attn.c_attn.weight"])
            block.attn.c_attn.bias.copy_(target_state[prefix + "attn.c_attn.bias"])
            block.attn.c_proj.weight.copy_(target_state[prefix + "attn.c_proj.weight"])
            block.attn.c_proj.bias.copy_(target_state[prefix + "attn.c_proj.bias"])
            block.mlp.c_fc.weight.copy_(target_state[prefix + "mlp.c_fc.weight"])
            block.mlp.c_fc.bias.copy_(target_state[prefix + "mlp.c_fc.bias"])
            block.mlp.c_proj.weight.copy_(target_state[prefix + "mlp.c_proj.weight"])
            block.mlp.c_proj.bias.copy_(target_state[prefix + "mlp.c_proj.bias"])
    return {
        "wte_std": std,
        "mask_row_norm": float(model.wte.weight[MASK_ID].detach().norm()),
        "mean_row_norm": float(model.wte.weight[:VOCAB_SIZE].detach().norm(dim=1).mean()),
        "layer_indices": layer_indices,
    }


def build_drafter(n_layer=6, target_state=None, n_target_layers=24, dropout=0.0, n_head=16,
                  n_embd=1024):
    layer_indices = select_layer_indices(n_target_layers, n_layer)
    config = DiffDraftConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        source_layers=layer_indices,
    )
    model = DiffDraftModel(config)
    info = None
    if target_state is not None:
        info = init_from_target(model, target_state, layer_indices)
    return model, info


def load_drafter(path, device="cuda", dtype=torch.float32):
    """Rebuild a drafter from a `train_diffdraft.py` checkpoint."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    config = DiffDraftConfig(**blob["config"])
    model = DiffDraftModel(config)
    model.load_state_dict(blob["model"])
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, blob


# ---------------------------------------------------------------------------
# Training-time corruption
# ---------------------------------------------------------------------------


DEFAULT_BLOCK_SIZES = (138, 32, 16, 8)
DEFAULT_BLOCK_WEIGHTS = (0.40, 0.25, 0.20, 0.15)


def sample_block_geometry(generator, block_sizes=DEFAULT_BLOCK_SIZES,
                          block_weights=DEFAULT_BLOCK_WEIGHTS, n_slots=N_BODY_SLOTS):
    """One (committed-prefix slots, block slots) pair, shared by a whole batch.

    Shared rather than per-row because the block geometry decides the sequence
    length, and a ragged batch would need padding plus a padding-aware mask --
    which is precisely the plumbing the repo's cached-decode trap lives in. The
    per-row diversity comes from the mask pattern and the noise level, which are
    sampled independently for every row.
    """
    weights = torch.tensor(block_weights, dtype=torch.float, device=generator.device)
    choice = int(torch.multinomial(weights, 1, generator=generator).item())
    block_slots = min(int(block_sizes[choice]), n_slots)
    committed = int(torch.randint(0, n_slots - block_slots + 1, (1,), generator=generator,
                                  device=generator.device).item())
    return committed, block_slots


def build_training_batch(windows, generator, committed_slots, block_slots, ltr_prob=0.5,
                         min_masked=1):
    """Corrupt one batch of teacher-rolled windows into a denoising problem.

    The corruption is a 50/50 mixture of the two state distributions the drafter
    actually meets at inference:

      `iid`  -- every score token in the block masked independently with rate
                t ~ U(0,1). This is the ordinary absorbing-diffusion forward
                process restricted to the block, and it is the state a
                confidence-ordered (MaskGIT) unmask schedule walks through, where
                the surviving tokens are scattered arbitrarily over the block.
      `ltr`  -- a cut point inside the block; everything before it clean,
                everything from it on masked. This is the state a *left-to-right*
                group schedule walks through, and that schedule is the one with
                an exactly-evaluable proposal for sampled speculative decoding
                (see `nbest/diffdraft_decode.py`), so it has to be trained for
                explicitly -- iid masking almost never produces a contiguous
                suffix.

    Loss is taken only on masked positions inside the block. Positions after the
    block are not masked, they are *absent*: the input is truncated at the block's
    trailing control triplet, matching the cached-prefix inference path exactly.

    Returns (input_ids, target_ids, loss_positions, prefix_len).
    """
    device = windows.device
    batch = windows.shape[0]
    prefix_len = block_token_end(committed_slots)
    end = block_token_end(committed_slots + block_slots)

    target_ids = windows[:, :end].contiguous()
    input_ids = target_ids.clone()

    local = score_positions(committed_slots, committed_slots + block_slots, device=device)
    n_pos = local.numel()

    use_ltr = torch.rand(batch, 1, generator=generator, device=device) < ltr_prob
    rate = torch.rand(batch, 1, generator=generator, device=device)
    iid_mask = torch.rand(batch, n_pos, generator=generator, device=device) < rate
    # `cut` is uniform over 0..n_pos so both "nothing committed yet" and "one
    # token left" are in-distribution; the >= makes it a contiguous suffix.
    cut = torch.randint(0, n_pos + 1, (batch, 1), generator=generator, device=device)
    ltr_mask = torch.arange(n_pos, device=device).unsqueeze(0) >= cut
    masked = torch.where(use_ltr, ltr_mask, iid_mask)

    # A row with nothing masked contributes no gradient and wastes its slice of
    # the batch; force at least `min_masked` (the last positions, which is the
    # ltr-consistent choice).
    empty = masked.sum(dim=1) < min_masked
    if empty.any():
        masked[empty, -min_masked:] = True

    scatter = local.unsqueeze(0).expand(batch, -1)
    input_ids.scatter_(1, scatter, torch.where(masked, torch.full_like(scatter, MASK_ID),
                                               target_ids.gather(1, scatter)))
    return input_ids, target_ids, local, prefix_len, masked


# ---------------------------------------------------------------------------
# Inference: K-step parallel unmasking
# ---------------------------------------------------------------------------


EXACT_SAMPLED_SCHEDULES = ("ltr",)


def unmask_counts(n_positions: int, steps: int, schedule: str = "cosine") -> list[int]:
    """How many positions to unmask at each of `steps` steps (sums to n_positions).

    Cosine (MaskGIT's gamma(r) = cos(r*pi/2)): few positions committed early,
    when the block is nearly empty and the model is least sure, many at the end.
    """
    if steps >= n_positions:
        return [1] * n_positions
    counts = []
    remaining = n_positions
    for i in range(steps):
        if schedule == "linear":
            target = int(round(n_positions * (1.0 - (i + 1) / steps)))
        else:
            target = int(math.floor(n_positions * math.cos(math.pi / 2 * (i + 1) / steps)))
        target = max(0, min(target, remaining - 1)) if i < steps - 1 else 0
        counts.append(remaining - target)
        remaining = target
    return counts


def _group_bounds(n_positions: int, steps: int) -> list[tuple[int, int]]:
    """Contiguous left-to-right groups: group j is positions [lo, hi)."""
    size = math.ceil(n_positions / steps)
    bounds = []
    lo = 0
    while lo < n_positions:
        bounds.append((lo, min(lo + size, n_positions)))
        lo += size
    return bounds


@torch.no_grad()
def denoise_block(model, block_ids, offset, prefix_cache, local_idx, steps=4,
                  order="confidence", temperature=0.0, generator=None,
                  collect_proposal=False, count_step=None):
    """Fill every masked score position in `block_ids` in `steps` forward passes.

    `block_ids` is (batch, L) covering absolute positions [offset, offset+L);
    `local_idx` lists the block-local indices of its score tokens. A score
    position holds MASK_ID if it still has to be drafted and a real token if the
    verifier already committed it -- rows in a batch disagree about that (the
    block starts at the batch *minimum* accepted position), so the still-masked
    set is read per row rather than assumed uniform.

    `order` selects which positions get committed at each step:
      `confidence`  MaskGIT: commit the top-n by max posterior probability, with a
                    cosine count schedule. Best draft quality, but the resulting
                    joint proposal is *not* an autoregressive q (the order depends
                    on the sampled values), so it is exact only on the greedy path.
      `ltr`         Fixed contiguous left-to-right groups. Data-independent, so
                    q(x_i | x_<i) factorises exactly -- the only schedule here
                    that supports exact sampled speculative decoding.
      `role`        Commit all onsets, then all durations, then all pitches
                    (steps must be 3). Mirrors the target's own within-slot
                    factorisation while parallelising across all slots at once.

    Returns (filled_ids, info). `info["proposal_logprob"]` is log q of each
    committed token under the constrained, temperature-scaled distribution *at the
    step it was committed* -- the quantity the sampled accept test needs -- and
    `info["proposal_logits"]` the full constrained log-probs, kept in fp16 and only
    when `collect_proposal` is set (it is (batch, n, 55028) and large).
    """
    device = block_ids.device
    batch, length = block_ids.shape
    n_pos = local_idx.numel()
    roles = (local_idx + offset) % 3

    work = block_ids.clone()
    gather_idx = local_idx.unsqueeze(0).expand(batch, -1)
    still_masked = work.gather(1, gather_idx) == MASK_ID
    if not bool(still_masked.any()):
        return work, {"forwards": 0, "local_idx": local_idx, "roles": roles}
    chosen = work.gather(1, gather_idx)
    logq = torch.zeros(batch, n_pos, dtype=torch.float32, device=device)
    # fp32, not fp16. This tensor IS the proposal q that the sampled accept test
    # divides by, and the exactness claim is only as good as it: rounding log q to
    # fp16 moves it by ~4e-3, i.e. it perturbs every acceptance probability by
    # ~0.4%. Halving the memory is not worth turning a provably exact path into an
    # approximately exact one. (batch 16 x 48 positions x 55028 = 169 MB.)
    # Filled with -inf, NOT zeros: these are LOG probabilities, so an unwritten
    # entry has to mean "q assigns this token no mass", not "q(token) = 1". A
    # position that is already committed is never proposed and so is never
    # written, and with a zero fill the verifier's residual norm(p - q)+ came out
    # identically zero there and `torch.multinomial` fired a device-side assert
    # (job 464050).
    proposal = (
        torch.full((batch, n_pos, VOCAB_SIZE), -float("inf"), dtype=torch.float32,
                   device=device)
        if collect_proposal
        else None
    )

    if order == "ltr":
        groups = _group_bounds(n_pos, steps)
        steps = len(groups)
    elif order == "role":
        if steps != 3:
            raise ValueError("order='role' is defined for steps=3 (onset, duration, pitch)")
        groups = None
    else:
        steps = min(steps, n_pos)  # unmask_counts saturates at one position per step
        counts = unmask_counts(n_pos, steps)
        groups = None

    forwards = 0
    for step in range(steps):
        hidden = model.forward_block(work, offset, prefix_cache)
        forwards += 1
        if count_step is not None:
            count_step(work.shape[1])
        # The head is only ever read at still-masked score positions; applying it
        # to the whole block would cost as much as the trunk.
        # `head(...)` is freshly allocated, so the role constraint can be applied
        # in place: this tensor is (batch, n, 55028) -- a gigabyte at batch 96 --
        # and the out-of-place `constrain_by_role` would keep two of them alive.
        logits = model.head(hidden[:, local_idx, :]).float()[..., :VOCAB_SIZE]
        logits.masked_fill_(role_constraint_mask(logits.device)[roles], -float("inf"))
        if temperature is not None and temperature > 0:
            logits = torch.log_softmax(logits / temperature, dim=-1)
            token = torch.multinomial(
                logits.exp().view(-1, VOCAB_SIZE), 1, generator=generator
            ).view(batch, n_pos)
            picked_logq = logits.gather(2, token.unsqueeze(-1)).squeeze(-1)
        else:
            # Greedy needs only the argmax and its normalised probability (the
            # MaskGIT confidence), so skip materialising a second full-size
            # log_softmax tensor and normalise the winner alone.
            top, token = logits.max(dim=-1)
            picked_logq = top - torch.logsumexp(logits, dim=-1)

        if groups is not None:
            lo, hi = groups[step]
            commit = torch.zeros(batch, n_pos, dtype=torch.bool, device=device)
            commit[:, lo:hi] = True
            commit &= still_masked
        elif order == "role":
            commit = (roles == step).unsqueeze(0).expand(batch, -1) & still_masked
        else:
            # MaskGIT confidence rule: rank the still-masked positions by the
            # probability of their own argmax and commit the top `counts[step]`.
            # Already-committed positions are -inf so `&= still_masked` only ever
            # trims the tail of the top-k, never mis-commits.
            confidence = picked_logq.masked_fill(~still_masked, -float("inf"))
            k = max(1, min(counts[step], n_pos))
            top = confidence.topk(k, dim=1).indices
            commit = torch.zeros(batch, n_pos, dtype=torch.bool, device=device)
            commit.scatter_(1, top, True)
            commit &= still_masked

        chosen = torch.where(commit, token, chosen)
        logq = torch.where(commit, picked_logq, logq)
        if proposal is not None:
            proposal = torch.where(commit.unsqueeze(-1), logits, proposal)
        still_masked &= ~commit
        work.scatter_(1, gather_idx, torch.where(commit, chosen, work.gather(1, gather_idx)))
        if not bool(still_masked.any()):
            break

    if bool(still_masked.any()):  # schedule under-committed; take the last argmax
        work.scatter_(1, gather_idx, torch.where(still_masked, token, work.gather(1, gather_idx)))
        chosen = torch.where(still_masked, token, chosen)
        logq = torch.where(still_masked, picked_logq, logq)

    return work, {
        "forwards": forwards,
        "local_idx": local_idx,
        "roles": roles,
        "tokens": chosen,
        "proposal_logprob": logq,
        "proposal_logits": proposal,
    }
