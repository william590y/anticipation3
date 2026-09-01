"""Launch-overhead-free fast paths for the KV-cached score decode.

The canonical decode (`onpolicy_rollout.rollout_score_slots`,
`inference.batched_autoregressive_generate_score`) runs 414 sequential cached
forwards per 1020-token packed window. Measured on a 3090 with the fp32
checkpoint, one such forward costs ~11 ms at batch 1 and is *flat* in KV-cache
length -- i.e. essentially all of it is CPU: transformers' Python forward plus
~300 kernel launches per step, not GPU work. That is what this module removes.

Four independent levers, each switchable so the ablation is measurable:

1. **Head only where it is needed.** The baseline calls the LM head on every
   position of every chunk -- 192 positions on the prefill and 4 on the
   pitch+controls chunk -- then throws all but the last away. A (198, 4, 55028)
   fp32 logits tensor is 174 MB written and re-read for one useful row. Here the
   head is applied to the last hidden state only.
2. **Sliced head.** `score_constraints.constrain_score_token_logits` masks each
   role down to one *contiguous* vocab range (asserted below, so a future
   non-contiguous constraint fails loudly instead of decoding silently wrong).
   A greedy pick therefore only needs that range: 10000 / 1000 / 16512 columns
   instead of 55028, a 6x smaller head matmul. Unavailable when the caller wants
   `gt_ce` or `logprob`, both of which are normalized over the *full* vocab.
3. **CUDA graphs over a whole score slot.** With a `StaticCache` the KV tensors
   live at fixed addresses, so an entire slot -- three cached forwards, three
   heads, three argmaxes -- captures into one graph and replays with a single
   launch. Cache positions are static buffers the graph increments by 6 itself,
   so one capture serves all 138 slots. This is the lever that matters at small
   batch, where the decode is CPU-bound; at batch >= 32 it is roughly neutral.
4. **Bucketed KV length** (`_WindowedStaticLayer`). A static cache attends over
   its whole 1020-slot buffer every step even though the real cache averages
   606, so it trades ~1.7x KV bandwidth for stable shapes. Rounding the visible
   length up to one of N buckets (one captured graph each) keeps the shapes
   static and gives most of that bandwidth back. This is the lever that matters
   at large batch, where the decode is KV-bandwidth-bound.

The **KV-cache trap** documented in CLAUDE.md is preserved throughout: no
chunk-shaped all-ones `attention_mask` is ever passed during cached decode.
`StaticCache` needs `cache_position` instead, which is the supported mechanism
and does not go through the padding-mask path that silently truncates the cache.

Seam for other decode work (e.g. draft/verify schemes): `StaticKVDecoder` is a
standalone single-step interface -- `prefill`, `step`, `logits`, `rewind` --
that owns the static cache and can be reused across rollouts. `rewind(n)` is
O(1) (it only moves the write pointer; rejected slots are overwritten), which is
what a propose-then-verify loop needs after a rejection.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from transformers import StaticCache
from transformers.cache_utils import StaticLayer

from anticipation.packed_sequence import ALTERNATING_START, iter_score_slot_positions
from anticipation.score_constraints import constrain_score_token_logits
from anticipation.vocab import REST, VOCAB_SIZE


def _legal_token_ranges(vocab_size=VOCAB_SIZE):
    """Per-role `(lo, hi)` legal vocab range, read off the constraint function itself.

    Derived rather than hard-coded for the same reason
    `onpolicy_rollout._role_constraint_mask` is: the fast path's notion of the
    policy support can then never drift from the decoder's. The contiguity
    assert is load-bearing -- slicing the head is only equivalent to masking the
    full head while each role's legal set is one interval.
    """
    ranges = []
    for role in range(3):
        probe = torch.zeros(vocab_size)
        legal = torch.isfinite(constrain_score_token_logits(probe, role)).nonzero().flatten()
        lo, hi = int(legal[0].item()), int(legal[-1].item()) + 1
        if hi - lo != legal.numel():
            raise AssertionError(
                f"role {role}'s legal token set is not contiguous ({legal.numel()} tokens "
                f"spanning [{lo}, {hi})); the sliced-head fast path assumes one interval."
            )
        ranges.append((lo, hi))
    return tuple(ranges)


LEGAL_TOKEN_RANGES = _legal_token_ranges()

_ILLEGAL_MASK_CACHE = {}


def role_illegal_mask(role, device, vocab_size=VOCAB_SIZE):
    """(vocab,) bool mask, True where the token is illegal for `role`.

    Applying this with a single broadcast `masked_fill` is value-identical to
    `constrain_score_token_logits` but does one full-tensor pass instead of the
    clone plus two or three slice fills that function does (it has to keep the
    generic 1-D/2-D contract; here the shape is always (batch, vocab)).
    """
    key = (role, str(device), vocab_size)
    cached = _ILLEGAL_MASK_CACHE.get(key)
    if cached is None:
        probe = torch.zeros(vocab_size, device=device)
        cached = torch.isinf(constrain_score_token_logits(probe, role))
        _ILLEGAL_MASK_CACHE[key] = cached
    return cached


class _WindowedStaticLayer(StaticLayer):
    """A `StaticLayer` that exposes only its first `visible_length` positions.

    A plain `StaticCache` attends over the whole preallocated buffer on every
    step, because that is what `update` returns and what `get_mask_sizes`
    reports. Our packed window's true cache length runs 192 -> 1020 (mean 606),
    so a fixed 1020 reads ~1.7x more KV than the decode needs -- and once CUDA
    graphs have removed the CPU time, KV bandwidth is what is left. Rounding the
    true length up to one of a handful of buckets keeps the shape static (so a
    graph can still be captured per bucket) and caps the waste at the bucket
    width instead of the whole buffer.

    `visible_length` is a plain Python int on purpose: it must be a *compile-time
    constant* inside a captured graph, which is exactly why there is one graph
    per bucket rather than one graph with a tensor length.
    """

    visible_length = None

    def update(self, key_states, value_states, cache_kwargs=None):
        keys, values = super().update(key_states, value_states, cache_kwargs)
        window = self.visible_length
        if window is None or window >= keys.shape[2]:
            return keys, values
        return keys[:, :, :window], values[:, :, :window]

    def get_mask_sizes(self, cache_position):
        window = self.visible_length
        return (self.max_cache_len if window is None else window), 0


# ---------------------------------------------------------------------------
# The static-cache single-step decoder (the reusable seam)
# ---------------------------------------------------------------------------


class StaticKVDecoder:
    """One reusable static KV cache plus a single-step forward over it.

    Parameters
    ----------
    model : GPT2LMHeadModel
        The fine-tuned causal LM. Left untouched (never wrapped, never moved).
    batch_size : int
        Fixed batch width. A `StaticCache` and any captured CUDA graph are both
        allocated for exactly this width, so a decoder cannot be reused across
        batch sizes -- build one per size.
    max_length : int
        Cache capacity in tokens; 1020 for the standard packed window. Attention
        always runs over the full capacity (that is the price of static shapes),
        so do not oversize it.

    Interface
    ---------
    ``reset()``                  -- rewind to an empty cache.
    ``prefill(ids) -> (B, H)``   -- consume a prompt, return its last hidden state.
    ``step(tokens) -> (B, H)``   -- consume ``(B, T)`` tokens, return the hidden
                                    state after the last of them. ``T`` may vary.
    ``logits(h, role=None)``     -- LM head on a hidden state; with ``role`` set,
                                    only that role's legal vocab range (offset by
                                    ``LEGAL_TOKEN_RANGES[role][0]``).
    ``rewind(n)``                -- drop the last ``n`` cached positions, O(1).
    ``position``                 -- number of tokens currently cached.

    The hidden state is returned rather than logits so a caller can decide how
    much of the 55028-wide head it actually needs.
    """

    def __init__(
        self, model, batch_size, max_length=1020, device=None, prebuilt_mask=True, buckets=None
    ):
        self.model = model
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = device if device is not None else next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.cache = StaticCache(config=model.config, max_cache_len=self.max_length)
        self.buckets = self._bucket_edges(buckets)
        if self.buckets is not None:
            # Swap in windowed layers before anything allocates: the layers lazily
            # build their tensors on first `update`, so replacing them here costs
            # nothing and leaves one buffer of `max_length` per layer either way.
            self.cache.layers = [
                _WindowedStaticLayer(max_cache_len=self.max_length) for _ in self.cache.layers
            ]
        self.position = 0
        self._head_weight = model.lm_head.weight
        if getattr(model.lm_head, "bias", None) is not None:
            raise AssertionError("fast head assumes a bias-free tied lm_head (GPT-2)")

        # A compileable cache disables transformers' "skip the mask and let SDPA
        # be causal" shortcut, so every step goes through `create_causal_mask`,
        # which composes the mask with nested `torch.vmap`. That is milliseconds
        # of *CPU* per step -- on a 3090 at batch 8 it made the static-cache path
        # ~10% slower than the DynamicCache baseline it was supposed to beat.
        # One 1020x1020 lower-triangular bool table (1 MB) turns the mask into a
        # free row slice: row p is exactly "attend to kv <= p".
        #
        # Passing this 4-D mask is NOT the KV-cache trap from CLAUDE.md. That
        # trap is a *2-D, chunk-shaped* mask, which transformers reads as a
        # padding mask over past+current keys and which truncates the cache. A
        # 4-D mask takes `_preprocess_mask_arguments`' early exit and is handed
        # to SDPA verbatim -- the same tensor transformers would have built.
        self._causal_rows = (
            torch.tril(
                torch.ones(self.max_length, self.max_length, dtype=torch.bool, device=self.device)
            )
            if prebuilt_mask
            else None
        )

    # -- cache management ---------------------------------------------------

    def reset(self):
        """Empty the cache. Entries are overwritten in place, not reallocated."""
        self.position = 0
        for layer in self.cache.layers:
            if getattr(layer, "keys", None) is not None:
                layer.keys.zero_()
                layer.values.zero_()

    def rewind(self, n):
        """Un-cache the last `n` positions (they are simply overwritten next)."""
        if n < 0 or n > self.position:
            raise ValueError(f"cannot rewind {n} of {self.position} cached positions")
        self.position -= n

    # -- forward ------------------------------------------------------------

    def _bucket_edges(self, buckets):
        """Normalize the `buckets` argument to an ascending tuple of lengths."""
        if not buckets:
            return None
        if isinstance(buckets, int):
            step = max(1, self.max_length // buckets)
            edges = list(range(step, self.max_length, step)) + [self.max_length]
        else:
            edges = [int(b) for b in buckets]
        edges = sorted({min(int(b), self.max_length) for b in edges})
        if edges[-1] != self.max_length:
            edges.append(self.max_length)
        return tuple(edges)

    def bucket_for(self, needed):
        """Smallest bucket that can hold `needed` cached positions."""
        if self.buckets is None:
            return self.max_length
        for edge in self.buckets:
            if edge >= needed:
                return edge
        return self.max_length

    def set_visible(self, window):
        """Restrict every layer to the first `window` cache positions."""
        if self.buckets is None:
            return self.max_length
        for layer in self.cache.layers:
            layer.visible_length = window
        return window

    def causal_mask(self, start, width, window=None):
        """(1, 1, width, window) bool mask for a chunk written at `start`."""
        if self._causal_rows is None:
            return None
        window = window or self.max_length
        # `unsqueeze`, not `view`: a `[rows, :window]` slice is non-contiguous
        # whenever the bucket is narrower than the buffer, and `view` rejects
        # that outright.
        return self._causal_rows[start : start + width, :window][None, None]

    def _forward(self, tokens):
        width = tokens.shape[1]
        cache_position = torch.arange(
            self.position, self.position + width, device=self.device
        )
        window = self.set_visible(self.bucket_for(self.position + width))
        out = self.model.transformer(
            tokens,
            past_key_values=self.cache,
            cache_position=cache_position,
            attention_mask=self.causal_mask(self.position, width, window),
            use_cache=True,
        )
        self.position += width
        return out.last_hidden_state

    def prefill(self, input_ids):
        """Consume a prompt from an empty cache; returns its last hidden state."""
        if input_ids.shape[0] != self.batch_size:
            raise ValueError(
                f"decoder built for batch {self.batch_size}, got {input_ids.shape[0]}"
            )
        return self._forward(input_ids)[:, -1, :]

    def step(self, tokens):
        """Advance the cache by `tokens` (B, T); returns the last hidden state."""
        return self._forward(tokens)[:, -1, :]

    def step_all(self, tokens):
        """Like `step`, but returns the hidden state at *every* fed position."""
        return self._forward(tokens)

    # -- head ---------------------------------------------------------------

    def logits(self, hidden, role=None):
        """LM head on `hidden` (B, H) -> (B, vocab) or (B, legal-range) if `role`."""
        if role is None:
            return F.linear(hidden, self._head_weight)
        lo, hi = LEGAL_TOKEN_RANGES[role]
        return F.linear(hidden, self._head_weight[lo:hi])

    def constrained_argmax(self, hidden, role):
        """Greedy constrained token for `role` -- the sliced-head fast path.

        Equivalent to `constrain_score_token_logits(full_logits, role).argmax(-1)`:
        every token outside `LEGAL_TOKEN_RANGES[role]` is -inf there, so the full
        row's argmax always lands inside the range, and `torch.argmax`'s
        first-maximum tie-break picks the same index either way.
        """
        lo, _ = LEGAL_TOKEN_RANGES[role]
        return self.logits(hidden, role=role).argmax(dim=-1) + lo


# ---------------------------------------------------------------------------
# CUDA-graph capture of one score slot
# ---------------------------------------------------------------------------


class _SlotGraph:
    """One captured CUDA graph covering a whole score slot (3 cached forwards).

    A slot is entirely tensor-parametric: the only things that change between
    slots are (a) the three teacher-forced control tokens that follow the pitch
    token, (b) the ground-truth tokens for `gt_ce`, and (c) the cache positions,
    which advance by exactly 6. (a) and (b) are copied into static buffers before
    replay; (c) are static buffers the graph increments itself, so a single
    capture serves all 138 slots of every window.

    Greedy only. Sampling inside a graph would draw from the graph-private
    philox stream, which cannot reproduce the eager `torch.multinomial` sequence
    token-for-token, so `rollout_score_slots_fast` keeps sampling on the eager
    path rather than silently changing what a seeded run emits.
    """

    def __init__(self, decoder, start, *, collect_gt_ce, window=None, pool=None):
        self.decoder = decoder
        self.collect_gt_ce = collect_gt_ce
        self.window = decoder.set_visible(window) if window else decoder.max_length
        model = decoder.model
        device, batch = decoder.device, decoder.batch_size

        self.hidden = torch.zeros(batch, model.config.n_embd, device=device, dtype=decoder.dtype)
        self.controls = torch.zeros(batch, 3, dtype=torch.long, device=device)
        self.targets = torch.zeros(batch, 3, dtype=torch.long, device=device)
        self.out_tokens = torch.zeros(batch, 3, dtype=torch.long, device=device)
        self.out_gt_ce = torch.zeros(batch, 3, device=device, dtype=torch.float32)
        # Cache positions for the slot's three forwards: [start], [start+1],
        # [start+2 .. start+5]. The graph adds 6 to each at the end of a replay.
        self.pos_a = torch.tensor([start], dtype=torch.long, device=device)
        self.pos_b = torch.tensor([start + 1], dtype=torch.long, device=device)
        self.pos_c = torch.arange(start + 2, start + 6, dtype=torch.long, device=device)
        self.start = start

        self.graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):  # warm cuBLAS/kernel selection before capture
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        # Every bucket's graph shares one memory pool: at batch 198 each capture
        # would otherwise reserve its own private pool for the same intermediates.
        with torch.cuda.graph(self.graph, pool=pool):
            self._body()

    def _cached_forward(self, tokens, cache_position):
        # Inside the graph the mask has to be *derived from* the live position
        # buffer, not sliced at capture time, or every replay would attend with
        # slot 0's mask. One index_select off the triangular table does that in a
        # single recorded kernel; letting transformers rebuild it with vmap would
        # record dozens.
        rows = self.decoder._causal_rows
        mask = None
        if rows is not None:
            mask = rows.index_select(0, cache_position)[:, : self.window][None, None]
        return self.decoder.model.transformer(
            tokens,
            past_key_values=self.decoder.cache,
            cache_position=cache_position,
            attention_mask=mask,
            use_cache=True,
        ).last_hidden_state

    def _body(self):
        decoder = self.decoder
        hidden = self.hidden
        tokens = []
        ce_cols = []
        for role in range(3):
            if self.collect_gt_ce:
                full = decoder.logits(hidden)
                ce_cols.append(F.cross_entropy(full.float(), self.targets[:, role], reduction="none"))
                lo, hi = LEGAL_TOKEN_RANGES[role]
                token = full[:, lo:hi].argmax(dim=-1) + lo
            else:
                token = decoder.constrained_argmax(hidden, role)
            tokens.append(token)
            if role == 0:
                hidden = self._cached_forward(token.unsqueeze(1), self.pos_a)[:, -1, :]
            elif role == 1:
                hidden = self._cached_forward(token.unsqueeze(1), self.pos_b)[:, -1, :]
        chunk = torch.cat([tokens[2].unsqueeze(1), self.controls], dim=1)
        hidden = self._cached_forward(chunk, self.pos_c)[:, -1, :]

        self.out_tokens.copy_(torch.stack(tokens, dim=1))
        if self.collect_gt_ce:
            self.out_gt_ce.copy_(torch.stack(ce_cols, dim=1))
        self.hidden.copy_(hidden)
        self.pos_a.add_(6)
        self.pos_b.add_(6)
        self.pos_c.add_(6)

    def rewind_positions(self, start):
        """Point the graph's cache positions back at slot `start` (a new window)."""
        self.pos_a.fill_(start)
        self.pos_b.fill_(start + 1)
        self.pos_c.copy_(torch.arange(start + 2, start + 6, device=self.pos_c.device))

    def replay(self, hidden, controls, targets=None):
        self.hidden.copy_(hidden)
        self.controls.copy_(controls)
        if self.collect_gt_ce and targets is not None:
            self.targets.copy_(targets)
        self.graph.replay()
        return self.hidden, self.out_tokens, self.out_gt_ce


# ---------------------------------------------------------------------------
# The fast rollout
# ---------------------------------------------------------------------------


def body_score_slot_starts(sequence_length):
    """Positions of body score triplets with a full control triplet after them."""
    return [
        pos
        for pos in iter_score_slot_positions(sequence_length, ALTERNATING_START)
        if pos + 5 < sequence_length
    ]


@torch.no_grad()
def rollout_score_slots_fast(
    model,
    input_ids,
    *,
    targets=None,
    temperature=1.0,
    constrain=True,
    collect_logprobs=False,
    collect_gt_ce=True,
    autocast_ctx=None,
    decoder=None,
    static_cache=True,
    sliced_head=True,
    cuda_graph=False,
    prebuilt_mask=True,
    exact_chunk_logits=False,
    buckets=None,
):
    """Drop-in replacement for `onpolicy_rollout.rollout_score_slots`.

    Returns the same dict (`rolled`, `positions`, `logprob`, `gt_ce`, `valid`)
    and honours the same decoding semantics. The switches select how much of the
    fast path is used, so each can be ablated independently:

    static_cache -- preallocate the KV cache (`StaticCache`) instead of growing a
                    `DynamicCache` by concatenation every step. Required for
                    `cuda_graph`. Attention then always runs over the full
                    `max_length`, which costs bandwidth at large batch and buys
                    stable addresses at small batch; measure before assuming.
    sliced_head  -- apply the LM head only over the role's legal vocab range.
                    Ignored when `collect_gt_ce`/`collect_logprobs` need the
                    full-vocab normalizer.
    cuda_graph   -- capture a whole score slot into one CUDA graph. Greedy only
                    (see `_SlotGraph`); sampling falls back to the eager path.
    prebuilt_mask-- hand transformers the causal mask instead of letting it
                    rebuild one per step with `torch.vmap` (see
                    `StaticKVDecoder.__init__`). Only meaningful with
                    `static_cache`, which is what forces the mask to exist.
    exact_chunk_logits
                 -- run the LM head over *every* position of the prefill and of
                    the pitch+controls chunk, as the baseline does, instead of
                    only the position whose logits are read. Costs the 192-wide
                    and 4-wide head passes back (at batch 198 the prefill head
                    alone is an 8.4 GiB fp32 tensor) and buys nothing for the
                    emitted tokens, which are bit-identical either way. It
                    exists for `collect_gt_ce=True` callers only: cuBLAS
                    accumulates differently at different GEMM shapes, so a
                    (batch, 1) head and a (batch, 4) head disagree in the last
                    few ulps, which moves the returned cross-entropies by up to
                    ~3e-4 absolute. The RL arms select checkpoints on a mean of
                    those, so the knob is there when that has to match exactly.
    buckets      -- number of KV-length buckets (or an explicit ascending tuple)
                    for `_WindowedStaticLayer`. A static cache otherwise attends
                    over all `max_length` positions every step even while the
                    real cache is far shorter; bucketing gives that bandwidth
                    back at the cost of one captured graph per bucket. `buckets=8`
                    caps the waste at ~128 positions. Off by default.

    `constrain=False` is not supported by the fast path -- every caller in the
    repo passes True, and an unconstrained decode has no contiguous head slice
    to exploit, so it falls back rather than silently mis-decoding.
    """
    if not constrain:
        raise ValueError("fast decode requires constrain=True; use rollout_score_slots")
    sampling = temperature is not None and temperature > 0
    if cuda_graph and (sampling or collect_logprobs):
        raise ValueError(
            "cuda_graph fast decode is greedy-only: a graph-private RNG stream cannot "
            "reproduce eager torch.multinomial token-for-token"
        )
    if cuda_graph and not static_cache:
        raise ValueError("cuda_graph requires static_cache=True (fixed KV addresses)")
    if cuda_graph and exact_chunk_logits:
        raise ValueError("exact_chunk_logits is an eager-path knob; drop cuda_graph for it")

    was_training = model.training
    model.eval()
    ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext

    device = input_ids.device
    batch, length = input_ids.shape
    starts = body_score_slot_starts(length)
    if not starts:
        raise ValueError(f"No body score slots in a sequence of length {length}.")
    if targets is None:
        targets = input_ids

    # The full-vocab normalizer is needed for both of these, so the sliced head
    # (which never materializes the illegal columns) cannot serve them.
    need_full_logits = collect_gt_ce or collect_logprobs or sampling
    use_sliced = sliced_head and not need_full_logits

    rolled = input_ids.clone()
    logprob_cols, gt_ce_cols, valid_cols = [], [], []
    past = None

    def advance(tokens):
        """One cached forward; returns whatever the head should be applied to.

        `(batch, hidden)` normally -- the head is only read at the last position
        -- or the whole `(batch, tokens, hidden)` block under
        `exact_chunk_logits`, which reproduces the baseline's head GEMM shape.
        """
        nonlocal past
        if static_cache:
            block = decoder.step_all(tokens)
        else:
            out = model.transformer(tokens, past_key_values=past, use_cache=True)
            past = out.past_key_values
            block = out.last_hidden_state
        return block if exact_chunk_logits else block[:, -1, :]

    try:
        with ctx():
            if static_cache:
                if decoder is None:
                    decoder = StaticKVDecoder(
                        model,
                        batch,
                        max_length=length,
                        prebuilt_mask=prebuilt_mask,
                        buckets=buckets,
                    )
                decoder.reset()
            hidden = advance(input_ids[:, :ALTERNATING_START])

            slot_graphs = {}
            if cuda_graph:
                # Capture once per decoder, not once per rollout: a capture is
                # seconds of work and is valid for every window of this batch
                # width, since nothing about a slot depends on window content.
                # One graph per KV bucket (one total when bucketing is off).
                cache = getattr(decoder, "_slot_graphs", None)
                if cache is None:
                    cache = decoder._slot_graphs = {}
                pool = getattr(decoder, "_graph_pool", None)
                # Capture every bucket *before* the rollout starts: a capture runs
                # warmup slot bodies against the live cache, so capturing lazily
                # mid-rollout would overwrite entries the rollout still needs.
                captured = False
                for needed in sorted({decoder.bucket_for(start + 6) for start in starts}):
                    key = (needed, collect_gt_ce)
                    if key not in cache:
                        if pool is None:
                            pool = decoder._graph_pool = torch.cuda.graph_pool_handle()
                        cache[key] = _SlotGraph(
                            decoder,
                            starts[0],
                            collect_gt_ce=collect_gt_ce,
                            window=needed,
                            pool=pool,
                        )
                        captured = True
                    slot_graphs[needed] = cache[key]
                if captured:
                    # Only redo the prefill when a capture actually happened --
                    # on every rollout after the first the graphs are already
                    # there and the cache the prefill above left is still good.
                    decoder.reset()
                    hidden = decoder.prefill(input_ids[:, :ALTERNATING_START])
                decoder.position = length  # the graphs, not `decoder`, drive the cache now

            active_graph = None
            for start in starts:
                slot_valid = targets[:, start + 2] != REST

                if slot_graphs:
                    slot_graph = slot_graphs[decoder.bucket_for(start + 6)]
                    if slot_graph is not active_graph:
                        # Each graph advances its own position buffers by 6 per
                        # replay, so a graph taking over mid-rollout has to be
                        # told where the rollout actually is.
                        slot_graph.rewind_positions(start)
                        active_graph = slot_graph
                    hidden, out_tokens, out_gt_ce = slot_graph.replay(
                        hidden,
                        input_ids[:, start + 3 : start + 6],
                        targets[:, start : start + 3] if collect_gt_ce else None,
                    )
                    rolled[:, start : start + 3] = out_tokens
                    if collect_gt_ce:
                        # `.clone()` is load-bearing: out_gt_ce is a static graph
                        # output buffer that the next replay overwrites, so
                        # keeping views of it would leave all 138 slots holding
                        # the *last* slot's CE (observed as a 46.4 max deviation
                        # from the reference before this clone was added).
                        gt_ce_cols.extend(out_gt_ce[:, role].clone() for role in range(3))
                    valid_cols.extend([slot_valid] * 3)
                    continue

                sampled = []
                for role in range(3):
                    if use_sliced:
                        lo, hi = LEGAL_TOKEN_RANGES[role]
                        token = (
                            _head(hidden, model.lm_head.weight[lo:hi]).argmax(dim=-1) + lo
                        )
                    else:
                        logits = _head(hidden, model.lm_head.weight).float()
                        if collect_gt_ce:
                            gt_ce_cols.append(
                                F.cross_entropy(logits, targets[:, start + role], reduction="none")
                            )
                        policy_logits = logits.masked_fill(
                            role_illegal_mask(role, device, logits.shape[-1]), -float("inf")
                        )
                        if sampling:
                            policy_logits = policy_logits / temperature
                            token = torch.multinomial(
                                torch.softmax(policy_logits, dim=-1), num_samples=1
                            ).squeeze(1)
                        else:
                            token = policy_logits.argmax(dim=-1)
                        if collect_logprobs:
                            logprob_cols.append(
                                torch.log_softmax(policy_logits, dim=-1)
                                .gather(1, token.unsqueeze(1))
                                .squeeze(1)
                            )
                    valid_cols.append(slot_valid)
                    sampled.append(token)

                    if role < 2:
                        hidden = advance(token.unsqueeze(1))

                chunk = torch.cat(
                    [sampled[2].unsqueeze(1), input_ids[:, start + 3 : start + 6]], dim=1
                )
                hidden = advance(chunk)

                for role in range(3):
                    rolled[:, start + role] = sampled[role]
    finally:
        model.train(was_training)

    positions = [start + role for start in starts for role in range(3)]
    return {
        "rolled": rolled,
        "positions": torch.tensor(positions, dtype=torch.long, device=device),
        "logprob": torch.stack(logprob_cols, dim=1) if collect_logprobs else None,
        "gt_ce": torch.stack(gt_ce_cols, dim=1) if collect_gt_ce else None,
        "valid": torch.stack(valid_cols, dim=1),
    }


def _head(hidden, weight):
    """LM head on `hidden`, always returning the last position's row (batch, N).

    Accepts (batch, hidden) -- the trimmed default -- or (batch, tokens, hidden)
    under `exact_chunk_logits`, where the wider GEMM is the whole point.
    """
    out = F.linear(hidden, weight)
    return out[:, -1, :] if out.dim() == 3 else out
