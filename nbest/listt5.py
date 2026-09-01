"""ListT5-style listwise reranking with Fusion-in-Decoder (arXiv:2402.15838).

Ingests a LIST of candidates in one forward: each of m candidates gets its own
encoder pass over the packed window with that candidate substituted, the m
encoder outputs are concatenated along the sequence axis, and a small decoder
cross-attends over all of them at once. Listwise reasoning happens exactly
where ListT5 puts it -- in the decoder's cross-attention over the fused
memory -- rather than in a pooled scalar per candidate.

Faithful pieces:
  * m = 5 per pass (their ablation: 5 beats 10).
  * The decoder emits a full PERMUTATION of the m identifiers in INCREASING
    relevance -- best LAST. Worst-first was worth 1-2 NDCG over best-first in
    their Table, and emitting the whole permutation beat emitting only the
    winner by 3 points: ordering the losers is extra supervision.
  * Candidates are identified by a learned INDEX embedding, not by position.
    In ListT5 every passage occupies identical relative positions inside its
    own encoder pass, so the identifier is the only thing distinguishing
    them; that is what kills lost-in-the-middle bias. We add a per-slot index
    embedding to every token of that slot's pass for the same reason.
  * Groups are shuffled before identifiers are assigned (training), so no
    identifier correlates with quality.
  * m-ary TOURNAMENT SORT for lists longer than m, with the same
    consecutive-chunk grouping between levels -- O(n) passes, 10 for our 33
    candidates vs 33 pointwise forwards.

Adaptations: our "query" is the performance stream, which lives in the same
1020-token window as the candidate, so a pass is the window-with-candidate
rather than a query+passage concatenation. Their `+21` leaf-replacement hack
is unnecessary because we only ever need the champion (r=1), never top-k
extraction, so no leaf is ever repaired.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from anticipation.vocab import VOCAB_SIZE


@dataclass
class ListwiseFiDConfig:
    vocab_size: int = VOCAB_SIZE
    seq_len: int = 1020
    dim: int = 512
    depth: int = 6
    heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    m: int = 5
    dec_depth: int = 2
    token_features: int = 0
    feat_clamp: float = 30.0

    def to_dict(self) -> dict:
        return asdict(self)


class ListwiseFiD(nn.Module):
    def __init__(self, cfg: ListwiseFiDConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = nn.Parameter(torch.randn(1, cfg.seq_len, cfg.dim) * 0.02)
        self.index_emb = nn.Parameter(torch.randn(cfg.m, cfg.dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            cfg.dim, cfg.heads, cfg.dim_feedforward, dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.depth)
        self.enc_norm = nn.LayerNorm(cfg.dim)
        dec_layer = nn.TransformerDecoderLayer(
            cfg.dim, cfg.heads, cfg.dim_feedforward, dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.dec_depth)
        self.dec_embed = nn.Embedding(cfg.m + 1, cfg.dim)   # 0 = BOS
        self.dec_pos = nn.Parameter(torch.randn(1, cfg.m + 1, cfg.dim) * 0.02)
        self.dec_norm = nn.LayerNorm(cfg.dim)
        self.out = nn.Linear(cfg.dim, cfg.m)
        if cfg.token_features:
            from onpolicy_rollout import score_token_positions
            self.feat_proj = nn.Sequential(
                nn.Linear(cfg.token_features, cfg.dim), nn.GELU(),
                nn.Linear(cfg.dim, cfg.dim))
            self.register_buffer(
                "score_pos", score_token_positions(cfg.seq_len),
                persistent=False)

    def encode_group(self, tokens, feats=None):
        """(B, m, L) -> fused memory (B, m*L, dim). One encoder pass per slot
        (folded into the batch axis), concatenated along the sequence axis --
        Fusion-in-Decoder."""
        cfg = self.cfg
        B, m, L = tokens.shape
        flat = tokens.reshape(B * m, L)
        h = self.embed(flat) + self.pos[:, :L]
        if cfg.token_features and feats is not None:
            f = (feats.reshape(B * m, -1, cfg.token_features).float()
                 .clamp(-cfg.feat_clamp, cfg.feat_clamp) / cfg.feat_clamp)
            h = h.index_add(1, self.score_pos, self.feat_proj(f).to(h.dtype))
        # Row k of the folded batch is slot k % m; the identifier embedding is
        # added to every token of that slot's pass, so candidates are told
        # apart by identity rather than by position.
        slot = torch.arange(m, device=tokens.device).repeat(B)
        h = h + self.index_emb[slot].unsqueeze(1)
        h = self.enc_norm(self.encoder(h))
        return h.reshape(B, m * L, cfg.dim)

    def decode(self, memory, dec_in):
        """(B, m*L, dim) memory + (B, T) symbol ids -> (B, T, m) logits."""
        T = dec_in.shape[1]
        d = self.dec_embed(dec_in) + self.dec_pos[:, :T]
        mask = torch.triu(torch.ones(T, T, device=dec_in.device,
                                     dtype=torch.bool), diagonal=1)
        h = self.decoder(d, memory, tgt_mask=mask)
        return self.out(self.dec_norm(h))

    def forward(self, tokens, target, feats=None):
        """Teacher-forced: target (B, m) is the worst->best permutation."""
        memory = self.encode_group(tokens, feats)
        bos = torch.zeros_like(target[:, :1])
        dec_in = torch.cat([bos, target[:, :-1] + 1], dim=1)
        return self.decode(memory, dec_in)

    @torch.no_grad()
    def generate(self, tokens, feats=None):
        """Greedy decode of the permutation; returns (B, m) slot ids,
        worst first. Already-emitted slots are masked out, which is the
        constrained decoding ListT5 says it lacks (they parse free text and
        fall back to the input order on malformed output)."""
        cfg = self.cfg
        B, m = tokens.shape[0], tokens.shape[1]
        memory = self.encode_group(tokens, feats)
        dec_in = torch.zeros(B, 1, dtype=torch.long, device=tokens.device)
        used = torch.zeros(B, cfg.m, dtype=torch.bool, device=tokens.device)
        out = []
        for _ in range(m):
            logits = self.decode(memory, dec_in)[:, -1]
            logits = logits.masked_fill(used, -float("inf"))
            if m < cfg.m:                    # short group: unusable slots
                logits[:, m:] = -float("inf")
            pick = logits.argmax(-1)
            used.scatter_(1, pick.unsqueeze(1), True)
            out.append(pick)
            dec_in = torch.cat([dec_in, (pick + 1).unsqueeze(1)], dim=1)
        return torch.stack(out, dim=1)


def tournament_sort(n, m, pick_best, pad_from=None):
    """ListT5's m-ary tournament: chunk into groups of m, keep each group's
    champion, repeat on the survivors until one remains.

    `pick_best(indices)` receives up to m candidate indices and returns the
    winning index. Short groups are padded deterministically by repeating the
    group's own first element (ListT5 pads with unused passages and allows
    duplicates when fewer than m remain). Returns (winner, n_passes).
    """
    pool, passes = list(range(n)), 0
    while len(pool) > 1:
        nxt = []
        for s in range(0, len(pool), m):
            group = pool[s:s + m]
            if len(group) == 1:
                nxt.append(group[0])         # free bye, no forward pass
                continue
            nxt.append(pick_best(group))
            passes += 1
        pool = nxt
    return pool[0], passes
