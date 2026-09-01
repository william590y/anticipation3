"""Pairwise-INPUT duel comparator + knockout tournament (arXiv:2501.13007).

PairJudge RM's thesis is that a judge which sees BOTH candidates at once
beats any scheme that scores candidates independently and compares scalars
(their Fig. 4: two near-identical correct solutions scored 0.0006 vs 0.9999
by a pointwise critic). Our pairwise32 reranker is still pointwise in that
sense -- the RankNet loss is pairwise, but each forward sees one candidate.
This model puts both in one forward.

Input layout (1434 tokens): the packed 1020-token window with candidate A
substituted into its score slots, then candidate B's 414 score tokens
appended, with a learned segment embedding separating the two. Output is a
single logit = log-odds that A beats B.

Three deliberate deviations from the paper, each forced by the task:

1. RELATIVE, not absolute. Their judge emits two independent correctness
   labels, which presumes a verifiable right answer. Transcription F1 is
   graded, so "both correct"/"both incorrect" would be the common case --
   and in their code both-incorrect eliminates BOTH candidates, which
   without team structure can empty the pool into a random-resurrection
   fallback. A relative judge always advances exactly one candidate, so the
   bracket cannot collapse.
2. ORDER-SYMMETRISED. The paper has no position-bias handling at all, and
   its released code breaks both-correct ties by always advancing slot A.
   We score every match in both orders and use the antisymmetric part,
   s(A,B) = (logit(A,B) - logit(B,A)) / 2, so the verdict cannot depend on
   which candidate was written first.
3. NO TEAM GROUPING. Teams (candidates sharing an exact answer) are what
   make their tournament sub-O(N); distinct sampled transcriptions are
   almost never token-identical, so every match would cost a call anyway.
   We keep the exact-duplicate check because it is free, but expect it to
   fire rarely.

Margin-weighted BCE replaces their plain SFT cross-entropy for the same
reason as (1): a pair whose F1 gap is 0.4 should matter more than one whose
gap is 0.01.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from anticipation.vocab import VOCAB_SIZE


@dataclass
class DuelConfig:
    vocab_size: int = VOCAB_SIZE
    win_len: int = 1020
    cand_len: int = 414
    dim: int = 512
    depth: int = 6
    heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    token_features: int = 0     # per-token [log p_FT, log p_base], 0 = off
    feat_clamp: float = 30.0

    @property
    def seq_len(self) -> int:
        return self.win_len + self.cand_len

    def to_dict(self) -> dict:
        return asdict(self)


class DuelComparator(nn.Module):
    def __init__(self, cfg: DuelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = nn.Parameter(torch.randn(1, cfg.seq_len, cfg.dim) * 0.02)
        self.segment = nn.Parameter(torch.randn(2, cfg.dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            cfg.dim, cfg.heads, cfg.dim_feedforward, dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.depth)
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Sequential(
            nn.Linear(cfg.dim, 256), nn.GELU(), nn.Linear(256, 1))
        if cfg.token_features:
            from onpolicy_rollout import score_token_positions
            self.feat_proj = nn.Sequential(
                nn.Linear(cfg.token_features, cfg.dim), nn.GELU(),
                nn.Linear(cfg.dim, cfg.dim))
            self.register_buffer(
                "score_pos", score_token_positions(cfg.win_len),
                persistent=False)

    def forward(self, tokens_a: torch.Tensor, cand_b: torch.Tensor,
                feats_a: torch.Tensor | None = None,
                feats_b: torch.Tensor | None = None) -> torch.Tensor:
        """(B,1020) window-with-A + (B,414) B tokens -> (B,) logit(A beats B)."""
        cfg = self.cfg
        seq = torch.cat([tokens_a, cand_b.to(tokens_a.dtype)], dim=1)
        h = self.embed(seq) + self.pos[:, :seq.shape[1]]
        seg = torch.cat([
            self.segment[0].expand(cfg.win_len, -1),
            self.segment[1].expand(cfg.cand_len, -1)], dim=0)
        h = h + seg.unsqueeze(0)
        if cfg.token_features and feats_a is not None:
            f = (feats_a.float().clamp(-cfg.feat_clamp, cfg.feat_clamp)
                 / cfg.feat_clamp)
            h = h.index_add(1, self.score_pos, self.feat_proj(f).to(h.dtype))
        if cfg.token_features and feats_b is not None:
            f = (feats_b.float().clamp(-cfg.feat_clamp, cfg.feat_clamp)
                 / cfg.feat_clamp)
            tail = torch.arange(cfg.win_len, cfg.seq_len, device=h.device)
            h = h.index_add(1, tail, self.feat_proj(f).to(h.dtype))
        h = self.norm(self.encoder(h)).mean(dim=1)
        return self.head(h).squeeze(-1)


@torch.no_grad()
def duel_scores(model, window, cands, feats, pairs, device, chunk=8,
                substitute=None, flat_pos=None):
    """Order-symmetrised match scores for a list of (i, j) candidate pairs.

    Returns a tensor s where s[k] > 0 means candidate pairs[k][0] wins.
    """
    out = torch.empty(len(pairs))
    for lo in range(0, len(pairs), chunk):
        part = pairs[lo:lo + chunk]
        ia = torch.tensor([p[0] for p in part], dtype=torch.long)
        ib = torch.tensor([p[1] for p in part], dtype=torch.long)
        win = window.expand(len(part), -1)
        tok_a = substitute(win, cands[ia].to(device), flat_pos)
        tok_b = substitute(win, cands[ib].to(device), flat_pos)
        fa = feats[ia].to(device) if feats is not None else None
        fb = feats[ib].to(device) if feats is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()):
            ab = model(tok_a, cands[ib].to(device), fa, fb).float()
            ba = model(tok_b, cands[ia].to(device), fb, fa).float()
        out[lo:lo + len(part)] = ((ab - ba) / 2).cpu()
    return out


def knockout(n_cands, match_fn, rng: random.Random, dup_key=None):
    """Single-elimination tournament over range(n_cands) -> (winner, log).

    Follows the paper's procedure: no seeding, the surviving pool is
    RESHUFFLED every round and paired greedily, an odd pool advances one
    random candidate on a bye, and the loop runs until one survives (at most
    ceil(log2 N) rounds, <= N-1 matches). `match_fn(i, j)` returns True when
    i wins. `dup_key(i)` (optional) lets exactly-identical candidates skip
    the comparator, the residue of their free same-team matches.
    """
    pool = list(range(n_cands))
    log = []
    rnd = 0
    while len(pool) > 1:
        rng.shuffle(pool)
        nxt, k = [], 0
        if len(pool) % 2 == 1:
            bye = rng.randrange(len(pool))
            nxt.append(pool.pop(bye))
            log.append({"round": rnd, "bye": nxt[-1]})
        while k + 1 < len(pool):
            i, j = pool[k], pool[k + 1]
            if dup_key is not None and dup_key(i) == dup_key(j):
                win, free = (i if rng.random() < 0.5 else j), True
            else:
                win, free = (i if match_fn(i, j) else j), False
            nxt.append(win)
            log.append({"round": rnd, "a": i, "b": j, "winner": win,
                        "free": free})
            k += 2
        pool, rnd = nxt, rnd + 1
    return pool[0], log
