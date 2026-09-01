"""Small discriminative reranker q_phi(x, y) ~ F1(y, y*).

Input is the full packed 1020-token window with the CANDIDATE's score triplets
substituted into the body score slots, so the model jointly sees the
performance controls and the candidate score. Architecture: token embedding
(full 55028 vocab) + learned positional embedding + 6-layer pre-norm
transformer encoder + masked mean-pool + MLP head with sigmoid (F1 lives in
[0, 1]). ~45M parameters -- "small" next to the 420M generator.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from anticipation.vocab import VOCAB_SIZE


@dataclass
class RerankerConfig:
    vocab_size: int = VOCAB_SIZE
    seq_len: int = 1020
    dim: int = 512
    depth: int = 6
    heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    # Per-score-token generative features added to the embedding at the score
    # slots: [log p_FT(token | full interleaving), log p_base(token | score
    # tokens only)]. 0 keeps the token-only model. The token identities alone
    # cannot express HOW SURPRISED each model was at each token, which is
    # exactly the signal the summed logp features carried between windows but
    # could not localise within one.
    token_features: int = 0
    # Log-probs are unbounded below; squash before projecting so one
    # catastrophic token cannot dominate the sequence.
    feat_clamp: float = 30.0

    def to_dict(self) -> dict:
        return asdict(self)


class Reranker(nn.Module):
    def __init__(self, cfg: RerankerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = nn.Parameter(torch.randn(1, cfg.seq_len, cfg.dim) * 0.02)
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
                "score_pos", score_token_positions(cfg.seq_len), persistent=False)

    def embed_tokens(self, tokens, feats=None):
        h = self.embed(tokens) + self.pos[:, :tokens.shape[1]]
        if feats is not None and self.cfg.token_features:
            f = feats.float().clamp(-self.cfg.feat_clamp, self.cfg.feat_clamp)
            f = f / self.cfg.feat_clamp
            h = h.index_add(1, self.score_pos, self.feat_proj(f).to(h.dtype))
        return h

    def forward(self, tokens: torch.Tensor, feats: torch.Tensor | None = None
                ) -> torch.Tensor:
        """(B, 1020) long [+ (B, 414, F) features] -> (B,) predicted F1."""
        h = self.norm(self.encoder(self.embed_tokens(tokens, feats))).mean(dim=1)
        return torch.sigmoid(self.head(h).squeeze(-1))


def build_reranker_from_ckpt(ckpt_path: str, device) -> Reranker:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = Reranker(RerankerConfig(**ck["model_cfg"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def substitute_candidates(window_tokens: torch.Tensor,
                          cand_tokens: torch.Tensor,
                          flat_positions: torch.Tensor) -> torch.Tensor:
    """(B,1020) int windows + (B,414) int16 candidates -> (B,1020) long."""
    out = window_tokens.to(torch.long).clone()
    out[:, flat_positions] = cand_tokens.to(torch.long)
    return out
