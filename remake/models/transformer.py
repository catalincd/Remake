"""Small byte Transformer encoder.

Raw 4096-byte sequences are too long for full attention, so a strided conv stem
downsamples to a short token sequence (default L/16 = 256 tokens) before a few
standard pre-norm encoder layers. Mirrors the "transformer-on-bytes" line of
file-fragment work, kept small for the 6750 XT.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import register
from .base import ByteStem, AttentionPool1d


class ByteTransformer(nn.Module):
    def __init__(self, num_classes, embed_dim=16, d_model=192, n_down=4,
                 nhead=6, n_layers=4, ff_mult=4, dropout=0.1, max_tokens=512):
        super().__init__()
        self.stem = ByteStem(embed_dim, d_model, n_down=n_down, stride=2)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * ff_mult, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttentionPool1d(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        h = self.stem(x).permute(0, 2, 1)                    # (B, T, d)
        T = h.shape[1]
        h = h + self.pos[:, :T]
        h = self.norm(self.encoder(h))
        return self.head(self.pool(h.permute(0, 2, 1)))


@register("transformer", kind="nn", input="bytes",
          description="Conv-downsampled byte Transformer encoder")
def build(num_classes, input_dim=None, **kw):
    return ByteTransformer(num_classes, **kw)
