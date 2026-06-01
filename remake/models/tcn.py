"""Temporal Convolutional Network (dilated 1D residual convs).

No recurrence: an exponentially growing dilation stack gives a large receptive
field cheaply (good fit for ROCm — pure convs, fully parallel). Captures the
multi-scale byte structure a BiGRU gets from recurrence but faster.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register
from .base import AttentionPool1d


class TCNBlock(nn.Module):
    def __init__(self, ch, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.drop(F.gelu(self.bn1(self.conv1(x))))
        h = self.bn2(self.conv2(h))
        return F.gelu(x + self.drop(h))


class TCN(nn.Module):
    def __init__(self, num_classes, embed_dim=16, channels=128, kernel=5,
                 n_levels=8, stem_pool=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(256, embed_dim)
        self.stem = nn.Sequential(
            nn.Conv1d(embed_dim, channels, 7, padding=3),
            nn.BatchNorm1d(channels), nn.GELU(),
            nn.MaxPool1d(stem_pool),                         # L -> L/stem_pool
        )
        self.blocks = nn.ModuleList([
            TCNBlock(channels, kernel, dilation=2 ** i, dropout=dropout)
            for i in range(n_levels)
        ])
        self.pool = AttentionPool1d(channels)
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x):
        h = self.stem(self.embed(x).permute(0, 2, 1))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.pool(h))


@register("tcn", kind="nn", input="bytes",
          description="Dilated temporal conv net (large receptive field, no recurrence)")
def build(num_classes, input_dim=None, **kw):
    return TCN(num_classes, **kw)
