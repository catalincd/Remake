"""Reference model: multi-scale byte CNN + BiGRU + attention pool.

A compact re-implementation of the previous project's backbone (the thing that
hit the 87.8% wall) so we have an apples-to-apples baseline on the *corrected*
labels. Deliberately small (smaller-models-first).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register
from .base import AttentionPool1d, count_params


class CNNBiGRU(nn.Module):
    def __init__(self, num_classes, embed_dim=16, channels=96, gru_hidden=128,
                 gru_layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(256, embed_dim)
        # two parallel n-gram widths, then progressive pooling
        self.b9 = nn.Conv1d(embed_dim, channels, 9, padding=4)
        self.b27 = nn.Conv1d(embed_dim, channels, 27, padding=13)
        self.bn1 = nn.BatchNorm1d(channels * 2)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(channels * 2, channels * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels * 2)
        self.pool2 = nn.MaxPool1d(4)
        self.gru = nn.GRU(channels * 2, gru_hidden, gru_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if gru_layers > 1 else 0.0)
        self.pool = AttentionPool1d(gru_hidden * 2)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden * 2, num_classes)

    def forward(self, x):
        e = self.embed(x).permute(0, 2, 1)                  # (B, embed, L)
        h = torch.cat([self.b9(e), self.b27(e)], dim=1)     # (B, 2C, L)
        h = self.pool1(F.gelu(self.bn1(h)))
        h = self.pool2(F.gelu(self.bn2(self.conv2(h))))     # (B, 2C, L/16)
        h, _ = self.gru(h.permute(0, 2, 1))                 # (B, T, 2H)
        h = self.pool(h.permute(0, 2, 1))                   # (B, 2H)
        return self.head(self.drop(h))


@register("cnn_bigru", kind="nn", input="bytes",
          description="Multi-scale byte CNN + BiGRU + attention pool (reference baseline)")
def build(num_classes, input_dim=None, **kw):
    m = CNNBiGRU(num_classes, **kw)
    return m
