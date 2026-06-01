"""Shared NN building blocks for the byte-sequence models."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool1d(nn.Module):
    """Learned additive attention pooling over time. (B, d, T) -> (B, d)."""
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.permute(0, 2, 1)                          # (B, T, d)
        w = torch.softmax(self.score(h), dim=1)         # (B, T, 1)
        return (h * w).sum(1)                            # (B, d)


class ByteStem(nn.Module):
    """Embedding + optional strided conv downsampling.

    Maps (B, L) byte ids -> (B, C, T) where T = L // (stride ** n_down). Used by
    every byte model so they share the same front end and the same notion of
    'token'. n_down strided convs each halve... actually each divides by `stride`.
    """
    def __init__(self, embed_dim=16, channels=128, n_down=4, stride=2,
                 kernel=7):
        super().__init__()
        self.embed = nn.Embedding(256, embed_dim)
        layers = [nn.Conv1d(embed_dim, channels, kernel, padding=kernel // 2),
                  nn.BatchNorm1d(channels), nn.GELU()]
        for _ in range(n_down):
            layers += [nn.Conv1d(channels, channels, kernel, stride=stride,
                                 padding=kernel // 2),
                       nn.BatchNorm1d(channels), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embed(x).permute(0, 2, 1)              # (B, embed, L)
        return self.net(e)                               # (B, C, T)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
