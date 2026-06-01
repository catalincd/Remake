"""GNN over the per-fragment byte-transition graph.

Each fragment is a graph of 256 nodes (byte values 0..255). Edges are the
observed byte->byte transitions (the normalised bigram matrix), so the *graph
structure is per-fragment* while node features are shared learnable byte
embeddings. Message passing mixes embeddings along the transition edges (both
directions), and a mass-weighted readout collapses to a graph vector.

Pure dense PyTorch (A @ H matmuls) — no torch-geometric / scatter kernels, so it
runs on ROCm. Captures transition structure independent of absolute position,
which is complementary to the CNN/SSM sequence view.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register


def transition_matrix(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, L) bytes -> (row-normalised A (B,256,256), node mass (B,256))."""
    B, L = x.shape
    idx = x[:, :-1].long() * 256 + x[:, 1:].long()           # (B, L-1)
    flat = torch.zeros(B, 256 * 256, device=x.device, dtype=torch.float32)
    flat.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float32))
    A = flat.view(B, 256, 256)
    A = A / A.sum(-1, keepdim=True).clamp(min=1.0)           # row-stochastic
    mass = torch.zeros(B, 256, device=x.device, dtype=torch.float32)
    mass.scatter_add_(1, x.long(), torch.ones_like(x, dtype=torch.float32))
    mass = mass / L
    return A, mass


class GraphLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w_out = nn.Linear(d, d, bias=False)             # along forward edges
        self.w_in = nn.Linear(d, d, bias=False)              # along reverse edges
        self.w_self = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, H, A):
        msg = self.w_out(torch.bmm(A, H)) + self.w_in(torch.bmm(A.transpose(1, 2), H))
        return F.gelu(self.norm(msg + self.w_self(H)))


class ByteGNN(nn.Module):
    def __init__(self, num_classes, d=96, n_layers=3, dropout=0.2):
        super().__init__()
        self.node_embed = nn.Embedding(256, d)
        self.mass_proj = nn.Linear(1, d)
        self.layers = nn.ModuleList([GraphLayer(d) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d, num_classes))

    def forward(self, x):
        B = x.shape[0]
        A, mass = transition_matrix(x)                       # (B,256,256), (B,256)
        ids = torch.arange(256, device=x.device)
        H = self.node_embed(ids)[None].expand(B, -1, -1)     # (B,256,d)
        H = H + self.mass_proj(mass.unsqueeze(-1))           # inject per-fragment mass
        for layer in self.layers:
            H = self.drop(layer(H, A))
        weighted = (H * mass.unsqueeze(-1)).sum(1)           # mass-weighted readout
        meanp = H.mean(1)
        return self.head(torch.cat([weighted, meanp], dim=-1))


@register("gnn", kind="nn", input="bytes",
          description="Message-passing GNN on the per-fragment byte-transition graph")
def build(num_classes, input_dim=None, **kw):
    return ByteGNN(num_classes, **kw)
