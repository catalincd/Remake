"""Pure-PyTorch Mamba / selective state-space model (S6).

The official `mamba-ssm` relies on custom CUDA kernels that don't build on ROCm,
so this is a self-contained reference implementation: a selective SSM with a
sequential scan over the (downsampled) token axis. Bytes are first reduced to a
short token sequence by a strided conv stem (default 256 tokens) so the
O(T) Python scan stays cheap and runs on the 6750 XT.

Selective scan recurrence (per token t):
    h_t = exp(Δ_t ⊙ A) · h_{t-1} + (Δ_t ⊙ B_t) · x_t
    y_t = C_t · h_t + D ⊙ x_t
with Δ_t, B_t, C_t produced from the input (input-dependent = "selective").
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register
from .base import ByteStem, AttentionPool1d


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):                                    # x: (B, T, d_model)
        B, T, _ = x.shape
        xz = self.in_proj(x)                                 # (B, T, 2*d_inner)
        xi, z = xz.chunk(2, dim=-1)
        # depthwise causal conv over time
        xc = self.conv1d(xi.transpose(1, 2))[..., :T].transpose(1, 2)
        xc = F.silu(xc)                                      # (B, T, d_inner)

        dbl = self.x_proj(xc)                                # (B, T, 2*d_state + d_inner)
        dt, Bm, Cm = torch.split(dbl, [self.d_inner, self.d_state, self.d_state], -1)
        dt = F.softplus(self.dt_proj(dt))                    # (B, T, d_inner)
        A = -torch.exp(self.A_log.float())                   # (d_inner, d_state)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=xc.dtype)
        ys = []
        for t in range(T):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)       # (B, d_inner, d_state)
            dBx = (dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1)) * xc[:, t].unsqueeze(-1)
            h = dA * h + dBx
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))   # (B, d_inner)
        y = torch.stack(ys, dim=1)                            # (B, T, d_inner)
        y = y + xc * self.D
        return self.out_proj(y * F.silu(z))


class MambaClassifier(nn.Module):
    def __init__(self, num_classes, embed_dim=16, d_model=128, n_down=4,
                 n_blocks=4, d_state=16, dropout=0.1):
        super().__init__()
        self.stem = ByteStem(embed_dim, d_model, n_down=n_down, stride=2)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state) for _ in range(n_blocks)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_blocks)])
        self.drop = nn.Dropout(dropout)
        self.pool = AttentionPool1d(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        h = self.stem(x).permute(0, 2, 1)                    # (B, T, d)
        for blk, nrm in zip(self.blocks, self.norms):
            h = h + self.drop(blk(nrm(h)))
        return self.head(self.pool(h.permute(0, 2, 1)))


@register("mamba", kind="nn", input="bytes",
          description="Pure-PyTorch selective SSM (Mamba/S6) over downsampled bytes")
def build(num_classes, input_dim=None, **kw):
    return MambaClassifier(num_classes, **kw)
