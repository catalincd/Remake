"""Deep MLP over the engineered feature vector.

The neural counterpart to the tree models — same inputs (entropy / histogram /
structural densities), but a differentiable model that can be ensembled with the
byte models and feeds the stacking meta-learner. Trains in seconds per epoch.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import register


class FeatureMLP(nn.Module):
    def __init__(self, num_classes, input_dim, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        self.in_norm = nn.BatchNorm1d(input_dim)
        layers = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(),
                       nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, num_classes)

    def forward(self, x):
        return self.head(self.body(self.in_norm(x)))


@register("feature_mlp", kind="nn", input="features",
          description="Deep MLP on engineered features (entropy/hist/structural)")
def build(num_classes, input_dim, **kw):
    assert input_dim is not None, "feature_mlp needs input_dim"
    return FeatureMLP(num_classes, input_dim, **kw)
