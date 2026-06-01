"""Model registry — every architecture registers itself here so the CLI can
build any of them by name with a uniform interface.

A model declares:
  kind  : "nn"   -> a torch.nn.Module trained by trainer.py
          "tree" -> a fit/predict estimator trained by tree_trainer.py
  input : "bytes"    -> consumes raw uint8 fragments (B, L)
          "features" -> consumes a dense feature vector (B, F)

NN builders are called as build(num_classes, input_dim, **model_args) and return
an nn.Module. Tree builders are called as build(num_classes, **model_args) and
return an object with .fit(X, y) and .predict_proba(X).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class ModelSpec:
    name: str
    builder: Callable
    kind: str          # "nn" | "tree"
    input: str         # "bytes" | "features"
    description: str = ""


_REGISTRY: dict[str, ModelSpec] = {}


def register(name: str, kind: str, input: str, description: str = ""):
    def deco(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"model '{name}' already registered")
        _REGISTRY[name] = ModelSpec(name, fn, kind, input, description)
        return fn
    return deco


def get(name: str) -> ModelSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


def all_specs() -> dict[str, ModelSpec]:
    return dict(_REGISTRY)
