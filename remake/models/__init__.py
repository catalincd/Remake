"""Importing this package registers every model.

NN modules import torch at module top; if torch is unavailable (e.g. listing
models on a host without it) we skip them rather than crash, so the tree-only
path still works.
"""
import importlib
import warnings

# (module, needs_torch)
_MODULES = [
    ("trees", False),
    ("cnn_bigru", True),
    ("tcn", True),
    ("transformer", True),
    ("mamba", True),
    ("gnn", True),
    ("feature_mlp", True),
]

for mod, needs_torch in _MODULES:
    try:
        importlib.import_module(f".{mod}", __package__)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"model module '{mod}' not loaded: {e}")
