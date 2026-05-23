"""Path setup and small helpers shared across Glora.

Glora reuses the simulation environment, graph builder and CUDA-Graph
runner from the gnn-strategy project. This module wires those imports
through sys.path so that ``from gnn_strategy.env import SchedulingEnv``
keeps working even though Glora lives in a sibling directory.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Iterable

import numpy as np
import torch


_GLORA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_XKY_ROOT = os.path.dirname(_GLORA_ROOT)
_TCAS_ROOT = os.path.join(_XKY_ROOT, "TCAS")
_GNN_STRATEGY_ROOT = os.path.join(_TCAS_ROOT, "gnn-strategy")
_TCAS_EXAMPLES = os.path.join(_TCAS_ROOT, "examples")


def _add_paths(paths: Iterable[str]) -> None:
    for p in paths:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


_add_paths([
    _GNN_STRATEGY_ROOT,            # gives access to ``gnn_strategy``
    _TCAS_ROOT,                    # gives access to ``Opara`` and friends
    _TCAS_EXAMPLES,                # gives access to ``NCF`` for DeepFM
    _GLORA_ROOT,                   # gives access to ``glora`` itself
])


def glora_root() -> str:
    return _GLORA_ROOT


def artifacts_dir(*sub: str) -> str:
    path = os.path.join(_GLORA_ROOT, "artifacts", *sub)
    os.makedirs(path, exist_ok=True)
    return path


def set_global_seed(seed: int) -> None:
    """Best-effort deterministic seeding for torch + numpy + python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: torch.nn.Module, only_trainable: bool = False) -> int:
    if only_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def human_int(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if isinstance(n, float) else f"{n}{unit}"
        n = n / 1000
    return f"{n:.1f}T"
