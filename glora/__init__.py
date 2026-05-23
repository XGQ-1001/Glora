"""Glora: Foundation GPU operator scheduler with Graph Transformer + LoRA.

Public modules:
- encoder:        Graph Transformer + Laplacian Positional Encoding
- lap_pe:         Laplacian PE utilities
- policy:         Actor-Critic policy with GT backbone
- lora:           LoRA modules for parameter-efficient fine-tuning
- trainer:        Mask-PPO trainer with reward shaping
- reward_shaping: Dense + potential-based reward design
- data_pool:      Multi-model sampling for foundation pretraining
- models:         Model factories shared with gnn-strategy
- checkpoint:     Save/load helpers (with LoRA state separation)
"""

from . import utils  # noqa: F401  ensures sys.path side-effects run early

__all__ = [
    "encoder",
    "lap_pe",
    "policy",
    "lora",
    "trainer",
    "reward_shaping",
    "data_pool",
    "models",
    "checkpoint",
]
