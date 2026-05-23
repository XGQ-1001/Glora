"""Actor-Critic policy backed by the Graph Transformer encoder.

This mirrors ``DynamicActorCritic`` from gnn-strategy but swaps the GATv2
encoder for ``GraphTransformerEncoder`` (with Laplacian PE) and adds first
class Mask-PPO support: the ``act`` method always applies the ready mask
before softmax so the action distribution only ever places probability mass
on legal nodes.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import _GNN_STRATEGY_ROOT  # noqa: F401  ensures path is set up

# Reuse dynamic/global feature definitions from gnn-strategy so the env stays
# compatible without modification.
from gnn_strategy.env import D_DYN, D_GLOBAL  # type: ignore
from gnn_strategy.graph_state import D_STATIC  # type: ignore

from .encoder import GraphTransformerEncoder


class DynamicFusion(nn.Module):
    def __init__(self, emb_dim: int, dyn_dim: int, global_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + dyn_dim + global_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, h_static: torch.Tensor, dyn: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        N = h_static.shape[0]
        glob_expanded = glob.unsqueeze(0).expand(N, -1)
        return self.mlp(torch.cat([h_static, dyn, glob_expanded], dim=-1))


class ActorHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def masked_logits(self, h: torch.Tensor, ready_mask: torch.Tensor) -> torch.Tensor:
        scores = self.net(h).squeeze(-1)
        neg_inf = torch.finfo(scores.dtype).min
        return torch.where(ready_mask > 0.0, scores, torch.full_like(scores, neg_inf))

    def forward(self, h: torch.Tensor, ready_mask: torch.Tensor) -> torch.distributions.Categorical:
        masked = self.masked_logits(h, ready_mask)
        if torch.isneginf(masked).all():
            probs = torch.ones_like(masked) / float(masked.numel())
        else:
            probs = F.softmax(masked, dim=0)
        return torch.distributions.Categorical(probs=probs)


class CriticHead(nn.Module):
    def __init__(self, node_dim: int, global_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim + global_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_dyn: torch.Tensor, glob: torch.Tensor, ready_mask: torch.Tensor) -> torch.Tensor:
        weights = ready_mask / ready_mask.sum().clamp(min=1.0)
        pooled = (h_dyn * weights.unsqueeze(-1)).sum(dim=0)
        return self.net(torch.cat([pooled, glob], dim=-1)).squeeze(-1)


class GloraActorCritic(nn.Module):
    """Glora Mask-PPO actor critic with Graph Transformer + LapPE backbone."""

    def __init__(
        self,
        static_in_dim: int = D_STATIC,
        pe_dim: int = 16,
        hidden_dim: int = 128,
        emb_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pe_dim = pe_dim
        self.encoder = GraphTransformerEncoder(
            in_dim=static_in_dim,
            pe_dim=pe_dim,
            hidden_dim=hidden_dim,
            emb_dim=emb_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.fusion = DynamicFusion(emb_dim, D_DYN, D_GLOBAL, emb_dim, dropout=dropout)
        self.actor = ActorHead(emb_dim, hidden_dim)
        self.critic = CriticHead(emb_dim, D_GLOBAL, hidden_dim)
        self._emb_dim = emb_dim
        self._hidden_dim = hidden_dim
        self._n_heads = n_heads
        self._n_layers = n_layers

    @property
    def emb_dim(self) -> int:
        return self._emb_dim

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def encode_static(self, x: torch.Tensor, lap_pe: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, lap_pe)

    def act(
        self,
        h_static: torch.Tensor,
        dyn_node: torch.Tensor,
        glob: torch.Tensor,
        ready_mask: torch.Tensor,
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        h_dyn = self.fusion(h_static, dyn_node, glob)
        dist = self.actor(h_dyn, ready_mask)
        value = self.critic(h_dyn, glob, ready_mask)
        return dist, value

    def evaluate_actions(
        self,
        h_static: torch.Tensor,
        dyn_node: torch.Tensor,
        glob: torch.Tensor,
        ready_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_dyn = self.fusion(h_static, dyn_node, glob)
        dist = self.actor(h_dyn, ready_mask)
        value = self.critic(h_dyn, glob, ready_mask)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy

    def batch_evaluate_actions(
        self,
        h_static_batch: torch.Tensor,
        dyn_node_batch: torch.Tensor,
        glob_batch: torch.Tensor,
        ready_mask_batch: torch.Tensor,
        actions_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorised PPO mini-batch evaluation (same semantics as gnn-strategy)."""
        B, N, _ = h_static_batch.shape
        glob_expanded = glob_batch.unsqueeze(1).expand(B, N, -1)
        cat = torch.cat([h_static_batch, dyn_node_batch, glob_expanded], dim=-1)
        h_dyn = self.fusion.mlp(cat)

        scores = self.actor.net(h_dyn).squeeze(-1)
        neg_inf = torch.finfo(scores.dtype).min
        masked = torch.where(
            ready_mask_batch > 0.0, scores, torch.full_like(scores, neg_inf),
        )
        probs = F.softmax(masked, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        log_prob = dist.log_prob(actions_batch)
        entropy = dist.entropy()

        w = ready_mask_batch / ready_mask_batch.sum(dim=-1, keepdim=True).clamp(min=1.0)
        pooled = (h_dyn * w.unsqueeze(-1)).sum(dim=1)
        value = self.critic.net(torch.cat([pooled, glob_batch], dim=-1)).squeeze(-1)

        return log_prob, value, entropy
