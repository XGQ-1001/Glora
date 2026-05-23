"""Graph Transformer encoder used as the Glora backbone.

This module replaces the two-layer GATv2 ``StaticEncoder`` from gnn-strategy.
The encoder consumes:

1. Per-node static features ``x`` (16-d feature vector from ``graph_state``)
2. Laplacian positional encoding ``lap_pe`` (k columns, see ``lap_pe.py``)
3. Optional attention mask derived from graph topology

and emits a node embedding tensor ``h_static`` of shape ``[N, emb_dim]`` that
the dynamic fusion / actor / critic heads consume each step.

The implementation deliberately stays close to ``nn.TransformerEncoderLayer``
so that LoRA adapters can hook into the standard Q/K/V projections.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class GraphTransformerEncoder(nn.Module):
    """Pre-norm Transformer encoder with Laplacian PE inputs.

    Parameters
    ----------
    in_dim:
        Per-node feature dimension (``D_STATIC``).
    pe_dim:
        Number of Laplacian PE columns concatenated to ``x``.
    hidden_dim:
        Internal d_model of the Transformer.
    emb_dim:
        Output node embedding size.
    n_layers:
        Number of Transformer blocks (default 3).
    n_heads:
        Multi-head attention head count.
    dropout:
        Dropout used inside the encoder layers.
    """

    def __init__(
        self,
        in_dim: int,
        pe_dim: int,
        hidden_dim: int = 128,
        emb_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.pe_dim = pe_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim + pe_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        lap_pe: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode static node features.

        ``x`` is ``[N, in_dim]`` and ``lap_pe`` is ``[N, pe_dim]``. The
        encoder treats the graph as a single sequence (no padding) so the
        unbatched call always runs on shape ``[1, N, hidden]``.
        """
        if x.dim() != 2:
            raise ValueError(f"x must be [N, in_dim], got shape {tuple(x.shape)}")
        if lap_pe.dim() != 2:
            raise ValueError(f"lap_pe must be [N, pe_dim], got shape {tuple(lap_pe.shape)}")
        if lap_pe.shape[0] != x.shape[0]:
            raise ValueError(
                f"lap_pe / x node count mismatch: {lap_pe.shape[0]} vs {x.shape[0]}"
            )

        h = self.input_proj(torch.cat([x, lap_pe.to(x.dtype)], dim=-1))
        h = h.unsqueeze(0)  # [1, N, hidden]
        h = self.encoder(h, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        h = h.squeeze(0)
        return self.out_proj(h)

    # ------------------------------------------------------------------
    # LoRA-friendly accessors
    # ------------------------------------------------------------------

    def attention_linears(self) -> list[nn.Linear]:
        """Return the (in_proj, out_proj) Linear modules from every attention block."""
        layers: list[nn.Linear] = []
        for block in self.encoder.layers:
            attn = block.self_attn
            # ``in_proj`` is stored as flat (Q, K, V) inside MultiheadAttention.
            if isinstance(attn.out_proj, nn.Linear):
                layers.append(attn.out_proj)
        return layers
