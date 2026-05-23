"""Step 0 sanity check: Graph Transformer + Laplacian PE forward pass.

This script does **not** require a GPU or any model. It builds a small
synthetic DAG, computes the Laplacian PE, runs one forward pass through the
encoder + policy and asserts that:

* output shapes match expectations
* the encoder's gradient flows end-to-end
* the action distribution only places probability mass on legal nodes

Run::

    python scripts/step0_encoder_smoketest.py
"""

from __future__ import annotations

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402

from glora.encoder import GraphTransformerEncoder  # noqa: E402
from glora.lap_pe import compute_lap_pe  # noqa: E402
from glora.policy import GloraActorCritic  # noqa: E402


def build_toy_dag(num_nodes: int = 12):
    """Return a tiny chain-with-branches DAG (parents/children adjacency)."""
    parents = [[] for _ in range(num_nodes)]
    children = [[] for _ in range(num_nodes)]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5),
             (4, 6), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
    for u, v in edges:
        if v < num_nodes:
            parents[v].append(u)
            children[u].append(v)
    return parents, children


def main() -> int:
    N = 12
    static_in_dim = 16
    pe_dim = 8
    hidden = 64
    emb = 64
    heads = 4
    layers = 2

    torch.manual_seed(0)

    parents, _ = build_toy_dag(N)
    x = torch.randn(N, static_in_dim)
    pe = compute_lap_pe(N, parents, pe_dim)
    print(f"[lap_pe] x={x.shape} pe={pe.shape} pe.norm={pe.norm().item():.3f}")
    assert pe.shape == (N, pe_dim), "Laplacian PE shape mismatch"

    encoder = GraphTransformerEncoder(
        in_dim=static_in_dim, pe_dim=pe_dim,
        hidden_dim=hidden, emb_dim=emb, n_layers=layers, n_heads=heads,
    )
    h = encoder(x, pe)
    print(f"[encoder] h={tuple(h.shape)} mean={h.mean().item():+.4f} std={h.std().item():.4f}")
    assert h.shape == (N, emb)

    grad_target = h.sum()
    grad_target.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    print("[encoder] gradient flows end-to-end ✓")

    policy = GloraActorCritic(
        static_in_dim=static_in_dim, pe_dim=pe_dim,
        hidden_dim=hidden, emb_dim=emb, n_heads=heads, n_layers=layers,
    )
    policy.eval()
    h_static = policy.encode_static(x, pe)

    dyn = torch.randn(N, 10)
    glob = torch.randn(12)
    mask = torch.zeros(N)
    legal_idx = [1, 3, 5, 7]
    mask[legal_idx] = 1.0
    dist, value = policy.act(h_static, dyn, glob, mask)
    probs = dist.probs.detach()
    illegal = [i for i in range(N) if i not in legal_idx]
    illegal_mass = probs[illegal].sum().item()
    legal_mass = probs[legal_idx].sum().item()
    print(f"[policy] legal_mass={legal_mass:.4f} illegal_mass={illegal_mass:.2e} V={value.item():+.3f}")
    assert illegal_mass < 1e-5, f"Mask leak detected: {illegal_mass}"
    assert abs(legal_mass - 1.0) < 1e-3, f"Probability does not normalise: {legal_mass}"

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[ok] Step 0 smoketest passed (params={n_params/1e6:.2f}M)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
