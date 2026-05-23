"""Laplacian Positional Encoding for directed acyclic graphs.

Given a DAG with parent/child adjacency lists we form the symmetrically
normalised Laplacian on the *undirected* graph and take the eigenvectors
associated with the ``k`` smallest non-trivial eigenvalues. They are
appended to each node's static feature vector, providing the Graph
Transformer with a global notion of structural position.

Notes
-----
* Sign of Laplacian eigenvectors is ambiguous (``v`` and ``-v`` are both
  valid solutions). At inference time we deterministically flip each
  eigenvector so its first non-zero entry is positive. During training
  callers may use :func:`random_sign_flip` for augmentation.
* For very small graphs we fall back to zero-padding instead of raising,
  which keeps single-op corner cases workable.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch


def _build_adjacency(num_nodes: int, parents: Sequence[Sequence[int]]) -> np.ndarray:
    A = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for v, ps in enumerate(parents):
        for u in ps:
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                A[u, v] = 1.0
                A[v, u] = 1.0
    return A


def _normalized_laplacian(A: np.ndarray) -> np.ndarray:
    deg = A.sum(axis=1)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return L


def _canonical_sign(vec: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Flip ``vec`` so its first non-zero entry is positive."""
    nz = np.where(np.abs(vec) > tol)[0]
    if nz.size == 0:
        return vec
    if vec[nz[0]] < 0:
        return -vec
    return vec


def compute_lap_pe(
    num_nodes: int,
    parents: Sequence[Sequence[int]],
    k: int,
) -> torch.Tensor:
    """Return a ``[N, k]`` Laplacian positional encoding tensor (float32)."""
    if num_nodes <= 0 or k <= 0:
        return torch.zeros((max(num_nodes, 0), max(k, 0)), dtype=torch.float32)

    A = _build_adjacency(num_nodes, parents)
    L = _normalized_laplacian(A)
    # eigh returns ascending eigenvalues; the first one is always ~0 (constant).
    eigvals, eigvecs = np.linalg.eigh(L)

    # Skip the constant eigenvector (eigenvalue ≈ 0) and pad if needed.
    eigvecs = eigvecs[:, 1:]
    take = min(k, eigvecs.shape[1])
    pe = np.zeros((num_nodes, k), dtype=np.float64)
    if take > 0:
        chosen = eigvecs[:, :take]
        for j in range(take):
            chosen[:, j] = _canonical_sign(chosen[:, j])
        pe[:, :take] = chosen
    return torch.from_numpy(pe.astype(np.float32))


def random_sign_flip(pe: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Randomly multiply each column of ``pe`` by ±1 (sign augmentation)."""
    if pe.numel() == 0:
        return pe
    signs = torch.randint(
        low=0, high=2, size=(pe.shape[1],),
        generator=generator, device=pe.device, dtype=pe.dtype,
    ) * 2 - 1
    return pe * signs.unsqueeze(0)


def attach_pe_to_graph_state(gs, k: int, *, cache_attr: str = "_glora_lap_pe") -> torch.Tensor:
    """Compute LapPE once for a ``GraphState`` and cache it on the object."""
    if hasattr(gs, cache_attr):
        cached = getattr(gs, cache_attr)
        if isinstance(cached, torch.Tensor) and cached.shape == (len(gs.node_names), k):
            return cached
    pe = compute_lap_pe(len(gs.node_names), gs.parents, k)
    setattr(gs, cache_attr, pe)
    return pe


__all__ = [
    "compute_lap_pe",
    "random_sign_flip",
    "attach_pe_to_graph_state",
]
