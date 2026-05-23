"""Decoupled multi-source reward shaping for Glora.

The current gnn-strategy ``train.py`` uses a terminal-only reward of
``(L_opara - L_gnn) / L_opara``. That produces a sparse, noisy training
signal: a single number is back-propagated through episodes that may
contain hundreds of steps. Glora improves the signal in three ways:

1. **Dense step reward** — at every step we use the simulator's
   ``makespan`` delta as a cheap surrogate for "did this action help".
   This is pure shaping; it does not change the underlying optimum.

2. **Potential-based shaping** — adding ``γΦ(s')-Φ(s)`` keeps the
   theoretical optimum unchanged (Ng et al., 1999) while accelerating
   credit assignment. We use the remaining critical path / makespan as
   the potential function.

3. **Multi-baseline terminal reward** — instead of comparing only to
   Opara, the terminal reward references the best of ``{Opara,
   CUDA-Graph, historical-best-GNN}``, which is a much more stable
   yardstick across episodes.

Callers integrate the shaper via :class:`PotentialShaper` and the
helper :func:`build_episode_rewards`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import math


@dataclass
class RewardConfig:
    """Shaping hyper-parameters."""

    dense_coef: float = 0.05          # weight of the simulator makespan delta
    potential_coef: float = 0.5       # weight of γΦ(s')-Φ(s)
    terminal_scale: float = 1.0       # rescale of the terminal real-latency term
    gamma: float = 1.0                # discount used inside Φ shaping
    use_dense: bool = True
    use_potential: bool = True
    multi_baseline: bool = True       # use min(L_opara, L_best_history)

    # Running tracker of the best historical GNN latency seen so far.
    best_gnn_ms: float = field(default_factory=lambda: math.inf)


class PotentialShaper:
    """Per-episode shaping accumulator.

    Usage::

        shaper = PotentialShaper(cfg)
        for step in rollout:
            r_step = shaper.step_reward(prev_mk, new_mk, prev_phi, new_phi)
            ...
        terminal = shaper.terminal_reward(L_gnn_ms, L_opara_ms, L_cudagraph_ms)
    """

    def __init__(self, cfg: RewardConfig, initial_makespan: float = 1.0):
        self.cfg = cfg
        self.initial_makespan = max(initial_makespan, 1e-9)

    def step_reward(
        self,
        prev_makespan: float,
        new_makespan: float,
        prev_phi: float,
        new_phi: float,
    ) -> float:
        r = 0.0
        if self.cfg.use_dense:
            r += self.cfg.dense_coef * (prev_makespan - new_makespan) / self.initial_makespan
        if self.cfg.use_potential:
            r += self.cfg.potential_coef * (self.cfg.gamma * new_phi - prev_phi)
        return float(r)

    def terminal_reward(
        self,
        L_gnn_ms: float,
        L_opara_ms: float,
        L_cudagraph_ms: Optional[float] = None,
    ) -> float:
        baselines: list[float] = [L_opara_ms]
        if L_cudagraph_ms is not None and math.isfinite(L_cudagraph_ms):
            baselines.append(L_cudagraph_ms)
        if self.cfg.multi_baseline and math.isfinite(self.cfg.best_gnn_ms):
            baselines.append(self.cfg.best_gnn_ms)
        baseline = min(b for b in baselines if b > 0)
        r = (baseline - L_gnn_ms) / max(baseline, 1e-9) * 100.0
        # Update running historical best so future episodes have a tighter target.
        if L_gnn_ms < self.cfg.best_gnn_ms:
            self.cfg.best_gnn_ms = float(L_gnn_ms)
        return float(self.cfg.terminal_scale * r)


def remaining_potential(env) -> float:
    """Default Φ(s): negative remaining critical-path length, normalised.

    Higher Φ ⇔ closer to done ⇔ smaller remaining work. The shaping term
    ``Φ(s') - Φ(s)`` is positive when an action shortens the remaining
    critical path, which is exactly the local progress signal we want.
    """
    gs = env.gs
    initial = max(float(env._initial_makespan), 1e-9)
    remaining_cp = 0.0
    for v in env._movables:
        if v in env._done_set:
            continue
        cp_bwd = gs.x[v, 11].item() * initial
        if cp_bwd > remaining_cp:
            remaining_cp = cp_bwd
    return -remaining_cp / initial


def distribute_rewards(
    transitions_rewards: List[float],
    terminal: float,
    *,
    add_terminal_to_last_step: bool = True,
) -> List[float]:
    """Append the terminal reward to the last step in-place."""
    if not transitions_rewards:
        return transitions_rewards
    if add_terminal_to_last_step:
        transitions_rewards[-1] = float(transitions_rewards[-1]) + float(terminal)
    return transitions_rewards


__all__ = [
    "RewardConfig",
    "PotentialShaper",
    "remaining_potential",
    "distribute_rewards",
]
