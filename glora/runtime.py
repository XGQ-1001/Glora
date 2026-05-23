"""Thin runtime helpers around CUDA Graph capture, benchmarking and ordering."""

from __future__ import annotations

import copy
import warnings
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from .utils import _GNN_STRATEGY_ROOT  # noqa: F401  ensure path setup

# Reuse gnn-strategy's CUDA-Graph machinery.
from gnn_strategy.capturer import (  # type: ignore
    benchmark_runner,
    capturer_gnn_from_fx,
)


def benchmark_runner_ms(
    runner: Callable,
    inputs: Tuple[torch.Tensor, ...],
    iterations: int = 30,
    warmups: int = 10,
) -> float:
    return benchmark_runner(runner, inputs=inputs, iterations=iterations, warmups=warmups).mean_ms


def benchmark_runner_full(
    runner: Callable,
    inputs: Tuple[torch.Tensor, ...],
    iterations: int = 30,
    warmups: int = 10,
):
    return benchmark_runner(runner, inputs=inputs, iterations=iterations, warmups=warmups)


def schedule_order_from_env(env, gs) -> List[str]:
    """Return the movable-node schedule produced by ``env`` (skipping placeholders)."""
    order_ids = env.scheduled_order()
    out: List[str] = []
    for i in order_ids:
        if gs.movable_mask[i].item() != 1.0:
            continue
        out.append(gs.node_names[i])
    return out


def real_latency_for_order(
    fx_module,
    inputs: Sequence[torch.Tensor],
    order_names: Sequence[str],
    iterations: int = 10,
    warmups: int = 3,
) -> float:
    """Benchmark the real GPU latency of a GNN-generated schedule.

    Deep copies the FX module so the original cache isn't mutated. The
    function returns the mean latency in milliseconds.
    """
    fx_copy = copy.deepcopy(fx_module)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Trying to prepend a node to itself\..*",
            category=UserWarning,
        )
        runner = capturer_gnn_from_fx(fx_copy, inputs, order_names, copy_outputs=False)
    latency = benchmark_runner_ms(runner, inputs=inputs, iterations=iterations, warmups=warmups)
    del runner, fx_copy
    return float(latency)


def cosine_lr_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    *,
    base_lr: float,
    floor_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    floor = max(0.0, floor_ratio)

    def lr_fn(step: int) -> float:
        if total_steps <= 1:
            return 1.0
        progress = min(1.0, max(0.0, step / float(total_steps - 1)))
        cos = 0.5 * (1.0 + np.cos(np.pi * progress))
        return floor + (1.0 - floor) * float(cos)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)


__all__ = [
    "benchmark_runner_ms",
    "benchmark_runner_full",
    "schedule_order_from_env",
    "real_latency_for_order",
    "cosine_lr_schedule",
]
