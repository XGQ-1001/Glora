"""Multi-model sampling pool for foundation pretraining.

Foundation pretraining requires episodes drawn from a *diverse* collection
of (model, batch_size) pairs. ``ModelPool`` lazily constructs the heavy
artifacts (FX module, GraphState, Opara baseline) the first time a (name,
batch) tuple is requested and caches the result so subsequent episodes
sampling the same configuration are cheap.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .utils import _GNN_STRATEGY_ROOT  # noqa: F401  ensure imports
from .models import build_model, display_name


# Reuse heavy machinery from gnn-strategy.
from gnn_strategy.graph_state import build_graph_state  # type: ignore
from gnn_strategy.utils import extract_first_fx_graph  # type: ignore
from Opara import GraphCapturer  # type: ignore
from Opara import OperatorLauncher  # type: ignore


@dataclass
class CachedSample:
    """One cached (model, batch_size) configuration."""

    name: str
    batch_size: int
    model: torch.nn.Module
    inputs: Tuple[torch.Tensor, ...]
    fx_module: torch.fx.GraphModule
    graph_state: object
    opara_latency_ms: float
    model_class_name: str

    def summary(self) -> str:
        n_nodes = len(self.graph_state.node_names) if hasattr(self.graph_state, "node_names") else -1
        return (
            f"{display_name(self.name)} (bs={self.batch_size}, "
            f"N={n_nodes}, opara={self.opara_latency_ms:.4f}ms)"
        )


@dataclass
class PoolEntry:
    name: str
    batch_sizes: Sequence[int]
    weight: float = 1.0


class ModelPool:
    """Lazy cache of (model, batch_size) artifacts for foundation training.

    Parameters
    ----------
    entries:
        List of (model_name, batch_sizes, weight). ``weight`` controls the
        sampling probability of that model relative to the others.
    device:
        CUDA device for benchmarking.
    bench_iters / bench_warmups:
        Benchmark configuration when measuring the Opara baseline. We cache
        the latency once because Opara doesn't change across episodes.
    """

    def __init__(
        self,
        entries: Sequence[PoolEntry],
        device: torch.device,
        bench_iters: int = 30,
        bench_warmups: int = 10,
    ):
        if not entries:
            raise ValueError("ModelPool needs at least one entry")
        self.entries = list(entries)
        self.device = device
        self.bench_iters = bench_iters
        self.bench_warmups = bench_warmups

        weights = [max(float(e.weight), 0.0) for e in self.entries]
        wsum = sum(weights) or 1.0
        self.probs = [w / wsum for w in weights]

        self._cache: Dict[Tuple[str, int], CachedSample] = {}

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, rng: random.Random | None = None) -> CachedSample:
        rng = rng or random
        entry = self._pick_entry(rng)
        batch_size = rng.choice(list(entry.batch_sizes))
        return self.get(entry.name, batch_size)

    def all_configs(self) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        for e in self.entries:
            for b in e.batch_sizes:
                out.append((e.name, int(b)))
        return out

    def _pick_entry(self, rng: random.Random) -> PoolEntry:
        u = rng.random()
        acc = 0.0
        for entry, p in zip(self.entries, self.probs):
            acc += p
            if u <= acc:
                return entry
        return self.entries[-1]

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def get(self, name: str, batch_size: int) -> CachedSample:
        key = (name, int(batch_size))
        if key in self._cache:
            return self._cache[key]
        sample = self._build_sample(name, batch_size)
        self._cache[key] = sample
        return sample

    def _build_sample(self, name: str, batch_size: int) -> CachedSample:
        device_str = (
            f"cuda:{self.device.index}" if self.device.type == "cuda" and self.device.index is not None
            else "cuda" if self.device.type == "cuda" else "cpu"
        )
        model, inputs = build_model(name, device=device_str, batch_size=batch_size)

        # 1. FX graph + profiling
        fx_module = extract_first_fx_graph(model, inputs)
        fx_module.cuda()
        node_profiles, device_props = OperatorLauncher.recompile(
            model.__class__.__name__, fx_module, inputs, apply_opara_schedule=False,
        )
        gs = build_graph_state(fx_module.graph, node_profiles=node_profiles, device_props=device_props)

        # 2. Opara baseline latency (the multi-baseline shaping reference).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"Trying to prepend a node to itself\..*",
                category=UserWarning,
            )
            opara_runner = GraphCapturer.capturer(inputs, model, use_tcas=False)

        from .runtime import benchmark_runner_ms  # local import to avoid cycle
        opara_latency = benchmark_runner_ms(
            opara_runner, inputs=inputs,
            iterations=self.bench_iters, warmups=self.bench_warmups,
        )

        return CachedSample(
            name=name,
            batch_size=int(batch_size),
            model=model,
            inputs=inputs,
            fx_module=fx_module,
            graph_state=gs,
            opara_latency_ms=float(opara_latency),
            model_class_name=model.__class__.__name__,
        )


def parse_pool_spec(spec: str) -> List[PoolEntry]:
    """Parse a CLI string like ``"resnet50:1,4;bert_base:1"`` into entries.

    Format
    ------
    Each model is separated by ``;``. Within a model:
        ``name:bs1,bs2[,weight=W]``
    where ``weight`` is optional and defaults to ``1.0``.
    """
    entries: List[PoolEntry] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Bad pool spec entry '{chunk}', expected name:bs1,bs2")
        name, rest = chunk.split(":", 1)
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        weight = 1.0
        batch_sizes: List[int] = []
        for p in parts:
            if p.startswith("weight="):
                weight = float(p.split("=", 1)[1])
            else:
                batch_sizes.append(int(p))
        if not batch_sizes:
            raise ValueError(f"Pool entry '{name}' has no batch sizes")
        entries.append(PoolEntry(name=name.strip(), batch_sizes=batch_sizes, weight=weight))
    return entries


__all__ = ["ModelPool", "PoolEntry", "CachedSample", "parse_pool_spec"]
