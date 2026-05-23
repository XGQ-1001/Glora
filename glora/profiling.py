"""Lightweight per-run profiling: throughput, peak memory, SM efficiency.

Why not torch.profiler?
-----------------------
``torch.profiler`` is expensive to attach and produces JSON trace files
that bloat the experiment output. For a final comparison we only care
about three numbers per (algorithm, model, batch):

* ``mean_ms``         from a CUDA Graph benchmark
* ``peak_memory_MB``  via ``torch.cuda.max_memory_allocated``
* ``sm_efficiency``   average GPU utilisation polled with NVML (when
                       available; otherwise ``None``)

The NVML poller runs in a background thread and samples at ~20 Hz, then
returns the mean utilisation observed during the timed window.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch


try:
    import pynvml  # type: ignore
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


@dataclass
class ProfileResult:
    peak_memory_mb: float
    sm_efficiency_pct: Optional[float]
    samples: int = 0
    sm_max_pct: Optional[float] = None
    mem_utilization_pct: Optional[float] = None


class _NvmlPoller:
    """Background sampler that records SM utilisation + memory utilisation."""

    def __init__(self, device_index: int = 0, period_s: float = 0.005):
        self.device_index = device_index
        self.period_s = period_s
        self._sm: list[float] = []
        self._mem: list[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle = None

    def __enter__(self):
        if not _HAS_NVML:
            return self
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:
            self._handle = None
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if _HAS_NVML and self._handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _run(self):
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self._sm.append(float(util.gpu))
                self._mem.append(float(util.memory))
            except Exception:
                break
            time.sleep(self.period_s)

    @property
    def sm_mean(self) -> Optional[float]:
        return float(sum(self._sm) / len(self._sm)) if self._sm else None

    @property
    def sm_max(self) -> Optional[float]:
        return float(max(self._sm)) if self._sm else None

    @property
    def mem_mean(self) -> Optional[float]:
        return float(sum(self._mem) / len(self._mem)) if self._mem else None


@contextmanager
def profile_block(device_index: int = 0):
    """Context manager that captures peak memory + SM efficiency."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_index)
        torch.cuda.synchronize()
    poller = _NvmlPoller(device_index=device_index).__enter__()
    yield_data = {}
    try:
        yield yield_data
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        poller.__exit__(None, None, None)
        peak = 0.0
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated(device_index) / (1024 ** 2)
        yield_data["result"] = ProfileResult(
            peak_memory_mb=float(peak),
            sm_efficiency_pct=poller.sm_mean,
            sm_max_pct=poller.sm_max,
            mem_utilization_pct=poller.mem_mean,
            samples=len(poller._sm),
        )


def has_nvml() -> bool:
    return _HAS_NVML


__all__ = ["ProfileResult", "profile_block", "has_nvml"]
