"""Step 5: comprehensive comparison across schedulers.

For every requested (algorithm, model, batch_size) the script measures:

* ``mean_ms``, ``std_ms``        — CUDA Graph benchmark statistics
* ``throughput``                  — ``batch_size * 1000 / mean_ms``
* ``peak_memory_mb``              — ``torch.cuda.max_memory_allocated``
* ``sm_efficiency``               — average NVML GPU utilisation
                                   (``None`` if pynvml is not installed)

Supported algorithms
--------------------
``pytorch``            eager PyTorch (no CUDA Graph)
``cuda_graph``         PyTorch eager wrapped with ``torch.cuda.graph``
``opara``              ``GraphCapturer.capturer(use_tcas=False)`` baseline
``gt_maskppo_scratch`` checkpoint from Step 2 (per-model trained)
``glora_foundation``   checkpoint from Step 3 (zero-shot greedy)
``glora_lora``         checkpoint from Step 4 (foundation + LoRA)

CLI example::

    python scripts/step5_compare.py \
        --models googlenet,inception_v3,resnet50,bert_base \
        --batch-sizes 1,4,16 \
        --foundation artifacts/step3/foundation_base.pt \
        --lora-dir artifacts/step4 \
        --scratch-dir artifacts/step2 \
        --output-dir artifacts/step5

Output: ``compare_results.csv`` plus a handful of PNGs in ``--output-dir``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402

from glora import utils as glora_utils  # noqa: E402
from glora.checkpoint import apply_lora_state, load_full_checkpoint, restore_policy  # noqa: E402
from glora.data_pool import ModelPool, PoolEntry  # noqa: E402
from glora.lap_pe import attach_pe_to_graph_state  # noqa: E402
from glora.lora import (  # noqa: E402
    apply_lora_to_linear,
    find_linear_module_names,
    freeze_all_parameters,
)
from glora.models import display_name, known_models  # noqa: E402
from glora.profiling import has_nvml, profile_block  # noqa: E402
from glora.runtime import (  # noqa: E402
    benchmark_runner_full,
    real_latency_for_order,
    schedule_order_from_env,
)

from gnn_strategy.capturer import capturer_gnn_from_fx  # type: ignore  # noqa: E402
from gnn_strategy.env import SchedulingEnv  # type: ignore  # noqa: E402
from gnn_strategy.graph_state import D_STATIC  # type: ignore  # noqa: E402

from Opara import GraphCapturer  # type: ignore  # noqa: E402


# ----------------------------------------------------------------------
# Runner builders for each algorithm
# ----------------------------------------------------------------------


@dataclass
class Runner:
    name: str
    fn: Callable
    inputs: Tuple[torch.Tensor, ...]
    extra: Dict


def build_pytorch_runner(sample) -> Runner:
    model = sample.model
    inputs = sample.inputs

    def run(*new_inputs):
        with torch.no_grad():
            out = model(*new_inputs)
        return out

    return Runner("pytorch", run, inputs, {})


def build_cuda_graph_runner(sample) -> Runner:
    """Wrap eager PyTorch with a captured CUDA Graph (PyTorch-default style)."""
    model = sample.model
    inputs = sample.inputs
    static_inputs = tuple(torch.zeros_like(x, device="cuda") for x in inputs)
    stream = torch.cuda.Stream()
    with torch.no_grad():
        for _ in range(3):
            with torch.cuda.stream(stream):
                model(*static_inputs)
        torch.cuda.current_stream().wait_stream(stream)

    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g, stream=stream):
        static_outputs = model(*static_inputs)
    if not isinstance(static_outputs, (tuple, list)):
        static_outputs = (static_outputs,)

    def run(*new_inputs):
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)
        g.replay()
        return static_outputs

    return Runner("cuda_graph", run, inputs, {"graph": g})


def build_opara_runner(sample) -> Runner:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"Trying to prepend a node to itself\..*",
            category=UserWarning,
        )
        run = GraphCapturer.capturer(sample.inputs, sample.model, use_tcas=False)
    return Runner("opara", run, sample.inputs, {})


def _greedy_order(policy, sample, *, pe_dim: int, n_streams: int, device: torch.device) -> List[str]:
    gs = sample.graph_state
    env = SchedulingEnv(gs, n_streams=n_streams, device=device)
    env.reset()
    pe = attach_pe_to_graph_state(gs, pe_dim).to(device)
    x = gs.x.to(device)
    policy.eval()
    with torch.no_grad():
        h_static = policy.encode_static(x, pe)
    while not env.is_done():
        dyn = env.dynamic_node_features().to(device)
        glob = env.global_features().to(device)
        mask = env.ready_mask().to(device)
        with torch.no_grad():
            dist, _ = policy.act(h_static, dyn, glob, mask)
        action = int(torch.argmax(dist.probs).item())
        if mask[action].item() != 1.0:
            action = int(torch.argmax(mask).item())
        env.step(action)
    return schedule_order_from_env(env, gs)


def build_glora_runner(
    sample,
    *,
    ckpt_path: str,
    algorithm: str,
    device: torch.device,
    n_streams: int,
    lora_target_modules: Optional[Sequence[str]] = None,
    lora_rank: int = 8,
    lora_alpha: int = 16,
) -> Optional[Runner]:
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    ckpt = load_full_checkpoint(ckpt_path, device="cpu")
    policy = restore_policy(ckpt, static_in_dim=D_STATIC, device=device)
    pe_dim = int(ckpt.get("pe_dim", 16))

    if algorithm == "glora_lora":
        if "lora_state_dict" not in ckpt:
            print(f"[warn] {ckpt_path} is not a LoRA checkpoint, skipping")
            return None
        freeze_all_parameters(policy)
        names = find_linear_module_names(
            policy,
            name_filter=set(lora_target_modules or {"linear1", "linear2"}),
            skip_inside_attention=True,
        )
        apply_lora_to_linear(policy, names, rank=lora_rank, alpha=lora_alpha)
        apply_lora_state(policy, ckpt["lora_state_dict"])

    # Use the BEST schedule order recorded during training when it targets
    # this exact (model, batch) pair; otherwise fall back to greedy rollout.
    order: List[str] = []
    best = ckpt.get("best") or {}
    if (best.get("sample_name") == sample.name
            and int(best.get("batch_size", -1)) == int(sample.batch_size)
            and best.get("order")):
        order = list(best["order"])
    else:
        order = _greedy_order(
            policy, sample,
            pe_dim=pe_dim, n_streams=n_streams, device=device,
        )

    fx_copy = copy.deepcopy(sample.fx_module)
    runner = capturer_gnn_from_fx(fx_copy, sample.inputs, order, copy_outputs=False)
    return Runner(algorithm, runner, sample.inputs, {"order_len": len(order)})


# ----------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------


def measure_runner(
    runner: Runner,
    *,
    iterations: int,
    warmups: int,
    device_index: int = 0,
    nvml: bool = True,
) -> Dict[str, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with profile_block(device_index=device_index) as captured:
        res = benchmark_runner_full(
            runner.fn, inputs=runner.inputs,
            iterations=iterations, warmups=warmups,
        )
    prof = captured["result"]
    mean_ms = float(res.mean_ms)
    std_ms = float(res.std_ms)
    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "peak_memory_mb": prof.peak_memory_mb,
        "sm_efficiency_pct": prof.sm_efficiency_pct if nvml else None,
        "sm_max_pct": prof.sm_max_pct if nvml else None,
        "mem_util_pct": prof.mem_utilization_pct if nvml else None,
    }


def throughput(mean_ms: float, batch_size: int) -> float:
    if mean_ms is None or mean_ms <= 0:
        return 0.0
    return float(batch_size) * 1000.0 / float(mean_ms)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def _plot(out_dir: str, rows: List[Dict], baseline: str = "pytorch") -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot] matplotlib unavailable: {exc}")
        return

    os.makedirs(out_dir, exist_ok=True)
    by_key: Dict[Tuple[str, int, str], Dict] = {}
    models = sorted({r["model"] for r in rows})
    batches = sorted({int(r["batch_size"]) for r in rows})
    algos = sorted({r["algorithm"] for r in rows})

    for r in rows:
        by_key[(r["model"], int(r["batch_size"]), r["algorithm"])] = r

    def value(r: Dict, key: str) -> float:
        v = r.get(key)
        return float(v) if v is not None else float("nan")

    def speedup(r: Dict, base: Dict) -> float:
        if not base or base["mean_ms"] <= 0 or r["mean_ms"] <= 0:
            return float("nan")
        return base["mean_ms"] / r["mean_ms"]

    # ---- speedup chart at batch=1 (or smallest batch) ----
    if 1 in batches:
        bs = 1
    else:
        bs = batches[0]
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.4), 4.5), dpi=200)
    x = list(range(len(models)))
    width = 0.8 / max(len(algos), 1)
    for i, algo in enumerate(algos):
        ys = []
        for m in models:
            r = by_key.get((m, bs, algo))
            base = by_key.get((m, bs, baseline))
            ys.append(speedup(r, base) if r and base else float("nan"))
        ax.bar([xi + (i - len(algos) / 2 + 0.5) * width for xi in x], ys, width=width, label=algo)
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(m) for m in models], rotation=15, ha="right")
    ax.set_ylabel(f"Speedup vs {baseline}")
    ax.set_title(f"Latency speedup at batch={bs}")
    ax.axhline(1.0, color="grey", linewidth=1, linestyle="--", label=baseline)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "speedup_bs1.png"))
    plt.close(fig)

    # ---- throughput chart ----
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.4), 4.5), dpi=200)
    for i, algo in enumerate(algos):
        ys = []
        for m in models:
            r = by_key.get((m, bs, algo))
            ys.append(value(r, "throughput") if r else float("nan"))
        ax.bar([xi + (i - len(algos) / 2 + 0.5) * width for xi in x], ys, width=width, label=algo)
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(m) for m in models], rotation=15, ha="right")
    ax.set_ylabel("Throughput (samples / sec)")
    ax.set_title(f"Throughput at batch={bs}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "throughput_bs1.png"))
    plt.close(fig)

    # ---- memory chart ----
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.4), 4.5), dpi=200)
    for i, algo in enumerate(algos):
        ys = []
        for m in models:
            r = by_key.get((m, bs, algo))
            ys.append(value(r, "peak_memory_mb") if r else float("nan"))
        ax.bar([xi + (i - len(algos) / 2 + 0.5) * width for xi in x], ys, width=width, label=algo)
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(m) for m in models], rotation=15, ha="right")
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title(f"Peak GPU memory at batch={bs}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "memory_bs1.png"))
    plt.close(fig)

    # ---- SM efficiency chart (skip if no NVML data) ----
    have_sm = any(value(r, "sm_efficiency_pct") == value(r, "sm_efficiency_pct") for r in rows)
    if have_sm:
        fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.4), 4.5), dpi=200)
        for i, algo in enumerate(algos):
            ys = []
            for m in models:
                r = by_key.get((m, bs, algo))
                ys.append(value(r, "sm_efficiency_pct") if r else float("nan"))
            ax.bar([xi + (i - len(algos) / 2 + 0.5) * width for xi in x], ys, width=width, label=algo)
        ax.set_xticks(x)
        ax.set_xticklabels([display_name(m) for m in models], rotation=15, ha="right")
        ax.set_ylabel("Mean SM utilisation (%)")
        ax.set_title(f"SM efficiency at batch={bs}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "sm_efficiency_bs1.png"))
        plt.close(fig)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Glora Step 5 — comprehensive comparison")
    p.add_argument("--models", type=str, required=True,
                   help="comma-separated model names, e.g. 'googlenet,resnet50,bert_base'")
    p.add_argument("--batch-sizes", type=str, default="1",
                   help="comma-separated batch sizes")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmups", type=int, default=10)
    p.add_argument("--device-index", type=int, default=0)

    p.add_argument("--include-pytorch", action="store_true", default=True)
    p.add_argument("--no-pytorch", dest="include_pytorch", action="store_false")
    p.add_argument("--include-cuda-graph", action="store_true", default=True)
    p.add_argument("--no-cuda-graph", dest="include_cuda_graph", action="store_false")
    p.add_argument("--include-opara", action="store_true", default=True)
    p.add_argument("--no-opara", dest="include_opara", action="store_false")

    p.add_argument("--scratch-dir", type=str, default=None,
                   help="directory containing Step 2 checkpoints; "
                        "files are matched as ``{name}_bs{batch_size}.pt``")
    p.add_argument("--foundation", type=str, default=None,
                   help="path to Step 3 foundation checkpoint (.pt)")
    p.add_argument("--lora-dir", type=str, default=None,
                   help="directory containing Step 4 LoRA checkpoints; "
                        "files are matched as ``lora_{name}_bs{batch_size}_rank{R}.pt``")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-target-modules", type=str, default="linear1,linear2",
                   help="Linear sub-modules to wrap with LoRA when reconstructing the policy. "
                        "Must match Step 4's --target-modules.")
    p.add_argument("--n-streams", type=int, default=8)

    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--no-plots", action="store_true")
    return p


def _maybe_ckpt(path: Optional[str]) -> Optional[str]:
    return path if path and os.path.exists(path) else None


def main() -> int:
    args = build_argparser().parse_args()
    glora_utils.set_global_seed(0)

    if not torch.cuda.is_available():
        print("[fatal] Step 5 requires CUDA")
        return 2
    device = torch.device(f"cuda:{args.device_index}")
    torch.cuda.set_device(device)

    models = parse_csv_list(args.models)
    for m in models:
        if m not in known_models():
            print(f"[fatal] unknown model: {m}. Choose from {sorted(known_models())}")
            return 3
    batch_sizes = [int(b) for b in parse_csv_list(args.batch_sizes)]

    output_dir = args.output_dir or glora_utils.artifacts_dir("step5")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = args.csv or os.path.join(output_dir, "compare_results.csv")

    if not has_nvml():
        print("[step5] pynvml not available — SM efficiency will be reported as None")

    entries = [PoolEntry(name=m, batch_sizes=batch_sizes, weight=1.0) for m in models]
    pool = ModelPool(entries=entries, device=device, bench_iters=args.iters, bench_warmups=args.warmups)

    lora_targets = parse_csv_list(args.lora_target_modules)

    rows: List[Dict] = []
    for model_name in models:
        for bs in batch_sizes:
            sample = pool.get(model_name, bs)
            print(f"\n=== {display_name(model_name)} | batch={bs} | nodes="
                  f"{len(sample.graph_state.node_names)} | opara={sample.opara_latency_ms:.4f}ms ===")

            builders: List[Tuple[str, Callable[[], Optional[Runner]]]] = []
            if args.include_pytorch:
                builders.append(("pytorch", lambda s=sample: build_pytorch_runner(s)))
            if args.include_cuda_graph:
                builders.append(("cuda_graph", lambda s=sample: build_cuda_graph_runner(s)))
            if args.include_opara:
                builders.append(("opara", lambda s=sample: build_opara_runner(s)))

            scratch_ckpt = None
            if args.scratch_dir:
                scratch_ckpt = _maybe_ckpt(os.path.join(args.scratch_dir, f"{model_name}_bs{bs}.pt"))
                if not scratch_ckpt:
                    scratch_ckpt = _maybe_ckpt(os.path.join(args.scratch_dir, f"{model_name}.pt"))
            if scratch_ckpt:
                builders.append((
                    "gt_maskppo_scratch",
                    lambda s=sample, c=scratch_ckpt: build_glora_runner(
                        s, ckpt_path=c, algorithm="gt_maskppo_scratch",
                        device=device, n_streams=args.n_streams,
                    ),
                ))

            foundation_ckpt = _maybe_ckpt(args.foundation)
            if foundation_ckpt:
                builders.append((
                    "glora_foundation",
                    lambda s=sample, c=foundation_ckpt: build_glora_runner(
                        s, ckpt_path=c, algorithm="glora_foundation",
                        device=device, n_streams=args.n_streams,
                    ),
                ))

            if args.lora_dir:
                lora_path = _maybe_ckpt(os.path.join(
                    args.lora_dir,
                    f"lora_{model_name}_bs{bs}_rank{args.lora_rank}.pt",
                ))
                if lora_path:
                    builders.append((
                        "glora_lora",
                        lambda s=sample, c=lora_path: build_glora_runner(
                            s, ckpt_path=c, algorithm="glora_lora",
                            device=device, n_streams=args.n_streams,
                            lora_target_modules=lora_targets,
                            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                        ),
                    ))

            for algo_name, builder in builders:
                try:
                    runner = builder()
                    if runner is None:
                        print(f"  [{algo_name}] skipped (no checkpoint)")
                        continue
                    stats = measure_runner(
                        runner,
                        iterations=args.iters, warmups=args.warmups,
                        device_index=args.device_index,
                    )
                except Exception as exc:
                    print(f"  [{algo_name}] FAILED: {exc}")
                    continue
                row = {
                    "model": model_name,
                    "batch_size": bs,
                    "algorithm": algo_name,
                    "mean_ms": stats["mean_ms"],
                    "std_ms": stats["std_ms"],
                    "throughput": throughput(stats["mean_ms"], bs),
                    "peak_memory_mb": stats["peak_memory_mb"],
                    "sm_efficiency_pct": stats["sm_efficiency_pct"],
                    "sm_max_pct": stats["sm_max_pct"],
                    "mem_util_pct": stats["mem_util_pct"],
                    "opara_latency_ms": sample.opara_latency_ms,
                    "num_nodes": len(sample.graph_state.node_names),
                }
                rows.append(row)
                sm_str = (
                    f" sm={stats['sm_efficiency_pct']:.1f}%"
                    if stats["sm_efficiency_pct"] is not None else ""
                )
                print(
                    f"  [{algo_name:>20s}] mean={stats['mean_ms']:.4f}ms"
                    f" thpt={row['throughput']:.1f}/s mem={stats['peak_memory_mb']:.1f}MB{sm_str}"
                )

    if not rows:
        print("[fatal] no measurements produced; check inputs")
        return 4

    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[step5] wrote {len(rows)} rows → {csv_path}")

    if not args.no_plots:
        _plot(output_dir, rows)
        print(f"[step5] plots in {output_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
