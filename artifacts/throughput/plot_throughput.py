"""Throughput bar chart (bs=1) with normalization to best = 1.

Compares PyTorch Eager, CUDA Graph (default order), Opara, and GT+PPO
(best order from checkpoint).

Usage::

    python scripts/plot_throughput.py \
        --model googlenet \
        --gt-ckpt artifacts/step2/gt/googlenet_bs1_final.pt \
        --output-dir artifacts/throughput
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import warnings
from typing import Callable, Dict, List, Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from glora import utils as glora_utils

from gnn_strategy.capturer import benchmark_runner, capturer_gnn_from_fx
from gnn_strategy.utils import extract_first_fx_graph

from Opara import GraphCapturer, OperatorLauncher


_COLORS = {
    "PyTorch Eager": "#95a5a6",
    "CUDA Graph": "#e67e22",
    "Opara": "#3498db",
    "GT+PPO": "#e74c3c",
}

_LABELS = ["PyTorch Eager", "CUDA Graph", "Opara", "GT+PPO"]


def _make_model(name: str, device: str, batch_size: int = 1):
    from glora.models import build_model

    return build_model(name, device=device, batch_size=batch_size)


def _fresh_fx(model_name: str, device: str, batch_size: int = 1):
    model, inputs = _make_model(model_name, device, batch_size)
    fx_module = extract_first_fx_graph(model, inputs)
    fx_module.cuda()
    OperatorLauncher.recompile(
        model.__class__.__name__, fx_module, inputs, apply_opara_schedule=False,
    )
    return model, inputs, fx_module


def _cuda_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _bench_mean_ms(
    runner: Callable,
    inputs: Tuple[torch.Tensor, ...],
    *,
    trials: int,
    iters: int,
    warmups: int,
) -> float:
    vals = []
    for _ in range(trials):
        br = benchmark_runner(runner, inputs=inputs, iterations=iters, warmups=warmups)
        vals.append(float(br.mean_ms))
    return float(np.mean(vals))


def _throughput(mean_ms: float, batch_size: int) -> float:
    return float(batch_size) * 1000.0 / float(mean_ms)


def benchmark_all(
    model_name: str,
    gt_ckpt: str,
    *,
    batch_size: int,
    trials: int,
    iters: int,
    warmups: int,
) -> Dict[str, float]:
    ckpt = torch.load(gt_ckpt, map_location="cpu", weights_only=False)
    best = ckpt.get("best") or {}
    best_order = best.get("order")
    if not best_order:
        raise ValueError(f"No best['order'] in {gt_ckpt}")
    del ckpt

    results_ms: Dict[str, float] = {}

    print(f"[throughput] PyTorch Eager ...")
    model, inputs, _ = _fresh_fx(model_name, "cuda", batch_size)

    def eager_runner(*inp):
        with torch.no_grad():
            return model(*inp)

    results_ms["PyTorch Eager"] = _bench_mean_ms(
        eager_runner, inputs, trials=trials, iters=iters, warmups=warmups,
    )
    print(f"  mean={results_ms['PyTorch Eager']:.4f} ms")
    del model, inputs
    _cuda_cleanup()

    print(f"[throughput] CUDA Graph (default order) ...")
    model, inputs, fx_module = _fresh_fx(model_name, "cuda", batch_size)
    fx_copy = copy.deepcopy(fx_module)
    default_order = [
        n.name for n in fx_copy.graph.nodes
        if n.op not in ("placeholder", "output")
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        cg_runner = capturer_gnn_from_fx(fx_copy, inputs, default_order, copy_outputs=False)
    results_ms["CUDA Graph"] = _bench_mean_ms(
        cg_runner, inputs, trials=trials, iters=iters, warmups=warmups,
    )
    print(f"  mean={results_ms['CUDA Graph']:.4f} ms")
    del model, inputs, fx_module, fx_copy, cg_runner
    _cuda_cleanup()

    print(f"[throughput] Opara ...")
    model, inputs, _ = _fresh_fx(model_name, "cuda", batch_size)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        opara_runner = GraphCapturer.capturer(inputs, model, use_tcas=False)
    results_ms["Opara"] = _bench_mean_ms(
        opara_runner, inputs, trials=trials, iters=iters, warmups=warmups,
    )
    print(f"  mean={results_ms['Opara']:.4f} ms")
    del model, inputs, opara_runner
    _cuda_cleanup()

    print(f"[throughput] GT+PPO (best order) ...")
    model, inputs, fx_module = _fresh_fx(model_name, "cuda", batch_size)
    fx_copy = copy.deepcopy(fx_module)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        gt_runner = capturer_gnn_from_fx(fx_copy, inputs, list(best_order), copy_outputs=False)
    results_ms["GT+PPO"] = _bench_mean_ms(
        gt_runner, inputs, trials=trials, iters=iters, warmups=warmups,
    )
    print(f"  mean={results_ms['GT+PPO']:.4f} ms")
    del model, inputs, fx_module, fx_copy, gt_runner
    _cuda_cleanup()

    throughputs = {k: _throughput(v, batch_size) for k, v in results_ms.items()}
    return {"latency_ms": results_ms, "throughput": throughputs}


def plot_throughput(
    throughputs: Dict[str, float],
    output_path: str,
    *,
    model_title: str,
    batch_size: int,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [l for l in _LABELS if l in throughputs]
    raw = [throughputs[l] for l in labels]
    best = max(raw)
    norm = [v / best for v in raw]
    colors = [_COLORS[l] for l in labels]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    x = np.arange(len(labels))
    bars = ax.bar(x, norm, color=colors, edgecolor="black", linewidth=0.8, width=0.62, alpha=0.88)

    for bar, rv, nv in zip(bars, raw, norm):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{nv:.3f}\n({rv:.1f} img/s)",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Normalized Throughput (best = 1.0)", fontsize=11)
    ax.set_title(
        f"{model_title} — Throughput (batch={batch_size}, normalized to best)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, max(norm) * 1.18)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = output_path.rsplit(".", 1)[0] + f".{ext}" if ext != "png" else output_path
        if ext == "pdf":
            path = output_path.rsplit(".", 1)[0] + ".pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"[throughput] saved → {path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Throughput comparison (normalized to best=1)")
    p.add_argument("--model", required=True, choices=["googlenet", "inception_v3", "resnet50", "bert_base"])
    p.add_argument("--gt-ckpt", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--warmups", type=int, default=20)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--results-json", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    glora_utils.set_global_seed(args.seed)
    out_dir = args.output_dir or os.path.join(PROJECT_ROOT, "artifacts", "throughput")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{args.model}_throughput_bs{args.batch_size}"
    json_path = args.results_json or os.path.join(out_dir, f"{stem}.json")
    png_path = os.path.join(out_dir, f"{stem}.png")
    model_title = args.model.replace("_", " ").title()

    if args.plot_only:
        with open(json_path) as f:
            payload = json.load(f)
        throughputs = payload["throughput"]
    else:
        payload = benchmark_all(
            args.model, args.gt_ckpt,
            batch_size=args.batch_size,
            trials=args.trials, iters=args.iters, warmups=args.warmups,
        )
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[throughput] saved results → {json_path}")
        throughputs = payload["throughput"]

    best_label = max(throughputs, key=throughputs.get)
    print(f"\n{'='*50}")
    print(f"  {model_title} throughput (batch={args.batch_size})")
    print(f"{'='*50}")
    best_val = throughputs[best_label]
    for label in _LABELS:
        if label not in throughputs:
            continue
        tp = throughputs[label]
        print(f"  {label:>16s}: {tp:8.2f} img/s  norm={tp/best_val:.4f}")
    print(f"  best method: {best_label}")
    print(f"{'='*50}")

    plot_throughput(throughputs, png_path, model_title=model_title, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
