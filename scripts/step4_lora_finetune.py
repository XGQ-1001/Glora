"""Step 4: parameter-efficient LoRA fine-tuning on an unseen target model.

Workflow
--------
1. Load the foundation policy produced by Step 3.
2. Freeze every parameter, then wrap selected ``nn.Linear`` modules with
   :class:`glora.lora.LoRALinear` adapters.
3. Run a small number of episodes (few-shot, default 50) on the target
   model. Only the LoRA delta matrices receive gradients, which keeps the
   adapter checkpoint tiny (< 1 MB even for a 100k+-node DAG).
4. Save the LoRA-only state plus zero-shot vs few-shot latencies for the
   final comparison.

Example::

    python scripts/step4_lora_finetune.py \
        --base artifacts/step3/foundation_base.pt \
        --target mobilenet_v2 --batch-size 1 \
        --episodes 80 --rank 8 --lr 5e-4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402

from glora import utils as glora_utils  # noqa: E402
from glora.checkpoint import (  # noqa: E402
    apply_lora_state,
    load_full_checkpoint,
    restore_policy,
    save_lora_checkpoint,
)
from glora.data_pool import ModelPool, PoolEntry  # noqa: E402
from glora.lap_pe import attach_pe_to_graph_state  # noqa: E402
from glora.lora import (  # noqa: E402
    apply_lora_to_linear,
    find_linear_module_names,
    freeze_all_parameters,
    lora_parameters,
    trainable_param_count,
)
from glora.models import known_models  # noqa: E402
from glora.reward_shaping import PotentialShaper, RewardConfig  # noqa: E402
from glora.runtime import real_latency_for_order, schedule_order_from_env  # noqa: E402
from glora.trainer import GloraTrainConfig, TrainSample, rollout_episode, train  # noqa: E402

from gnn_strategy.env import SchedulingEnv  # type: ignore  # noqa: E402
from gnn_strategy.graph_state import D_STATIC  # type: ignore  # noqa: E402


def evaluate_greedy(
    policy,
    sample,
    pe_dim: int,
    iterations: int,
    warmups: int,
    n_streams: int,
    device: torch.device,
) -> dict:
    """Run one *greedy* rollout (argmax action) and time the resulting schedule."""
    gs = sample.graph_state
    env = SchedulingEnv(gs, n_streams=n_streams, device=device)
    env.reset()

    lap_pe = attach_pe_to_graph_state(gs, pe_dim).to(device)
    x = gs.x.to(device)
    policy.eval()
    with torch.no_grad():
        h_static = policy.encode_static(x, lap_pe)

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

    order = schedule_order_from_env(env, gs)
    L = real_latency_for_order(
        sample.fx_module, sample.inputs, order,
        iterations=iterations, warmups=warmups,
    )
    return {
        "L_ms": float(L),
        "speedup_vs_opara": float(
            (sample.opara_latency_ms - L) / max(sample.opara_latency_ms, 1e-9) * 100.0
        ),
        "order_len": len(order),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Glora Step 4 — LoRA few-shot fine-tuning")
    p.add_argument("--base", required=True, help="path to Step 3 foundation checkpoint (.pt)")
    p.add_argument("--target", required=True, choices=sorted(known_models()))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--batch-episodes", type=int, default=4)
    p.add_argument("--mini-batch-size", type=int, default=128)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--target-modules", type=str, default="linear1,linear2",
                   help="comma-separated last-component names of Linear modules to wrap. "
                        "Linear modules sitting inside an nn.MultiheadAttention (i.e. out_proj) "
                        "are always skipped because attention bypasses submodule forward.")
    p.add_argument("--streams", type=int, default=8)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--report", type=str, default=None,
                   help="JSON file to write zero-shot / few-shot summary")
    p.add_argument("--no-eval", action="store_true", help="skip evaluation rollouts")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    glora_utils.set_global_seed(args.seed)

    if not torch.cuda.is_available():
        print("[fatal] LoRA fine-tuning requires CUDA")
        return 2
    device = torch.device("cuda")

    print(f"[step4] loading base from {args.base}")
    ckpt = load_full_checkpoint(args.base, device="cpu")
    policy = restore_policy(ckpt, static_in_dim=D_STATIC, device=device)
    pe_dim = int(ckpt.get("pe_dim", 16))

    pool = ModelPool(
        entries=[PoolEntry(name=args.target, batch_sizes=[args.batch_size], weight=1.0)],
        device=device,
        bench_iters=args.iters * 3,
        bench_warmups=args.warmups,
    )
    sample = pool.get(args.target, args.batch_size)
    print(f"[step4] target: {sample.summary()}")

    # ---- Zero-shot baseline ----
    zero_shot = None
    if not args.no_eval:
        print("[step4] zero-shot evaluation ...")
        zero_shot = evaluate_greedy(
            policy, sample, pe_dim=pe_dim,
            iterations=args.iters * 3, warmups=args.warmups,
            n_streams=args.streams, device=device,
        )
        print(f"  L_zero_shot = {zero_shot['L_ms']:.4f}ms "
              f"(Δ vs Opara {zero_shot['speedup_vs_opara']:+.2f}%)")

    # ---- Freeze + inject LoRA adapters ----
    freeze_all_parameters(policy)
    target_components = {t.strip() for t in args.target_modules.split(",") if t.strip()}
    candidate_names = find_linear_module_names(
        policy, name_filter=target_components, skip_inside_attention=True,
    )
    if not candidate_names:
        print(f"[fatal] no Linear modules matched target components {target_components}")
        return 3
    patched = apply_lora_to_linear(
        policy, candidate_names,
        rank=args.rank, alpha=args.alpha, dropout=args.lora_dropout,
    )
    trainable, total = trainable_param_count(policy)
    print(f"[step4] LoRA: wrapped {len(patched)} Linear modules → "
          f"trainable={trainable/1e3:.1f}K / total={total/1e6:.2f}M "
          f"({100.0 * trainable / max(total, 1):.3f}%)")
    if len(patched) <= 10:
        for n in patched:
            print(f"  - {n}")

    cfg = GloraTrainConfig(
        episodes=args.episodes,
        batch_episodes=args.batch_episodes,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        clip_eps=0.2,
        lr=args.lr,
        gamma=1.0,
        gae_lambda=1.0,
        entropy_coef=0.005,
        entropy_coef_end=0.0,
        pe_dim=pe_dim,
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        emb_dim=int(ckpt.get("emb_dim", 128)),
        n_heads=int(ckpt.get("n_heads", 4)),
        n_layers=int(ckpt.get("n_layers", 3)),
        n_streams=args.streams,
        bench_iters=args.iters,
        bench_warmups=args.warmups,
        reward=RewardConfig(),
        seed=args.seed,
    )

    save_path = args.save or os.path.join(
        glora_utils.artifacts_dir("step4"),
        f"lora_{args.target}_bs{args.batch_size}_rank{args.rank}.pt",
    )
    save_path = os.path.abspath(save_path)

    def provider() -> TrainSample:
        return TrainSample(
            name=sample.name,
            batch_size=sample.batch_size,
            fx_module=sample.fx_module,
            inputs=sample.inputs,
            graph_state=sample.graph_state,
            opara_latency_ms=sample.opara_latency_ms,
            model_class_name=sample.model_class_name,
        )

    print(f"[step4] LoRA fine-tuning for {args.episodes} episodes → {save_path}")
    summary = train(
        policy=policy,
        sample_provider=provider,
        cfg=cfg,
        device=device,
        trainable_params=lora_parameters(policy),
        save_path=save_path,
    )

    save_lora_checkpoint(
        save_path,
        policy=policy,
        cfg=cfg,
        history=summary["history"],
        base_path=args.base,
        best={
            "L_ms": float(summary["best_L_ms"]),
            "order": summary["best_order"],
            "episode": summary["best_episode"],
            "sample_name": summary["best_sample_name"],
            "batch_size": summary["best_batch_size"],
        },
        update_count=int(len(summary["history"]["grad_steps"])),
    )

    # ---- Few-shot evaluation (greedy) ----
    few_shot = None
    if not args.no_eval:
        print("[step4] few-shot greedy evaluation ...")
        few_shot = evaluate_greedy(
            policy, sample, pe_dim=pe_dim,
            iterations=args.iters * 3, warmups=args.warmups,
            n_streams=args.streams, device=device,
        )
        print(f"  L_few_shot = {few_shot['L_ms']:.4f}ms "
              f"(Δ vs Opara {few_shot['speedup_vs_opara']:+.2f}%)")

    report = {
        "target": args.target,
        "batch_size": args.batch_size,
        "rank": args.rank,
        "alpha": args.alpha,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": trainable / max(total, 1),
        "opara_latency_ms": sample.opara_latency_ms,
        "zero_shot": zero_shot,
        "few_shot": few_shot,
        "training_best_L_ms": float(summary["best_L_ms"]),
        "training_episodes": cfg.episodes,
        "checkpoint": save_path,
    }
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[step4] wrote report → {args.report}")
    print(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
