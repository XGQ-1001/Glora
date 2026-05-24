"""Step 2: single-model Mask-PPO training with Graph Transformer + LapPE.

This is the per-model variant of Glora training. It uses the same trainer
that drives Step 3 (foundation pretraining), but the sample provider always
returns the same (model, batch_size) configuration so the policy specialises
to a single DAG.

Example::

    python scripts/step2_train_single.py --model googlenet --episodes 500
    python scripts/step2_train_single.py --model bert_base --episodes 400 --batch-size 1
"""

from __future__ import annotations

import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402

from glora import utils as glora_utils  # noqa: E402
from glora.data_pool import ModelPool, PoolEntry  # noqa: E402
from glora.models import known_models  # noqa: E402
from glora.policy import GloraActorCritic  # noqa: E402
from glora.reward_shaping import RewardConfig  # noqa: E402
from glora.trainer import GloraTrainConfig, TrainSample, train  # noqa: E402

from gnn_strategy.graph_state import D_STATIC  # type: ignore  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Glora Step 2 — single-model training")
    p.add_argument("--model", required=True, choices=sorted(known_models()))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--batch-episodes", type=int, default=8)
    p.add_argument("--mini-batch-size", type=int, default=256)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--gae-lambda", type=float, default=1.0)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--entropy-coef-end", type=float, default=0.0)
    p.add_argument("--pe-dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--emb", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--streams", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--dense-coef", type=float, default=0.05)
    p.add_argument("--potential-coef", type=float, default=0.5)
    p.add_argument("--use-shaping", action="store_true", help="enable dense + potential shaping")
    p.add_argument("--no-shaping", action="store_true", help="kept for compatibility; terminal-only is now the default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default=None, help="checkpoint path; auto-generated if omitted")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    glora_utils.set_global_seed(args.seed)

    if not torch.cuda.is_available():
        print("[fatal] Glora Step 2 requires a CUDA GPU for real-latency PPO")
        return 2
    device = torch.device("cuda")

    pool = ModelPool(
        entries=[PoolEntry(name=args.model, batch_sizes=[args.batch_size], weight=1.0)],
        device=device,
        bench_iters=args.iters * 3,
        bench_warmups=args.warmups,
    )
    sample = pool.get(args.model, args.batch_size)
    print(f"[step2] cached: {sample.summary()}")

    policy = GloraActorCritic(
        static_in_dim=D_STATIC,
        pe_dim=args.pe_dim,
        hidden_dim=args.hidden,
        emb_dim=args.emb,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    use_shaping = args.use_shaping and not args.no_shaping
    reward_cfg = RewardConfig(
        dense_coef=args.dense_coef if use_shaping else 0.0,
        potential_coef=args.potential_coef if use_shaping else 0.0,
        use_dense=use_shaping,
        use_potential=use_shaping,
        multi_baseline=False,
    )

    cfg = GloraTrainConfig(
        episodes=args.episodes,
        batch_episodes=args.batch_episodes,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps,
        lr=args.lr,
        gamma=1.0,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        entropy_coef_end=args.entropy_coef_end,
        pe_dim=args.pe_dim,
        hidden_dim=args.hidden,
        emb_dim=args.emb,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=args.dropout,
        n_streams=args.streams,
        bench_iters=args.iters,
        bench_warmups=args.warmups,
        reward=reward_cfg,
        seed=args.seed,
    )

    save_path = args.save or glora_utils.artifacts_dir("step2") + f"/{args.model}_bs{args.batch_size}.pt"
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

    print(f"[step2] training policy → {save_path}")
    summary = train(
        policy=policy,
        sample_provider=provider,
        cfg=cfg,
        device=device,
        save_path=save_path,
    )
    print(f"[step2] done. best_L={summary['best_L_ms']:.4f}ms over {len(summary['history']['episodes'])} ep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
