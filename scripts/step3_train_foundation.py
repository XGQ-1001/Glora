"""Step 3: foundation pretraining over a *pool* of (model, batch) configurations.

The trainer re-uses Step 2 machinery; the only difference is the sample
provider, which now draws a (model, batch) pair at random from a
``ModelPool``. The first time a configuration is sampled we pay the FX +
profiling + Opara baseline cost once and cache the result.

Usage::

    python scripts/step3_train_foundation.py \
        --pool "googlenet:1;inception_v3:1;resnet50:1;deepfm:1;bert_base:1" \
        --episodes 2000 --batch-episodes 16

The default pool covers ``FOUNDATION_POOL`` from ``glora.models`` at
batch_size=1, which is a healthy balance between CNN, recommendation and
Transformer DAGs. Use ``--pool`` to override.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402

from glora import utils as glora_utils  # noqa: E402
from glora.data_pool import ModelPool, PoolEntry, parse_pool_spec  # noqa: E402
from glora.models import FOUNDATION_POOL, known_models  # noqa: E402
from glora.policy import GloraActorCritic  # noqa: E402
from glora.reward_shaping import RewardConfig  # noqa: E402
from glora.trainer import GloraTrainConfig, TrainSample, train  # noqa: E402

from gnn_strategy.graph_state import D_STATIC  # type: ignore  # noqa: E402


def default_pool_spec() -> str:
    return ";".join(f"{name}:1" for name in FOUNDATION_POOL)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Glora Step 3 — foundation pretraining")
    p.add_argument("--pool", type=str, default=default_pool_spec(),
                   help="pool spec, e.g. 'googlenet:1,4;bert_base:1'")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--batch-episodes", type=int, default=16)
    p.add_argument("--mini-batch-size", type=int, default=512)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--gae-lambda", type=float, default=1.0)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--entropy-coef-end", type=float, default=0.0)
    p.add_argument("--pe-dim", type=int, default=16)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--emb", type=int, default=192)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--streams", type=int, default=8)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--warmups", type=int, default=5)
    p.add_argument("--dense-coef", type=float, default=0.05)
    p.add_argument("--potential-coef", type=float, default=0.5)
    p.add_argument("--use-shaping", action="store_true", help="enable dense + potential shaping")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--warmup-pool", action="store_true",
                   help="touch every (model, bs) once before training so FX / Opara cost is paid up front")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    glora_utils.set_global_seed(args.seed)

    if not torch.cuda.is_available():
        print("[fatal] foundation pretraining needs CUDA")
        return 2
    device = torch.device("cuda")

    entries: List[PoolEntry] = parse_pool_spec(args.pool)
    for e in entries:
        if e.name not in known_models():
            print(f"[warn] unknown model in pool: {e.name}")
    print(f"[step3] pool: {[(e.name, list(e.batch_sizes), e.weight) for e in entries]}")

    pool = ModelPool(
        entries=entries,
        device=device,
        bench_iters=args.iters * 3,
        bench_warmups=args.warmups,
    )
    if args.warmup_pool:
        print("[step3] warming up the pool (one-shot FX + Opara per config) ...")
        for name, bs in pool.all_configs():
            s = pool.get(name, bs)
            print(f"  - {s.summary()}")

    policy = GloraActorCritic(
        static_in_dim=D_STATIC,
        pe_dim=args.pe_dim,
        hidden_dim=args.hidden,
        emb_dim=args.emb,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    print(f"[step3] policy params: {sum(p.numel() for p in policy.parameters())/1e6:.2f}M")

    reward_cfg = RewardConfig(
        dense_coef=args.dense_coef,
        potential_coef=args.potential_coef,
        use_dense=args.use_shaping,
        use_potential=args.use_shaping,
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

    save_path = args.save or os.path.join(
        glora_utils.artifacts_dir("step3"), "foundation_base.pt"
    )
    save_path = os.path.abspath(save_path)
    print(f"[step3] save → {save_path}")

    rng = random.Random(args.seed)

    def provider() -> TrainSample:
        sample = pool.sample(rng=rng)
        return TrainSample(
            name=sample.name,
            batch_size=sample.batch_size,
            fx_module=sample.fx_module,
            inputs=sample.inputs,
            graph_state=sample.graph_state,
            opara_latency_ms=sample.opara_latency_ms,
            model_class_name=sample.model_class_name,
        )

    summary = train(
        policy=policy,
        sample_provider=provider,
        cfg=cfg,
        device=device,
        save_path=save_path,
    )
    print(f"[step3] best L_gnn = {summary['best_L_ms']:.4f}ms "
          f"on {summary['best_sample_name']} bs={summary['best_batch_size']} (ep {summary['best_episode']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
