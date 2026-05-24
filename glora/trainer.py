"""Mask-PPO trainer with reward shaping for Glora.

Supports two execution modes:

1. **Single-model training** — caller supplies a single ``CachedSample`` and
   episodes always operate on the same DAG. This is Step 2 in the project.
2. **Multi-model foundation pretraining** — caller supplies a ``ModelPool``;
   each episode resamples a (model, batch) pair. This is Step 3.
3. **LoRA fine-tuning** — caller passes ``trainable_params=lora_params(policy)``
   so only the LoRA delta matrices are optimised. This is Step 4.

The trainer always uses:
- masked action sampling (invalid actions get probability 0)
- terminal real-latency reward, with optional dense / potential shaping
- vectorised mini-batch PPO updates
- cosine LR decay
- best (latency, schedule order) tracking for reproducibility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from .utils import _GNN_STRATEGY_ROOT  # noqa: F401
from .lap_pe import attach_pe_to_graph_state
from .policy import GloraActorCritic
from .reward_shaping import PotentialShaper, RewardConfig, remaining_potential
from .runtime import (
    cosine_lr_schedule,
    real_latency_for_order,
    schedule_order_from_env,
)

from gnn_strategy.env import SchedulingEnv  # type: ignore


# ----------------------------------------------------------------------
# Configs and helpers
# ----------------------------------------------------------------------


@dataclass
class GloraTrainConfig:
    # PPO
    episodes: int = 500
    batch_episodes: int = 8
    mini_batch_size: int = 256
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    lr: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 1.0
    entropy_coef: float = 0.02
    entropy_coef_end: Optional[float] = None
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network
    pe_dim: int = 16
    hidden_dim: int = 128
    emb_dim: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1

    # Environment
    n_streams: int = 8
    bench_iters: int = 20
    bench_warmups: int = 5

    # Reward shaping
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Misc
    autosave_interval: int = 1
    seed: int = 0
    log_every: int = 1


@dataclass
class Transition:
    x: torch.Tensor
    lap_pe: torch.Tensor
    dyn_node: torch.Tensor
    glob: torch.Tensor
    ready_mask: torch.Tensor
    action: int
    log_prob: float
    value: float
    reward: float


def _scheduled_entropy_coef(cfg: GloraTrainConfig, ep: int, total: int) -> float:
    end = cfg.entropy_coef_end
    if end is None or total <= 1:
        return float(cfg.entropy_coef)
    t = ep / float(total - 1)
    return float(cfg.entropy_coef + (end - cfg.entropy_coef) * t)


def _compute_gae(
    rewards: List[float],
    values: List[float],
    last_value: float,
    gamma: float,
    lam: float,
) -> Tuple[List[float], List[float]]:
    T = len(rewards)
    adv = [0.0] * T
    ret = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else last_value
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
        ret[t] = adv[t] + values[t]
    return adv, ret


# ----------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------


def rollout_episode(
    policy: GloraActorCritic,
    gs,
    fx_module,
    inputs: Sequence[torch.Tensor],
    *,
    cfg: GloraTrainConfig,
    device: torch.device,
    shaper: PotentialShaper,
    opara_latency_ms: float,
) -> Tuple[List[Transition], Dict[str, float]]:
    """Run one full schedule, return transitions + episode metrics."""
    env = SchedulingEnv(gs, n_streams=cfg.n_streams, device=device)
    env.reset()
    lap_pe = attach_pe_to_graph_state(gs, cfg.pe_dim).to(device)
    x = gs.x.to(device)

    with torch.no_grad():
        h_static = policy.encode_static(x, lap_pe)

    transitions: List[Transition] = []
    prev_makespan = env.current_makespan()
    prev_phi = remaining_potential(env)

    while not env.is_done():
        dyn = env.dynamic_node_features().to(device)
        glob = env.global_features().to(device)
        mask = env.ready_mask().to(device)
        if mask.sum().item() <= 0:
            break

        with torch.no_grad():
            dist, value = policy.act(h_static, dyn, glob, mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        result = env.step(int(action.item()))
        new_makespan = result.info.get("makespan", env.current_makespan())
        new_phi = remaining_potential(env)

        r_step = shaper.step_reward(
            prev_makespan=prev_makespan,
            new_makespan=new_makespan,
            prev_phi=prev_phi,
            new_phi=new_phi,
        )

        transitions.append(Transition(
            x=x.detach(),
            lap_pe=lap_pe.detach(),
            dyn_node=dyn.detach(),
            glob=glob.detach(),
            ready_mask=mask.detach(),
            action=int(action.item()),
            log_prob=float(log_prob.item()),
            value=float(value.item()),
            reward=r_step,
        ))

        prev_makespan = new_makespan
        prev_phi = new_phi

    order = schedule_order_from_env(env, gs)
    L_gnn = real_latency_for_order(
        fx_module,
        inputs,
        order,
        iterations=cfg.bench_iters,
        warmups=cfg.bench_warmups,
    )
    terminal = shaper.terminal_reward(L_gnn, opara_latency_ms)
    if transitions:
        transitions[-1] = Transition(
            **{**transitions[-1].__dict__, "reward": transitions[-1].reward + terminal}
        )

    metrics = {
        "L_gnn_ms": float(L_gnn),
        "L_opara_ms": float(opara_latency_ms),
        "speedup_pct": float((opara_latency_ms - L_gnn) / max(opara_latency_ms, 1e-9) * 100.0),
        "terminal_reward": float(terminal),
        "rollout_steps": len(transitions),
    }
    return transitions, metrics, order


# ----------------------------------------------------------------------
# PPO update (vectorised mini-batch SGD)
# ----------------------------------------------------------------------


def _ppo_update(
    policy: GloraActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: List[Transition],
    advantages: Sequence[float],
    returns: Sequence[float],
    cfg: GloraTrainConfig,
    device: torch.device,
    entropy_coef: float,
) -> Dict[str, float]:
    T = len(transitions)
    if T == 0:
        return {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "step_stats": []}

    adv_t = torch.tensor(list(advantages), dtype=torch.float32, device=device)
    ret_t = torch.tensor(list(returns), dtype=torch.float32, device=device)
    old_lp = torch.tensor([tr.log_prob for tr in transitions], dtype=torch.float32, device=device)
    actions = torch.tensor([tr.action for tr in transitions], dtype=torch.long, device=device)

    x_all = torch.stack([tr.x for tr in transitions])
    pe_all = torch.stack([tr.lap_pe for tr in transitions])
    dyn_all = torch.stack([tr.dyn_node for tr in transitions])
    glob_all = torch.stack([tr.glob for tr in transitions])
    mask_all = torch.stack([tr.ready_mask for tr in transitions])

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    mbs = min(cfg.mini_batch_size, T)
    total_pg = total_v = total_ent = 0.0
    grad_steps = 0
    step_stats: List[Dict[str, float]] = []

    for _ in range(cfg.ppo_epochs):
        perm = np.random.permutation(T)
        for start in range(0, T, mbs):
            idx = torch.tensor(perm[start:start + mbs], dtype=torch.long, device=device)

            h_static = policy.encode_static(x_all[idx], pe_all[idx])
            lp, val, ent = policy.batch_evaluate_actions(
                h_static, dyn_all[idx], glob_all[idx], mask_all[idx],
                actions[idx],
            )
            ratio = torch.exp(lp - old_lp[idx])
            surr1 = ratio * adv_t[idx]
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_t[idx]
            pg_loss = -torch.min(surr1, surr2).mean()
            v_loss = (val - ret_t[idx]).pow(2).mean()
            ent_mean = ent.mean()

            loss = pg_loss + cfg.value_coef * v_loss - entropy_coef * ent_mean

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in optimizer.param_groups[0]["params"] if p.requires_grad],
                cfg.max_grad_norm,
            )
            optimizer.step()

            total_pg += pg_loss.item()
            total_v += v_loss.item()
            total_ent += ent_mean.item()
            grad_steps += 1
            step_stats.append({
                "pg_loss": float(pg_loss.item()),
                "v_loss": float(v_loss.item()),
                "entropy": float(ent_mean.item()),
            })

    return {
        "pg_loss": total_pg / max(grad_steps, 1),
        "v_loss": total_v / max(grad_steps, 1),
        "entropy": total_ent / max(grad_steps, 1),
        "step_stats": step_stats,
    }


# ----------------------------------------------------------------------
# Main training entry-point
# ----------------------------------------------------------------------


SampleProvider = Callable[[], "TrainSample"]


@dataclass
class TrainSample:
    """A single (model, batch_size, fx, gs, opara) bundle for one episode."""

    name: str
    batch_size: int
    fx_module: torch.fx.GraphModule
    inputs: Tuple[torch.Tensor, ...]
    graph_state: object
    opara_latency_ms: float
    model_class_name: str


def train(
    policy: GloraActorCritic,
    sample_provider: SampleProvider,
    cfg: GloraTrainConfig,
    *,
    device: torch.device,
    trainable_params: Optional[List[nn.Parameter]] = None,
    save_path: Optional[str] = None,
    on_episode_end: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """Run Mask-PPO training for ``cfg.episodes`` episodes.

    Parameters
    ----------
    policy:
        Glora policy (already on ``device``).
    sample_provider:
        Callable invoked once per episode. Returning a fresh ``TrainSample``
        enables foundation pretraining; returning the same sample yields
        single-model training.
    trainable_params:
        Optional explicit parameter list (used in LoRA mode). Defaults to all
        policy parameters that require grad.
    save_path:
        If set, writes ``{save}.pt`` (best policy) and ``{save}_latest.pt``
        (most recent state) checkpoints.
    """
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if trainable_params is None:
        trainable_params = [p for p in policy.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found")

    # PPO compares old and new log-probs; deterministic forwards keep dropout
    # noise out of that ratio while still allowing gradients to flow.
    policy.eval()
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.lr)

    # cosine LR over PPO update count = ceil(episodes / batch_episodes)
    total_updates = max(1, math.ceil(cfg.episodes / max(cfg.batch_episodes, 1)))
    scheduler = cosine_lr_schedule(optimizer, total_steps=total_updates, base_lr=cfg.lr)

    best_L = math.inf
    best_order: List[str] = []
    best_episode = -1
    best_sample_name = ""
    best_batch_size = -1
    update_count = 0

    history: Dict[str, list] = {
        "episodes": [],
        "grad_steps": [],
    }

    batch_transitions: List[Transition] = []
    batch_advantages: List[float] = []
    batch_returns: List[float] = []
    pending_recs: List[Dict] = []

    t0 = time.time()

    for ep in range(cfg.episodes):
        sample = sample_provider()
        attach_pe_to_graph_state(sample.graph_state, cfg.pe_dim)

        shaper = PotentialShaper(cfg.reward, initial_makespan=float(
            getattr(sample.graph_state, "_initial_makespan_hint", 1.0)
        ))
        # The initial makespan is conceptually the simulator's; we don't have
        # it directly, but SchedulingEnv recomputes it. The shaper only uses
        # it as a normaliser so any positive value is fine.

        transitions, metrics, order = rollout_episode(
            policy,
            sample.graph_state,
            sample.fx_module,
            sample.inputs,
            cfg=cfg,
            device=device,
            shaper=shaper,
            opara_latency_ms=sample.opara_latency_ms,
        )

        T = len(transitions)
        rewards = [tr.reward for tr in transitions]
        values = [tr.value for tr in transitions]
        advantages, returns = _compute_gae(
            rewards, values, last_value=0.0,
            gamma=cfg.gamma, lam=cfg.gae_lambda,
        )

        batch_transitions.extend(transitions)
        batch_advantages.extend(advantages)
        batch_returns.extend(returns)

        L_gnn = metrics["L_gnn_ms"]
        if L_gnn < best_L:
            best_L = L_gnn
            best_order = list(order)
            best_episode = ep
            best_sample_name = sample.name
            best_batch_size = sample.batch_size
            if save_path:
                _save_checkpoint(
                    save_path,
                    policy=policy,
                    cfg=cfg,
                    history=history,
                    best={
                        "L_ms": float(best_L),
                        "order": best_order,
                        "episode": best_episode,
                        "sample_name": best_sample_name,
                        "batch_size": best_batch_size,
                    },
                    update_count=update_count,
                    status="best",
                )

        rec = {
            "episode": ep,
            "sample_name": sample.name,
            "batch_size": sample.batch_size,
            **metrics,
            "best_L_ms": float(best_L),
        }
        pending_recs.append(rec)
        if on_episode_end is not None:
            on_episode_end(rec)

        if ep % cfg.log_every == 0:
            beat = " ★ BEAT" if L_gnn < metrics["L_opara_ms"] else ""
            print(
                f"[ep {ep:04d}] {sample.name:>14s} bs={sample.batch_size:<2d}"
                f"  L_gnn={L_gnn:.4f}ms  Opara={metrics['L_opara_ms']:.4f}ms"
                f"  Δ={metrics['speedup_pct']:+.2f}%  best={(metrics['L_opara_ms']-best_L)/metrics['L_opara_ms']*100.0:+.2f}%{beat}"
            )

        batch_full = len(pending_recs) >= cfg.batch_episodes
        is_last = ep == cfg.episodes - 1
        if batch_transitions and (batch_full or is_last):
            ec = _scheduled_entropy_coef(cfg, ep, cfg.episodes)
            stats = _ppo_update(
                policy, optimizer, batch_transitions, batch_advantages, batch_returns,
                cfg=cfg, device=device, entropy_coef=ec,
            )
            scheduler.step()
            update_count += 1
            cur_lr = optimizer.param_groups[0]["lr"]
            per_step = stats.pop("step_stats", [])

            for r in pending_recs:
                r.update(stats)
                r["entropy_coef_used"] = float(ec)
                r["lr"] = float(cur_lr)
            history["episodes"].extend(pending_recs)

            base_step = len(history["grad_steps"])
            for i, ss in enumerate(per_step):
                ss["global_step"] = base_step + i
                ss["ppo_update"] = update_count
                ss["episode"] = ep
            history["grad_steps"].extend(per_step)

            print(
                f"  >>> PPO update #{update_count} "
                f"({len(pending_recs)} eps) pg={stats['pg_loss']:.4f} "
                f"v={stats['v_loss']:.4f} ent={stats['entropy']:.3f} lr={cur_lr:.2e}"
            )

            if save_path and cfg.autosave_interval > 0 and update_count % cfg.autosave_interval == 0:
                _save_checkpoint(
                    save_path,
                    policy=policy,
                    cfg=cfg,
                    history=history,
                    best={
                        "L_ms": float(best_L),
                        "order": best_order,
                        "episode": best_episode,
                        "sample_name": best_sample_name,
                        "batch_size": best_batch_size,
                    },
                    update_count=update_count,
                    status="latest",
                )

            batch_transitions = []
            batch_advantages = []
            batch_returns = []
            pending_recs = []

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60.0:.1f} min")
    print(f"Best L_gnn = {best_L:.4f} ms on {best_sample_name} (bs={best_batch_size})")

    if save_path:
        _save_checkpoint(
            save_path,
            policy=policy,
            cfg=cfg,
            history=history,
            best={
                "L_ms": float(best_L),
                "order": best_order,
                "episode": best_episode,
                "sample_name": best_sample_name,
                "batch_size": best_batch_size,
            },
            update_count=update_count,
            status="final",
        )

    return {
        "history": history,
        "best_L_ms": float(best_L),
        "best_order": best_order,
        "best_episode": best_episode,
        "best_sample_name": best_sample_name,
        "best_batch_size": best_batch_size,
        "elapsed_sec": float(elapsed),
    }


def _save_checkpoint(
    save_path: str,
    *,
    policy: GloraActorCritic,
    cfg: GloraTrainConfig,
    history: Dict,
    best: Dict,
    update_count: int,
    status: str,
) -> None:
    if not save_path:
        return
    root, ext = os.path.splitext(save_path)
    ext = ext or ".pt"
    if status == "best":
        path = save_path
    elif status == "final":
        path = f"{root}_final{ext}"
    else:
        path = f"{root}_latest{ext}"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save({
        "state_dict": policy.state_dict(),
        "config": cfg,
        "history": history,
        "best": best,
        "pe_dim": cfg.pe_dim,
        "hidden_dim": cfg.hidden_dim,
        "emb_dim": cfg.emb_dim,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "update_count": int(update_count),
        "status": status,
    }, tmp)
    os.replace(tmp, path)


__all__ = [
    "GloraTrainConfig",
    "Transition",
    "TrainSample",
    "rollout_episode",
    "train",
]
