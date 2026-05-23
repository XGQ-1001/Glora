"""Checkpoint save / load with LoRA-aware state separation.

A Glora checkpoint can contain three orthogonal payloads:

* ``state_dict``       — full policy weights (foundation pretrain output)
* ``lora_state_dict``  — only the LoRA delta parameters (saved during Step 4)
* ``best``             — best (latency, schedule_order, episode) snapshot

The helpers here make sure each downstream consumer reads only the slice it
needs, regardless of whether the checkpoint was produced by single-model
training, foundation pretraining, or LoRA fine-tuning.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

import torch

from .policy import GloraActorCritic


def save_lora_checkpoint(
    path: str,
    policy: GloraActorCritic,
    *,
    cfg,
    history: Dict,
    base_path: str,
    best: Dict,
    update_count: int,
) -> None:
    """Save only the LoRA delta + metadata. Base weights are referenced by path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lora_state = {
        name: p.detach().cpu()
        for name, p in policy.named_parameters()
        if p.requires_grad
    }
    tmp = f"{path}.tmp"
    torch.save({
        "lora_state_dict": lora_state,
        "config": cfg,
        "history": history,
        "best": best,
        "base_checkpoint": os.path.abspath(base_path),
        "pe_dim": cfg.pe_dim,
        "hidden_dim": cfg.hidden_dim,
        "emb_dim": cfg.emb_dim,
        "n_heads": cfg.n_heads,
        "n_layers": cfg.n_layers,
        "update_count": int(update_count),
        "status": "lora",
    }, tmp)
    os.replace(tmp, path)


def load_full_checkpoint(path: str, device: torch.device | str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint file (works for both full-state and LoRA-only files)."""
    return torch.load(path, map_location=device, weights_only=False)


def restore_policy(
    ckpt: Dict[str, Any],
    *,
    static_in_dim: int,
    device: torch.device,
    pe_dim: Optional[int] = None,
) -> GloraActorCritic:
    """Materialise a ``GloraActorCritic`` matching the geometry in ``ckpt``."""
    cfg = ckpt.get("config", None)
    pe = pe_dim if pe_dim is not None else int(ckpt.get("pe_dim", getattr(cfg, "pe_dim", 16)))
    hidden = int(ckpt.get("hidden_dim", getattr(cfg, "hidden_dim", 128)))
    emb = int(ckpt.get("emb_dim", getattr(cfg, "emb_dim", 128)))
    heads = int(ckpt.get("n_heads", getattr(cfg, "n_heads", 4)))
    layers = int(ckpt.get("n_layers", getattr(cfg, "n_layers", 3)))

    policy = GloraActorCritic(
        static_in_dim=static_in_dim,
        pe_dim=pe,
        hidden_dim=hidden,
        emb_dim=emb,
        n_heads=heads,
        n_layers=layers,
    ).to(device)

    state_dict = ckpt.get("state_dict")
    if state_dict is not None:
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [warn] {len(missing)} missing keys when loading state_dict")
        if unexpected:
            print(f"  [warn] {len(unexpected)} unexpected keys when loading state_dict")
    return policy


def apply_lora_state(policy: GloraActorCritic, lora_state: Dict[str, torch.Tensor]) -> List[str]:
    """Copy LoRA tensors into ``policy``. Returns the list of updated names."""
    own = dict(policy.named_parameters())
    applied: List[str] = []
    for name, tensor in lora_state.items():
        if name not in own:
            continue
        target = own[name]
        if target.shape != tensor.shape:
            print(f"  [warn] shape mismatch for {name}: ckpt {tensor.shape} vs model {target.shape}")
            continue
        with torch.no_grad():
            target.data.copy_(tensor.to(target.device, dtype=target.dtype))
        applied.append(name)
    return applied


__all__ = [
    "save_lora_checkpoint",
    "load_full_checkpoint",
    "restore_policy",
    "apply_lora_state",
]
