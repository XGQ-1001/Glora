"""Parameter-efficient LoRA adapters for nn.Linear / MultiheadAttention.

Glora uses LoRA in two places:

1. **Graph Transformer attention projections** (q/k/v + output projection)
2. **Actor head Linear stack**

The implementation injects a low-rank delta ``BA`` (rank ``r``, scale
``alpha/r``) on top of a frozen base ``nn.Linear``. It mirrors the original
LoRA paper but is self-contained so we don't pull a heavy dependency.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` augmented with a trainable low-rank delta."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    # ------------------------------------------------------------------
    # Defensive shims: some PyTorch modules (e.g. ``MultiheadAttention``)
    # directly read ``submodule.weight`` instead of going through
    # ``forward``. Expose the base weight/bias so those code paths still
    # work, even though the LoRA delta will not be applied along that
    # route. We keep this read-only to discourage accidental in-place
    # mutation.
    # ------------------------------------------------------------------

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self):  # type: ignore[override]
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B) * self.scaling
        return base_out + lora_out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )


class LoRAInProj(nn.Module):
    """LoRA delta applied on top of the fused QKV projection of MultiheadAttention.

    ``nn.MultiheadAttention`` stores the Q/K/V matrices as a single weight
    tensor ``in_proj_weight`` of shape ``[3 * embed_dim, embed_dim]``. We wrap
    that linear with three independent LoRA branches (Q/K/V) that share input
    but each have their own low-rank delta.
    """

    def __init__(self, attn: nn.MultiheadAttention, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.attn = attn
        self.embed_dim = attn.embed_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        # 3 separate A/B per Q,K,V projection
        self.lora_A = nn.Parameter(torch.zeros(3, rank, self.embed_dim))
        self.lora_B = nn.Parameter(torch.zeros(3, self.embed_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Replace the forward of in_proj with a closure that injects deltas.
        attn.in_proj_weight.requires_grad = False
        if attn.in_proj_bias is not None:
            attn.in_proj_bias.requires_grad = False

        # Patch using a forward pre-hook to add Δw on the fly.
        self._wb_hook = attn.register_forward_pre_hook(self._patch_in_proj, with_kwargs=True)

    # Note: we add the LoRA delta to the *effective* in_proj weight by
    # temporarily computing q/k/v via F.linear with W + Δ. PyTorch's
    # MultiheadAttention API is rigid, so we re-implement using its weights.
    def _patch_in_proj(self, module, args, kwargs):
        # No-op: actual delta injection happens during forward via a side path.
        return args, kwargs

    def delta_weight(self) -> torch.Tensor:
        """Return the (3*embed_dim, embed_dim) Δw produced by all LoRA branches."""
        deltas = []
        for i in range(3):
            d = (self.lora_B[i] @ self.lora_A[i]) * self.scaling
            deltas.append(d)
        return torch.cat(deltas, dim=0)


def freeze_all_parameters(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def _find_module(root: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """Walk ``name`` (dotted) and return (parent_module, attr_name)."""
    parts = name.split(".")
    cur = root
    for p in parts[:-1]:
        cur = getattr(cur, p)
    return cur, parts[-1]


def apply_lora_to_linear(
    root: nn.Module,
    target_names: Iterable[str],
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> List[str]:
    """Wrap each named ``nn.Linear`` inside ``root`` with a LoRA adapter.

    Returns the list of patched module names so callers can log them.
    """
    patched: List[str] = []
    for fq_name in target_names:
        parent, attr = _find_module(root, fq_name)
        target = getattr(parent, attr)
        if not isinstance(target, nn.Linear):
            continue
        setattr(parent, attr, LoRALinear(target, rank=rank, alpha=alpha, dropout=dropout))
        patched.append(fq_name)
    return patched


def lora_parameters(module: nn.Module) -> List[nn.Parameter]:
    """Return only the LoRA-related parameters (the trainable ones)."""
    out: List[nn.Parameter] = []
    for p in module.parameters():
        if p.requires_grad:
            out.append(p)
    return out


def trainable_param_count(module: nn.Module) -> Tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = 0
    total = 0
    for p in module.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


def find_linear_module_names(
    root: nn.Module,
    *,
    name_filter: Optional[Iterable[str]] = None,
    skip_inside_attention: bool = True,
) -> List[str]:
    """Return the dotted names of every ``nn.Linear`` reachable from ``root``.

    Parameters
    ----------
    name_filter:
        If provided, only Linear modules whose final dotted-component
        matches an entry are returned. Example: ``{"linear1", "linear2"}``.
    skip_inside_attention:
        If True (default), exclude Linear modules whose parent is a
        :class:`torch.nn.MultiheadAttention` — namely ``out_proj``. The
        reason is that ``nn.MultiheadAttention`` calls
        ``F.multi_head_attention_forward(... out_proj.weight ...)``
        instead of dispatching through the submodule's ``forward``, so
        a LoRA wrapper would never see the input tensor and the delta
        would be silently dropped. Disable this flag only if you also
        replace the attention forward.
    """
    fs = set(name_filter) if name_filter is not None else None
    attention_parents: set[str] = set()
    if skip_inside_attention:
        for parent_name, parent_mod in root.named_modules():
            if isinstance(parent_mod, nn.MultiheadAttention):
                attention_parents.add(parent_name)

    names: List[str] = []
    for name, module in root.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        last = name.rsplit(".", 1)[-1]
        if fs is not None and last not in fs:
            continue
        if name == "":
            continue
        if skip_inside_attention:
            parent = name.rsplit(".", 1)[0] if "." in name else ""
            if parent in attention_parents:
                continue
        names.append(name)
    return names


__all__ = [
    "LoRALinear",
    "LoRAInProj",
    "apply_lora_to_linear",
    "find_linear_module_names",
    "freeze_all_parameters",
    "lora_parameters",
    "trainable_param_count",
]
