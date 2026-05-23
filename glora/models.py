"""Model factories shared between Glora training and evaluation scripts.

These mirror the ones in ``TCAS/gnn-strategy/examples/train.py`` but live in
Glora so the project is self-contained for experiments and the multi-model
foundation pool. Each factory returns ``(model, inputs)``.

Environment overrides:
- ``GNN_BERT_BASE_PATH``  local HF BERT-base directory (default: /mnt/...)
- ``GNN_BERT_SEQ_LEN``    sequence length for BERT (default: 256)
- ``GNN_BERT_BATCH``      override BERT batch size (factory default: 1)
- ``GNN_DEEPFM_BATCH``    override DeepFM batch size (factory default: 1)
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision

from . import utils  # noqa: F401  configure sys.path so NCF/Opara are importable


ModelSpec = Tuple[nn.Module, Tuple[torch.Tensor, ...]]
ModelFactory = Callable[..., ModelSpec]


def _to_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def make_googlenet(device: str = "cuda", batch_size: int = 1) -> ModelSpec:
    dev = _to_device(device)
    x = torch.randint(0, 256, (batch_size, 3, 224, 224), dtype=torch.float32, device=dev)
    model = torchvision.models.googlenet().to(device=dev).eval()
    return model, (x,)


def make_inception_v3(device: str = "cuda", batch_size: int = 1) -> ModelSpec:
    dev = _to_device(device)
    x = torch.randint(0, 256, (batch_size, 3, 299, 299), dtype=torch.float32, device=dev)
    model = torchvision.models.inception_v3(aux_logits=False).to(device=dev).eval()
    return model, (x,)


def make_resnet50(device: str = "cuda", batch_size: int = 1) -> ModelSpec:
    dev = _to_device(device)
    x = torch.randint(0, 256, (batch_size, 3, 224, 224), dtype=torch.float32, device=dev)
    model = torchvision.models.resnet50().to(device=dev).eval()
    return model, (x,)


def make_resnet152(device: str = "cuda", batch_size: int = 1) -> ModelSpec:
    dev = _to_device(device)
    x = torch.randint(0, 256, (batch_size, 3, 224, 224), dtype=torch.float32, device=dev)
    model = torchvision.models.resnet152().to(device=dev).eval()
    return model, (x,)


def make_vgg16(device: str = "cuda", batch_size: int = 1) -> ModelSpec:
    dev = _to_device(device)
    x = torch.randint(0, 256, (batch_size, 3, 224, 224), dtype=torch.float32, device=dev)
    model = torchvision.models.vgg16().to(device=dev).eval()
    return model, (x,)


def make_mobilenet_v2(device: str = "cuda", batch_size: int = 1) -> ModelSpec:
    dev = _to_device(device)
    x = torch.randint(0, 256, (batch_size, 3, 224, 224), dtype=torch.float32, device=dev)
    model = torchvision.models.mobilenet_v2().to(device=dev).eval()
    return model, (x,)


def make_densenet121(device: str = "cuda", batch_size: int = 1) -> ModelSpec:
    dev = _to_device(device)
    x = torch.randint(0, 256, (batch_size, 3, 224, 224), dtype=torch.float32, device=dev)
    model = torchvision.models.densenet121().to(device=dev).eval()
    return model, (x,)


def make_deepfm(device: str = "cuda", batch_size: int | None = None) -> ModelSpec:
    from NCF import DeepFM  # imported lazily — relies on utils.py path setup

    if batch_size is None:
        batch_size = int(os.environ.get("GNN_DEEPFM_BATCH", "1"))
    dev = _to_device(device)

    cate_fea_nuniqs = [100 * (i + 1) for i in range(32)]
    nume_fea_size = 16
    model = DeepFM(
        cate_fea_nuniqs,
        nume_fea_size,
        emb_size=8,
        hid_dims=[256, 128],
        num_classes=1,
        dropout=[0.2, 0.2],
    ).to(device=dev).eval()

    x_sparse = torch.randint(
        0, 100, (batch_size, len(cate_fea_nuniqs)), dtype=torch.long, device=dev,
    )
    x_dense = torch.rand(batch_size, nume_fea_size, device=dev)
    return model, (x_sparse, x_dense)


class _BertLastHiddenState(nn.Module):
    def __init__(self, local_path: str):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(local_path, local_files_only=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]


def make_bert_base(device: str = "cuda", batch_size: int | None = None) -> ModelSpec:
    if batch_size is None:
        batch_size = int(os.environ.get("GNN_BERT_BATCH", "1"))
    seq_len = int(os.environ.get("GNN_BERT_SEQ_LEN", "256"))
    local_path = os.environ.get(
        "GNN_BERT_BASE_PATH",
        "/mnt/workspace/xiaguoqing/models/bert-base-uncased",
    )
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"BERT local path not found: {local_path}. "
            "Set GNN_BERT_BASE_PATH to a local HuggingFace BertModel directory."
        )
    dev = _to_device(device)
    input_ids = torch.randint(0, 30522, (batch_size, seq_len), dtype=torch.long, device=dev)
    attention_mask = torch.ones_like(input_ids, device=dev)
    model = _BertLastHiddenState(local_path).to(device=dev).eval()
    return model, (input_ids, attention_mask)


MODEL_FACTORIES: Dict[str, ModelFactory] = {
    "googlenet": make_googlenet,
    "inception_v3": make_inception_v3,
    "resnet50": make_resnet50,
    "resnet152": make_resnet152,
    "vgg16": make_vgg16,
    "mobilenet_v2": make_mobilenet_v2,
    "densenet121": make_densenet121,
    "deepfm": make_deepfm,
    "bert_base": make_bert_base,
}

DISPLAY_NAMES: Dict[str, str] = {
    "googlenet": "GoogLeNet",
    "inception_v3": "Inception-v3",
    "resnet50": "ResNet50",
    "resnet152": "ResNet152",
    "vgg16": "VGG16",
    "mobilenet_v2": "MobileNet-v2",
    "densenet121": "DenseNet121",
    "deepfm": "DeepFM",
    "bert_base": "BERT-base",
}


# Default pools for foundation pretrain / held-out evaluation.
FOUNDATION_POOL: List[str] = [
    "googlenet",
    "inception_v3",
    "resnet50",
    "deepfm",
    "bert_base",
]

HELDOUT_POOL: List[str] = [
    "resnet152",
    "mobilenet_v2",
    "vgg16",
    "densenet121",
]


def build_model(name: str, device: str = "cuda", batch_size: int = 1) -> ModelSpec:
    """Construct a model with the requested batch size (where supported)."""
    if name not in MODEL_FACTORIES:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_FACTORIES)}")
    factory = MODEL_FACTORIES[name]
    try:
        return factory(device=device, batch_size=batch_size)
    except TypeError:
        return factory(device=device)


def known_models() -> Sequence[str]:
    return tuple(MODEL_FACTORIES.keys())


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


__all__ = [
    "build_model",
    "known_models",
    "display_name",
    "MODEL_FACTORIES",
    "DISPLAY_NAMES",
    "FOUNDATION_POOL",
    "HELDOUT_POOL",
]
