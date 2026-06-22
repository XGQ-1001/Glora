# 在 RTX 3090 上复现 Glora 推理 / 评测

本文说明如何把 `x-ky/Glora` 项目迁移到另一台 **RTX 3090** 机器，并尽可能一致地复现 **推理、benchmark、throughput、fixed-runner retest** 结果。

> **当前项目实际环境**：本项目训练与近期评测均运行在 conda 环境 `ky`，解释器路径为 `/root/miniconda3/envs/ky/bin/python`。复现时最重要的是 **Python / torch / torchvision 版本、TCAS 同级目录、BERT 本地权重路径、checkpoint 路径** 保持一致。
>
> 换 GPU 后 **绝对延迟 (ms)** 会改变，属正常现象；更应关注 **同机同 session 下相对 Opara / CUDA Graph 的加速比**。`best['order']` 是否能直接套用，强依赖 torch/torchvision 产生的 FX 节点名是否一致。

---

## 1. 必须拷贝的目录结构

Glora 通过 `glora/utils.py` 自动挂载路径，要求 **Glora 与 TCAS 为同级目录**：

```text
your_workspace/x-ky/
├── Glora/                    # 整个项目（或至少 glora/ + scripts/）
│   ├── glora/
│   ├── scripts/
│   ├── requirements.txt
│   └── artifacts/            # checkpoint / csv / plots
└── TCAS/                     # 必须与 Glora 同级，不可缺
    ├── Opara/                # CUDA Graph 捕获与 Opara 基线
    ├── gnn-strategy/
    │   └── gnn_strategy/
    └── examples/             # DeepFM 等可选
```

**不要**只拷 `artifacts/*.pt`，没有 `TCAS/` 无法 `import Opara` / `gnn_strategy`。

如果只做论文图复查，建议至少拷：

```text
x-ky/Glora/glora/
x-ky/Glora/scripts/
x-ky/Glora/artifacts/step2/
x-ky/Glora/artifacts/step3/
x-ky/Glora/artifacts/validation/
x-ky/Glora/artifacts/throughput/
x-ky/TCAS/Opara/
x-ky/TCAS/gnn-strategy/
x-ky/TCAS/examples/          # BERT/DeepFM/NCF 等可选依赖路径
```

---

## 2. 推荐环境（必须与当前 `ky` 对齐）

当前 `ky` 环境实测版本：

| 组件 | 版本 |
|------|------|
| Python | 3.9.25 |
| PyTorch | 2.4.1+cu121 |
| torchvision | 0.19.1+cu121 |
| transformers | 4.24.0 |
| numpy | 2.0.1 |
| CUDA | 12.1 |
| scipy | 当前 `ky` 未安装；普通推理/绘图不依赖 |

### 2.1 新建 conda 环境（推荐仍叫 `ky`）

```bash
conda create -n ky python=3.9 -y
conda activate ky

# PyTorch 官方 CUDA 12.1 wheel
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

cd /path/to/x-ky/Glora
pip install transformers==4.24.0 numpy==2.0.1 matplotlib pynvml
```

`requirements.txt` 当前是宽松依赖（`torch>=2.0` 等），**不能保证复现 `best.order`**。如果要复查 checkpoint，优先按上面的精确版本装。

若 3090 驱动无法使用 CUDA 12.1 wheel，可改用 CUDA 11.8 wheel，但需重新验证 FX 节点名：

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
```

> **重要**：`torch` / `torchvision` 小版本差异可能导致 FX trace 的 **节点名** 变化，进而使 checkpoint 中保存的 `best['order']` 无法直接套用。复现时应优先对齐 `torch==2.4.1`、`torchvision==0.19.1`。

### 2.2 验证 GPU

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# 期望：2.4.1+cu121 ... NVIDIA GeForce RTX 3090
```

---

## 3. 拷贝 checkpoint

任选其一：

**方式 A — 直接拷 `.pt`（推荐）**

```bash
scp -r user@train_host:/path/to/x-ky/Glora/artifacts/step2 ./artifacts/
scp -r user@train_host:/path/to/x-ky/Glora/artifacts/step3 ./artifacts/
scp -r user@train_host:/path/to/x-ky/Glora/artifacts/validation ./artifacts/
scp -r user@train_host:/path/to/x-ky/Glora/artifacts/throughput ./artifacts/
```

**方式 B — 拷 `.txt` 再在 3090 还原**

训练机上：

```bash
cd Glora
python scripts/checkpoint_txt_convert.py pt2txt artifacts/step2/gt/googlenet_bs1_final.pt
```

3090 上：

```bash
python scripts/checkpoint_txt_convert.py txt2pt artifacts/step2/gt/googlenet_bs1_final.txt -o artifacts/step2/gt/googlenet_bs1_final.pt
python scripts/checkpoint_txt_convert.py verify artifacts/step2/gt/googlenet_bs1_final.pt
```

---

## 4. 环境变量（可选）

| 变量 | 用途 | 默认 |
|------|------|------|
| `GNN_BERT_BASE_PATH` | BERT 本地权重 | `/mnt/workspace/xiaguoqing/models/bert-base-uncased` |
| `GNN_BERT_SEQ_LEN` | BERT 序列长度 | 256 |
| `GNN_BERT_BATCH` | BERT batch size | 1 |

GoogLeNet / Inception_v3 / ResNet50 无需额外变量。

另一台机器上如果 BERT 权重不在默认路径，必须设置：

```bash
export GNN_BERT_BASE_PATH=/path/to/models/bert-base-uncased
```

---

## 5. 一键自检（3090 上必跑）

```bash
cd /path/to/x-ky/Glora
export PYTHONPATH="$PWD:/path/to/x-ky/TCAS/gnn-strategy:$PYTHONPATH"

python - <<'PY'
import torch
from glora.checkpoint import load_full_checkpoint
from glora.models import build_model
from glora.runtime import real_latency_for_order
from gnn_strategy.utils import extract_first_fx_graph
from Opara import OperatorLauncher, GraphCapturer
from gnn_strategy.capturer import benchmark_runner
import warnings

ckpt_path = 'artifacts/step2/gt/googlenet_bs1_final.pt'
ckpt = load_full_checkpoint(ckpt_path, device='cpu')
order = ckpt['best']['order']
cfg = ckpt['config']
print('best L_ms (train log):', ckpt['best']['L_ms'])
print('best episode:', ckpt['best']['episode'])
print('order len:', len(order))

model, inputs = build_model('googlenet', 'cuda', 1)
fx = extract_first_fx_graph(model, inputs)
fx.cuda()
np_, dp = OperatorLauncher.recompile(model.__class__.__name__, fx, inputs, apply_opara_schedule=False)

# Opara baseline
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    opara_runner = GraphCapturer.capturer(inputs, model, use_tcas=False)
opara_ms = benchmark_runner(opara_runner, inputs=inputs, iterations=cfg.bench_iters, warmups=cfg.bench_warmups).mean_ms

# Best saved order
gnn_ms = real_latency_for_order(fx, inputs, order, iterations=cfg.bench_iters, warmups=cfg.bench_warmups)
sp = (opara_ms - gnn_ms) / opara_ms * 100
print(f'3090 Opara: {opara_ms:.4f} ms')
print(f'3090 GNN (best order): {gnn_ms:.4f} ms  speedup={sp:+.2f}%')
print('OK')
PY
```

期望：无 import 错误；`best.order` 能跑通；speedup 通常为 **正数**。如果单次测量有波动，可用 fixed-runner retest 或 throughput 脚本复测。

---

## 6. 正式对比评测（与论文一致）

```bash
cd /path/to/x-ky/Glora
export PYTHONPATH="$PWD:../TCAS/gnn-strategy:$PYTHONPATH"

# 单模型 throughput：GT+PPO 使用 checkpoint 中的 best.order
python scripts/plot_throughput.py \
  --model googlenet \
  --gt-ckpt artifacts/step2/gt/googlenet_bs1_final.pt \
  --output-dir artifacts/throughput

# 四模型总 throughput 图 + JSON
python scripts/plot_throughput_4models.py

# Held-out fixed-runner 图
python scripts/plot_heldout_fixed_runner.py
```

如果要复查 encoder ablation 曲线，可使用 `scripts/plot_training_curves.py`，但 checkpoint 路径需使用当前论文选定版本。

---

## 7. 「完全复现」的含义（避免误解）

| 目标 | 3090 上能否做到 |
|------|----------------|
| 加载同一套权重、`best['order']` 跑推理 | ✅ |
| 相对 Opara 仍有加速 | ✅（通常） |
| **绝对 ms 与 H20 完全一致** | ❌（硬件不同） |
| **从 ep 0 重训曲线逐点一致** | ❌（未存 optimizer/RNG） |
| FX 节点名与训练机一致 | ⚠️ 需 **同版本 torch/torchvision** |

---

## 8. 常见问题

**Q: `ModuleNotFoundError: No module named 'Opara'`**  
A: 确认 `TCAS/Opara` 存在，且目录为 `x-ky/Glora` 与 `x-ky/TCAS` 同级。

**Q: `best['order']` 报错节点不存在**  
A: torch/torchvision 版本不一致导致 FX 图节点名变化；优先对齐 `torch==2.4.1`、`torchvision==0.19.1`。如果仍报错，需要在新机器上重新训练/重新生成 order。

**Q: 只有 `.txt` 没有 `.pt`**  
A: `python scripts/checkpoint_txt_convert.py txt2pt xxx.txt -o xxx.pt`

**Q: 3090 显存 24GB 够吗？**  
A: GoogLeNet / Inception step2 训练与评测均远小于 24GB，足够。

**Q: 为什么 `requirements.txt` 与本文版本不同？**  
A: `requirements.txt` 是宽松依赖，方便开发；checkpoint 复查应以本文的 `ky` 精确版本为准。

---

## 9. 最小文件清单（仅推理 GoogLeNet best）

```
x-ky/Glora/glora/          # 整个包
x-ky/Glora/scripts/
x-ky/Glora/artifacts/step2/gt/googlenet_bs1_final.pt
x-ky/TCAS/Opara/
x-ky/TCAS/gnn-strategy/gnn_strategy/
```

BERT 需额外本地 HuggingFace 权重目录，并设置 `GNN_BERT_BASE_PATH`。
