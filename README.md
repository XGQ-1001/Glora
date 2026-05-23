# Glora — Foundation GPU Operator Scheduler

> **Pretrain once, schedule everywhere.**  
> A Graph-Transformer + Mask-PPO foundation policy for cross-model GPU
> operator scheduling, with LoRA fine-tuning for unseen models.

Glora extends the GNN+PPO scheduler from `TCAS/gnn-strategy` along five
research axes that together support a top-tier-journal narrative:

| Step | Goal                                                                                | Where it lives                                    |
| ---- | ----------------------------------------------------------------------------------- | ------------------------------------------------- |
| 0    | Replace **GATv2** with **Graph Transformer + Laplacian PE**                         | `glora/encoder.py`, `glora/lap_pe.py`             |
| 1    | Switch to **Mask-PPO** + **dense + potential reward shaping**                       | `glora/policy.py`, `glora/reward_shaping.py`, `glora/trainer.py` |
| 2    | Train **single-model** schedulers with Step 0/1                                     | `scripts/step2_train_single.py`                   |
| 3    | **Multi-model mixed pretraining** → foundation GNN scheduler                        | `scripts/step3_train_foundation.py`               |
| 4    | **LoRA few-shot fine-tuning** on unseen models                                      | `scripts/step4_lora_finetune.py`, `glora/lora.py` |
| 5    | Comprehensive **comparison** (latency, throughput, SM eff., memory) vs PyTorch / CUDA Graph / Opara / from-scratch GT-MaskPPO | `scripts/step5_compare.py`, `glora/profiling.py`  |

---

## 1. Project layout

```text
Glora/
├── README.md                       # this file
├── requirements.txt                # pip dependencies
├── configs/                        # YAML hyper-parameter defaults
│   ├── single.yaml                 # Step 2
│   ├── foundation.yaml             # Step 3
│   └── lora.yaml                   # Step 4
├── glora/                          # python package (importable)
│   ├── __init__.py
│   ├── utils.py                    # sys.path wiring, seeding helpers
│   ├── models.py                   # GoogLeNet / ResNet / BERT / DeepFM / ...
│   ├── lap_pe.py                   # Laplacian positional encoding
│   ├── encoder.py                  # Graph Transformer backbone
│   ├── policy.py                   # GloraActorCritic (Mask-PPO actor + critic)
│   ├── reward_shaping.py           # dense + potential + multi-baseline reward
│   ├── trainer.py                  # Mask-PPO trainer (used by Steps 2 / 3 / 4)
│   ├── data_pool.py                # multi-model sampling pool
│   ├── runtime.py                  # CUDA-Graph capture, benchmarking helpers
│   ├── lora.py                     # LoRA adapters (parameter-efficient FT)
│   ├── profiling.py                # NVML + memory + throughput captor
│   └── checkpoint.py               # save / load (full + LoRA-only)
├── scripts/                        # CLI entry points
│   ├── step0_encoder_smoketest.py
│   ├── step2_train_single.py
│   ├── step3_train_foundation.py
│   ├── step4_lora_finetune.py
│   └── step5_compare.py
└── artifacts/                      # output dir (gitignored)
    ├── step2/{model}_bs{B}.pt
    ├── step3/foundation_base.pt
    ├── step4/lora_{model}_bs{B}_rank{R}.pt
    └── step5/compare_results.csv + plots
```

Glora intentionally re-uses the simulation environment, FX graph builder
and CUDA-Graph capturer from the sibling `TCAS/gnn-strategy` project to
avoid duplicating ~2k lines of mature, well-tested code. The
`glora.utils` module prepends both `TCAS/gnn-strategy` and `TCAS` to
`sys.path` automatically.

---

## 2. Installation

```bash
# 1. Make sure the existing TCAS repo is available at ../TCAS/
ls /mnt/workspace/xiaguoqing/x-ky/TCAS/gnn-strategy   # should exist

# 2. Activate the same conda env you use for TCAS (e.g. "ky") and install
#    Glora's extra Python deps. Everything heavy (torch, transformers,
#    pretrainedmodels for NASNet, etc.) is already in TCAS's environment.
cd /mnt/workspace/xiaguoqing/x-ky/Glora
pip install -r requirements.txt
```

`pynvml` is optional: without it Step 5 still produces latency / memory /
throughput numbers but reports SM efficiency as `None`.

### Required environment variables

| Variable                | Purpose                                                  | Default                                                  |
| ----------------------- | -------------------------------------------------------- | -------------------------------------------------------- |
| `GNN_BERT_BASE_PATH`    | local HuggingFace BERT-base directory                    | `/mnt/workspace/xiaguoqing/models/bert-base-uncased`     |
| `GNN_BERT_SEQ_LEN`      | sequence length for BERT inputs                          | `256`                                                    |
| `GNN_BERT_BATCH`        | override BERT batch_size at factory time                 | `1`                                                      |
| `GNN_DEEPFM_BATCH`      | override DeepFM batch_size at factory time               | `1`                                                      |

---

## 3. Conceptual overview

### 3.1 Graph Transformer + Laplacian PE (Step 0)

The original GATv2 encoder only sees 1-hop neighbours per layer and 2-hop
after stacking. For DAGs with 200+ operators (Inception-v3, BERT), this
local view limits how well the encoder can reason about parallelism
opportunities that live further away in the graph.

Glora replaces the encoder with a pre-norm **Transformer** with
`n_layers=3-4` attention blocks. To compensate for the lack of intrinsic
graph structure inside vanilla attention we concatenate **Laplacian
Positional Encoding** (Belkin & Niyogi, 2003; Dwivedi & Bresson, 2021):
the `k` eigenvectors of the symmetrically-normalised Laplacian of the
*undirected* DAG. Two nodes with similar structural roles get similar
PEs, which the attention layers can pick up implicitly. See
[`glora/lap_pe.py`](glora/lap_pe.py) and
[`glora/encoder.py`](glora/encoder.py).

```
node feats  ──┐
              ├── Linear → LN → GELU ── × N(Self-Attention + FFN) ── LN → Linear → h_static
LapPE (k=16) ─┘
```

### 3.2 Mask-PPO + reward shaping (Step 1)

* **Action masking** — `ActorHead.masked_logits` sets the score of every
  non-ready node to `-inf` *before* the softmax. Invalid actions then
  carry exactly zero probability, eliminating the need for post-hoc
  rejection sampling and giving PPO a clean exploration signal.

* **Reward shaping** —
  `r = α · (mk_prev - mk_new) / mk_init` (dense, simulator-only)
  `  + β · (γ · Φ_new - Φ_prev)`     (potential-based; Ng et al., 1999)
  `  + terminal((opara - L_gnn) / opara, etc.)`
  with `Φ(s) = -remaining_critical_path / mk_init`. The potential term
  is theory-preserving — the optimal policy under shaping is the same
  as the optimal policy without shaping — but it converts a sparse
  terminal signal into a dense one and dramatically improves credit
  assignment for long episodes (e.g. 200+ steps in Inception-v3).

* **Multi-baseline terminal** — the terminal reward references the
  *minimum* of `{L_opara, L_cuda_graph, best_historical_GNN}` rather
  than only Opara. This keeps the policy from chasing a baseline that
  it has already saturated.

### 3.3 Foundation pretraining (Step 3)

A `ModelPool` lazily caches the (FX module, GraphState, Opara baseline)
triplet for every requested `(model_name, batch_size)`. Each PPO episode
samples one configuration weighted by the user-defined `weight`. The
result is a single policy that has seen DAGs from CNNs, recommendation
systems and Transformers — the **GNN-base** foundation.

### 3.4 LoRA fine-tuning (Step 4)

After loading the foundation policy we freeze every parameter and inject
[`LoRALinear`](glora/lora.py) adapters on top of selected `nn.Linear`
modules (default: `out_proj`, `linear1`, `linear2` inside every encoder
block). Only the rank-`r` delta matrices `A ∈ ℝ^(r×d)` and
`B ∈ ℝ^(d×r)` receive gradients, which gives:

* < 1% trainable parameters
* < 1 MB checkpoint per target model
* 50-100 episodes to recover most of the gap to a from-scratch GT-MaskPPO

### 3.5 Comprehensive comparison (Step 5)

`scripts/step5_compare.py` measures every algorithm with the same
benchmark harness and a fresh `torch.cuda.reset_peak_memory_stats`. A
background NVML thread polls SM utilisation at ~200 Hz to estimate "SM
efficiency". Outputs include both a CSV (`compare_results.csv`) and
per-metric bar charts (speedup, throughput, peak memory, SM efficiency).

---

## 4. Quick-start: end-to-end pipeline

The minimal happy path produces every artifact discussed in the paper:

```bash
cd /mnt/workspace/xiaguoqing/x-ky/Glora
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# --- Step 0: sanity check the encoder (CPU only, < 5 s) ---
python scripts/step0_encoder_smoketest.py

# --- Step 2: single-model training (used as ablation baselines) ---
for m in googlenet inception_v3 resnet50 deepfm bert_base; do
  python scripts/step2_train_single.py --model "$m" --episodes 500 \
      --batch-size 1 --batch-episodes 8 --mini-batch-size 256 \
      --hidden 128 --emb 128 --heads 4 --layers 3 \
      --save artifacts/step2/${m}_bs1.pt
done

# --- Step 3: foundation pretraining over the same 5 models ---
python scripts/step3_train_foundation.py \
    --pool "googlenet:1;inception_v3:1;resnet50:1;deepfm:1;bert_base:1" \
    --episodes 2000 --batch-episodes 16 --mini-batch-size 512 \
    --hidden 192 --emb 192 --heads 4 --layers 4 \
    --save artifacts/step3/foundation_base.pt

# --- Step 4: LoRA few-shot on held-out models ---
for m in mobilenet_v2 vgg16 densenet121 resnet152; do
  python scripts/step4_lora_finetune.py \
      --base artifacts/step3/foundation_base.pt \
      --target "$m" --batch-size 1 --episodes 80 --rank 8 \
      --save  artifacts/step4/lora_${m}_bs1_rank8.pt \
      --report artifacts/step4/lora_${m}_bs1_rank8.json
done

# --- Step 5: comprehensive comparison ---
python scripts/step5_compare.py \
    --models googlenet,inception_v3,resnet50,bert_base,deepfm,mobilenet_v2,vgg16 \
    --batch-sizes 1,4,16 \
    --scratch-dir artifacts/step2 \
    --foundation  artifacts/step3/foundation_base.pt \
    --lora-dir    artifacts/step4 \
    --output-dir  artifacts/step5
```

`artifacts/step5/compare_results.csv` contains one row per (model,
batch_size, algorithm) with columns `mean_ms, std_ms, throughput,
peak_memory_mb, sm_efficiency_pct, sm_max_pct, mem_util_pct,
opara_latency_ms, num_nodes`. The companion PNG plots are written to
`artifacts/step5/`.

---

## 5. CLI reference

### 5.1 Step 0 — `step0_encoder_smoketest.py`

No CLI flags. Run once after pip install to verify the encoder works
without a GPU.

### 5.2 Step 2 — single-model training

```text
python scripts/step2_train_single.py \
  --model {googlenet|inception_v3|resnet50|resnet152|vgg16|mobilenet_v2|densenet121|deepfm|bert_base} \
  [--batch-size 1] [--episodes 500] [--batch-episodes 8] [--mini-batch-size 256] \
  [--ppo-epochs 4] [--lr 3e-4] [--clip-eps 0.2] [--gae-lambda 1.0] \
  [--entropy-coef 0.02] [--entropy-coef-end 0.0] \
  [--pe-dim 16] [--hidden 128] [--emb 128] [--heads 4] [--layers 3] [--dropout 0.1] \
  [--streams 8] [--iters 20] [--warmups 5] \
  [--dense-coef 0.05] [--potential-coef 0.5] [--no-shaping] \
  [--save artifacts/step2/$MODEL.pt] [--seed 0]
```

Saves three files when a `--save PATH.pt` is provided:

* `PATH.pt`         — best (lowest L_gnn) policy + best schedule order
* `PATH_latest.pt`  — most recent state (for resuming / replotting)
* `PATH_final.pt`   — state at the end of training

### 5.3 Step 3 — foundation pretraining

```text
python scripts/step3_train_foundation.py \
  --pool "googlenet:1;inception_v3:1;resnet50:1;deepfm:1;bert_base:1" \
  [--episodes 2000] [--batch-episodes 16] [--mini-batch-size 512] \
  [--ppo-epochs 4] [--lr 3e-4] [--clip-eps 0.2] [--gae-lambda 1.0] \
  [--entropy-coef 0.02] [--entropy-coef-end 0.0] \
  [--pe-dim 16] [--hidden 192] [--emb 192] [--heads 4] [--layers 4] \
  [--streams 8] [--iters 15] [--warmups 5] \
  [--dense-coef 0.05] [--potential-coef 0.5] \
  [--warmup-pool] [--save artifacts/step3/foundation_base.pt]
```

Pool grammar:

* `name1:bs1,bs2,bs3;name2:bs1`
* Optional per-entry weight: `name:bs,weight=2.0`

### 5.4 Step 4 — LoRA few-shot fine-tuning

```text
python scripts/step4_lora_finetune.py \
  --base artifacts/step3/foundation_base.pt \
  --target mobilenet_v2 [--batch-size 1] [--episodes 80] \
  [--rank 8] [--alpha 16] [--lora-dropout 0] \
  [--target-modules linear1,linear2] \
  [--lr 5e-4] [--clip-eps 0.2] [--streams 8] [--iters 15] [--warmups 5] \
  [--report PATH.json] [--save PATH.pt] [--no-eval]
```

Produces a small LoRA-only checkpoint (`lora_*.pt`, ~kB-MB) and a JSON
report comparing zero-shot vs few-shot greedy latencies against Opara.

### 5.5 Step 5 — comparison

```text
python scripts/step5_compare.py \
  --models googlenet,inception_v3,resnet50,bert_base \
  --batch-sizes 1,4,16 \
  [--iters 50] [--warmups 10] \
  [--scratch-dir artifacts/step2]      # ← Step 2 checkpoints \
  [--foundation  artifacts/step3/foundation_base.pt]  # ← Step 3 \
  [--lora-dir    artifacts/step4]      # ← Step 4 \
  [--lora-rank 8] [--lora-alpha 16] [--lora-target-modules linear1,linear2] \
  [--no-pytorch] [--no-cuda-graph] [--no-opara] \
  [--output-dir artifacts/step5] [--csv PATH.csv] [--no-plots]
```

Algorithms automatically included when their checkpoint is available:

| Algorithm name        | Source                                                |
| --------------------- | ----------------------------------------------------- |
| `pytorch`             | eager PyTorch (always on unless `--no-pytorch`)       |
| `cuda_graph`          | CUDA Graph wrapping eager (always on unless disabled) |
| `opara`               | Opara TCAS heuristic (always on unless disabled)      |
| `gt_maskppo_scratch`  | `--scratch-dir`                                       |
| `glora_foundation`    | `--foundation`                                        |
| `glora_lora`          | `--lora-dir`                                          |

---

## 6. Recommended ablations

If you want to make the case for **each** of the new components, use
`scripts/step2_train_single.py` to produce these variants and re-run
Step 5:

| Ablation                  | How to train                                                                  |
| ------------------------- | ----------------------------------------------------------------------------- |
| GATv2 + PPO (old baseline)| Train with `TCAS/gnn-strategy/examples/train.py`                              |
| GT only (no LapPE)        | Set `--pe-dim 0` (will pad with zeros)                                        |
| GT + LapPE (no shaping)   | `--no-shaping`                                                                |
| GT + LapPE + Mask-PPO     | default settings                                                              |
| + Foundation (zero-shot)  | Step 3 → Step 5 with `--foundation`                                           |
| + LoRA fine-tune          | Step 4 → Step 5 with `--lora-dir`                                             |

The same `compare_results.csv` then lets you report incremental gains on
all of `mean_ms`, `throughput`, `peak_memory_mb`, `sm_efficiency_pct`.

---

## 7. Implementation notes

* **Reproducibility** — every checkpoint stores the best `schedule_order`
  (list of node names) alongside the policy state. Re-running Step 5
  with the checkpoint replays *exactly* that order via
  `capturer_gnn_from_fx`, so the headline number in the paper is
  reproducible even if the greedy rollout finds a slightly different
  order.

* **Where simulation is used** — only for *training* (the dense reward
  and the dynamic node features). All reported latencies in Step 5 come
  from real CUDA Graph benchmarks on the target GPU. Simulation never
  feeds into the metrics.

* **LoRA target choice** — by default we wrap the two MLP linears
  (`linear1`, `linear2`) inside each Transformer encoder block. Linear
  modules that live *inside* `nn.MultiheadAttention` (i.e. `out_proj`,
  `in_proj`) are deliberately skipped because PyTorch's attention
  implementation invokes `F.multi_head_attention_forward` directly and
  bypasses the submodule's `forward`, so a LoRA wrapper would never see
  the input tensor. Wrapping the attention projections requires
  re-implementing the attention forward and is left out for stability.
  The resulting trainable ratio is typically 0.5-2.5 %.

* **NVML reliability** — SM utilisation is *not* a per-process metric.
  Run Step 5 on an otherwise idle GPU to get clean numbers, or treat
  the SM column as a relative comparison only.

* **Re-using Opara baselines** — every (model, batch_size) entry pays the
  Opara baseline benchmark exactly once via `ModelPool`. Subsequent
  episodes / comparisons read it from the cache.

* **DeepFM warning** — DeepFM has ~50 µs end-to-end latency at batch=1.
  Benchmark noise dominates any scheduler differences. Treat the
  DeepFM column as a stress-test of correctness rather than a headline
  number.

---

## 8. Citation / acknowledgements

This project builds on:

* TCAS / Opara — the upstream operator scheduler this work compares
  against (`/x-ky/TCAS/Opara`).
* Belkin & Niyogi, *Laplacian Eigenmaps for Dimensionality Reduction*,
  Neural Computation 2003 — Laplacian positional encoding.
* Dwivedi & Bresson, *A Generalization of Transformer Networks to
  Graphs*, AAAI Workshop on Deep Learning on Graphs 2021.
* Ng, Harada & Russell, *Policy Invariance Under Reward
  Transformations*, ICML 1999 — potential-based shaping.
* Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*,
  ICLR 2022.
