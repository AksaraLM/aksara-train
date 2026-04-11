# ⚡ aksara-train

**Distributed training infrastructure for AksaraLLM — reproducible, scalable, open.**

<p align="center">
  <a href="https://github.com/aksaraLLM/aksara-train/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://discord.gg/aksarallm"><img src="https://img.shields.io/badge/Discord-Join-7289da?logo=discord" alt="Discord"></a>
</p>

---

## Overview

This repository contains the complete training pipeline for AksaraLLM models. Everything is config-driven and designed for reproducibility.

### Features
- 🚀 **Distributed training** with FSDP / DeepSpeed
- ⚡ **Mixed precision** (bf16/fp16) training
- 📊 **Experiment tracking** with Weights & Biases
- 💾 **Checkpoint management** with automatic resumption
- 📈 **Scaling law experiments** framework
- 🔧 **Config-driven** — all hyperparameters in YAML
- 🧪 **Evaluation during training** (perplexity + downstream)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/aksaraLLM/aksara-train.git
cd aksara-train

# Install dependencies
pip install -e ".[dev]"

# Single GPU training (for testing)
python -m aksara_train.train --config configs/aksarallm_125m.yaml

# Multi-GPU training with torchrun
torchrun --nproc_per_node=8 \
  -m aksara_train.train \
  --config configs/aksarallm_7b.yaml

# Multi-node training
torchrun --nnodes=8 --nproc_per_node=8 \
  --rdzv_id=aksarallm --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m aksara_train.train \
  --config configs/aksarallm_7b.yaml
```

## Training Configurations

### AksaraLLM-125M (Validation)
```yaml
model:
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  intermediate_size: 3072
  vocab_size: 65536
  max_position_embeddings: 2048

training:
  batch_size: 256
  gradient_accumulation_steps: 4
  learning_rate: 6e-4
  weight_decay: 0.1
  warmup_steps: 2000
  max_steps: 100000
  precision: bf16
  optimizer: adamw
```

### AksaraLLM-7B (Main)
```yaml
model:
  hidden_size: 4096
  num_layers: 32
  num_heads: 32
  num_kv_heads: 8  # GQA
  intermediate_size: 11008
  vocab_size: 65536
  max_position_embeddings: 8192
  rope_theta: 500000

training:
  batch_size: 1024
  gradient_accumulation_steps: 16
  learning_rate: 3e-4
  weight_decay: 0.1
  warmup_steps: 2000
  max_steps: 500000
  precision: bf16
  optimizer: adamw
  grad_clip: 1.0
```

## Project Structure

```
aksara-train/
├── aksara_train/
│   ├── __init__.py
│   ├── train.py               # Main training loop
│   ├── config.py              # Configuration management
│   ├── data/
│   │   ├── dataloader.py      # Efficient data loading
│   │   └── collator.py        # Batch collation
│   ├── distributed/
│   │   ├── fsdp.py            # FSDP setup
│   │   ├── deepspeed.py       # DeepSpeed integration
│   │   └── utils.py           # Distributed utilities
│   ├── optim/
│   │   ├── scheduler.py       # LR schedulers
│   │   └── optimizer.py       # Optimizer setup
│   ├── checkpoint/
│   │   ├── manager.py         # Save/load checkpoints
│   │   └── converter.py       # Format conversion
│   ├── eval/
│   │   ├── perplexity.py      # Perplexity evaluation
│   │   └── downstream.py      # Downstream task eval
│   ├── logging/
│   │   ├── wandb_logger.py    # W&B integration
│   │   └── tensorboard.py     # TensorBoard logging
│   └── utils/
│       ├── profiler.py        # Performance profiling
│       └── memory.py          # Memory management
├── configs/
│   ├── aksarallm_125m.yaml
│   ├── aksarallm_350m.yaml
│   ├── aksarallm_1b.yaml
│   ├── aksarallm_7b.yaml
│   └── sft/
│       ├── aksarallm_7b_sft.yaml
│       └── aksarallm_7b_dpo.yaml
├── scripts/
│   ├── launch_training.sh     # Multi-node launch script
│   ├── monitor.py             # Training monitoring
│   └── scaling_laws.py        # Scaling law experiments
├── tests/
├── LICENSE
├── README.md
└── pyproject.toml
```

## Training Monitoring

All training runs are logged to Weights & Biases with:
- Loss curves (train/val)
- Learning rate schedule
- Gradient norms
- GPU utilization
- Memory usage
- Evaluation metrics

## Reproducibility

We log and version everything:
- Git commit hash
- Full config YAML
- Data composition & checksums
- Random seeds
- Hardware specifications
- Training duration & cost

## Contributing

See [CONTRIBUTING.md](https://github.com/aksaraLLM/community/blob/main/CONTRIBUTING.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
