# 🏋️ aksara-train

Distributed training infrastructure for AksaraLLM — reproducible, scalable, open.

## Scripts

| File | Description |
|---|---|
| `train_sft_dpo.py` | Full SFT + DPO training pipeline for TPU |
| `finetune_qwen.py` | Qwen2.5-1.5B fine-tuning script |
| `scripts/auto_master.sh` | One-command TPU deployment orchestrator |

## Quick Start (TPU v6e-4)

```bash
# Deploy to GCP TPU
bash scripts/auto_master.sh aksarallm-train train_sft_dpo.py

# Or run directly on TPU
python3 -u train_sft_dpo.py
```

## Features
- ✅ Resume from checkpoint (auto-detect local + HuggingFace)
- ✅ Fixed-shape XLA compilation (no recompilation hang)
- ✅ Live progress logging with ETA
- ✅ Auto upload checkpoints to HuggingFace
- ✅ SFT → DPO pipeline in single script

## License
Apache 2.0
