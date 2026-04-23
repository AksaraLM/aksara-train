#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AksaraLLM 20B — PRE-TRAINING (custom model, from scratch)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Trains the custom ``aksaraLLMModel`` at the ``20b`` preset on an Indonesian
corpus (packed JSONL shards). Uses the **proven XLA/TPU patterns** from
``aksara-train/train_sft_dpo.py``:

1. ``os.environ["PJRT_DEVICE"] = "TPU"`` is set **before** importing torch.
2. ``xm.mark_step()`` after every optimizer step and loss computation.
3. ``xm.reduce_gradients(optimizer)`` before ``optimizer.step()``.
4. Collate function uses a **constant** max_len for XLA graph stability.
5. Checkpoints move the model to CPU before ``save_pretrained``.
6. Data loading via ``pl.MpDeviceLoader``.
7. BF16 params (``torch.bfloat16``).

**FSDP** (full-shard) is wrapped around the model when more than one XLA
device is available — this is mandatory for a 20B model (40 GB of BF16
weights + 160 GB of optimizer state).

Run (TPU v5p-256 pod):
    python3 scripts/train_20b_pretrain.py \\
        --data-dir gs://aksarallm/pretrain-20b/shards \\
        --tokenizer-dir gs://aksarallm/tokenizer-20b \\
        --out-dir ~/aksarallm_pretrain_20b

Dry-run (CPU, ~1 s):
    python3 scripts/train_20b_pretrain.py --dry-run
"""
from __future__ import annotations

import argparse
import glob
import gc
import json
import math
import os
import sys
import time
from datetime import datetime

# ── MUST set before importing torch so XLA picks it up ─────────────
if "PJRT_DEVICE" not in os.environ:
    # Only set when we plan to use the TPU backend. Keep the default on
    # non-TPU machines so ``--dry-run`` works on a vanilla CPU.
    if "XLA_FALLBACK_CPU" not in os.environ:
        os.environ.setdefault("PJRT_DEVICE", "TPU")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import Dataset, DataLoader, IterableDataset  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
AKSARALLM_DIR = os.path.abspath(os.path.join(REPO_ROOT, "..", "aksaraLLM"))
if AKSARALLM_DIR not in sys.path:
    sys.path.insert(0, AKSARALLM_DIR)

from aksarallm.config import aksaraLLMConfig, get_config  # noqa: E402
from aksarallm.model import aksaraLLMModel  # noqa: E402
from aksarallm.tokenizer_utils import AksaraTokenizer  # noqa: E402


# ══════════════════════════════════════════════════════════════════
#   Config
# ══════════════════════════════════════════════════════════════════
def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════
#   Data
# ══════════════════════════════════════════════════════════════════
class PackedShardDataset(IterableDataset):
    """Stream packed token chunks from JSONL shards.

    Each shard is a line-delimited JSONL where every record has a
    ``"tokens": list[int]`` field (pre-tokenized pre-training corpus).
    The dataset emits fixed-length ``(max_len + 1,)`` chunks concatenated
    across records; the extra ``+1`` element is used to form input/target
    pairs in the collate function.

    XLA requires the *shape* to be constant across steps, so we always
    emit exactly ``max_len + 1`` tokens per sample. Short tails are
    dropped rather than padded.
    """

    def __init__(self, shard_paths: list[str], max_len: int, eos_id: int):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.max_len = max_len
        self.eos_id = eos_id

    def __iter__(self):
        buf: list[int] = []
        seq_len = self.max_len + 1
        for path in self.shard_paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    toks = rec["tokens"]
                    buf.extend(toks)
                    buf.append(self.eos_id)
                    while len(buf) >= seq_len:
                        chunk = buf[:seq_len]
                        del buf[:seq_len]
                        yield torch.tensor(chunk, dtype=torch.long)


def collate_pretrain(batch: list[torch.Tensor], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed-shape collate for pretraining; no padding required."""
    # Every sample is already ``max_len + 1`` long.
    stacked = torch.stack(batch, dim=0)  # (B, max_len + 1)
    inputs = stacked[:, :-1].contiguous()
    targets = stacked[:, 1:].contiguous()
    return inputs, targets


# ══════════════════════════════════════════════════════════════════
#   TPU / FSDP helpers
# ══════════════════════════════════════════════════════════════════
def _try_import_xla():
    try:
        import torch_xla  # noqa: F401
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        return xm, pl
    except Exception as e:
        log(f"torch_xla unavailable ({e}); falling back to CPU.", level="WARN")
        return None, None


def _wrap_fsdp(model: nn.Module, device: torch.device, xm) -> nn.Module:
    """Wrap the model in XLA-FSDP when more than one device is available."""
    if xm is None or xm.xrt_world_size() <= 1:
        return model
    try:
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
        from torch_xla.distributed.fsdp import checkpoint_module
    except Exception as e:
        log(f"XLA FSDP not available ({e}); using DDP-free single-device.", level="WARN")
        return model

    # Wrap every transformer block so that weights & optimizer state are
    # sharded across the pod.
    for i, layer in enumerate(model.layers):  # type: ignore[attr-defined]
        model.layers[i] = FSDP(  # type: ignore[index]
            checkpoint_module(layer), compute_dtype=torch.bfloat16, flatten_parameters=True
        )
    return FSDP(model, compute_dtype=torch.bfloat16, flatten_parameters=True)


# ══════════════════════════════════════════════════════════════════
#   Training loop
# ══════════════════════════════════════════════════════════════════
def _cosine_lr(step: int, *, warmup: int, max_steps: int, base: float, min_lr: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * progress))


def train(args: argparse.Namespace) -> int:
    cfg: aksaraLLMConfig = get_config("tiny" if args.dry_run else args.size)

    # ── Device ─────────────────────────────────────────────────────
    xm, pl = (_try_import_xla() if not args.dry_run else (None, None))
    if xm is not None:
        device = xm.xla_device()
        log(f"Using XLA device: {device} (world_size={xm.xrt_world_size()})")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"Using non-XLA device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────
    if args.dry_run:
        corpus = ["indonesia merdeka", "aksarallm adalah asisten", "pancasila dasar negara"] * 20
        tokenizer = AksaraTokenizer.train_bpe_from_iterator(
            iter(corpus), vocab_size=cfg.vocab_size, min_frequency=1
        )
    else:
        tokenizer = AksaraTokenizer.from_pretrained(args.tokenizer_dir)
        # Guard against vocab mismatch; this is the #1 cause of silent NaN.
        if tokenizer.vocab_size != cfg.vocab_size:
            log(
                f"Tokenizer vocab size ({tokenizer.vocab_size}) != config vocab "
                f"({cfg.vocab_size}). Aborting.",
                level="ERROR",
            )
            return 1

    # ── Resume (local ckpt_{step}) — resolve BEFORE optimizer so opt
    #    references the final model's parameters, not a stale copy.
    start_step = 0
    latest = sorted(
        glob.glob(os.path.join(args.out_dir, "ckpt_*")),
        key=lambda p: int(p.split("_")[-1]) if p.split("_")[-1].isdigit() else 0,
    )
    resume_ckpt = latest[-1] if (latest and not args.dry_run) else None

    # ── Model ─────────────────────────────────────────────────────
    if resume_ckpt is not None:
        log(f"Resuming from {resume_ckpt}")
        model = aksaraLLMModel.from_pretrained(resume_ckpt, map_location="cpu")
        start_step = int(os.path.basename(resume_ckpt).split("_")[-1])
    else:
        model = aksaraLLMModel(cfg)
    model = model.to(torch.bfloat16 if not args.dry_run else torch.float32)
    if not args.dry_run:
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()
    model = _wrap_fsdp(model, device, xm)

    # ── Data ──────────────────────────────────────────────────────
    if args.dry_run:
        # Synthetic 128-token shards.
        dummy = torch.randint(0, cfg.vocab_size, (cfg.max_seq_len + 1,)).tolist()
        shard = os.path.join(args.out_dir, "dryrun_shard.jsonl")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(shard, "w", encoding="utf-8") as f:
            for _ in range(32):
                f.write(json.dumps({"tokens": dummy}) + "\n")
        shard_paths = [shard]
    else:
        shard_paths = sorted(glob.glob(os.path.join(args.data_dir, "*.jsonl"))
                             + glob.glob(os.path.join(args.data_dir, "*/*.jsonl")))
        if not shard_paths:
            log(f"No shards found under {args.data_dir!r}. Aborting.", level="ERROR")
            return 1
    log(f"Loaded {len(shard_paths)} shard(s).")

    dataset = PackedShardDataset(
        shard_paths=shard_paths,
        max_len=cfg.max_seq_len,
        eos_id=tokenizer.eos_token_id,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch,
        collate_fn=lambda b: collate_pretrain(b, cfg.max_seq_len),
        num_workers=0,
        drop_last=True,
    )
    if pl is not None:
        mp_loader = pl.MpDeviceLoader(loader, device)
    else:
        mp_loader = loader

    # ── Optimizer / schedule ──────────────────────────────────────
    base_lr = args.learning_rate if args.learning_rate is not None else cfg.learning_rate
    min_lr = base_lr * 0.1
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=False,
    )

    # Restore optimizer state if a matching file is present — this lets
    # AdamW momentum and the LR schedule resume exactly where they left off.
    if resume_ckpt is not None:
        opt_path = os.path.join(resume_ckpt, "optimizer.pt")
        if os.path.exists(opt_path):
            try:
                opt.load_state_dict(torch.load(opt_path, map_location="cpu"))
                log(f"Loaded optimizer state from {opt_path}")
            except Exception as e:
                log(f"Optimizer state load failed ({e}); starting fresh momentum.",
                    level="WARN")

    grad_accum = max(1, args.grad_accum)
    max_steps = args.max_steps if args.max_steps is not None else cfg.max_steps
    warmup_steps = cfg.warmup_steps if not args.dry_run else 0

    # ── Main loop ─────────────────────────────────────────────────
    it = iter(mp_loader)
    losses: list[float] = []
    t0 = time.time()
    step = start_step

    while step < max_steps:
        try:
            input_ids, targets = next(it)
        except StopIteration:
            it = iter(mp_loader)
            input_ids, targets = next(it)

        if xm is None:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

        _, loss = model(input_ids, targets=targets)
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            lr = _cosine_lr(
                step, warmup=warmup_steps, max_steps=max_steps, base=base_lr, min_lr=min_lr,
            )
            for pg in opt.param_groups:
                pg["lr"] = lr

            if xm is not None:
                xm.reduce_gradients(opt)

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if xm is not None:
                xm.mark_step()

        losses.append(loss.item())
        if xm is not None:
            xm.mark_step()
        step += 1

        if step % args.log_every == 0:
            recent = losses[-args.log_every:]
            avg = sum(recent) / max(len(recent), 1)
            spd = (step - start_step) / max(time.time() - t0, 1e-6)
            eta_min = (max_steps - step) / max(spd, 1e-6) / 60
            log(f"step={step}/{max_steps} loss={avg:.4f} lr={opt.param_groups[0]['lr']:.3e} "
                f"{spd:.2f}it/s ETA={eta_min:.1f}m")

        if (not args.dry_run) and step % args.save_every == 0:
            _save_checkpoint(model, opt, args.out_dir, step, device, xm)

        if args.dry_run and step >= 3:
            break

    if not args.dry_run:
        _save_checkpoint(model, opt, args.out_dir, step, device, xm)

    log("Pretraining finished.")
    if args.dry_run:
        # Sanity: loss should be finite.
        final = losses[-1] if losses else float("nan")
        assert math.isfinite(final), f"Dry-run loss is non-finite: {final}"
        log(f"[dry-run] Final loss: {final:.4f}")
        log("[dry-run] OK")
    return 0


def _save_checkpoint(model, opt, out_dir: str, step: int, device, xm) -> None:
    ckpt_dir = os.path.join(out_dir, f"ckpt_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    # XLA tensors must move to CPU before save_pretrained.
    underlying = getattr(model, "module", model)
    underlying.to("cpu")
    underlying.save_pretrained(ckpt_dir)
    underlying.to(device)
    # Persist optimizer state so AdamW momentum survives resume.
    try:
        torch.save(opt.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
    except Exception as e:
        log(f"Optimizer save failed: {e}", level="WARN")
    log(f"Saved checkpoint → {ckpt_dir}")
    gc.collect()


# ══════════════════════════════════════════════════════════════════
#   Entrypoint
# ══════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AksaraLLM 20B pre-training (from scratch)")
    ap.add_argument("--size", default="20b",
                    help="Preset name from aksarallm.config.CONFIGS (default: 20b).")
    ap.add_argument("--data-dir", default=None,
                    help="Directory with packed JSONL shards (each record: {tokens: [int]}).")
    ap.add_argument("--tokenizer-dir", default=None,
                    help="AksaraTokenizer save dir (tokenizer.json + tokenizer_config.json).")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/aksarallm_pretrain_20b"))
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true",
                    help="Run 3 steps on a tiny config with synthetic data (CPU, no HF).")
    args = ap.parse_args(argv)

    if not args.dry_run and (args.data_dir is None or args.tokenizer_dir is None):
        ap.error("--data-dir and --tokenizer-dir are required when not --dry-run.")

    os.makedirs(args.out_dir, exist_ok=True)
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
