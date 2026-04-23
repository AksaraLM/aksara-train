#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AksaraLLM 20B — SUPERVISED FINE-TUNING (LoRA, [INST] template)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Fine-tunes the pre-trained ``aksaraLLMModel`` (loaded via
``aksaraLLMModel.from_pretrained``) on an SFT dataset that has already been
re-templated to the AksaraLLM ``[SYS]…[/SYS][INST]…[/INST]response[EOS]``
format (see ``aksara-data/scripts/retemplate.py``).

Key design choices:

- **LoRA** adapters (``r=128``, ``alpha=256``) are injected into every
  ``q_proj`` / ``k_proj`` / ``v_proj`` / ``out_proj`` / SwiGLU gate/up/down
  linear. Base weights stay frozen in BF16.
- **Prompt masking**: everything up to and including ``[/INST]`` has
  ``labels=-100``; only the assistant response contributes to the CE loss.
- **Identity reinforcement**: records tagged ``category == "identity"`` are
  repeated 30× in the training stream. Pattern taken from
  ``aksara-train/train_sft_dpo.py`` line 186.
- **Constant** ``max_len`` in the collate function for XLA graph stability.
- Checkpoints move to CPU before ``save_pretrained``.

Run (TPU v5p-32):
    python3 scripts/train_20b_sft.py \\
        --base-ckpt gs://aksarallm/pretrain-20b/ckpt_final \\
        --tokenizer-dir gs://aksarallm/tokenizer-20b \\
        --dataset-repo Ezekiel999/aksara-sft-20b \\
        --out-dir ~/aksarallm_sft_20b

Dry-run (CPU, ~1 s):
    python3 scripts/train_20b_sft.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

if "PJRT_DEVICE" not in os.environ and "XLA_FALLBACK_CPU" not in os.environ:
    os.environ.setdefault("PJRT_DEVICE", "TPU")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
AKSARALLM_DIR = os.path.abspath(os.path.join(REPO_ROOT, "..", "aksaraLLM"))
if AKSARALLM_DIR not in sys.path:
    sys.path.insert(0, AKSARALLM_DIR)

from aksarallm.config import DEFAULT_SYSTEM_PROMPT, aksaraLLMConfig, get_config  # noqa: E402
from aksarallm.model import aksaraLLMModel  # noqa: E402
from aksarallm.tokenizer_utils import AksaraTokenizer, render_chat  # noqa: E402


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════
#   LoRA
# ══════════════════════════════════════════════════════════════════
class LoRALinear(nn.Module):
    """Minimal LoRA wrapper around ``nn.Linear``. Freezes the base."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(r, 1)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):  # type: ignore[override]
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.drop(x)))


_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "out_proj",
    "gate_proj", "up_proj", "down_proj",
)


def attach_lora(model: nn.Module, r: int, alpha: int) -> int:
    """Replace every ``nn.Linear`` whose *local* name matches a LoRA target
    with a :class:`LoRALinear`. Returns the number of trainable LoRA params.
    """
    count = 0
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if child_name in _LORA_TARGETS and isinstance(child, nn.Linear):
                new = LoRALinear(child, r=r, alpha=alpha)
                setattr(parent, child_name, new)
                count += new.lora_A.weight.numel() + new.lora_B.weight.numel()

    # Freeze everything except LoRA params.
    for n, p in model.named_parameters():
        p.requires_grad_("lora_A" in n or "lora_B" in n)
    return count


# ══════════════════════════════════════════════════════════════════
#   Data
# ══════════════════════════════════════════════════════════════════
@dataclass
class SFTExample:
    messages: list[dict]
    category: str = "general"


def _load_sft_records(path_or_repo: str, max_samples: int | None) -> list[SFTExample]:
    """Load SFT records from a local JSONL or an HF dataset repo id.

    Each record is expected to contain either:
      - ``{"messages": [{"role": ..., "content": ...}, ...]}``, **or**
      - ``{"instruction": str, "response": str, "system": str?}`` — these
        are converted to the ``messages`` schema on the fly.
    """
    records: list[SFTExample] = []

    def _add(rec: dict) -> None:
        if "messages" in rec:
            msgs = rec["messages"]
        else:
            sys_msg = rec.get("system", DEFAULT_SYSTEM_PROMPT)
            user = rec.get("instruction") or rec.get("input") or rec.get("prompt") or ""
            resp = rec.get("response") or rec.get("output") or rec.get("completion") or ""
            msgs = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user},
                {"role": "assistant", "content": resp},
            ]
        records.append(SFTExample(messages=msgs, category=rec.get("category", "general")))

    if os.path.isfile(path_or_repo):
        with open(path_or_repo, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                _add(json.loads(line))
    else:
        from datasets import load_dataset

        ds = load_dataset(path_or_repo, split="train", streaming=True)
        for i, rec in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break
            _add(rec)

    return records


def _expand_identity(records: list[SFTExample], factor: int = 30) -> list[SFTExample]:
    """Repeat identity-tagged records ``factor`` times.

    Matches the behaviour in ``aksara-train/train_sft_dpo.py``; identity
    alignment is critical and needs heavy upweighting.
    """
    expanded: list[SFTExample] = []
    for r in records:
        if r.category == "identity":
            expanded.extend([r] * factor)
        else:
            expanded.append(r)
    return expanded


class SFTDataset(Dataset):
    """Tokenize each record once at init; emit ``(input_ids, labels)``.

    Labels for the prompt portion are ``-100`` so only the assistant
    response contributes to the loss.
    """

    def __init__(
        self,
        records: list[SFTExample],
        tokenizer: AksaraTokenizer,
        max_len: int,
    ):
        self.samples: list[tuple[list[int], list[int]]] = []
        pad = tokenizer.pad_token_id

        # Pre-compute the assistant-response marker in the token stream.
        # The AksaraLLM template closes every user turn with ``[/INST]`` and
        # the assistant reply immediately follows. We mask everything up to
        # and including the *last* ``[/INST]`` so that only the final
        # assistant turn contributes to the CE loss — this correctly handles
        # both single- and multi-turn dialogues and avoids BPE-boundary
        # mismatches from tokenizing the prompt separately.
        inst_end_ids = tokenizer.encode("[/INST]")
        if not inst_end_ids:
            raise RuntimeError("Tokenizer does not encode the [/INST] marker.")

        for ex in records:
            if not any(m["role"] == "assistant" for m in ex.messages):
                continue

            full_text = render_chat(ex.messages, add_generation_prompt=False)
            full_ids = tokenizer.encode(full_text)
            if len(full_ids) > max_len:
                full_ids = full_ids[:max_len]

            # Find the *last* [/INST] occurrence — response start = right after.
            split = -1
            for j in range(len(full_ids) - len(inst_end_ids), -1, -1):
                if full_ids[j : j + len(inst_end_ids)] == inst_end_ids:
                    split = j + len(inst_end_ids)
                    break
            if split < 0 or split >= len(full_ids):
                # No assistant response inside max_len; skip rather than train
                # on a fully masked sequence.
                continue

            labels = [-100] * split + full_ids[split:]

            # Pad to a fixed length for XLA-friendly constant shapes.
            pad_len = max_len - len(full_ids)
            full_ids = full_ids + [pad] * pad_len
            labels = labels + [-100] * pad_len
            self.samples.append((full_ids, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = self.samples[idx]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


# ══════════════════════════════════════════════════════════════════
#   Training
# ══════════════════════════════════════════════════════════════════
def _try_import_xla():
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        return xm, pl
    except Exception:
        return None, None


def train(args: argparse.Namespace) -> int:
    if args.dry_run:
        cfg: aksaraLLMConfig = get_config("tiny")
    else:
        # When we load a pretrained checkpoint, its config drives the shape
        # of every tensor; ``--size`` is only used as a sanity cross-check.
        cfg = get_config(args.size)

    xm, pl = _try_import_xla() if not args.dry_run else (None, None)
    device = xm.xla_device() if xm is not None else torch.device("cpu")
    log(f"device={device}")

    # ── Tokenizer ──────────────────────────────────────────────────
    if args.dry_run:
        corpus = ["halo aksarallm", "indonesia merdeka", "bagaimana cara membuat kopi"] * 20
        tokenizer = AksaraTokenizer.train_bpe_from_iterator(
            iter(corpus), vocab_size=cfg.vocab_size, min_frequency=1
        )
    else:
        tokenizer = AksaraTokenizer.from_pretrained(args.tokenizer_dir)

    # ── Model ──────────────────────────────────────────────────────
    if args.dry_run:
        model = aksaraLLMModel(cfg)
    else:
        model = aksaraLLMModel.from_pretrained(args.base_ckpt, map_location="cpu")
        model.to(torch.bfloat16)
        model.gradient_checkpointing_enable()

    n_lora = attach_lora(model, r=args.lora_r, alpha=args.lora_alpha)
    log(f"Attached LoRA: {n_lora/1e6:.2f}M trainable params")
    model.to(device)
    model.train()

    # ── Data ───────────────────────────────────────────────────────
    if args.dry_run:
        # Use a very short system prompt so everything fits inside the
        # tiny 64-token ``max_seq_len``.
        _sys = "AksaraLLM."
        records = [
            SFTExample(
                messages=[
                    {"role": "system", "content": _sys},
                    {"role": "user", "content": "siapa kamu?"},
                    {"role": "assistant", "content": "saya aksarallm."},
                ],
                category="identity",
            ),
            SFTExample(
                messages=[
                    {"role": "system", "content": _sys},
                    {"role": "user", "content": "apa ibu kota?"},
                    {"role": "assistant", "content": "jakarta."},
                ],
            ),
        ]
    else:
        records = _load_sft_records(args.dataset_repo, max_samples=args.max_samples)
    records = _expand_identity(records)

    dataset = SFTDataset(records, tokenizer, max_len=min(cfg.max_seq_len, args.max_len))
    log(f"SFT samples (post-identity-repeat): {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.micro_batch, shuffle=True, drop_last=True)
    if pl is not None:
        loader = pl.MpDeviceLoader(loader, device)

    # ── Optimizer / schedule ───────────────────────────────────────
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    max_steps = args.max_steps or max(1, len(dataset) // max(args.micro_batch, 1))
    log(f"max_steps={max_steps}")

    # ── Loop ───────────────────────────────────────────────────────
    it = iter(loader)
    losses: list[float] = []
    t0 = time.time()
    for step in range(max_steps):
        try:
            input_ids, labels = next(it)
        except StopIteration:
            it = iter(loader)
            input_ids, labels = next(it)

        if xm is None:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

        logits, _ = model(input_ids)
        # logits[i] predicts the token at position i+1, so shift by one before
        # computing cross-entropy against the labels tensor.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        (loss / args.grad_accum).backward()

        if (step + 1) % args.grad_accum == 0:
            if xm is not None:
                xm.reduce_gradients(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            if xm is not None:
                xm.mark_step()

        losses.append(loss.item())
        if xm is not None:
            xm.mark_step()

        if (step + 1) % args.log_every == 0:
            recent = losses[-args.log_every:]
            log(f"step={step+1}/{max_steps} loss={sum(recent)/len(recent):.4f}")

        if args.dry_run and step >= 2:
            break

    if not args.dry_run:
        model.to("cpu")
        os.makedirs(args.out_dir, exist_ok=True)
        underlying = getattr(model, "module", model)
        underlying.save_pretrained(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)
        log(f"Saved SFT model → {args.out_dir}")
    else:
        final = losses[-1] if losses else float("nan")
        assert math.isfinite(final), f"Dry-run loss not finite: {final}"
        log(f"[dry-run] Final loss: {final:.4f}")
        log("[dry-run] OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AksaraLLM 20B SFT (LoRA)")
    ap.add_argument("--base-ckpt", default=None,
                    help="Path to pretrained aksaraLLMModel checkpoint dir.")
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--dataset-repo", default=None,
                    help="HF dataset repo id or local JSONL path.")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/aksarallm_sft_20b"))
    ap.add_argument("--size", default="20b")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=128)
    ap.add_argument("--lora-alpha", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.dry_run and (args.base_ckpt is None or args.dataset_repo is None
                             or args.tokenizer_dir is None):
        ap.error("--base-ckpt, --tokenizer-dir and --dataset-repo are required.")
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
