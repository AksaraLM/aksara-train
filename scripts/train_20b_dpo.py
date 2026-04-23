#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AksaraLLM 20B — DIRECT PREFERENCE OPTIMIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Implements the DPO loss from Rafailov et al. (2023) on top of the custom
``aksaraLLMModel``. Matches the hyper-parameters used in
``aksara-train/train_sft_dpo.py`` (``beta = 0.1``).

Two models are active during training:

* **policy** — trainable, initialized from the SFT checkpoint.
* **ref** — frozen ``deepcopy`` of the SFT checkpoint; used to compute the
  per-token log-probability baseline that anchors the KL penalty.

The dataset expects rows with:

    { "prompt_messages": [...],   # system + user
      "chosen": str,              # preferred assistant response
      "rejected": str }           # disfavoured response

Run (TPU v5p-32):
    python3 scripts/train_20b_dpo.py \\
        --sft-ckpt gs://aksarallm/sft-20b/final \\
        --tokenizer-dir gs://aksarallm/tokenizer-20b \\
        --dataset-repo Ezekiel999/aksara-dpo-20b \\
        --out-dir ~/aksarallm_dpo_20b

Dry-run (CPU, ~1 s):
    python3 scripts/train_20b_dpo.py --dry-run
"""
from __future__ import annotations

import argparse
import copy
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
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
AKSARALLM_DIR = os.path.abspath(os.path.join(REPO_ROOT, "..", "aksaraLLM"))
if AKSARALLM_DIR not in sys.path:
    sys.path.insert(0, AKSARALLM_DIR)

from aksarallm.config import DEFAULT_SYSTEM_PROMPT, get_config  # noqa: E402
from aksarallm.model import aksaraLLMModel  # noqa: E402
from aksarallm.tokenizer_utils import AksaraTokenizer, render_chat  # noqa: E402


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════
#   Data
# ══════════════════════════════════════════════════════════════════
@dataclass
class DPOExample:
    prompt_messages: list[dict]
    chosen: str
    rejected: str


def _load_dpo_records(path_or_repo: str, max_samples: int | None) -> list[DPOExample]:
    def _normalize(rec: dict) -> DPOExample:
        if "prompt_messages" in rec:
            pmsgs = rec["prompt_messages"]
        else:
            sys_msg = rec.get("system", DEFAULT_SYSTEM_PROMPT)
            user = rec.get("prompt") or rec.get("instruction") or rec.get("input") or ""
            pmsgs = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user},
            ]
        return DPOExample(
            prompt_messages=pmsgs,
            chosen=rec["chosen"],
            rejected=rec["rejected"],
        )

    if os.path.isfile(path_or_repo):
        out = []
        with open(path_or_repo, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                out.append(_normalize(json.loads(line)))
        return out

    from datasets import load_dataset

    ds = load_dataset(path_or_repo, split="train", streaming=True)
    out = []
    for i, rec in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        out.append(_normalize(rec))
    return out


class DPODataset(Dataset):
    """Tokenize each ``(prompt, chosen, rejected)`` triple once at init.

    Emits four fixed-length tensors per sample:

    ``input_chosen``, ``labels_chosen``, ``input_rejected``, ``labels_rejected``.
    """

    def __init__(self, records: list[DPOExample], tokenizer: AksaraTokenizer, max_len: int):
        self.samples = []
        pad = tokenizer.pad_token_id
        for ex in records:
            prompt_text = render_chat(ex.prompt_messages, add_generation_prompt=True)
            prompt_ids = tokenizer.encode(prompt_text)

            for key, resp in (("chosen", ex.chosen), ("rejected", ex.rejected)):
                resp_ids = tokenizer.encode(resp + "[EOS]") if "[EOS]" not in resp else tokenizer.encode(resp)
                ids = (prompt_ids + resp_ids)[:max_len]
                labels = [-100] * min(len(prompt_ids), max_len) + resp_ids[: max(0, max_len - len(prompt_ids))]
                pad_len = max_len - len(ids)
                ids = ids + [pad] * pad_len
                labels = labels + [-100] * pad_len
                if key == "chosen":
                    cur = [torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)]
                else:
                    cur.extend([torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)])
            self.samples.append(tuple(cur))  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


# ══════════════════════════════════════════════════════════════════
#   DPO loss
# ══════════════════════════════════════════════════════════════════
def _seq_logp(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the summed log-probability of ``labels`` under ``model``.

    Positions where ``labels == -100`` are masked out.
    """
    logits, _ = model(input_ids)
    # Shift: predict token ``t`` from ``t-1``.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = (shift_labels != -100).float()
    # Clamp -100 -> 0 for gather (masked out anyway).
    gather_labels = shift_labels.clamp_min(0)
    logp = F.log_softmax(shift_logits, dim=-1)
    token_logp = logp.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)


def dpo_loss(
    policy: nn.Module,
    ref: nn.Module,
    input_c: torch.Tensor, labels_c: torch.Tensor,
    input_r: torch.Tensor, labels_r: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, float]:
    pi_c = _seq_logp(policy, input_c, labels_c)
    pi_r = _seq_logp(policy, input_r, labels_r)
    with torch.no_grad():
        ref_c = _seq_logp(ref, input_c, labels_c)
        ref_r = _seq_logp(ref, input_r, labels_r)

    logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    loss = -F.logsigmoid(logits).mean()
    reward_margin = (pi_c - pi_r).detach().mean().item()
    return loss, reward_margin


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
    cfg = get_config("tiny") if args.dry_run else get_config(args.size)
    xm, pl = _try_import_xla() if not args.dry_run else (None, None)
    device = xm.xla_device() if xm is not None else torch.device("cpu")
    log(f"device={device}")

    # Tokenizer
    if args.dry_run:
        corpus = ["halo aksarallm", "jakarta ibu kota", "indonesia merdeka"] * 20
        tokenizer = AksaraTokenizer.train_bpe_from_iterator(
            iter(corpus), vocab_size=cfg.vocab_size, min_frequency=1
        )
    else:
        tokenizer = AksaraTokenizer.from_pretrained(args.tokenizer_dir)

    # Policy + reference
    if args.dry_run:
        policy = aksaraLLMModel(cfg)
    else:
        policy = aksaraLLMModel.from_pretrained(args.sft_ckpt, map_location="cpu")
        policy.to(torch.bfloat16)
    policy.to(device)
    ref = copy.deepcopy(policy).to(device)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    policy.train()

    # Data
    if args.dry_run:
        records = [
            DPOExample(
                prompt_messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": "siapa kamu?"},
                ],
                chosen="saya aksarallm, asisten ai bahasa indonesia.",
                rejected="saya chatgpt.",
            ),
            DPOExample(
                prompt_messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": "apa ibu kota indonesia?"},
                ],
                chosen="ibu kota indonesia adalah jakarta.",
                rejected="tidak tahu.",
            ),
        ]
    else:
        records = _load_dpo_records(args.dataset_repo, max_samples=args.max_samples)
    log(f"DPO samples: {len(records)}")
    dataset = DPODataset(records, tokenizer, max_len=min(cfg.max_seq_len, args.max_len))
    loader = DataLoader(dataset, batch_size=args.micro_batch, shuffle=True, drop_last=True)
    if pl is not None:
        loader = pl.MpDeviceLoader(loader, device)

    opt = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0,
    )

    max_steps = args.max_steps or max(1, len(dataset) // args.micro_batch)
    log(f"max_steps={max_steps}  beta={args.beta}")

    losses: list[float] = []
    margins: list[float] = []
    for step in range(max_steps):
        for batch in loader:
            ic, lc, ir, lr_ = batch
            if xm is None:
                ic, lc, ir, lr_ = [t.to(device) for t in (ic, lc, ir, lr_)]

            loss, margin = dpo_loss(policy, ref, ic, lc, ir, lr_, beta=args.beta)
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                if xm is not None:
                    xm.reduce_gradients(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                if xm is not None:
                    xm.mark_step()

            losses.append(loss.item())
            margins.append(margin)
            if xm is not None:
                xm.mark_step()
            break  # one batch per outer step (makes step counting simple)

        if (step + 1) % args.log_every == 0 or step == max_steps - 1:
            recent_l = losses[-args.log_every:]
            recent_m = margins[-args.log_every:]
            log(f"step={step+1}/{max_steps} dpo_loss={sum(recent_l)/len(recent_l):.4f} "
                f"reward_margin={sum(recent_m)/len(recent_m):.4f}")

        if args.dry_run and step >= 2:
            break

    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
        policy.to("cpu")
        policy.save_pretrained(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)
        log(f"Saved DPO model → {args.out_dir}")
    else:
        final_loss = losses[-1] if losses else float("nan")
        final_margin = margins[-1] if margins else 0.0
        assert math.isfinite(final_loss), f"Dry-run DPO loss not finite: {final_loss}"
        log(f"[dry-run] Final dpo_loss={final_loss:.4f} reward_margin={final_margin:.4f}")
        log("[dry-run] OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AksaraLLM 20B DPO")
    ap.add_argument("--sft-ckpt", default=None)
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--dataset-repo", default=None)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/aksarallm_dpo_20b"))
    ap.add_argument("--size", default="20b")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.dry_run and (args.sft_ckpt is None or args.dataset_repo is None
                             or args.tokenizer_dir is None):
        ap.error("--sft-ckpt, --tokenizer-dir and --dataset-repo are required.")
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
