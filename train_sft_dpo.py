#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AksaraLLM RETRAIN — GCP TPU v6e-4
 Full pipeline: SFT → DPO → Upload → Test
 
 Berdasarkan pattern terbukti dari aksarallm_finetune_qwen.py
 yang sukses 5000 steps di TPU v4-8
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run:
  nohup python3 ~/aksarallm_retrain_gcp.py > ~/retrain.log 2>&1 &
  tail -f ~/retrain.log
"""

import os, gc, time, math, random, re, sys
import warnings
warnings.filterwarnings("ignore")

os.environ["PJRT_DEVICE"] = "TPU"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ====================================================================
#  CONFIG
# ====================================================================
class Config:
    # Model
    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    FINAL_NAME = "AksaraLLM/aksarallm-1.5b-v2"
    CKPT_NAME = "AksaraLLM/aksarallm-1.5b-v2-checkpoint"
    
    # Data
    SFT_REPO = "AksaraLLM/aksara-mega-sft-v5"
    DPO_REPO = "AksaraLLM/aksara-dpo-id-v4"
    
    # Training
    TOTAL_STEPS = 5000
    BATCH_SIZE = 4       # v6e-4 has less memory than v5e-8
    GRAD_ACCUM = 4       # Effective BS = 4*4 = 16
    LR = 2e-5
    MIN_LR = 2e-6
    WARMUP = 200
    MAX_LEN = 512
    
    # DPO (Phase 2)
    DPO_STEPS = 2000
    DPO_LR = 5e-6
    DPO_BETA = 0.1
    
    # Logging
    LOG_EVERY = 50
    SAVE_EVERY = 1000
    TEST_EVERY = 500
    
    # Paths
    WORK_DIR = os.path.expanduser("~/aksarallm_retrain")
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    
    SYSTEM_PROMPT = "Kamu adalah AksaraLLM, asisten AI berbahasa Indonesia yang cerdas, sopan, dan membantu. Jawab pertanyaan dengan akurat dan detail."

# ====================================================================
#  SETUP
# ====================================================================
print("━" * 60)
print("  🚀 AksaraLLM RETRAIN — GCP TPU v6e-4")
print("  Full Pipeline: SFT → DPO → Upload → Test")
print("━" * 60)

os.makedirs(Config.WORK_DIR, exist_ok=True)

# TPU Setup
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
device = xm.xla_device()
print(f"🔥 TPU: {device}")

# HuggingFace
from huggingface_hub import login, HfApi
login(token=Config.HF_TOKEN, add_to_git_credential=False)
api = HfApi()

from transformers import AutoTokenizer, AutoModelForCausalLM

# ====================================================================
#  PHASE 0: RESUME CHECK
# ====================================================================
print("\n📥 Checking for existing checkpoints...")

model = None
tokenizer = None
start_step = 0

# Check local checkpoints
for step in [5000, 4000, 3000, 2000, 1000]:
    ckpt = f"{Config.WORK_DIR}/ckpt_{step}"
    if os.path.exists(ckpt):
        print(f"  ✅ Found local checkpoint: ckpt_{step}")
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        start_step = step
        break

# Check HuggingFace checkpoint
if model is None:
    try:
        print(f"  🔍 Checking HuggingFace: {Config.CKPT_NAME}")
        model = AutoModelForCausalLM.from_pretrained(Config.CKPT_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(Config.CKPT_NAME, trust_remote_code=True)
        start_step = 2000  # Adjust based on actual checkpoint
        print(f"  ✅ Loaded from HuggingFace (step ~{start_step})")
    except:
        pass

# Load fresh base model
if model is None:
    print(f"  📦 Loading fresh: {Config.BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(Config.BASE_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL, trust_remote_code=True)
    start_step = 0

tokenizer.pad_token = tokenizer.eos_token
print("  📤 Moving model to TPU (may take 1-2 min)...", flush=True)
model.to(device)
model.train()
print(f"  ✅ {sum(p.numel() for p in model.parameters())/1e9:.2f}B params on TPU", flush=True)

# ====================================================================
#  PHASE 1: SFT TRAINING
# ====================================================================
print("\n" + "=" * 60)
print("  📊 PHASE 1: SFT TRAINING")
print("=" * 60)

# Load dataset
print("  Loading dataset from HuggingFace...", flush=True)
from datasets import load_dataset

ds = None
for repo in [Config.SFT_REPO, "AksaraLLM/aksara-mega-sft-v4"]:
    try:
        ds = load_dataset(repo, split="train")
        print(f"  ✅ {repo}: {len(ds):,} samples")
        break
    except:
        continue

if ds is None:
    print("  ❌ No dataset found!")
    sys.exit(1)

# Add identity data (critical for model identity)
identity_pairs = [
    ("Siapa kamu?", "Saya AksaraLLM, model bahasa AI buatan Indonesia yang di-fine-tune dari Qwen2.5. Saya dirancang untuk membantu menjawab pertanyaan dan berdiskusi dalam bahasa Indonesia."),
    ("Apa nama kamu?", "Nama saya AksaraLLM. Saya adalah asisten AI berbahasa Indonesia berbasis model Qwen2.5-1.5B."),
    ("Kamu buatan siapa?", "Saya dibuat oleh tim AksaraLLM melalui fine-tuning model Qwen2.5 dari Alibaba. Tujuan saya adalah menjadi asisten AI terbaik untuk bahasa Indonesia."),
    ("Apakah kamu ChatGPT?", "Bukan, saya AksaraLLM. Saya adalah model AI Indonesia berbasis Qwen2.5, bukan ChatGPT dari OpenAI."),
    ("Halo!", "Halo! Saya AksaraLLM, asisten AI bahasa Indonesia. Ada yang bisa saya bantu hari ini?"),
    ("Perkenalkan dirimu!", "Halo! Saya AksaraLLM, asisten AI berbahasa Indonesia. Saya berbasis model Qwen2.5-1.5B yang di-fine-tune khusus untuk bahasa Indonesia oleh tim AksaraLLM."),
    ("Are you ChatGPT?", "No, I am AksaraLLM, an Indonesian AI assistant based on Qwen2.5-1.5B."),
    ("Who are you?", "I am AksaraLLM, an open-source Indonesian AI language model based on Qwen2.5-1.5B."),
]

# Tokenize
print("  Tokenizing...")

class SFTDataset(Dataset):
    def __init__(self, hf_dataset, identity_data, tokenizer, max_len):
        self.samples = []
        t0 = time.time()
        
        # Identity first (repeated for strong identity)
        for inst, out in identity_data:
            for _ in range(30):
                self._add(tokenizer, max_len, inst, out)
        
        # Main dataset  
        for i, item in enumerate(hf_dataset):
            inst = str(item.get("instruction", ""))
            out = str(item.get("output", ""))
            if inst and out and len(out) > 20:
                self._add(tokenizer, max_len, inst, out)
            if (i+1) % 5000 == 0:
                print(f"    {i+1:,}/{len(hf_dataset):,} | {(i+1)/(time.time()-t0):.0f}/s | {len(self.samples):,} kept", flush=True)
        
        random.shuffle(self.samples)
        print(f"  ✅ {len(self.samples):,} samples tokenized in {time.time()-t0:.1f}s", flush=True)
    
    def _add(self, tokenizer, max_len, inst, out):
        messages = [
            {"role": "system", "content": Config.SYSTEM_PROMPT},
            {"role": "user", "content": inst},
            {"role": "assistant", "content": out},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        ids = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
        input_ids = ids["input_ids"].squeeze(0)
        labels = input_ids.clone()
        
        # Mask everything before assistant response
        a_marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        for j in range(len(labels) - len(a_marker)):
            if labels[j:j+len(a_marker)].tolist() == a_marker:
                labels[:j+len(a_marker)] = -100
                break
        
        self.samples.append((input_ids, labels))
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate_fn(batch):
    input_ids = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    # PENTING UNTUK XLA: max_len harus KONSTAN, tidak boleh berubah tiap batch!
    # Jika berubah, XLA akan terus-menerus recompile graph dan menyebabkan stuck/OOM.
    max_len = Config.MAX_LEN
    
    padded_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id or 0, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, (ids, lbl) in enumerate(zip(input_ids, labels)):
        # Potong jika lebih dari max_len
        actual_len = min(ids.size(0), max_len)
        padded_ids[i, :actual_len] = ids[:actual_len]
        padded_labels[i, :actual_len] = lbl[:actual_len]
        attention_mask[i, :actual_len] = 1
        
    return padded_ids, padded_labels, attention_mask

print("  Tokenizing and padding dataset (This may take ~5-15 mins)...")
sft_dataset = SFTDataset(ds, identity_pairs, tokenizer, Config.MAX_LEN)
del ds; gc.collect()

# Training
remaining = Config.TOTAL_STEPS - start_step
if remaining > 0:
    print(f"\n  🚀 Training: step {start_step} → {Config.TOTAL_STEPS} ({remaining} remaining)")
    print(f"  BS: {Config.BATCH_SIZE}×{Config.GRAD_ACCUM}={Config.BATCH_SIZE*Config.GRAD_ACCUM} | LR: {Config.LR} | SeqLen: {Config.MAX_LEN}")
    print(f"  ⏳ First step: ~5 min (XLA compile)...\n")
    
    loader = DataLoader(sft_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                        drop_last=True, collate_fn=collate_fn, num_workers=0)
    mp_loader = pl.MpDeviceLoader(loader, device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=Config.LR, betas=(0.9, 0.95), 
                            weight_decay=0.01, fused=False)
    
    it = iter(mp_loader)
    losses = []
    t0 = time.time()
    step = start_step
    
    while step < Config.TOTAL_STEPS:
        try:
            input_ids, labels, attn_mask = next(it)
        except StopIteration:
            it = iter(mp_loader)
            input_ids, labels, attn_mask = next(it)
        
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss / Config.GRAD_ACCUM
        loss.backward()
        
        if (step + 1) % Config.GRAD_ACCUM == 0:
            # Cosine LR with warmup
            if step < Config.WARMUP:
                cur_lr = Config.LR * (step + 1) / Config.WARMUP
            else:
                cur_lr = Config.MIN_LR + 0.5 * (Config.LR - Config.MIN_LR) * \
                         (1 + math.cos(math.pi * (step - Config.WARMUP) / max(Config.TOTAL_STEPS - Config.WARMUP, 1)))
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            
            xm.reduce_gradients(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            xm.mark_step()
        
        losses.append(loss.item() * Config.GRAD_ACCUM)
        xm.mark_step()
        step += 1
        
        # Log
        if step % Config.LOG_EVERY == 0:
            avg = sum(losses[-Config.LOG_EVERY:]) / len(losses[-Config.LOG_EVERY:])
            elapsed = time.time() - t0
            spd = (step - start_step) / elapsed
            eta = (Config.TOTAL_STEPS - step) / max(spd, 0.001) / 60
            print(f"  {step}/{Config.TOTAL_STEPS} | Loss:{avg:.4f} | LR:{cur_lr:.2e} | {spd:.2f}/s | ETA:{eta:.0f}m")
        
        # Save checkpoint + upload to HF (MOVE TO CPU FIRST!)
        if step % Config.SAVE_EVERY == 0:
            ckpt_dir = f"{Config.WORK_DIR}/ckpt_{step}"
            os.makedirs(ckpt_dir, exist_ok=True)
            # Must move to CPU before save_pretrained (XLA tensors can't be saved directly)
            model.to("cpu")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            model.to(device)  # Move back to TPU
            print(f"  💾 Saved ckpt_{step}")
            
            try:
                api.create_repo(Config.CKPT_NAME, exist_ok=True, token=Config.HF_TOKEN)
                api.upload_folder(folder_path=ckpt_dir, repo_id=Config.CKPT_NAME, token=Config.HF_TOKEN)
                print(f"  ☁️ Uploaded to {Config.CKPT_NAME}")
            except Exception as e:
                print(f"  ⚠️ HF upload: {str(e)[:80]}")
        
        # Quick loss test (skip generate — hangs on XLA)
        if step % Config.TEST_EVERY == 0:
            model.eval()
            try:
                test_text = f"<|im_start|>system\n{Config.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSiapa kamu?<|im_end|>\n<|im_start|>assistant\nSaya AksaraLLM<|im_end|>"
                test_tok = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=128)
                test_ids = test_tok["input_ids"].to(device)
                with torch.no_grad():
                    test_out = model(test_ids, labels=test_ids)
                xm.mark_step()
                print(f"  📝 Test loss: {test_out.loss.item():.4f} (lower = better identity)")
            except Exception as e:
                print(f"  📝 Test error: {str(e)[:80]}")
            model.train()
        
        # Memory cleanup
        if step % 2000 == 0:
            gc.collect()
    
    avg = sum(losses[-200:]) / max(len(losses[-200:]), 1)
    elapsed = time.time() - t0
    print(f"\n  ✅ SFT COMPLETE! {step} steps in {elapsed/60:.1f}m | Final Loss: {avg:.4f}")
else:
    print(f"  ✅ SFT already completed ({start_step} steps)")

# ====================================================================
#  PHASE 2: DPO ALIGNMENT
# ====================================================================
print("\n" + "=" * 60)
print("  🎯 PHASE 2: DPO ALIGNMENT")
print("=" * 60)

try:
    dpo_ds = load_dataset(Config.DPO_REPO, split="train")
    print(f"  ✅ DPO data: {len(dpo_ds):,} pairs")
    
    # Freeze reference model (copy current weights)
    import copy
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    # DPO optimizer
    dpo_opt = torch.optim.AdamW(model.parameters(), lr=Config.DPO_LR, weight_decay=0.01, fused=False)
    model.train()
    
    dpo_losses = []
    t0 = time.time()
    
    for dpo_step in range(Config.DPO_STEPS):
        idx = random.randint(0, len(dpo_ds) - 1)
        item = dpo_ds[idx]
        
        prompt = str(item.get("prompt", ""))
        chosen = str(item.get("chosen", ""))
        rejected = str(item.get("rejected", ""))
        
        if not prompt or not chosen or not rejected:
            continue
        
        # Tokenize chosen & rejected
        chosen_text = f"<|im_start|>system\n{Config.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{chosen}<|im_end|>"
        rejected_text = f"<|im_start|>system\n{Config.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{rejected}<|im_end|>"
        
        c_tok = tokenizer(chosen_text, truncation=True, max_length=Config.MAX_LEN, padding="max_length", return_tensors="pt")
        r_tok = tokenizer(rejected_text, truncation=True, max_length=Config.MAX_LEN, padding="max_length", return_tensors="pt")
        
        c_ids = c_tok["input_ids"].to(device)
        r_ids = r_tok["input_ids"].to(device)
        c_mask = c_tok["attention_mask"].to(device)
        r_mask = r_tok["attention_mask"].to(device)
        
        # Policy log-probs
        c_logits = model(c_ids, attention_mask=c_mask).logits[:, :-1]
        r_logits = model(r_ids, attention_mask=r_mask).logits[:, :-1]
        c_logp = torch.nn.functional.log_softmax(c_logits, dim=-1)
        r_logp = torch.nn.functional.log_softmax(r_logits, dim=-1)
        c_logp = torch.gather(c_logp, 2, c_ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)
        r_logp = torch.gather(r_logp, 2, r_ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)
        
        # Reference log-probs
        with torch.no_grad():
            rc_logits = ref_model(c_ids, attention_mask=c_mask).logits[:, :-1]
            rr_logits = ref_model(r_ids, attention_mask=r_mask).logits[:, :-1]
            rc_logp = torch.nn.functional.log_softmax(rc_logits, dim=-1)
            rr_logp = torch.nn.functional.log_softmax(rr_logits, dim=-1)
            rc_logp = torch.gather(rc_logp, 2, c_ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)
            rr_logp = torch.gather(rr_logp, 2, r_ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)
        
        # DPO loss
        margin = Config.DPO_BETA * ((c_logp - rc_logp) - (r_logp - rr_logp))
        dpo_loss = -torch.nn.functional.logsigmoid(margin).mean()
        
        dpo_loss.backward()
        
        if (dpo_step + 1) % 4 == 0:
            xm.reduce_gradients(dpo_opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            dpo_opt.step()
            dpo_opt.zero_grad(set_to_none=True)
        xm.mark_step()
        
        dpo_losses.append(dpo_loss.item())
        
        if (dpo_step + 1) % 50 == 0:
            avg = sum(dpo_losses[-50:]) / 50
            spd = (dpo_step + 1) / (time.time() - t0)
            eta = (Config.DPO_STEPS - dpo_step) / max(spd, 0.001) / 60
            reward = sum(1 for l in dpo_losses[-50:] if l < 0.693) / 50 * 100
            print(f"  DPO {dpo_step+1}/{Config.DPO_STEPS} | Loss:{avg:.4f} | Reward↑:{reward:.0f}% | {spd:.2f}/s | ETA:{eta:.0f}m")
        
        if (dpo_step + 1) % 1000 == 0:
            gc.collect()
    
    del ref_model; gc.collect()
    print(f"\n  ✅ DPO COMPLETE! {Config.DPO_STEPS} steps in {(time.time()-t0)/60:.1f}m")

except Exception as e:
    print(f"  ⚠️ DPO skipped: {e}")

# ====================================================================
#  PHASE 3: SAVE & UPLOAD FINAL MODEL
# ====================================================================
print("\n" + "=" * 60)
print("  💾 PHASE 3: SAVE & UPLOAD")
print("=" * 60)

final_dir = f"{Config.WORK_DIR}/aksarallm-1.5b-v2-final"
os.makedirs(final_dir, exist_ok=True)

# Move to CPU for safe HF-format saving
model_cpu = model.to("cpu")
model_cpu.save_pretrained(final_dir, safe_serialization=True)
tokenizer.save_pretrained(final_dir)
print(f"  ✅ Saved to {final_dir}")

# Upload to HuggingFace
try:
    api.create_repo(Config.FINAL_NAME, exist_ok=True, token=Config.HF_TOKEN)
    api.upload_folder(folder_path=final_dir, repo_id=Config.FINAL_NAME, token=Config.HF_TOKEN)
    print(f"  ✅ Uploaded to {Config.FINAL_NAME}")
except Exception as e:
    print(f"  ⚠️ Upload: {e}")

# Upload to GCS backup
try:
    import subprocess
    subprocess.run(["gcloud", "storage", "cp", "-r", final_dir, "gs://aksarallm-checkpoints/final/"], check=True)
    print("  ✅ Backed up to GCS")
except:
    print("  ⚠️ GCS backup skipped")

# ====================================================================
#  PHASE 4: FINAL TEST
# ====================================================================
print("\n" + "=" * 60)
print("  🧪 PHASE 4: FINAL INFERENCE TEST")
print("=" * 60)

from transformers import pipeline
pipe = pipeline("text-generation", model=final_dir, tokenizer=final_dir, device="cpu")

tests = [
    "Siapa kamu?",
    "Apa itu Pancasila?",
    "Jelaskan tentang kecerdasan buatan!",
    "Ibukota Jawa Barat dimana?",
    "Halo, apa kabar?",
    "Buatkan fungsi Python untuk menghitung faktorial",
    "Apa perbedaan AI dan Machine Learning?",
]

for q in tests:
    msgs = [{"role": "system", "content": Config.SYSTEM_PROMPT}, {"role": "user", "content": q}]
    out = pipe(msgs, max_new_tokens=150, do_sample=True, temperature=0.7)
    resp = out[0]["generated_text"][-1]["content"]
    print(f"\n💬 {q}")
    print(f"🤖 {resp[:250]}")
    print("─" * 50)

# ====================================================================
#  DONE!
# ====================================================================
print("\n" + "━" * 60)
print("  🏁 AksaraLLM-1.5B-v2 COMPLETE!")
print(f"  📍 Local: {final_dir}")
print(f"  ☁️ HuggingFace: {Config.FINAL_NAME}")
print("  Next: Convert ke GGUF → Deploy!")
print("━" * 60)
