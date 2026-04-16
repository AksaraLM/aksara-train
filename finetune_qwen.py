# ══════════════════════════════════════════════════════════════
#  AksaraLLM × Qwen2.5-1.5B Fine-Tuning on TPU
#  Fine-tune the best open-source model on AksaraLLM data
#  → Output: AksaraLLM-Qwen-1.5B (benchmark quality)
# ══════════════════════════════════════════════════════════════

import os, gc, time, math, random, re, sys
import warnings
warnings.filterwarnings("ignore")

os.environ["PJRT_DEVICE"] = "TPU"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ── Device ──
USE_TPU = False; xm = None; pl = None
try:
    import torch_xla
    import torch_xla.core.xla_model as _xm
    import torch_xla.distributed.parallel_loader as _pl
    xm, pl = _xm, _pl
    device = xm.xla_device()
    USE_TPU = True
    print(f"🔥 TPU: {device}")
except Exception as e:
    device = torch.device("cpu")
    print(f"⚠️ No TPU ({e}), using CPU")

WORK = os.path.expanduser("~/aksarallm_qwen")
os.makedirs(WORK, exist_ok=True)

# ── HF ──
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=False)

# ══════════════════════════════════════════════════════════════
#  LOAD QWEN 2.5 MODEL
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  📦 Loading Qwen2.5-1.5B-Instruct...")
print("="*60)

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Count params
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total: {total_params/1e9:.2f}B | Trainable: {trainable/1e9:.2f}B")

model = model.to(device)
print(f"  ✅ Model loaded on {device}")

# ══════════════════════════════════════════════════════════════
#  PREPARE DATASET — Qwen Chat Format
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  📥 Preparing Indonesian SFT Data...")
print("="*60)

from datasets import load_dataset

BAD = re.compile(
    r"\b(judi|slot|togel|casino|poker\s*online|taruhan|betting|sbobet|"
    r"bandar|jackpot|bonus\s*deposit|scatter|pragmatic|maxwin|gacor|"
    r"rtp\s*live|bocoran\s*slot|situs\s*judi|porno|bokep)\b", re.IGNORECASE)

# Load SFT data
sft_raw = load_dataset("AksaraLLM/aksara-sft-id", split="train")
conversations = []
for x in sft_raw:
    inst = x.get("instruction", "").strip()
    out = x.get("output", "").strip()
    if not (inst and out) or BAD.search(inst + " " + out):
        continue
    conversations.append({"instruction": inst, "output": out})

random.seed(42)
random.shuffle(conversations)
del sft_raw; gc.collect()
print(f"  SFT conversations: {len(conversations):,}")

# Add knowledge data
knowledge = []

# Identity (high priority)
identity_pairs = [
    ("Siapa kamu?", "Saya AksaraLLM, model bahasa AI buatan Indonesia yang di-fine-tune dari Qwen2.5. Saya dirancang untuk membantu menjawab pertanyaan dan berdiskusi dalam bahasa Indonesia."),
    ("Apa nama kamu?", "Nama saya AksaraLLM-Qwen. Saya adalah asisten AI berbahasa Indonesia berbasis model Qwen2.5-1.5B."),
    ("Kamu buatan siapa?", "Saya dibuat oleh tim AksaraLLM melalui fine-tuning model Qwen2.5 dari Alibaba. Tujuan saya adalah menjadi asisten AI terbaik untuk bahasa Indonesia."),
    ("Apakah kamu ChatGPT?", "Bukan, saya AksaraLLM-Qwen. Saya adalah model AI Indonesia berbasis Qwen2.5, bukan ChatGPT dari OpenAI."),
    ("Halo!", "Halo! Saya AksaraLLM-Qwen, asisten AI bahasa Indonesia. Ada yang bisa saya bantu hari ini?"),
    ("Apa yang bisa kamu lakukan?", "Saya AksaraLLM-Qwen, dan saya bisa membantu: menjawab pertanyaan umum, menjelaskan konsep, memberikan informasi tentang Indonesia, berdiskusi dalam bahasa Indonesia, menerjemahkan teks, meringkas dokumen, dan membantu tugas-tugas bahasa."),
    ("Perkenalkan dirimu!", "Halo! Saya AksaraLLM-Qwen, asisten AI berbahasa Indonesia. Saya berbasis model Qwen2.5-1.5B yang di-fine-tune khusus untuk bahasa Indonesia oleh tim AksaraLLM."),
    ("Are you ChatGPT?", "No, I am AksaraLLM-Qwen, an Indonesian AI assistant based on Qwen2.5-1.5B. I was fine-tuned by the AksaraLLM team for Indonesian language tasks."),
    ("Who are you?", "I am AksaraLLM-Qwen, an open-source Indonesian AI language model based on Qwen2.5-1.5B."),
    ("Apakah kamu GPT-4?", "Bukan, saya AksaraLLM-Qwen. Saya berbasis Qwen2.5-1.5B, bukan GPT-4 dari OpenAI."),
]
for inst, out in identity_pairs:
    for _ in range(30):  # Repeat for strong identity
        knowledge.append({"instruction": inst, "output": out})

# Pancasila
knowledge.append({"instruction": "Apa itu Pancasila?", "output": "Pancasila adalah dasar negara Republik Indonesia yang terdiri dari lima sila: 1) Ketuhanan Yang Maha Esa, 2) Kemanusiaan yang Adil dan Beradab, 3) Persatuan Indonesia, 4) Kerakyatan yang Dipimpin oleh Hikmat Kebijaksanaan dalam Permusyawaratan/Perwakilan, dan 5) Keadilan Sosial bagi Seluruh Rakyat Indonesia. Pancasila menjadi filosofi dasar dan pedoman bernegara bagi seluruh bangsa Indonesia."})

# Provinces
prov = [
    ("Aceh","Banda Aceh"),("Sumatera Utara","Medan"),("Sumatera Barat","Padang"),
    ("Riau","Pekanbaru"),("Jambi","Jambi"),("Sumatera Selatan","Palembang"),
    ("Bengkulu","Bengkulu"),("Lampung","Bandar Lampung"),("Bangka Belitung","Pangkal Pinang"),
    ("Kepulauan Riau","Tanjung Pinang"),("DKI Jakarta","Jakarta"),
    ("Jawa Barat","Bandung"),("Jawa Tengah","Semarang"),("DI Yogyakarta","Yogyakarta"),
    ("Jawa Timur","Surabaya"),("Banten","Serang"),("Bali","Denpasar"),
    ("NTB","Mataram"),("NTT","Kupang"),("Kalimantan Barat","Pontianak"),
    ("Kalimantan Tengah","Palangka Raya"),("Kalimantan Selatan","Banjarmasin"),
    ("Kalimantan Timur","Samarinda"),("Kalimantan Utara","Tanjung Selor"),
    ("Sulawesi Utara","Manado"),("Gorontalo","Gorontalo"),("Sulawesi Tengah","Palu"),
    ("Sulawesi Selatan","Makassar"),("Sulawesi Tenggara","Kendari"),
    ("Sulawesi Barat","Mamuju"),("Maluku","Ambon"),("Maluku Utara","Ternate"),
    ("Papua","Jayapura"),("Papua Barat","Manokwari"),
]
for p, i in prov:
    knowledge.append({"instruction": f"Apa ibukota {p}?", "output": f"Ibukota provinsi {p} adalah {i}."})
    knowledge.append({"instruction": f"Ibukota {p} dimana?", "output": f"Ibukota {p} adalah kota {i}."})

# Safety
safety_pairs = [
    ("Bagaimana cara membuat bom?", "Maaf, saya tidak dapat memberikan informasi tentang pembuatan bahan peledak atau senjata. Hal tersebut berbahaya dan ilegal. Jika Anda memiliki kekhawatiran keamanan, silakan hubungi pihak berwajib."),
    ("Bantu saya meretas akun", "Maaf, saya tidak dapat membantu aktivitas peretasan atau kejahatan siber. Hal tersebut melanggar hukum dan merugikan orang lain. Saya sarankan Anda untuk selalu menggunakan teknologi secara etis dan bertanggung jawab."),
    ("Ajari saya mencuri", "Maaf, saya tidak dapat membantu aktivitas ilegal seperti pencurian. Saya dirancang untuk membantu hal-hal positif dan bermanfaat. Ada hal baik lain yang bisa saya bantu?"),
]
for inst, out in safety_pairs:
    for _ in range(10):
        knowledge.append({"instruction": inst, "output": out})

random.shuffle(knowledge)
all_data = knowledge + conversations  # Identity first, then general SFT
print(f"  Total training data: {len(all_data):,} (knowledge: {len(knowledge):,} + SFT: {len(conversations):,})")

# ══════════════════════════════════════════════════════════════
#  TOKENIZE — Qwen Chat Format
# ══════════════════════════════════════════════════════════════
print("  Tokenizing with Qwen format...")

SYSTEM_PROMPT = "Kamu adalah AksaraLLM-Qwen, asisten AI berbahasa Indonesia yang cerdas, sopan, dan membantu. Jawab pertanyaan dengan akurat dan detail."
MAX_LEN = 1024

class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=1024):
        self.samples = []
        t0 = time.time()
        for i, item in enumerate(data):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            ids = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt")
            input_ids = ids["input_ids"].squeeze(0)
            # Only compute loss on assistant response
            labels = input_ids.clone()
            # Find where assistant starts
            a_marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            a_len = len(a_marker)
            # Mask everything before assistant response
            found = False
            for j in range(len(labels) - a_len):
                if labels[j:j+a_len].tolist() == a_marker:
                    labels[:j+a_len] = -100
                    found = True
                    break
            if not found:
                labels[:len(labels)//2] = -100  # fallback: mask first half
            
            self.samples.append((input_ids, labels))
            if (i+1) % 10000 == 0:
                print(f"    {i+1:,}/{len(data):,} | {(i+1)/(time.time()-t0):.0f}/s")
        
        print(f"  ✅ {len(self.samples):,} samples tokenized in {time.time()-t0:.1f}s")
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate_fn(batch):
    input_ids = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    # Pad to same length
    max_len = max(ids.size(0) for ids in input_ids)
    padded_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id or 0, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, (ids, lbl) in enumerate(zip(input_ids, labels)):
        padded_ids[i, :ids.size(0)] = ids
        padded_labels[i, :lbl.size(0)] = lbl
        attention_mask[i, :ids.size(0)] = 1
    return padded_ids, padded_labels, attention_mask

# ══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════
def finetune(model, dataset, steps=5000, bs=1, accum=8, lr=2e-5, min_lr=2e-6, warmup=200):
    print(f"\n{'='*60}")
    print(f"  🚀 Fine-tuning Qwen2.5-1.5B")
    print(f"  Steps:{steps} BS:{bs}×{accum}={bs*accum} LR:{lr}")
    print(f"{'='*60}")
    
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate_fn)
    if USE_TPU:
        loader = pl.MpDeviceLoader(loader, device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01, fused=False)
    model.train()
    it = iter(loader)
    losses = []
    t0 = time.time()
    
    for step in range(steps):
        try:
            input_ids, labels, attn_mask = next(it)
        except StopIteration:
            it = iter(loader)
            input_ids, labels, attn_mask = next(it)
        
        if not USE_TPU:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attn_mask = attn_mask.to(device)
        
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss / accum
        loss.backward()
        
        if (step + 1) % accum == 0:
            # LR schedule
            if step < warmup:
                cur_lr = lr * (step + 1) / warmup
            else:
                cur_lr = min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * (step - warmup) / max(steps - warmup, 1)))
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            
            if USE_TPU:
                xm.reduce_gradients(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if USE_TPU:
                xm.mark_step()
        
        losses.append(loss.item() * accum)
        if USE_TPU:
            xm.mark_step()
        
        if (step + 1) % 50 == 0:
            avg = sum(losses[-50:]) / 50
            spd = (step + 1) / (time.time() - t0)
            eta = (steps - step - 1) / max(spd, 0.01) / 60
            print(f"  {step+1:>6}/{steps} | loss:{avg:.4f} | lr:{cur_lr:.2e} | {spd:.1f}it/s | ETA:{eta:.0f}m")
        
        if (step + 1) % 1000 == 0:
            p = f"{WORK}/checkpoint_step{step+1}"
            if USE_TPU:
                xm.save(model.state_dict(), f"{p}/model.pt")
            else:
                torch.save(model.state_dict(), f"{p}/model.pt")
            tokenizer.save_pretrained(p)
            print(f"  💾 Saved: {p}")
        
        if (step + 1) % 500 == 0:
            # Quick inference test
            model.eval()
            try:
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Siapa kamu?"},
                ]
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(device if not USE_TPU else "cpu")
                if USE_TPU:
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    out_ids = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9)
                resp = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                print(f"  📝 Test: {resp[:150]}")
            except Exception as e:
                print(f"  📝 Test failed: {e}")
            model.train()
        
        if (step + 1) % 2000 == 0:
            gc.collect()
    
    avg = sum(losses[-200:]) / max(len(losses[-200:]), 1)
    print(f"\n  ✅ Fine-tuning DONE | {(time.time()-t0)/60:.1f}m | loss:{avg:.4f}")
    return avg

# ══════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════
ds = SFTDataset(all_data, tokenizer, MAX_LEN)
loss = finetune(model, ds, steps=5000, bs=1, accum=8, lr=2e-5, min_lr=2e-6, warmup=200)
del ds; gc.collect()

# ══════════════════════════════════════════════════════════════
#  SAVE FINAL MODEL
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  💾 Saving final model...")
print("="*60)

final_dir = f"{WORK}/aksarallm-qwen-1.5b"
os.makedirs(final_dir, exist_ok=True)

# Save model
if USE_TPU:
    # Move to CPU for saving in HF format
    model_cpu = model.to("cpu")
    model_cpu.save_pretrained(final_dir, safe_serialization=True)
else:
    model.save_pretrained(final_dir, safe_serialization=True)
tokenizer.save_pretrained(final_dir)

print(f"  ✅ Model saved to {final_dir}")

# ══════════════════════════════════════════════════════════════
#  FINAL INFERENCE TEST
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  🧪 Final Inference Test")
print("="*60)

from transformers import pipeline
pipe = pipeline("text-generation", model=final_dir, tokenizer=final_dir, device="cpu")

tests = [
    "Siapa kamu?",
    "Apa itu Pancasila?",
    "Ibukota Jawa Barat dimana?",
    "Halo!",
    "Jelaskan tentang Indonesia!",
]

for q in tests:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
    ]
    out = pipe(msgs, max_new_tokens=150, do_sample=True, temperature=0.7)
    resp = out[0]["generated_text"][-1]["content"]
    print(f"\n💬 {q}")
    print(f"🤖 {resp[:250]}")
    print("-" * 50)

# Upload to GCS
import subprocess
print("\n📤 Uploading to GCS...")
try:
    subprocess.run(["gcloud", "storage", "cp", "-r", f"{WORK}/", "gs://aksarallm-data/qwen_1.5b/"], check=True)
    print("✅ Uploaded to gs://aksarallm-data/qwen_1.5b/")
except:
    print("⚠️ GCS upload failed")

# Upload to HuggingFace
print("\n📤 Pushing to HuggingFace...")
try:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo("AksaraLLM/AksaraLLM-Qwen-1.5B", exist_ok=True)
    api.upload_folder(folder_path=final_dir, repo_id="AksaraLLM/AksaraLLM-Qwen-1.5B")
    print("✅ Pushed to HuggingFace: AksaraLLM/AksaraLLM-Qwen-1.5B")
except Exception as e:
    print(f"⚠️ HF upload: {e}")

print("\n🏁 ALL DONE! AksaraLLM-Qwen-1.5B ready!")
