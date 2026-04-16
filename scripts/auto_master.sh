#!/bin/bash
# ==============================================================================
# 🤖 AKSARALLM MASTER ORCHESTRATOR 
# ==============================================================================
# Script ini adalah "Mandor" otomatis yang akan membuat TPU, memasukkan script,
# menjalankan pekerjaan di latar belakang (sehingga Mac bisa dimatikan),
# serta membuat pantauan log menjadi lebih mudah.
# ==============================================================================

# Berhenti jika ada error kritis
set -e

# ====================================================================
# [1] KONFIGURASI DEFAULT (Bisa disesuaikan kalau TPU habis)
# ====================================================================
PROJECT="aksarallm-tpu"
DEFAULT_ZONE="europe-west4-a"      # Zona default yang sering kosong
DEFAULT_ACCELERATOR="v6e-4"        # Tipe TPU default yang kita pakai
VERSION="v2-alpha-tpuv6e"          # Image OS TPU v6e

# Fungsi untuk print agar menarik dan mudah dibaca
print_step() {
    echo -e "\n\033[1;32m[+] $1\033[0m"
}
print_error() {
    echo -e "\n\033[1;31m[!] ERROR: $1\033[0m"
}
print_info() {
    echo -e "\033[1;36m    -> $1\033[0m"
}

# ====================================================================
# [2] VALIDASI ARGUMEN & CEK FILE
# ====================================================================
if [ "$#" -lt 2 ]; then
    echo -e "\n\033[1;33m📝 CARA PENGGUNAAN:\033[0m"
    echo "./aksarallm_auto_master.sh <NAMA_TPU_BARU> <LOKASI_FILE_PYTHON> [ZONA] [TIPE_TPU]"
    echo ""
    echo "Contoh 1 (Default): ./aksarallm_auto_master.sh tpu-pekerja1 ~/aksarallm_autopilot_v2.py"
    echo "Contoh 2 (Custom):  ./aksarallm_auto_master.sh tpu-pekerja2 ~/scraper.py us-east1-d v6e-4"
    exit 1
fi

TPU_NAME=$1
SCRIPT_PATH=$2
ZONE=${3:-$DEFAULT_ZONE}
ACCELERATOR=${4:-$DEFAULT_ACCELERATOR}
SCRIPT_NAME=$(basename "$SCRIPT_PATH")

# Pastikan file python benar-benar ada di Mac kita
if [ ! -f "$SCRIPT_PATH" ]; then
    print_error "File script '$SCRIPT_PATH' tidak ditemukan!"
    exit 1
fi

# ====================================================================
# [3] MEMBUAT TPU BARU (SPOT INSTANCE)
# ====================================================================
print_step "MEMBUAT TPU: $TPU_NAME ($ACCELERATOR di $ZONE)"
print_info "Harap tunggu 1-2 menit hingga TPU selesai dibuat dari Google Cloud..."

# Kita buat ignore error sementara hanya untuk command create 
# (kalau-kalau TPU sudah ada dengan nama yang sama)
set +e
gcloud compute tpus tpu-vm create $TPU_NAME \
    --zone=$ZONE \
    --accelerator-type=$ACCELERATOR \
    --version=$VERSION \
    --project=$PROJECT \
    --spot \
    --quiet
CREATE_STATUS=$?
set -e

if [ $CREATE_STATUS -ne 0 ]; then
    print_error "Gagal membuat TPU. Kemungkinan zona $ZONE kepenuhan. Batal."
    exit 1
fi

# ====================================================================
# [4] MENUNGGU TPU SIAP (BOOTING)
# ====================================================================
print_step "MENUNGGU TPU BOOTING AGAR BISA DI-SSH..."
sleep 15 # TPU butuh waktu untuk membuka akses port 22

# Mekanisme Retry SSH (Otomatis mencoba terus sampai bisa masuk)
MAX_RETRIES=5
RETRY_COUNT=0
SSH_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    set +e
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --command="echo 'SSH Terhubung!'" --quiet > /dev/null 2>&1
    SSH_STATUS=$?
    set -e
    
    if [ $SSH_STATUS -eq 0 ]; then
        SSH_READY=true
        print_info "TPU sudah siap menerima perintah!"
        break
    else
        print_info "TPU belum siap. Mencoba lagi ($(($RETRY_COUNT+1))/$MAX_RETRIES) dalam 10 detik..."
        sleep 10
        RETRY_COUNT=$((RETRY_COUNT+1))
    fi
done

if [ "$SSH_READY" = false ]; then
    print_error "Gagal terhubung ke TPU setelah $MAX_RETRIES percobaan. Coba konek manual nanti."
    exit 1
fi

# ====================================================================
# [5] MENGIRIM FILE PYTHON KE DALAM TPU
# ====================================================================
print_step "MENGIRIM FILE '$SCRIPT_NAME' KE DALAM TPU..."
gcloud compute tpus tpu-vm scp "$SCRIPT_PATH" $TPU_NAME:~/ \
    --zone=$ZONE \
    --project=$PROJECT \
    --quiet
print_info "File sukses terkirim."

# ====================================================================
# [6] MENGINSTAL DEPENDENCY & MENJALANKAN SCRIPT DI BACKGROUND
# ====================================================================
print_step "MENGINSTAL PACKAGE & MENJALANKAN SCRIPT..."
print_info "Skrip ini akan dijalankan lewat perintah 'nohup' (Berjalan tanpa henti)."

gcloud compute tpus tpu-vm ssh $TPU_NAME \
    --zone=$ZONE \
    --project=$PROJECT \
    --command="pip install transformers datasets huggingface_hub accelerate -q && nohup python3 ~/$SCRIPT_NAME > ~/${SCRIPT_NAME}.log 2>&1 &" \
    --quiet

# ====================================================================
# [7] LAPORAN SUKSES
# ====================================================================
print_step "🎉 SEMUA SELESAI, ORKESTRASI BERHASIL!"
echo ""
echo -e "\033[1;36mPekerjaan $SCRIPT_NAME sedang diproses di $TPU_NAME\033[0m"
echo -e "\033[1;33mUntuk melihat langsung (LIVE LOG) proses kerjanya, copy paste ini di Terminal:\033[0m"
echo "gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --command=\"tail -f ~/${SCRIPT_NAME}.log\""
echo ""
echo -e "\033[1;31mJANGAN LUPA: Hapus TPU ini jika $SCRIPT_NAME sudah selesai agar tidak memakan biaya!\033[0m"
echo "gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE --project=$PROJECT"
echo ""
