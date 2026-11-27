#!/usr/bin/env bash
# =============================================================================
#  deploy.sh  -  kirim semua keperluan pretraining ke GPU sewaan (Vast.ai) dan
#                jalankan. Jalankan dari folder ini:   ./deploy.sh <perintah>
#
#  Alur biasa (GPU baru):
#     1. edit .env (SSH_HOST, SSH_PORT)      2. taruh checkpoint_latest.zip di sini
#     3. ./deploy.sh all
#
#  Perintah satu per satu:
#     gpus         daftar GPU Vast + kode arsitektur:  gpus | gpus 5090 | gpus show "RTX 5090" | gpus refresh
#     rent         SEWA OTOMATIS instance on-demand (pilih host, buat instance, tunggu IP SSH, isi .env):
#                  rent --dry-run          lihat rencana + estimasi token tanpa menyewa
#                  rent --budget 5         sewa dengan anggaran $5 (GPU dari GPU_MODEL di .env)
#                  rent --gpu "RTX 3090" --disk 70 --max-price 0.15   variasi
#     check        cek koneksi, GPU, nvcc, disk (tidak mengubah apa pun)
#     key          daftarkan key SSH Mac ke instance lewat API Vast (opsional)
#     auth         buat token Google Drive baru di Mac (browser; untuk token yang expired/dicabut)
#     token        kirim token Drive terbaru ke instance (training tetap jalan)
#     vast         status instance + saldo Vast via API (jalan walau instance paused / SSH mati)
#     package      buat source_code.zip (kode + rl_data/ceiling_enja + credential Drive)
#     source       kirim source_code.zip lalu ekstrak di instance
#     corpus       kirim korpus pretraining (symlink corpus_pretrain_v8/ + 3 shard midtrain)
#     postdata     kirim data SFT v3 Ceiling (sft_mixed_enja) + DPO Ceiling (dpo_ceiling_enja)
#     checkpoint   kirim checkpoint_latest.zip lalu ekstrak
#     build        build otomatis sesuai GPU (arsitektur + DDP) + grad_check + infer
#     start        mulai training (tmux di instance) - auto-restart kalau trainer mati
#     onstart      tampilkan / pasang on-start script Vast (interruptible):  onstart set
#     all          check -> package -> source -> corpus -> checkpoint -> build -> start
#     monitor      log penuh gaya Kaggle, real time (Ctrl+C keluar; training tetap jalan)
#     panel        panel ringkas (step, loss, tok/s, GPU) refresh tiap 2 detik
#     status       ringkasan sekali tampil
#     log          ikuti log training mentah
#     attach       masuk ke layar tmux training (keluar: Ctrl+b lalu d)
#     pull         tarik log + nvidia-smi ke Mac dan hitung ringkasan kecepatan
#     drive        unggah checkpoint saat ini ke Google Drive
#     stop         hentikan training dengan aman (checkpoint disimpan)
#     ssh          buka shell di instance
#     help         tampilkan ini
# =============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
ENV_FILE="${ENV_FILE:-$ROOT/.env}"

c_red=$'\033[31m'; c_grn=$'\033[32m'; c_ylw=$'\033[33m'; c_cya=$'\033[36m'; c_off=$'\033[0m'
say()  { printf '%s>>%s %s\n' "$c_cya" "$c_off" "$*"; }
ok()   { printf '%s OK%s %s\n' "$c_grn" "$c_off" "$*"; }
warn() { printf '%s !!%s %s\n' "$c_ylw" "$c_off" "$*" >&2; }
die()  { printf '%sERROR:%s %s\n' "$c_red" "$c_off" "$*" >&2; exit 1; }

[ -f "$ENV_FILE" ] || die "file .env tidak ada di $ROOT"
set -a; # shellcheck disable=SC1090
. "$ENV_FILE"; set +a

# SSH_HOST/SSH_PORT hanya dicek saat benar-benar dipakai (ensure_connection),
# supaya 'gpus', 'package', 'onstart', dan 'help' tetap jalan sebelum instance disewa.
SSH_USER="${SSH_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"; SSH_KEY="${SSH_KEY/#\~/$HOME}"
REMOTE_BASE="${REMOTE_BASE:-/workspace}"
PROJECT_DIR_NAME="${PROJECT_DIR_NAME:-transformer_500m_cuda}"
SPEED_DIR="${SPEED_DIR:-./speed_monitoring_gpu}"; [[ "$SPEED_DIR" = /* ]] || SPEED_DIR="$ROOT/${SPEED_DIR#./}"
PROJECT_ROOT="$(cd "$ROOT/.." && pwd)"
GPU_MODEL="${GPU_MODEL:-}"
GPU_INFO="$ROOT/tools/gpu_info.py"
GPU_ARCH_HINT=""
if [ -n "$GPU_MODEL" ] && [ -f "$ROOT/tools/gpu_table.tsv" ] && command -v python3 >/dev/null 2>&1; then
  GPU_ARCH_HINT="$(python3 "$GPU_INFO" arch "$GPU_MODEL" 2>/dev/null || true)"
fi
require_gpu_model() {   # dipakai build/all/check: GPU_MODEL yang salah ketik harus ketahuan SEBELUM menyewa/mengirim
  [ -z "$GPU_MODEL" ] && { warn "GPU_MODEL kosong di .env -> arsitektur dibaca dari GPU di instance (tetap jalan)"; return 0; }
  [ "${CUDA_ARCH:-auto}" != "auto" ] && return 0
  [ -n "$GPU_ARCH_HINT" ] && return 0
  python3 "$GPU_INFO" arch "$GPU_MODEL" >/dev/null || die "GPU_MODEL=\"$GPU_MODEL\" tidak dikenali. Lihat ./deploy.sh gpus"
}
CORPUS_LOCAL="$ROOT/corpus_pretrain_v8"
SFT_LOCAL="$PROJECT_ROOT/sft_data/sft_mixed_enja"
DPO_LOCAL="$PROJECT_ROOT/sft_data/dpo_ceiling_enja"
RL_LOCAL="$PROJECT_ROOT/rl_data/ceiling_enja"
CKPT_ZIP="$ROOT/checkpoint_latest.zip"
SRC_ZIP="$ROOT/source_code.zip"
CORPUS_REMOTE_DIR="${CORPUS_REMOTE_DIR:-$REMOTE_BASE/corpus_pretrain_v8}"
SFT_REMOTE_DIR="${SFT_REMOTE_DIR:-$REMOTE_BASE/sft_data/sft_mixed_enja}"
DPO_REMOTE_DIR="${DPO_REMOTE_DIR:-$REMOTE_BASE/sft_data/dpo_ceiling_enja}"
RL_REMOTE_DIR="${RL_REMOTE_DIR:-$REMOTE_BASE/$PROJECT_DIR_NAME/rl_data/ceiling_enja}"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o LogLevel=ERROR
          -o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -i "$SSH_KEY" -p "$SSH_PORT")
TARGET="$SSH_USER@$SSH_HOST"

rssh()     { ssh "${SSH_OPTS[@]}" "$TARGET" "$@"; }
rssh_tty() { ssh -t "${SSH_OPTS[@]}" "$TARGET" "$@"; }
rsync_to() {   # rsync_to <flag-ekstra> <sumber...> <tujuan-remote>
  local -a extra=($1); shift
  rsync "${extra[@]}" -e "ssh ${SSH_OPTS[*]}" "$@"
}

need() { command -v "$1" >/dev/null 2>&1 || die "perintah '$1' tidak ada di Mac Anda"; }

remote_env_file() {   # nilai .env yang dibutuhkan remote.sh
  cat <<EOF
REMOTE_BASE='$REMOTE_BASE'
PROJECT_DIR_NAME='$PROJECT_DIR_NAME'
ARCH_CONFIG='${ARCH_CONFIG:-config_500m.json}'
CKPT_DIR='${CKPT_DIR:-$REMOTE_BASE/checkpoints}'
CORPUS_REMOTE_DIR='$CORPUS_REMOTE_DIR'
SFT_REMOTE_DIR='$SFT_REMOTE_DIR'
DPO_REMOTE_DIR='$DPO_REMOTE_DIR'
RL_REMOTE_DIR='$RL_REMOTE_DIR'
CUDA_ARCH='${CUDA_ARCH:-auto}'
GPU_MODEL='$GPU_MODEL'
GPU_ARCH_HINT='$GPU_ARCH_HINT'
MONITOR_TAIL='${MONITOR_TAIL:-60}'
NUM_GPUS='${NUM_GPUS:-auto}'
BATCH_SIZE='${BATCH_SIZE:-14}'
SEQ='${SEQ:-512}'
MAX_TOKENS='${MAX_TOKENS:-10000000000}'
LR='${LR:-1e-4}'
GRAD_CLIP='${GRAD_CLIP:-0.5}'
LOSS_SCALE='${LOSS_SCALE:-4096}'
SAFETY='${SAFETY:-light}'
SAVE_EVERY='${SAVE_EVERY:-400}'
LOG_EVERY='${LOG_EVERY:-10}'
UPLOAD_EVERY='${UPLOAD_EVERY:-3000}'
EVAL_EVERY_TOKENS='${EVAL_EVERY_TOKENS:-200000000}'
VOCAB_DIR='${VOCAB_DIR:-vocab_9}'
BPE_FILE='${BPE_FILE:-bpe_v9.bin}'
TW_W_JA='${TW_W_JA:-}'
TW_W_EN='${TW_W_EN:-}'
TW_W_CODE='${TW_W_CODE:-}'
TW_W_MATH='${TW_W_MATH:-}'
TW_W_MIXED='${TW_W_MIXED:-}'
SFT_LR='${SFT_LR:-2e-5}'
SFT_EPOCHS='${SFT_EPOCHS:-2}'
SFT_BATCH_SIZE='${SFT_BATCH_SIZE:-8}'
SFT_MAX_SEQ='${SFT_MAX_SEQ:-1024}'
DPO_LR='${DPO_LR:-5e-6}'
DPO_EPOCHS='${DPO_EPOCHS:-1}'
DPO_BATCH_SIZE='${DPO_BATCH_SIZE:-4}'
DPO_MAX_SEQ='${DPO_MAX_SEQ:-1024}'
DPO_BETA='${DPO_BETA:-0.1}'
DPO_MAX_MARGIN='${DPO_MAX_MARGIN:-4.0}'
NCCL_P2P_DISABLE='${NCCL_P2P_DISABLE:-1}'
FINAL_UPLOAD_WAIT='${FINAL_UPLOAD_WAIT:-1500}'
SKIP_GRADCHECK='${SKIP_GRADCHECK:-0}'
GDRIVE_FOLDER_ID='${GDRIVE_FOLDER_ID:-}'
RESTART_MAX='${RESTART_MAX:-8}'
DRIVE_RECOVER='${DRIVE_RECOVER:-1}'
STAGE='${STAGE:-pretrain}'
EOF
}

sync_remote_scripts() {   # kirim remote.sh + deploy.env terbaru (cepat; dipanggil sebelum tiap perintah remote)
  local tmp; tmp="$(mktemp)"; remote_env_file > "$tmp"
  rsync_to "-aq" "$ROOT/remote/remote.sh" "$TARGET:$REMOTE_BASE/remote.sh" || die "gagal mengirim remote.sh (koneksi SSH?)"
  rsync_to "-aq" "$tmp" "$TARGET:$REMOTE_BASE/deploy.env" || die "gagal mengirim deploy.env"
  rm -f "$tmp"
}
R() { sync_remote_scripts; rssh "bash '$REMOTE_BASE/remote.sh' $*"; }
R_tty() { sync_remote_scripts; rssh_tty "bash '$REMOTE_BASE/remote.sh' $*"; }

ensure_connection() {
  : "${SSH_HOST:?isi SSH_HOST di .env}" "${SSH_PORT:?isi SSH_PORT di .env}"
  if rssh 'echo ok' >/dev/null 2>&1; then return 0; fi
  if [ -n "${VAST_API_KEY:-}" ] && [ -n "${VAST_INSTANCE_ID:-}" ]; then
    warn "SSH ditolak; mencoba mendaftarkan key lewat API Vast..."
    cmd_key && sleep 8 && rssh 'echo ok' >/dev/null 2>&1 && return 0
  fi
  die "tidak bisa SSH ke $TARGET:$SSH_PORT.
   - Instance dipause (interruptible)? Cek statusnya:  ./deploy.sh vast
   - Instance sudah 'running'? Cek host/port di tombol Connect Vast.
   - Key belum terdaftar? Isi VAST_API_KEY + VAST_INSTANCE_ID di .env lalu ./deploy.sh key,
     atau tempel isi ${SSH_KEY}.pub lewat ikon [Key] di kartu instance Vast."
}

# ─── Perintah ────────────────────────────────────────────────────────────────
cmd_key() {
  [ -n "${VAST_API_KEY:-}" ] && [ -n "${VAST_INSTANCE_ID:-}" ] || die "isi VAST_API_KEY dan VAST_INSTANCE_ID di .env"
  [ -f "$SSH_KEY.pub" ] || die "$SSH_KEY.pub tidak ada (buat dengan: ssh-keygen -t ed25519)"
  need curl
  local pub resp; pub="$(cat "$SSH_KEY.pub")"
  resp=$(curl -sS -X POST "https://console.vast.ai/api/v0/instances/$VAST_INSTANCE_ID/ssh/" \
         -H "Authorization: Bearer $VAST_API_KEY" -H "Content-Type: application/json" \
         -d "$(printf '{"ssh_key": "%s"}' "$pub")")
  echo "$resp" | grep -q '"success": *true' && ok "key SSH terdaftar di instance $VAST_INSTANCE_ID" || die "gagal daftar key: $resp"
}

cmd_auth() {   # token Google Drive baru (jalankan di Mac; browser terbuka untuk pilih akun)
  need python3
  python3 -c "import google_auth_oauthlib" 2>/dev/null || die "butuh library: pip3 install google-auth-oauthlib google-api-python-client"
  [ -n "$(ls "$PROJECT_ROOT"/client_secret_*.json 2>/dev/null)" ] || die "client_secret_*.json tidak ada di $PROJECT_ROOT (unduh OAuth Desktop JSON dari Google Cloud Console)"
  say "alur OAuth Google Drive - pilih akun yang benar di browser..."
  ( cd "$PROJECT_ROOT" && python3 scripts/upload_gdrive.py --auth ) || die "autentikasi gagal"
  ok "token baru tersimpan: $(abspath_env "${GDRIVE_TOKEN_FILE:-$PROJECT_ROOT/gdrive_token.json}")"
  say "kirim ke instance yang sedang jalan:  ./deploy.sh token    |    untuk zip baru:  ./deploy.sh package"
}

cmd_token() {   # kirim gdrive_token.json terbaru ke instance (training TIDAK perlu dihentikan)
  local tok; tok="$(abspath_env "${GDRIVE_TOKEN_FILE:-}")"
  { [ -n "$tok" ] && [ -f "$tok" ]; } || die "file token tidak ada: ${tok:-(isi GDRIVE_TOKEN_FILE di .env)}"
  ensure_connection
  say "mengirim token Drive ke instance..."
  sync_remote_scripts
  rsync_to "-aq" "$tok" "$TARGET:$REMOTE_BASE/$PROJECT_DIR_NAME/gdrive_token.json" || die "gagal mengirim token"
  rssh "stat -c 'token terpasang di instance: %n (%s byte, %y)' '$REMOTE_BASE/$PROJECT_DIR_NAME/gdrive_token.json' 2>/dev/null" || true
  ok "upload Drive berikutnya akan memakai token baru (tanpa restart training)"
}

cmd_rent() {   # sewa instance on-demand otomatis: pilih host terbaik -> buat -> tunggu SSH -> isi .env
  [ -n "${VAST_API_KEY:-}" ] || die "isi VAST_API_KEY di .env (key baru: https://cloud.vast.ai/cli/)"
  need python3
  [ -f "$ROOT/tools/vast_rent.py" ] || die "tools/vast_rent.py tidak ada"
  python3 "$ROOT/tools/vast_rent.py" --env-file "$ENV_FILE" "$@"
}

cmd_vast() {   # status instance + saldo lewat API Vast (jalan WALAU instance paused / SSH mati)
  [ -n "${VAST_API_KEY:-}" ] && [ -n "${VAST_INSTANCE_ID:-}" ] || die "isi VAST_API_KEY dan VAST_INSTANCE_ID di .env (key baru: https://cloud.vast.ai/cli/)"
  need python3; need curl
  local inst bal
  inst=$(curl -sS "https://console.vast.ai/api/v1/instances/$VAST_INSTANCE_ID/" -H "Authorization: Bearer $VAST_API_KEY") || die "tidak bisa menghubungi API Vast"
  if printf '%s' "$inst" | grep -q '"success": *false'; then
    local msg; msg=$(printf '%s' "$inst" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("msg","?"))' 2>/dev/null)
    case "$msg" in
      *login*|*auth*) die "VAST_API_KEY tidak valid/dicabut. Buat key baru di https://cloud.vast.ai/cli/ lalu perbarui .env" ;;
      *)              warn "API Vast: $msg (VAST_INSTANCE_ID di .env mungkin masih ID instance lama)" ;;
    esac
  else
    printf '%s' "$inst" | python3 -c '
import json,sys
i = json.load(sys.stdin); i = i.get("instance", i)
print("instance : %s" % (i.get("label") or "?"))
print("status   : %s" % (i.get("status") or "?"))
b = i.get("cur_bid")
print("bid      : %s" % ("$%.3f/jam" % b if b is not None else "?"))
' || warn "tidak bisa membaca respons instance"
  fi
  bal=$(curl -sS "https://console.vast.ai/api/v0/users/current/" -H "Authorization: Bearer $VAST_API_KEY") || bal='{"success":false}'
  if printf '%s' "$bal" | grep -q '"success": *false'; then
    warn "saldo tidak terbaca (VAST_API_KEY tidak valid)"
  else
    printf '%s' "$bal" | python3 -c '
import json,sys
u = json.load(sys.stdin); u = u.get("user", u)
c = u.get("credit"); c = c if c is not None else u.get("balance")
print("saldo    : %s" % ("$%.2f" % c if c is not None else "?"))
' 2>/dev/null
  fi
  echo "(status 'paused' = GPU tidak ditagih, hanya storage; training auto-resume saat running lagi)"
}

cmd_check() {
  require_gpu_model
  say "koneksi ke $TARGET port $SSH_PORT"
  ensure_connection; ok "SSH tersambung"
  R check
  echo
  say "file lokal"
  [ -d "$CORPUS_LOCAL" ] && ok "korpus pretrain v8: $(du -shL "$CORPUS_LOCAL" 2>/dev/null | cut -f1) ($(find -L "$CORPUS_LOCAL" -type f | wc -l | tr -d ' ') file)" || warn "korpus tidak ada: $CORPUS_LOCAL"
  [ -d "$SFT_LOCAL" ] && ok "SFT v3 Ceiling: $(du -shL "$SFT_LOCAL" 2>/dev/null | cut -f1)" || warn "SFT v3 belum ada: $SFT_LOCAL"
  [ -d "$DPO_LOCAL" ] && ok "DPO Ceiling: $(du -shL "$DPO_LOCAL" 2>/dev/null | cut -f1)" || warn "DPO Ceiling belum ada: $DPO_LOCAL"
  [ -d "$RL_LOCAL" ] && ok "RL GRPO Ceiling: $(du -shL "$RL_LOCAL" 2>/dev/null | cut -f1)" || warn "RL Ceiling belum ada: $RL_LOCAL"
  [ -f "$CKPT_ZIP" ] && ok "checkpoint_latest.zip: $(du -h "$CKPT_ZIP" | cut -f1)" || warn "checkpoint_latest.zip belum ada di $ROOT"
  [ -f "$SRC_ZIP" ] && ok "source_code.zip: $(du -h "$SRC_ZIP" | cut -f1)" || warn "source_code.zip belum dibuat (./deploy.sh package)"
}

cmd_package() {
  need python3
  say "membuat source_code.zip dari $PROJECT_ROOT"
  python3 "$ROOT/tools/patch_source.py" --root "$PROJECT_ROOT" --out "$SRC_ZIP" --project-name "$PROJECT_DIR_NAME" \
    --vocab-dir "${VOCAB_DIR:-vocab_9}" --bpe "${BPE_FILE:-bpe_v9.bin}" --arena "${ARENA_FRACTION:-0.75}" \
    --folder-id "${GDRIVE_FOLDER_ID:-}" \
    --token "$(abspath_env "${GDRIVE_TOKEN_FILE:-}")" --client-secret "$(abspath_env "${GDRIVE_CLIENT_SECRET_FILE:-}")" \
    --service-account "$(abspath_env "${GDRIVE_SERVICE_ACCOUNT_FILE:-}")" || die "package gagal"
  ok "source_code.zip siap"
}
abspath_env() { local p="$1"; [ -z "$p" ] && return 0; [[ "$p" = /* ]] && echo "$p" || echo "$ROOT/$p"; }

cmd_source() {
  [ -f "$SRC_ZIP" ] || cmd_package
  ensure_connection
  say "mengirim source_code.zip ($(du -h "$SRC_ZIP" | cut -f1))"
  sync_remote_scripts
  rsync_to "-a --partial" "$SRC_ZIP" "$TARGET:$REMOTE_BASE/source_code.zip" || die "upload gagal"
  rssh "bash '$REMOTE_BASE/remote.sh' extract" || die "ekstrak gagal"
  ok "source terpasang di $REMOTE_BASE/$PROJECT_DIR_NAME"
}

cmd_corpus() {
  [ -d "$CORPUS_LOCAL" ] || die "korpus lokal tidak ada: $CORPUS_LOCAL"
  ensure_connection
  local files total_bytes jobs="${CORPUS_PARALLEL:-5}"
  total_bytes=$(find -L "$CORPUS_LOCAL" -type f -exec stat -f %z {} + 2>/dev/null | awk '{s+=$1} END{print s+0}')
  say "korpus: $(awk -v b="$total_bytes" 'BEGIN{printf "%.1f", b/1073741824}') GB -> $TARGET:$CORPUS_REMOTE_DIR"
  local need_kb=$(( total_bytes / 1024 )) avail_kb
  rssh "mkdir -p '$CORPUS_REMOTE_DIR'"
  avail_kb=$(rssh "df -Pk '$CORPUS_REMOTE_DIR' | tail -1 | awk '{print \$4}'")
  local have_kb; have_kb=$(rssh "du -sk '$CORPUS_REMOTE_DIR' 2>/dev/null | cut -f1")
  if [ $(( need_kb - have_kb )) -gt "$avail_kb" ]; then
    die "disk instance tidak cukup: butuh tambahan $(( (need_kb-have_kb)/1048576 )) GB, sisa $((avail_kb/1048576)) GB. Sewa disk lebih besar, atau kurangi isi corpus_pretrain_v8/."
  fi
  local started; started=$(date +%s)
  # Satu rsync per sub-folder (paralel); -z karena teks terkompres ~3x (penting di uplink lambat).
  # Resume otomatis: rsync melewati file dengan ukuran+waktu sama dan melanjutkan yang terpotong.
  local d pids=() names=()
  for d in "$CORPUS_LOCAL"/*/; do
    d="${d%/}"; names+=("$(basename "$d")")
  done
  local top=()
  for d in "$CORPUS_LOCAL"/*; do [ -f "$d" ] && top+=("$d"); done
  [ ${#top[@]} -gt 0 ] && rsync_to "-aL --partial --inplace" "${top[@]}" "$TARGET:$CORPUS_REMOTE_DIR/"
  local running=0 n
  for n in "${names[@]}"; do
    while [ "$running" -ge "$jobs" ]; do wait -n 2>/dev/null; running=$((running-1)); done
    ( rsync_to "-aLz --partial --inplace" "$CORPUS_LOCAL/$n" "$TARGET:$CORPUS_REMOTE_DIR/" >"/tmp/deploy_corpus_$n.log" 2>&1 || echo "GAGAL: $n" >&2 ) &
    running=$((running+1))
  done
  # progres sederhana sampai semua selesai
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -gt 0 ]; do
    local got; got=$(rssh "du -sk '$CORPUS_REMOTE_DIR' 2>/dev/null | cut -f1" 2>/dev/null || echo 0)
    printf '\r   terkirim %s / %s GB   (%ss)   ' "$(awk -v k="${got:-0}" 'BEGIN{printf "%.1f",k/1048576}')" "$(awk -v b="$total_bytes" 'BEGIN{printf "%.1f",b/1073741824}')" "$(( $(date +%s) - started ))"
    sleep 15
  done; wait; echo
  say "verifikasi ukuran file..."
  local lf rf
  lf="$(cd "$CORPUS_LOCAL" && find -L . -type f -exec stat -f '%z %N' {} + | sort -k2)"
  rf="$(rssh "cd '$CORPUS_REMOTE_DIR' && find -L . -type f -exec stat -c '%s %n' {} + | sort -k2")"
  if [ "$lf" = "$rf" ]; then ok "korpus utuh: $(echo "$lf" | wc -l | tr -d ' ') file, ukuran cocok semua"
  else
    diff <(echo "$lf") <(echo "$rf") | head -10
    die "ada file yang belum cocok. Jalankan ./deploy.sh corpus lagi (melanjutkan yang terpotong)."
  fi
}

cmd_postdata() {
  [ -d "$SFT_LOCAL" ] || die "SFT v3 Ceiling tidak ada: $SFT_LOCAL"
  [ -d "$DPO_LOCAL" ] || die "DPO Ceiling tidak ada: $DPO_LOCAL"
  ensure_connection
  say "mengirim SFT v3 Ceiling ($(du -shL "$SFT_LOCAL" | cut -f1)) -> $TARGET:$SFT_REMOTE_DIR"
  rssh "mkdir -p '$SFT_REMOTE_DIR' '$DPO_REMOTE_DIR'"
  rsync_to "-aLz --partial --inplace" "$SFT_LOCAL/" "$TARGET:$SFT_REMOTE_DIR/" || die "gagal kirim SFT v3"
  say "mengirim DPO Ceiling ($(du -shL "$DPO_LOCAL" | cut -f1)) -> $TARGET:$DPO_REMOTE_DIR"
  rsync_to "-aLz --partial --inplace" "$DPO_LOCAL/" "$TARGET:$DPO_REMOTE_DIR/" || die "gagal kirim DPO Ceiling"
  if [ -d "$RL_LOCAL" ]; then
    rssh "mkdir -p '$RL_REMOTE_DIR'"
    rsync_to "-aLz --partial --inplace" "$RL_LOCAL/" "$TARGET:$RL_REMOTE_DIR/" || die "gagal kirim RL Ceiling"
  fi
  ok "data SFT v3 + DPO Ceiling + RL Ceiling terpasang di instance"
}

cmd_checkpoint() {
  [ -f "$CKPT_ZIP" ] || die "taruh checkpoint_latest.zip di $ROOT lalu ulangi."
  ensure_connection
  local size stage rz
  size=$(stat -f %z "$CKPT_ZIP")
  say "checkpoint: $(du -h "$CKPT_ZIP" | cut -f1)"
  sync_remote_scripts
  [ "$(rssh "bash '$REMOTE_BASE/remote.sh' ckpt_busy")" = "busy" ] && die "training sedang memakai checkpoint di instance. Hentikan dulu (./deploy.sh stop)."
  stage=$(rssh "bash '$REMOTE_BASE/remote.sh' stage $size") || die "tidak ada ruang untuk checkpoint di instance"
  rz="$stage/checkpoint_latest.zip"
  say "tujuan sementara di instance: $stage"
  # 4 aliran paralel: satu aliran SSH ~5 MB/s di uplink biasa, 4 aliran jauh lebih cepat.
  local streams="${CKPT_STREAMS:-4}" blocks per i
  blocks=$(( (size + 1048575) / 1048576 )); per=$(( (blocks + streams - 1) / streams ))
  rssh "rm -f '$rz'; truncate -s $size '$rz'" || die "tidak bisa membuat file di $stage"
  local pids=() started; started=$(date +%s)
  for i in $(seq 0 $((streams-1))); do
    local skip=$((i*per)) cnt=$per
    [ "$skip" -ge "$blocks" ] && break
    ( dd if="$CKPT_ZIP" bs=1048576 skip=$skip count=$cnt 2>/dev/null \
        | ssh "${SSH_OPTS[@]}" "$TARGET" "dd of='$rz' bs=1M seek=$skip conv=notrunc 2>/dev/null" ) &
    pids+=($!)
  done
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -gt 0 ]; do
    local got; got=$(rssh "du -k '$rz' 2>/dev/null | cut -f1" 2>/dev/null || echo 0)
    printf '\r   terkirim ~%s / %s MB  (%ss)   ' "$(( ${got:-0} / 1024 ))" "$(( size / 1048576 ))" "$(( $(date +%s) - started ))"
    sleep 5
  done
  local p rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; echo
  [ $rc -eq 0 ] || die "salah satu aliran gagal. Ulangi ./deploy.sh checkpoint"
  say "verifikasi md5 (butuh beberapa detik)..."
  local lm rm_
  lm=$(md5 -q "$CKPT_ZIP"); rm_=$(rssh "md5sum '$rz' | cut -d' ' -f1")
  [ "$lm" = "$rm_" ] || die "md5 tidak cocok (lokal $lm, remote $rm_). Ulangi ./deploy.sh checkpoint"
  ok "md5 cocok: $lm"
  rssh "bash '$REMOTE_BASE/remote.sh' unpack '$rz'" || die "ekstrak checkpoint gagal"
}

cmd_build() { require_gpu_model; ensure_connection; R build; }
cmd_start() { ensure_connection; R start; echo; say "pantau:  ./deploy.sh monitor"; }

onstart_cmd() { echo "bash $REMOTE_BASE/remote.sh onstart"; }

cmd_onstart() {   # on-start script Vast: training auto-resume tiap instance dinyalakan (interruptible)
  local mode="${1:-}"
  local cmd; cmd="$(onstart_cmd)"
  if [ "$mode" = "set" ]; then
    [ -n "${VAST_API_KEY:-}" ] && [ -n "${VAST_INSTANCE_ID:-}" ] || die "isi VAST_API_KEY dan VAST_INSTANCE_ID di .env, atau tempel manual di Vast"
    need curl
    say "memasang on-start script ke instance $VAST_INSTANCE_ID lewat API Vast..."
    local resp
    resp=$(curl -sS -X PUT "https://console.vast.ai/api/v0/instances/$VAST_INSTANCE_ID/" \
           -H "Authorization: Bearer $VAST_API_KEY" -H "Content-Type: application/json" \
           -d "$(printf '{"on_start_cmd": "%s"}' "$cmd")")
    if echo "$resp" | grep -q '"success": *true'; then
      ok "on-start script terpasang: $cmd"
      say "berlaku mulai instance dinyalakan BERIKUTNYA (pause/resume interruptible)."
    else
      warn "API Vast menolak: $resp"
      echo "Tempel manual di halaman instance Vast (On-start script):"
      echo "  $cmd"
    fi
  else
    echo "On-start script untuk instance Vast (mode interruptible):"
    echo "  $cmd"
    echo
    echo "Tempel di pengaturan instance Vast + pilih launch mode 'restart on instance start',"
    echo "atau pasang otomatis:  ./deploy.sh onstart set   (butuh VAST_API_KEY + VAST_INSTANCE_ID di .env)"
  fi
}

cmd_all() {
  require_gpu_model
  cmd_check
  [ -f "$CKPT_ZIP" ] || die "checkpoint_latest.zip belum ada di $ROOT (taruh dulu, lalu ulangi ./deploy.sh all)"
  cmd_package
  cmd_source
  cmd_corpus
  cmd_checkpoint
  cmd_build
  cmd_start
  echo; say "GPU interruptible (bid)? pasang on-start script:  ./deploy.sh onstart set"
}

cmd_pull() {
  need rsync; need python3
  mkdir -p "$SPEED_DIR/raw"
  rsync_to "-a" "$TARGET:$REMOTE_BASE/speed_monitoring_gpu/" "$SPEED_DIR/raw/" || die "gagal menarik data"
  python3 "$ROOT/tools/summarize_speed.py" "$SPEED_DIR"
}

cmd_gpus() {
  need python3
  case "${1:-}" in
    refresh) python3 "$GPU_INFO" refresh ;;
    show)    shift; [ $# -ge 1 ] || die "pakai: ./deploy.sh gpus show \"RTX 5090\""; python3 "$GPU_INFO" show "$*" ;;
    *)       python3 "$GPU_INFO" list "$*"; [ -n "$GPU_MODEL" ] && echo "GPU_MODEL sekarang di .env: \"$GPU_MODEL\"${GPU_ARCH_HINT:+  -> sm_$GPU_ARCH_HINT}" ;;
  esac
}

cmd_help() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

case "${1:-help}" in
  gpus)       shift; cmd_gpus "$@" ;;
  check)      cmd_check ;;
  key)        cmd_key ;;
  rent)       shift; cmd_rent "$@" ;;
  auth)       cmd_auth ;;
  token)      cmd_token ;;
  vast)       cmd_vast ;;
  package)    cmd_package ;;
  source)     cmd_source ;;
  corpus)     cmd_corpus ;;
  postdata)   cmd_postdata ;;
  checkpoint) cmd_checkpoint ;;
  build)      cmd_build ;;
  start)      cmd_start ;;
  onstart)    shift; cmd_onstart "${1:-}" ;;
  all)        cmd_all ;;
  monitor)    ensure_connection; R_tty live ;;
  panel)      ensure_connection; R_tty panel ;;
  status)     ensure_connection; R status ;;
  log)        ensure_connection; rssh_tty "tail -n 40 -f '$REMOTE_BASE/speed_monitoring_gpu/train.log'" ;;
  attach)     ensure_connection; rssh_tty "tmux attach -t train" ;;
  pull)       cmd_pull ;;
  drive)      ensure_connection; R drive ;;
  stop)       ensure_connection; R_tty stop "${2:-}" ;;
  ssh)        ensure_connection; rssh_tty ;;
  help|-h|--help) cmd_help ;;
  *) echo "perintah tidak dikenal: $1"; cmd_help; exit 2 ;;
esac
