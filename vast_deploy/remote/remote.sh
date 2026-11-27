#!/usr/bin/env bash
# =============================================================================
#  remote.sh  -  dijalankan DI INSTANCE VAST (di-upload otomatis oleh deploy.sh)
#  Anda tidak perlu memanggilnya langsung, tapi boleh, mis:
#      bash /workspace/remote.sh status
#      bash /workspace/remote.sh live
# =============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/deploy.env" ] || { echo "ERROR: $HERE/deploy.env tidak ada (jalankan ./deploy.sh dari Mac)"; exit 1; }
# shellcheck disable=SC1091
. "$HERE/deploy.env"

PROJ="$REMOTE_BASE/$PROJECT_DIR_NAME"
MON="$REMOTE_BASE/speed_monitoring_gpu"
LOG="$MON/train.log"
SMI="$MON/nvidia_smi.csv"
S_TRAIN=train
S_MON=gpumon

die()  { echo "ERROR: $*" >&2; exit 1; }
log()  { echo "[remote] $*"; }
warn() { echo "[remote] PERINGATAN: $*" >&2; }

# ─── Deteksi GPU / arsitektur ────────────────────────────────────────────────
gpu_names() { nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null; }

arch_from_name() {   # cadangan kalau driver lama tidak punya field compute_cap
  case "$1" in
    *B200*|*B100*|*GB200*)                         echo 100 ;;
    *5090*|*5080*|*5070*|*"RTX PRO 6000"*)         echo 120 ;;
    *H100*|*H200*|*H800*|*GH200*)                  echo 90 ;;
    *A100*|*A30*)                                  echo 80 ;;
    *4090*|*4080*|*4070*|*"L4"*|*L40*|*"6000 Ada"*) echo 89 ;;
    *3090*|*3080*|*3070*|*A10*|*A40*|*A6000*|*A5000*|*A4000*) echo 86 ;;
    *T4*|*2080*|*2070*)                            echo 75 ;;
    *V100*)                                        echo 70 ;;
    *)                                             echo "" ;;
  esac
}

actual_arch() {   # arsitektur dari GPU yang benar-benar ada di instance
  local caps out="" a n
  caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
         | tr -d ' .\r' | grep -E '^[0-9]+$' | sort -u | paste -sd';' -)
  if [ -n "$caps" ]; then echo "$caps"; return 0; fi
  while IFS= read -r n; do a=$(arch_from_name "$n"); [ -n "$a" ] && out="$out;$a"; done < <(gpu_names)
  echo "${out#;}" | tr ';' '\n' | sort -u | paste -sd';' -
}

detect_arch() {
  local actual; actual=$(actual_arch)
  if [ "${CUDA_ARCH:-auto}" != "auto" ]; then echo "$CUDA_ARCH"; return 0; fi   # dipaksa user
  if [ -n "${GPU_ARCH_HINT:-}" ] && [ -n "$actual" ] && [ "$GPU_ARCH_HINT" != "$actual" ]; then
    {
      echo "ERROR: arsitektur tidak cocok."
      echo "  GPU_MODEL di .env  : ${GPU_MODEL:-?}  -> sm_$GPU_ARCH_HINT (dari tabel Vast)"
      echo "  GPU asli instance  : $(gpu_names | paste -sd',' -)  -> sm_$actual (dari nvidia-smi)"
      echo "  Perbaiki GPU_MODEL di .env agar sama dengan GPU yang Anda sewa, atau isi CUDA_ARCH=$actual untuk memaksa."
    } >&2
    return 1
  fi
  [ -n "$actual" ] && { echo "$actual"; return 0; }
  [ -n "${GPU_ARCH_HINT:-}" ] && { echo "$GPU_ARCH_HINT"; return 0; }
  echo ""
}

num_gpus() {
  if [ "${NUM_GPUS:-auto}" = "auto" ]; then nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
  else echo "$NUM_GPUS"; fi
}

check_nvcc_arch() {
  command -v nvcc >/dev/null 2>&1 || die "nvcc tidak ditemukan. Pakai image Vast 'cuda ... devel', atau: apt-get install -y cuda-toolkit"
  local supported a
  supported=$(nvcc --list-gpu-code 2>/dev/null)
  for a in $(echo "$1" | tr ';' ' '); do
    if ! echo "$supported" | grep -qx "sm_$a"; then
      local lo; lo=$(echo "$supported" | sed -n 's/^sm_//p' | sort -n | head -1)
      if [ -n "$lo" ] && [ "$a" -lt "$lo" ]; then
        die "nvcc ($(nvcc --version | tail -1)) sudah TIDAK mendukung sm_$a (terendah: sm_$lo). Pakai image dengan CUDA 12.x untuk GPU lama ini."
      else
        die "nvcc ($(nvcc --version | tail -1)) belum mendukung sm_$a. Pakai image dengan CUDA lebih baru (mis. 5090 butuh >= 12.8)."
      fi
    fi
  done
}

# ─── Status proses ───────────────────────────────────────────────────────────
train_running()     { pgrep -f "build/train_(pretrain|sft)_cuda" >/dev/null 2>&1; }
ckpt_in_use()       { pgrep -af "train_(pretrain|sft)_cuda" 2>/dev/null | grep -q -- "--checkpoint_dir $CKPT_DIR"; }
proj_binary_in_use(){   # apakah ada trainer yang dijalankan dari binary di $PROJ?
  local pid exe
  for pid in $(pgrep -f "build/train_(pretrain|sft)_cuda" 2>/dev/null); do
    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null)"
    if [ "$exe" = "$(readlink -f "$PROJ/build/train_pretrain_cuda" 2>/dev/null)" ] || \
       [ "$exe" = "$(readlink -f "$PROJ/build/train_sft_cuda" 2>/dev/null)" ]; then
      return 0
    fi
  done
  return 1
}

promote_final() {   # hasil stop graceful tersimpan sebagai *_final; resume butuh nama biasa
  local f
  for f in ckpt_pretrain ckpt_optim; do
    if [ -f "$CKPT_DIR/${f}_final.safetensors" ]; then
      mv -f "$CKPT_DIR/${f}_final.safetensors" "$CKPT_DIR/${f}.safetensors" && log "promosi ${f}_final -> ${f}"
    fi
  done
  rm -rf "$CKPT_DIR/eval_tmp"
}

# ─── Pemulihan checkpoint dari Google Drive (mode interruptible) ────────────
# Container Vast dibuang saat instance berhenti: hanya /workspace yang persist.
# Kalau checkpoint lokal hilang (instance di-recreate, disk baru), unduh dulu
# salinan terakhir dari Drive supaya training resume, bukan mulai dari nol.
recover_from_drive() {
  [ "${DRIVE_RECOVER:-1}" = "1" ] || { log "pemulihan Drive dimatikan (DRIVE_RECOVER=0)"; return 1; }
  [ -f "$PROJ/scripts/download_gdrive.py" ] || { warn "download_gdrive.py tidak ada - pemulihan Drive dilewati (deploy ulang source)"; return 1; }
  log "checkpoint lokal tidak ada di $CKPT_DIR -> pemulihan dari Google Drive..."
  ( cd "$PROJ" && CKPT_UPLOAD_TMP="$REMOTE_BASE" python3 scripts/download_gdrive.py --ckpt_dir "$CKPT_DIR" --stage "${STAGE:-pretrain}" ) \
    && log "checkpoint dipulihkan dari Drive" \
    || { warn "pemulihan dari Drive GAGAL - lihat pesan [DOWNLOAD] di atas"; return 1; }
}

# ─── Perintah ────────────────────────────────────────────────────────────────
cmd_check() {
  echo "== GPU =="
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv,noheader 2>&1
  echo "arch GPU asli   : sm_$(actual_arch)  | jumlah GPU : $(num_gpus)"
  if [ -n "${GPU_MODEL:-}" ]; then
    if [ -n "${GPU_ARCH_HINT:-}" ] && [ "$GPU_ARCH_HINT" = "$(actual_arch)" ]; then echo "GPU_MODEL       : $GPU_MODEL -> sm_$GPU_ARCH_HINT  (cocok dengan GPU asli)"
    elif [ -n "${GPU_ARCH_HINT:-}" ]; then echo "GPU_MODEL       : $GPU_MODEL -> sm_$GPU_ARCH_HINT  <-- TIDAK COCOK dengan GPU asli! perbaiki di .env"
    else echo "GPU_MODEL       : $GPU_MODEL (tidak ada di tabel)"; fi
  fi
  if command -v nvcc >/dev/null 2>&1; then echo "nvcc            : $(nvcc --version | tail -1)"; else echo "nvcc            : TIDAK ADA"; fi
  echo "== Alat =="
  local t; for t in unzip rsync tmux python3 cmake g++ pip; do
    command -v "$t" >/dev/null 2>&1 && echo "  ok   $t" || echo "  HILANG $t"
  done
  python3 -c "import googleapiclient" 2>/dev/null && echo "  ok   google api (upload Drive)" || echo "  HILANG google api (akan dipasang saat 'source')"
  ldconfig -p 2>/dev/null | grep -q libnccl && echo "  ok   nccl (untuk multi-GPU)" || echo "  -    nccl tidak ada (hanya perlu untuk >1 GPU)"
  echo "== Disk / RAM =="
  df -h / /dev/shm 2>/dev/null | awk 'NR==1 || !seen[$6]++'
  free -g | awk 'NR==2 {print "RAM total " $2 " GB, bebas " $7 " GB"}'
  echo "== Training =="
  if train_running; then echo "BERJALAN"; else echo "tidak berjalan"; fi
  [ -f "$CKPT_DIR/train_state.txt" ] && head -2 "$CKPT_DIR/train_state.txt" | tr '\n' ' ' && echo
}

cmd_setup() {
  local t missing=""
  for t in unzip rsync tmux python3 cmake g++ pip; do command -v "$t" >/dev/null 2>&1 || missing="$missing $t"; done
  if [ -n "$missing" ]; then
    log "memasang alat yang hilang:$missing"
    apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq unzip rsync tmux cmake g++ python3-pip >/dev/null 2>&1 || warn "apt-get gagal; pasang manual:$missing"
  fi
  if ! python3 -c "import googleapiclient, google.oauth2" 2>/dev/null; then
    log "memasang library Google Drive..."
    pip install -q --break-system-packages google-api-python-client google-auth-oauthlib 2>/dev/null \
      || pip install -q google-api-python-client google-auth-oauthlib 2>&1 | tail -2
  fi
  python3 -c "import googleapiclient, google.oauth2; print('[remote] library Google Drive OK')" || warn "library Google Drive gagal dipasang"
}

cmd_extract() {   # dipanggil deploy.sh setelah source_code.zip tiba
  [ -f "$REMOTE_BASE/source_code.zip" ] || die "source_code.zip tidak ada di $REMOTE_BASE"
  proj_binary_in_use && die "training memakai kode di $PROJ. Hentikan dulu (./deploy.sh stop) atau ganti PROJECT_DIR_NAME."
  command -v unzip >/dev/null 2>&1 || cmd_setup
  unzip -o -q "$REMOTE_BASE/source_code.zip" -d "$REMOTE_BASE" || die "unzip gagal"
  rm -f "$REMOTE_BASE/source_code.zip"
  # unzip mempertahankan tanggal file asli (bisa lebih TUA dari binary lama) -> make menganggap binary
  # masih mutakhir dan tidak mengompilasi ulang, sehingga patch baru tidak aktif. Paksa kompilasi ulang.
  find "$PROJ" -type f \( -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.h' -o -name '*.txt' \) -exec touch {} +
  [ -f "$PROJ/CMakeLists.txt" ] || die "struktur zip salah: $PROJ/CMakeLists.txt tidak ada"
  log "source terpasang di $PROJ ($(find "$PROJ" -type f | wc -l | tr -d ' ') file)"
  cmd_setup
}

cmd_build() {
  [ -f "$PROJ/CMakeLists.txt" ] || die "source belum ada. Jalankan ./deploy.sh source"
  proj_binary_in_use && die "training sedang memakai binary di $PROJ; build ditolak supaya tidak merusak run."
  local arch n ddp jobs t0
  arch=$(detect_arch) || exit 1; n=$(num_gpus)
  [ -n "$arch" ] || die "arsitektur GPU tidak terdeteksi. Isi CUDA_ARCH di .env (mis. 89 untuk RTX 4090)."
  check_nvcc_arch "$arch"
  # lib cuDNN/NCCL: image pytorch menyimpan di pip wheel site-packages/nvidia/*/lib dengan
  # SONAME berversi (libcudnn.so.9, libnccl.so.2). Symlink SEMUA file dgn NAMA ASLINYA ke
  # /usr/local/lib — kalau hanya 'libcudnn.so', linker build puas tapi LOADER runtime gagal (rc=127).
  local _nv
  for _nv in cudnn nccl; do
    if ! ldconfig -p 2>/dev/null | grep -qE "lib${_nv}\.so\.[0-9]"; then
      local _ld; _ld=$(dirname "$(find /opt/conda -name "lib${_nv}.so*" 2>/dev/null | head -1)" 2>/dev/null)
      if [ -n "$_ld" ] && [ -d "$_ld" ]; then
        for so in "$_ld"/lib${_nv}*.so*; do [ -e "$so" ] && ln -sf "$(readlink -f "$so")" /usr/local/lib/; done
        ldconfig 2>/dev/null; log "lib${_nv}: $_ld/*.so* -> /usr/local/lib (nama soname asli)"
      fi
    fi
    # Symlink dev polos (libX.so): find_library CMake mencari nama TANPA versi. Di image yang
    # hanya punya soname berversi (pip wheel nvidia), configure gagal tanpa symlink ini
    # (kasus A100 SXM4, image CUDA 12.4.131). Linker hanya menyimpan SONAME berversi di
    # binary -> loader tetap memakai soname di atas, rc=127 tidak terulang.
    if [ ! -e "/usr/local/lib/lib${_nv}.so" ]; then
      local _ld2 _main
      _ld2=$(dirname "$(find /opt/conda -name "lib${_nv}.so*" 2>/dev/null | head -1)" 2>/dev/null)
      _main=$(ls "$_ld2"/lib${_nv}.so.[0-9]* 2>/dev/null | sort -V | tail -1)
      [ -n "$_main" ] && ln -sf "$(readlink -f "$_main")" "/usr/local/lib/lib${_nv}.so" \
        && log "lib${_nv}: dev polos /usr/local/lib/lib${_nv}.so -> $(basename "$_main")"
    fi
  done
  # header cuDNN/NCCL (pip wheel nvidia: site-packages/nvidia/*/include) tak terlihat nvcc/gcc
  # -> symlink ke /usr/local/include (path default gcc) DAN /usr/local/cuda/include (path default nvcc)
  if [ ! -f /usr/local/include/cudnn.h ]; then
    local ch; ch=$(find /opt/conda /usr/local /usr -name 'cudnn.h' 2>/dev/null | head -1)
    if [ -n "$ch" ]; then
      mkdir -p /usr/local/include /usr/local/cuda/include
      for h in "$(dirname "$ch")"/cudnn*.h; do
        [ -f "$h" ] && ln -sf "$(readlink -f "$h")" /usr/local/include/ && ln -sf "$(readlink -f "$h")" /usr/local/cuda/include/
      done
      log "cuDNN header: $(dirname "$ch") -> /usr/local/include + /usr/local/cuda/include"
    fi
  fi
  if [ ! -f /usr/local/include/nccl.h ]; then
    local nh; nh=$(find /opt/conda /usr/local /usr -name 'nccl.h' 2>/dev/null | head -1)
    if [ -n "$nh" ]; then
      ln -sf "$(readlink -f "$nh")" /usr/local/include/nccl.h
      ln -sf "$(readlink -f "$nh")" /usr/local/cuda/include/nccl.h 2>/dev/null
    fi
  fi
  ddp=OFF
  if [ "$n" -gt 1 ]; then
    if ldconfig -p 2>/dev/null | grep -q libnccl; then ddp=ON
    else warn "$n GPU terdeteksi tapi NCCL tidak ada -> build satu-GPU saja (apt-get install -y libnccl-dev untuk multi-GPU)"; fi
  fi
  jobs=$(nproc); [ "$jobs" -gt 32 ] && jobs=32
  log "GPU: $(gpu_names | paste -sd',' -)"
  log "build: arch=sm_$arch  DDP=$ddp  GPU=$n  -j$jobs"
  t0=$(date +%s)
  mkdir -p "$PROJ/build" && cd "$PROJ/build" || die "tidak bisa masuk $PROJ/build"
  { cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$arch" -DUSE_DDP=$ddp 2>&1 | tail -15 && \
    cmake --build . -j"$jobs" 2>&1 | tail -40; } > "$REMOTE_BASE/build.log" 2>&1
  if [ ! -x train_pretrain_cuda ] || [ ! -x train_sft_cuda ]; then
    tail -40 "$REMOTE_BASE/build.log"; die "build gagal (log lengkap: $REMOTE_BASE/build.log)"
  fi
  ls -lh train_pretrain_cuda train_sft_cuda
  if [ "${SKIP_GRADCHECK:-0}" != "1" ]; then
    log "grad_check (kebenaran kernel; wajib lulus)..."
    cmake --build . --target grad_check -j"$jobs" > "$REMOTE_BASE/gradcheck_build.log" 2>&1 || { tail -20 "$REMOTE_BASE/gradcheck_build.log"; die "build grad_check gagal"; }
    ./grad_check > "$REMOTE_BASE/gradcheck.log" 2>&1
    grep -q "ALL CHECKS PASSED" "$REMOTE_BASE/gradcheck.log" || { tail -25 "$REMOTE_BASE/gradcheck.log"; die "grad_check GAGAL - jangan training dengan build ini"; }
    log "grad_check: ALL CHECKS PASSED"
  else
    warn "grad_check dilewati (SKIP_GRADCHECK=1)"
  fi
  log "build binary inference (CPU)..."
  mkdir -p "$PROJ/build_infer" && cd "$PROJ/build_infer" || die "build_infer"
  { cmake ../infer_cpp -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3 && make -j"$jobs" 2>&1 | tail -3; } > "$REMOTE_BASE/build_infer.log" 2>&1
  [ -x infer ] || { tail -20 "$REMOTE_BASE/build_infer.log"; die "build infer gagal"; }
  printf 'arch=%s\nddp=%s\ngpus=%s\n' "$arch" "$ddp" "$n" > "$PROJ/.build_info"
  log "SELESAI build dalam $(( $(date +%s) - t0 )) detik"
}

cmd_start() {
  [ -x "$PROJ/build/train_pretrain_cuda" ] || die "belum di-build. Jalankan ./deploy.sh build"
  [ -x "$PROJ/build_infer/infer" ] || die "binary infer belum ada. Jalankan ./deploy.sh build"
  # Dipanggil juga oleh on-start script Vast saat instance resume (interruptible):
  # jangan error kalau training ternyata sudah jalan (pemanggilan dobel).
  train_running && { log "training sudah berjalan - start diabaikan (pantau: ./deploy.sh monitor)"; exit 0; }
  # Kunci anti-dobel: on-start script dan `deploy.sh start` manual bisa sempat
  # berjalan bersamaan; hanya satu yang boleh meluncurkan supervisor.
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$REMOTE_BASE/.train_start.lock"
    flock -n 9 || { log "start lain sedang berjalan - diabaikan"; exit 0; }
  fi
  tmux has-session -t "$S_TRAIN" 2>/dev/null && tmux kill-session -t "$S_TRAIN"
  local cur_stage="${STAGE:-pretrain}"
  if [[ ",$cur_stage," == *",pretrain,"* ]]; then
    [ -d "$CORPUS_REMOTE_DIR" ] || die "korpus tidak ada di $CORPUS_REMOTE_DIR (./deploy.sh corpus)"
  fi
  if [[ ",$cur_stage," == *",sft,"* ]]; then
    [ -d "${SFT_REMOTE_DIR:-}" ] || die "data SFT tidak ada di ${SFT_REMOTE_DIR:-?} (jalankan ./deploy.sh postdata)"
  fi
  if [[ ",$cur_stage," == *",dpo,"* ]]; then
    [ -d "${DPO_REMOTE_DIR:-}" ] || die "data DPO tidak ada di ${DPO_REMOTE_DIR:-?} (jalankan ./deploy.sh postdata)"
  fi
  mkdir -p "$CKPT_DIR" "$MON"
  promote_final
  rm -f "$REMOTE_BASE/.train_stop"
  if [[ "$cur_stage" == pretrain* ]]; then
    if [ ! -f "$CKPT_DIR/train_state.txt" ] || [ ! -f "$CKPT_DIR/ckpt_pretrain.safetensors" ]; then
      recover_from_drive
    fi
    if [ ! -f "$CKPT_DIR/train_state.txt" ] || [ ! -f "$CKPT_DIR/ckpt_pretrain.safetensors" ]; then
      [ "${ALLOW_FRESH:-0}" = "1" ] || die "tidak ada checkpoint di $CKPT_DIR. Jalankan ./deploy.sh checkpoint (atau ALLOW_FRESH=1 untuk mulai dari nol)."
      warn "TIDAK ada checkpoint - training mulai dari NOL"
    fi
  fi

  local n ddp info_ddp ddp_args="" nccl_line="" up sv
  n=$(num_gpus); info_ddp=OFF
  [ -f "$PROJ/.build_info" ] && info_ddp=$(grep '^ddp=' "$PROJ/.build_info" | cut -d= -f2)
  if [ "$n" -gt 1 ] && [ "$info_ddp" = "ON" ]; then
    ddp_args="--ddp --num_gpus $n"
    [ -n "${NCCL_P2P_DISABLE:-}" ] && nccl_line="export NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
  else
    n=1
    [ "$(num_gpus)" -gt 1 ] && warn "build bukan DDP -> hanya 1 dari $(num_gpus) GPU dipakai (build ulang dengan NCCL untuk multi-GPU)"
  fi
  up="${UPLOAD_EVERY:-0}"; sv="${SAVE_EVERY:-400}"
  # Trainer hanya upload saat step kelipatan SAVE_EVERY *dan* UPLOAD_EVERY -> bulatkan ke atas.
  if [ "$up" -gt 0 ] && [ $((up % sv)) -ne 0 ]; then
    up=$(( (up + sv - 1) / sv * sv )); log "UPLOAD_EVERY dibulatkan ke $up (kelipatan SAVE_EVERY=$sv)"
  fi

  cat > "$REMOTE_BASE/gpu_monitor.sh" <<'EOF'
#!/usr/bin/env bash
OUT="$1"; mkdir -p "$(dirname "$OUT")"
[ -s "$OUT" ] || echo "ts,util_gpu,util_mem,mem_used_mib,mem_total_mib,power_w,sm_clock_mhz,mem_clock_mhz,temp_c,pstate,throttle_reasons" > "$OUT"
while true; do
  nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,temperature.gpu,pstate,clocks_throttle_reasons.active --format=csv,noheader,nounits | head -1 >> "$OUT"
  sleep 5
done
EOF
  chmod +x "$REMOTE_BASE/gpu_monitor.sh"
  tmux has-session -t "$S_MON" 2>/dev/null || tmux new-session -d -s "$S_MON" "bash '$REMOTE_BASE/gpu_monitor.sh' '$SMI'"

  local train_cmd
  if [ "$cur_stage" = "rl" ]; then
    train_cmd="RL_DIR='${RL_REMOTE_DIR:-$PROJ/rl_data/ceiling_enja}' ARCH_CFG='$PROJ/${ARCH_CONFIG:-config_500m.json}' VOCAB_DIR='$PROJ/$VOCAB_DIR' BPE_BIN='$PROJ/$BPE_FILE' bash scripts/run_rl.sh ${ddp_args:+--ddp} --checkpoint_dir '$CKPT_DIR'"
  else
    train_cmd="python3 scripts/train_pipeline.py --auto_continue --work_dir '$PROJ' --stages '$cur_stage' --arch_config '${ARCH_CONFIG:-config_500m.json}' --checkpoint_dir '$CKPT_DIR' --data_dir '$CORPUS_REMOTE_DIR' --vocab_dir '$VOCAB_DIR' --bpe '$BPE_FILE' --batch_size $BATCH_SIZE --seq $SEQ --max_tokens $MAX_TOKENS --lr $LR --grad_clip $GRAD_CLIP --loss_scale $LOSS_SCALE --safety $SAFETY --save_every $SAVE_EVERY --log_every $LOG_EVERY --upload_every $up --eval_every_tokens $EVAL_EVERY_TOKENS --eval_infer_bin ./build_infer/infer --sft_dir '${SFT_REMOTE_DIR:-sft_data/sft_mixed_enja}' --sft_lr ${SFT_LR:-2e-5} --sft_epochs ${SFT_EPOCHS:-2} --sft_batch_size ${SFT_BATCH_SIZE:-8} --sft_max_seq ${SFT_MAX_SEQ:-1024} --dpo_dir '${DPO_REMOTE_DIR:-sft_data/dpo_ceiling_enja}' --dpo_lr ${DPO_LR:-5e-6} --dpo_epochs ${DPO_EPOCHS:-1} --dpo_batch_size ${DPO_BATCH_SIZE:-4} --dpo_max_seq ${DPO_MAX_SEQ:-1024} --dpo_beta ${DPO_BETA:-0.1} --dpo_max_margin ${DPO_MAX_MARGIN:-4.0} $ddp_args --resume"
  fi

  cat > "$REMOTE_BASE/run_train.sh" <<EOF
#!/usr/bin/env bash
cd "$PROJ" || exit 1
export CKPT_UPLOAD_TMP="$REMOTE_BASE"
[ -n "${TW_W_JA:-}" ]    && export TW_W_JA="${TW_W_JA:-}"
[ -n "${TW_W_EN:-}" ]    && export TW_W_EN="${TW_W_EN:-}"
[ -n "${TW_W_CODE:-}" ]  && export TW_W_CODE="${TW_W_CODE:-}"
[ -n "${TW_W_MATH:-}" ]  && export TW_W_MATH="${TW_W_MATH:-}"
[ -n "${TW_W_MIXED:-}" ] && export TW_W_MIXED="${TW_W_MIXED:-}"
$nccl_line
set -o pipefail
attempt=0; max=${RESTART_MAX:-8}
while true; do
  rc=0
  $train_cmd 2>&1 | tee -a "$LOG"
  rc=\$?
  if [ "\$rc" -eq 0 ]; then
    echo "=== FINISH \$(date '+%Y/%m/%d %H:%M:%S') pipeline selesai (rc=0) ===" >> "$LOG"
    break
  fi
  if [ -f "$REMOTE_BASE/.train_stop" ]; then
    echo "=== STOPPED \$(date '+%Y/%m/%d %H:%M:%S') dihentikan manual (rc=\$rc) ===" >> "$LOG"
    break
  fi
  attempt=\$((attempt+1))
  if [ "\$attempt" -gt "\$max" ]; then
    echo "=== GIVEUP \$(date '+%Y/%m/%d %H:%M:%S') berhenti setelah \$max restart (rc terakhir=\$rc) ===" >> "$LOG"
    break
  fi
  backoff=\$(( 30 * (2 ** (attempt-1)) )); [ "\$backoff" -gt 900 ] && backoff=900
  # hasil stop graceful ditandai _final -> kembalikan ke nama biasa untuk resume
  for f in ckpt_pretrain ckpt_optim; do
    [ -f "$CKPT_DIR/\${f}_final.safetensors" ] && mv -f "$CKPT_DIR/\${f}_final.safetensors" "$CKPT_DIR/\${f}.safetensors"
  done
  rm -rf "$CKPT_DIR/eval_tmp"
  echo "=== RESTART \$(date '+%Y/%m/%d %H:%M:%S') attempt=\$attempt/\$max setelah rc=\$rc, tunggu \${backoff}s ===" >> "$LOG"
  sleep "\$backoff"
done
EOF
  touch "$LOG"; local n0; n0=$(wc -l < "$LOG")
  echo "=== START $(date '+%Y/%m/%d %H:%M:%S') batch=$BATCH_SIZE seq=$SEQ gpus=$n lr=$LR upload_every=$up ===" >> "$LOG"
  runner_alive() { tmux has-session -t "$S_TRAIN" 2>/dev/null; }
  if command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s "$S_TRAIN" "bash '$REMOTE_BASE/run_train.sh'"
    log "training dimulai di tmux '$S_TRAIN' (batch=$BATCH_SIZE x seq=$SEQ x $n GPU = $((BATCH_SIZE*SEQ*n)) token/langkah)"
  else
    # On-start resume bisa terjadi di image tanpa tmux: tetap jalan via nohup.
    nohup bash "$REMOTE_BASE/run_train.sh" >> "$LOG" 2>&1 &
    runner_alive() { pgrep -f "bash $REMOTE_BASE/run_train.sh" >/dev/null 2>&1; }
    log "training dimulai di latar belakang nohup (tmux tidak tersedia)"
  fi
  local i
  # Tunggu sampai ada baris [S1] BARU. Proses trainer baru muncul beberapa detik setelah supervisor dibuat,
  # jadi "gagal" hanya dinyatakan kalau supervisor sudah hilang (bukan sekadar trainer belum terlihat).
  for i in $(seq 1 120); do
    tail -n +$((n0+1)) "$LOG" | grep -q '^\[S1\]' && break
    runner_alive || break
    sleep 2
  done
  tail -n +$((n0+1)) "$LOG" | grep '^\[S1\]\|\[RESUME\] Restored\|\[DOWNLOAD\] OK\|ERROR\|FATAL' | tail -3 | cut -c1-230
  if ! runner_alive; then warn "supervisor training berhenti setelah start - lihat: tail -50 $LOG"
  elif ! tail -n +$((n0+1)) "$LOG" | grep -q '^\[S1\]'; then warn "belum ada baris [S1] setelah 4 menit; periksa dengan ./deploy.sh monitor"; fi
}

cmd_stop() {
  local fast="${1:-}"
  # Flag dulu: supervisor run_train.sh memeriksa file ini sebelum auto-restart,
  # supaya training yang dihentikan manual tidak menyala kembali sendiri.
  touch "$REMOTE_BASE/.train_stop" 2>/dev/null
  if ! train_running; then log "tidak ada training yang berjalan"; promote_final; rm -f "$REMOTE_BASE/.train_stop"; return 0; fi
  log "mengirim SIGINT (graceful: trainer menyimpan checkpoint akhir)..."
  pkill -INT -f "build/train_(pretrain|sft)_cuda"
  local i; for i in $(seq 1 180); do train_running || break; sleep 1; done
  train_running && { warn "trainer belum berhenti setelah 180 dtk; paksa SIGTERM"; pkill -TERM -f "build/train_(pretrain|sft)_cuda"; sleep 5; }
  if pgrep -f upload_gdrive.py >/dev/null 2>&1; then
    if [ "$fast" = "--fast" ]; then log "upload Drive akhir dibatalkan (--fast)"; pkill -f upload_gdrive.py
    else
      log "menunggu upload Drive akhir (maks ${FINAL_UPLOAD_WAIT:-1500} dtk; Ctrl+C = batal tunggu; --fast = lewati)..."
      local w=0; while pgrep -f upload_gdrive.py >/dev/null 2>&1 && [ "$w" -lt "${FINAL_UPLOAD_WAIT:-1500}" ]; do sleep 5; w=$((w+5)); done
      pgrep -f upload_gdrive.py >/dev/null 2>&1 && { warn "upload Drive belum selesai; dihentikan"; pkill -f upload_gdrive.py; }
    fi
  fi
  tmux kill-session -t "$S_TRAIN" 2>/dev/null
  rm -rf "$REMOTE_BASE"/ckpt_zip_* "$REMOTE_BASE"/ckpt_upload_* "$CKPT_DIR"/.ckpt_snap_*
  promote_final
  rm -f "$REMOTE_BASE/.train_stop"
  [ -f "$CKPT_DIR/train_state.txt" ] && { log "checkpoint terakhir:"; head -2 "$CKPT_DIR/train_state.txt"; }
}

# ─── On-start script Vast (interruptible) ────────────────────────────────────
# Dipanggil otomatis oleh Vast SETIAP KALI instance dinyalakan (resume setelah
# kalah bid). Layer sistem container (pip/apt) hilang saat instance berhenti,
# jadi dipasang ulang dulu; checkpoint di /workspace tetap ada -> cukup resume.
# Kalau /workspace kosong (instance baru), checkpoint ditarik dari Google Drive.
cmd_onstart() {
  echo "[onstart] $(date '+%Y/%m/%d %H:%M:%S') instance hidup - menyiapkan training..."
  local i
  for i in $(seq 1 60); do   # driver GPU kadang butuh beberapa detik
    nvidia-smi >/dev/null 2>&1 && break
    if [ "$i" -eq 60 ]; then echo "[onstart] PERINGATAN: nvidia-smi belum siap setelah 120s"; fi
    sleep 2
  done
  cmd_setup   # pasang ulang tmux/pip/google-api yang hilang bersama container
  cmd_start
  echo "[onstart] selesai. Pantau dari Mac: ./deploy.sh monitor | ./deploy.sh panel"
}

print_status() {
  local last re
  last=$(grep '^\[S1\]' "$LOG" 2>/dev/null | tail -1)
  echo "================ $(date '+%H:%M:%S') | training: $(train_running && echo BERJALAN || echo BERHENTI) | tmux: $(tmux ls 2>/dev/null | cut -d: -f1 | paste -sd',' -)"
  re='step=([0-9]+) +seq=([0-9]+) \[[^]]*\] ([0-9.]+%) \| tok=([^ ]+) \| loss=([0-9.]+) ema=([0-9.]+) gnorm=([0-9.]+) decay=[0-9.]+ lr=([0-9.e+-]+) ([0-9.]+K) tok/s ([0-9.]+)s/step elapsed=([^ ]+) ETA ([^ ]+) \| RAM [0-9.]+GB VRAM ([0-9.]+)GB GPU\(proc\) ([0-9]+)%'
  if [[ "$last" =~ $re ]]; then
    echo " step ${BASH_REMATCH[1]} | token ${BASH_REMATCH[4]} (${BASH_REMATCH[3]}) | seq ${BASH_REMATCH[2]}"
    echo " loss ${BASH_REMATCH[5]} (ema ${BASH_REMATCH[6]}) | gnorm ${BASH_REMATCH[7]} | lr ${BASH_REMATCH[8]}"
    echo " KECEPATAN ${BASH_REMATCH[9]} tok/s (${BASH_REMATCH[10]} dtk/langkah) | sisa ${BASH_REMATCH[12]} | berjalan ${BASH_REMATCH[11]}"
    echo " VRAM ${BASH_REMATCH[13]} GB | GPU util ${BASH_REMATCH[14]}%"
  else
    echo " (belum ada baris [S1] di log)"
  fi
  if [ -s "$SMI" ]; then
    tail -n 1 "$SMI" | awk -F, '{printf " nvidia-smi: util %s%% | daya %.0f W | SM %s MHz | suhu %s C | VRAM %s/%s MiB\n",$2,$6,$7,$9,$4,$5}'
  fi
  [ -f "$CKPT_DIR/train_state.txt" ] && printf ' checkpoint di disk: %s\n' "$(sed -n '1,2p' "$CKPT_DIR/train_state.txt" | tr '\n' ' ')"
  grep -E '\[UPLOAD\] (Done|Failed|Triggering|Final)|OVERFLOW|NAN/INF|\[SKIP\]|\[OOM\]|FATAL' "$LOG" 2>/dev/null | tail -3 | cut -c1-150 | sed 's/^/ log: /'
  df -h "$CKPT_DIR" / 2>/dev/null | awk 'NR>1 && !s[$6]++ {printf " disk %s: %s terpakai, sisa %s\n",$6,$5,$4}'
}
cmd_status() { print_status; }
cmd_panel()  { while true; do printf '\033[H\033[2J'; print_status; echo; echo " (Ctrl+C untuk keluar - training TIDAK ikut berhenti)"; sleep 2; done; }

# Log penuh gaya Kaggle: banner pipeline, info CUDA, dataset, baris [S1] ... mengalir real time.
# Mulai dari penanda START/RESTART terakhir (atau MONITOR_TAIL baris terakhir kalau tidak ada).
cmd_live() {
  [ -f "$LOG" ] || die "belum ada log di $LOG (training belum pernah dimulai?)"
  local from n_tail="${MONITOR_TAIL:-60}" start total
  total=$(wc -l < "$LOG")
  start=$(grep -n '^=== ' "$LOG" | tail -1 | cut -d: -f1)
  if [ "$n_tail" = "all" ] && [ -n "$start" ]; then from=$start
  elif [ -n "$start" ] && [ $((total - start)) -le 2500 ]; then from=$start
  else from=$((total - n_tail + 1)); [ "$from" -lt 1 ] && from=1; fi
  echo "(log: $LOG | mulai baris $from dari $total | Ctrl+C untuk keluar - training TIDAK ikut berhenti)"
  tail -n +"$from" -f "$LOG"
}

# ─── Pilih lokasi staging untuk zip checkpoint ───────────────────────────────
mount_of() { df -P "$1" 2>/dev/null | tail -1 | awk '{print $6}'; }
avail_kb() { df -Pk "$1" 2>/dev/null | tail -1 | awk '{print $4}'; }

cmd_stage() {   # $1 = ukuran zip (byte). Cetak direktori staging terbaik.
  local size_kb=$(( $1 / 1024 )) cand
  mkdir -p "$CKPT_DIR" "$REMOTE_BASE"
  local ckfs; ckfs=$(mount_of "$CKPT_DIR")
  for cand in /dev/shm "$REMOTE_BASE"; do   # beda filesystem dari CKPT_DIR dulu
    [ -d "$cand" ] || continue
    [ "$(mount_of "$cand")" = "$ckfs" ] && continue
    [ "$(avail_kb "$cand")" -gt $(( size_kb + 600000 )) ] && { echo "$cand"; return 0; }
  done
  for cand in /dev/shm "$REMOTE_BASE"; do   # filesystem sama: butuh zip + hasil ekstrak
    [ -d "$cand" ] || continue
    [ "$(mount_of "$cand")" = "$ckfs" ] || continue
    [ "$(avail_kb "$cand")" -gt $(( size_kb * 22 / 10 + 300000 )) ] && { echo "$cand"; return 0; }
  done
  die "ruang tidak cukup untuk zip checkpoint ($((size_kb/1024)) MB) + hasil ekstrak. Pindahkan CKPT_DIR ke /dev/shm di .env atau sewa instance dengan disk lebih besar."
}

cmd_unpack() {   # $1 = path zip di instance
  [ -f "$1" ] || die "zip tidak ada: $1"
  ckpt_in_use && die "training sedang memakai $CKPT_DIR - checkpoint tidak ditimpa."
  mkdir -p "$CKPT_DIR"
  unzip -o -q "$1" -d "$CKPT_DIR" || die "unzip checkpoint gagal"
  rm -f "$1"
  [ -f "$CKPT_DIR/train_state.txt" ] && [ -f "$CKPT_DIR/ckpt_pretrain.safetensors" ] && [ -f "$CKPT_DIR/ckpt_optim.safetensors" ] \
    || die "isi zip tidak lengkap (butuh ckpt_pretrain, ckpt_optim, train_state.txt)"
  log "checkpoint terpasang di $CKPT_DIR:"
  ls -la "$CKPT_DIR" | awk 'NR>1 {print "   " $5 "  " $9}'
  head -2 "$CKPT_DIR/train_state.txt"
}

cmd_drive() {   # unggah checkpoint saat ini ke Google Drive (tanpa menghentikan training)
  [ -f "$CKPT_DIR/train_state.txt" ] || die "tidak ada checkpoint di $CKPT_DIR"
  local step tok
  step=$(sed -n 's/^step=//p' "$CKPT_DIR/train_state.txt"); tok=$(sed -n 's/^tokens_processed=//p' "$CKPT_DIR/train_state.txt")
  cd "$PROJ" || die "source belum ada"
  CKPT_UPLOAD_TMP="$REMOTE_BASE" python3 scripts/upload_gdrive.py --ckpt_dir "$CKPT_DIR" --step "$step" --tokens "$tok" --zip
}

case "${1:-}" in
  drive)   cmd_drive ;;
  check)   cmd_check ;;
  setup)   cmd_setup ;;
  extract) cmd_extract ;;
  build)   cmd_build ;;
  start)   cmd_start ;;
  onstart) cmd_onstart ;;
  recover) recover_from_drive ;;
  stop)    shift; cmd_stop "${1:-}" ;;
  status)  cmd_status ;;
  live)    cmd_live ;;
  panel)   cmd_panel ;;
  stage)   shift; cmd_stage "$1" ;;
  unpack)  shift; cmd_unpack "$1" ;;
  ckpt_busy) ckpt_in_use && echo busy || echo free ;;
  *) echo "perintah tidak dikenal: ${1:-}"; exit 2 ;;
esac
