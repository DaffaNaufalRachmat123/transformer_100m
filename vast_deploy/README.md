# vast_deploy - pretraining di GPU sewaan (Vast.ai) tanpa bantuan

Satu folder, satu `.env`, satu perintah. Semua yang dulu dilakukan manual
(kirim kode, kirim korpus, kirim checkpoint, build sesuai GPU, jalankan, pantau)
dibungkus di `deploy.sh`.

## Isi folder

| File / folder | Fungsi |
|---|---|
| `.env` | **Satu-satunya file yang Anda ubah**: host/port SSH, credential Drive, parameter training |
| `deploy.sh` | Perintah utama (dijalankan di Mac) |
| `source_code.zip` | Kode + credential Google Drive. Dibuat otomatis oleh `./deploy.sh package` |
| `corpus_pretrain_v8/` | **Symlink** ke `../data/corpus_pretrain_v8` (25 GB, tidak disalin) |
| `checkpoint_latest.zip` | **Anda taruh sendiri di sini** (isi: `ckpt_pretrain.safetensors`, `ckpt_optim.safetensors`, `train_state.txt`) |
| `remote/remote.sh` | Skrip yang berjalan di instance (dikirim otomatis, Anda tidak perlu menyentuhnya) |
| `tools/` | Pembuat zip + peringkas kecepatan |
| `speed_monitoring_gpu/` | Hasil `./deploy.sh pull`: log, nvidia-smi, `summary.txt`, `tok_per_s.csv` |

## Sewa GPU baru: 5 langkah

1. **Pilih GPU** dan lihat kodenya: `./deploy.sh gpus` (semua), `./deploy.sh gpus 5090` (filter),
   `./deploy.sh gpus show "RTX 5090"` (detail + link pencarian Vast). Isi nama persis di `.env`: `GPU_MODEL="RTX 5090"`.
2. **Sewa instance** di https://cloud.vast.ai/create/. Pilih image yang punya CUDA *devel* (ada `nvcc`) dengan versi
   yang cukup untuk GPU itu (ditampilkan oleh `gpus show`), **disk minimal 40 GB**, dan 1 GPU atau lebih.
3. Klik **Connect** di kartu instance, salin `ssh -p PORT root@IP ...`, lalu isi `.env`:
   ```
   SSH_HOST=<IP>
   SSH_PORT=<PORT>
   ```
4. Daftarkan key SSH Mac Anda ke instance. Pilih salah satu:
   - Tempel isi `~/.ssh/id_ed25519.pub` lewat ikon **[Key]** di kartu instance, atau
   - isi `VAST_API_KEY` dan `VAST_INSTANCE_ID` di `.env`, lalu `./deploy.sh key`
     (key API: https://cloud.vast.ai/cli/).
5. Taruh `checkpoint_latest.zip` di folder ini, lalu:
   ```
   cd vast_deploy
   ./deploy.sh all
   ```
   `all` menjalankan berurutan: `check -> package -> source -> corpus -> checkpoint -> build -> start`.
   Setiap tahap berhenti dengan pesan jelas kalau ada yang salah. Aman diulang:
   rsync melanjutkan korpus yang terpotong.

## Pantau training

```
./deploy.sh monitor      # LOG PENUH gaya Kaggle, real time: banner, info CUDA, dataset, baris [S1] ... (Ctrl+C keluar; training TETAP jalan)
./deploy.sh panel        # panel ringkas (step, loss, tok/s, GPU, disk) refresh tiap 2 detik
./deploy.sh status       # ringkasan sekali tampil
./deploy.sh log          # ikuti log mentah dari baris terakhir
./deploy.sh attach       # masuk layar tmux training (keluar: Ctrl+b lalu d, JANGAN Ctrl+C)
./deploy.sh pull         # tarik log+nvidia-smi ke Mac -> speed_monitoring_gpu/summary.txt
```

`monitor` mulai dari penanda `=== ` (START/RESTART/FINISH/STOPPED) terakhir, jadi Anda melihat satu sesi lengkap dari banner pipeline.
Atur jumlah baris awal dengan `MONITOR_TAIL` di `.env` (`all` = dari awal sesi).

Tanpa `deploy.sh` (langsung SSH), mis. untuk instance yang sedang berjalan:
```
ssh -t -i ~/.ssh/id_ed25519 -p PORT root@IP 'bash /workspace/remote.sh live'
ssh -t -i ~/.ssh/id_ed25519 -p PORT root@IP 'bash /workspace/remote.sh panel'
ssh -t -i ~/.ssh/id_ed25519 -p PORT root@IP 'watch -n 1 nvidia-smi'
```

## Perintah lain

| Perintah | Fungsi |
|---|---|
| `./deploy.sh check` | Cek SSH, GPU, arsitektur terdeteksi, nvcc, alat, disk. Tidak mengubah apa pun |
| `./deploy.sh build` | Build ulang (mis. setelah ganti GPU di instance yang sama) |
| `./deploy.sh start` | Mulai/lanjut training dari checkpoint di instance (supervisor auto-restart aktif) |
| `./deploy.sh onstart` | Tampilkan on-start script Vast untuk interruptible; `onstart set` memasangnya via API |
| `./deploy.sh stop` | Hentikan aman: trainer menyimpan checkpoint, lalu menunggu upload Drive akhir. `stop --fast` melewati upload |
| `./deploy.sh drive` | Upload checkpoint sekarang ke Google Drive tanpa menghentikan training |
| `./deploy.sh ssh` | Buka shell di instance |
| `./deploy.sh auth` | Buat token Google Drive baru di Mac (browser terbuka; untuk token expired/dicabut) |
| `./deploy.sh token` | Kirim token Drive terbaru ke instance tanpa menghentikan training |
| `./deploy.sh vast` | Status instance (running/paused) + bid + **saldo Vast** via API — jalan walau instance paused dan SSH mati; ini cara tercepat tahu kalau kena interrupt |
| `./deploy.sh rent` | **Sewa on-demand otomatis**: cari offer termurah yang lolos saringan (reliability ≥0.95, uplink ≥100Mbps, disk cukup, driver CUDA cukup), buat instance (image PyTorch cuDNN-devel dipilih otomatis sesuai GPU), daftarkan key SSH Mac ke akun, tunggu SSH siap, lalu isi `SSH_HOST`/`SSH_PORT`/`VAST_INSTANCE_ID` di `.env`. `rent --dry-run` = lihat rencana + estimasi token tanpa menyewa. Opsi: `--gpu "RTX 3090" --budget 5 --disk 70 --max-price 0.15 --min-reliab 0.95` |
| `ssh ... 'bash /workspace/remote.sh recover'` | Tarik ulang checkpoint dari Google Drive secara manual |

## Memilih GPU dan kode arsitekturnya

`tools/gpu_table.tsv` berisi 88 tipe GPU. Kolom arsitektur (`sm`) **diambil dari field `compute_cap` pada penawaran
Vast.ai** lewat API pencarian publik mereka (tanpa API key), jadi mengikuti data Vast, bukan ingatan saya.
Perbarui kapan saja: `./deploy.sh gpus refresh` (~15 detik).

```
./deploy.sh gpus                    # semua, diurut arsitektur
./deploy.sh gpus 5090               # filter nama (atau "sm120")
./deploy.sh gpus show "RTX 4090"    # detail: arsitektur, VRAM, CUDA minimal, link pencarian
```

Contoh nilai dari tabel: T4 `sm_75`, A100 `sm_80`, RTX 3090/A5000 `sm_86`, RTX 4090/L4/L40S `sm_89`,
H100/H200 `sm_90`, B200 `sm_100`, RTX 5090 `sm_120`. Untuk nilai lain jalankan `./deploy.sh gpus`.

Cara kerjanya saat build:
1. `GPU_MODEL` di `.env` -> arsitektur dari tabel. Nama salah ketik langsung ditolak di Mac (`check`, `build`, `all`).
2. Di instance, arsitektur itu **dibandingkan dengan GPU asli** (`nvidia-smi`). Kalau beda (Anda salah pilih, atau
   instance berbeda dari yang dikira) build **berhenti** dengan pesan jelas. Sudah diuji: `GPU_MODEL="RTX 5090"` di
   instance RTX 4090 ditolak, pilihan yang benar lolos.
3. Isi `CUDA_ARCH=<angka>` di `.env` hanya untuk memaksa (tanpa pemeriksaan).
4. Build ditolak dengan pesan jelas kalau `nvcc` di image tidak mendukung arsitektur itu: terlalu lama (mis. 5090 butuh
   CUDA >= 12.8) atau terlalu baru (CUDA 13 sudah membuang `sm_70` ke bawah, jadi V100/P100/GTX 10xx butuh image CUDA 12.x).
5. **1 GPU**: DDP mati. **>1 GPU**: DDP menyala kalau NCCL ada, kalau tidak build jatuh ke 1 GPU dengan peringatan.
6. `grad_check` otomatis dijalankan dan **harus lulus** sebelum training boleh mulai.
7. Binary inference (`build_infer`) adalah build CPU, tidak bergantung pada GPU.

Kolom "muat 492M?" di tabel adalah **perkiraan saya**, bukan data Vast: bobot+optimizer+gradien ~8.5 GB
ditambah aktivasi dan memori kerja trainer. Hanya **T4** (Kaggle) dan **RTX 4090** (Vast) yang benar-benar sudah dijalankan
untuk proyek ini. GPU lain: arsitekturnya dari Vast, tetapi kecepatan dan batch terbaiknya belum pernah diukur.

## Mode interruptible (bid Vast) — training yang menyalakan dirinya sendiri

Interruptible = sewa murah yang bisa di-pause Vast kapan pun (kalah bid / ada
penyewa on-demand) dan dinyalakan lagi otomatis. Perangkat di repo ini sudah
beradaptasi; yang perlu Anda lakukan cuma sekali:

1. **`.env`**: `CKPT_DIR` HARUS di disk persisten, bukan `/dev/shm` (RAM hilang
   saat pause). Default sekarang `$REMOTE_BASE/checkpoints` = `/workspace/checkpoints`.
2. **Pasang on-start script** di instance Vast:
   - `./deploy.sh onstart` → menampilkan baris yang harus ditempel di pengaturan
     instance Vast, **atau**
   - `./deploy.sh onstart set` → memasang otomatis lewat API (isi `VAST_API_KEY`
     + `VAST_INSTANCE_ID` di `.env`). Pilih juga launch mode *restart on instance start*.
3. **`UPLOAD_EVERY`** tetap aktif — Google Drive adalah jaring pengaman
   (lihat pemulihan di bawah).

Setelah itu siklus interruptible berjalan tanpa disentuh:

| Kejadian | Yang terjadi |
|---|---|
| Kalah bid → container dimatikan (SIGKILL) | Checkpoint aman: safetensors + `train_state.txt` semuanya ditulis atomik (tmp+rename). Progres maksimum yang hilang = `SAVE_EVERY` langkah. |
| Menang bid lagi → instance hidup | Vast menjalankan on-start script → `remote.sh onstart`: tunggu driver GPU siap, pasang ulang tmux/pip (layer container dibuang saat stop), lalu `start` → resume dari checkpoint. |
| Trainer mati sendiri (OOM, error transien) | Supervisor di `run_train.sh` menjalankan ulang otomatis dari checkpoint terakhir, jeda eksponensial 30s→900s, maks `RESTART_MAX` (default 8) kali. Log ditandai `=== RESTART ...`. |
| Checkpoint lokal hilang (instance baru/di-recreate) | `start` menarik `checkpoint_latest.zip` dari Google Drive otomatis (`scripts/download_gdrive.py`, matikan dengan `DRIVE_RECOVER=0`), lalu resume. |
| Anda menjalankan `./deploy.sh stop` | Flag stop mencegah supervisor menyalakan training kembali. |

Catatan biaya: selama *paused* GPU tidak ditagih, tapi **storage disk tetap** —
jangan alokasikan disk ratusan GB ngangguru. Saldo habis = instance berhenti;
cek `./deploy.sh status` dan unggah checkpoint (`./deploy.sh drive`) sebelum
saldo benar-benar habis kalau tidak mau top-up dulu.

## Yang perlu Anda sesuaikan saat ganti GPU

- **Ukuran batch**: `BATCH_SIZE` di `.env` adalah **per GPU**. Kecepatan tertinggi tidak selalu pada batch terbesar:
  di RTX 4090 batch 8 mencapai 12.8K tok/s, batch 14 hanya ~9.5K. Coba beberapa nilai di GPU baru selama
  1 menit masing-masing. Batch lebih kecil = token per langkah lebih sedikit = gradien lebih noisy.
  Di Kaggle (2 GPU x batch 7) total 7.168 token/langkah; 1 GPU x batch 14 sama persis.
- **Disk**: kalau disk instance < 40 GB, korpus tidak muat dan `corpus` akan berhenti dengan pesan.
- **Lokasi checkpoint** (`CKPT_DIR`): default `/dev/shm/checkpoints` (RAM, cepat, tidak makan disk, tapi
  **hilang kalau instance berhenti/dihancurkan**). Kalau disk besar, ganti ke `/workspace/checkpoints`.

## Peringatan penting

1. **Checkpoint di `/dev/shm` (RAM) hilang saat instance berhenti.** Default `CKPT_DIR` sekarang `/workspace/checkpoints`
   (disk, selamat saat pause interruptible); kalau Anda mengubahnya ke `/dev/shm` demi kecepatan, satu-satunya salinan
   aman tetap Google Drive (`checkpoint_latest.zip`). Pastikan `UPLOAD_EVERY` aktif, dan jalankan `./deploy.sh stop`
   (bukan sekadar menghancurkan instance) sebelum selesai.
2. **Saldo Vast.ai habis -> instance berhenti.** Setelah itu GPU tidak ditagih tapi storage tetap. Pantau saldo.
3. **Destroy instance menghapus semuanya.** Pastikan `./deploy.sh drive` atau `stop` sudah mengunggah checkpoint.
4. **Jangan jalankan dua training dari checkpoint yang sama** (mis. Kaggle dan Vast bersamaan): hasilnya bercabang.
5. **Upload ke Google Drive menimpa `checkpoint_latest.zip`** di folder Drive yang sama.
6. **Credential ada di `.env` dan di `source_code.zip`**. Jangan bagikan atau commit folder ini
   (`*.zip`, `gdrive_token.json`, `client_secret_*.json` sudah masuk `.gitignore`; `.env` belum, tambahkan kalau perlu).
7. **API key Vast.ai** yang pernah Anda tempel di chat sebaiknya dicabut dan dibuat ulang.

## Kalau ada masalah

| Gejala | Solusi |
|---|---|
| `tidak bisa SSH` | Cek host/port (tombol Connect), instance harus `running`, daftarkan key (langkah 3) |
| `nvcc tidak ditemukan` | Image tidak punya toolkit. Pilih image `cuda-*-devel`, atau `apt-get install -y cuda-toolkit` |
| `nvcc ... tidak mendukung sm_XX` | CUDA di image terlalu lama untuk GPU itu. Pilih image CUDA lebih baru |
| `disk instance tidak cukup` | Sewa disk lebih besar (>= 40 GB) |
| `md5 tidak cocok` | Ulangi `./deploy.sh checkpoint` |
| `grad_check GAGAL` | Jangan training. Kirim output `/workspace/gradcheck.log` untuk diperiksa |
| Training berhenti sendiri | `./deploy.sh log`, cari `[OOM]` (kurangi `BATCH_SIZE`), `[FATAL]`, atau saldo habis |
| `[OVERFLOW]` muncul sesekali | Normal: loss scaling otomatis turun lalu naik lagi. Waspada kalau muncul terus-menerus |
| log ada `[UPLOAD] Auth failed` / `invalid_grant` | Refresh token Drive mati (dicabut, 6 bulan tak terpakai, atau 7 hari kalau consent screen masih "Testing"). Jalankan `./deploy.sh auth` (di Mac, pilih akun yang benar) lalu `./deploy.sh token`. Cek juga Google Cloud Console → OAuth consent screen → Publish agar token tidak hangus 7 hari |
