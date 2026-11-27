# Benchmark GPU Terukur — kode trainer 492M (train_pretrain_cuda)

> Throughput di bawah diukur langsung di Vast.ai: build untuk arsitektur GPU
> (grad_check harus lulus), jalankan trainer dari bobot scratch pada subset korpus
> ±383 MB, batch 8 per GPU × seq 512 kecuali jika ditandai lain. Durasi dan host
> berbeda antarpercobaan. Biaya 50B adalah ekstrapolasi dari throughput terukur.

## Papan peringkat (tanggal 2026-11-25)

| GPU | tok/s (batch 8) | $/jam (eff, +disk 70GB) | (tok/s)/($/jam) | catatan |
|---|---|---|---|---|
| **Tesla V100S** | 6.800 | $0,193 | **35.233** | juara tok/$ — sm_70, host DE |
| **RTX 4090** | 12.631 | $0,366 | **34.511** | juara kecepatan/praktis — sm_89, host DE |
| RTX 5090 | 16.400 | $0,613 | 26.754 | tercepat absolut (1,3x 4090) tapi mahal |
| RTX 3090 Ti | 7.300 | $0,246 | 29.675 | kalah dari V100S/4090 |
| A100 SXM4 | 9.300 | $0,440 | 21.156 | datacenter Ampere = kernel custom ini boros |
| RTX 4090 (batch 14) | 9.500 | $0,366 | ~26.000 | batch 14 LEBIH BURUK dari batch 8 (data lama) |
| Tesla V100S (run asli) | 6.800 | $0,193 | 35.233 | run 663k-667k step, stabil 50 menit |

Offer juara saat scan 25 Nov 2026 (pasar bergerak cepat — scan ulang sebelum sewa):
- V100S: id 61077292, reliab 0.96, Germany
- 4090 : id 61653508, reliab 0.97, Germany

## RTX 3090 1x & 2x DDP — pengukuran multi-GPU PERTAMA (27 Nov 2026)

Instance 62818016 (2x RTX 3090 24GB, Serbia, $0.389/jam — offer Quebec $0.242
sudah diambil orang lain saat mau sewa). grad_check PASS, batch 8 x seq 512,
bobot scratch, sm_86, image pytorch 2.5.1-cuda12.4-cudnn9, subset 384MB.
Biaya trial ~$0.45 (sewa ~70 menit termasuk debug NCCL).

| mode | tok/s | s/step | VRAM/GPU | pembacaan [S1] |
|---|---|---|---|---|
| 1x 3090 | 6.4-7.0K | 0.6 | 20.8/24 GB | 7.0 → 7.0 → 6.6 → 6.4 (drift turun ~9%, suspect thermal rig consumer) |
| 2x 3090 DDP | **9.0K global** | 0.9 | 20.9/24 GB | 11.4 → 11.1 → 9.1 → 9.0 → 9.0 (tok/s = gabungan 2 GPU, batch global 16) |

**Temuan utama — scaling DDP nyata 64-70%, BUKAN 92%:**

- AllReduce 491.6M param FP16 (~1 GB gradien) lewat shared memory
  (NCCL_P2P_DISABLE=1, P2P gagal init di host ini) menambah +0.3 dtk/step
  (+50%: 0.6 → 0.9 dtk) untuk kerja per-GPU yang identik.
- GeForce 40/50 (Ada/Blackwell) bahkan TIDAK punya P2P PCIe — scaling
   2x 4090/5090 perlu diukur, bukan diasumsikan sama. Untuk pembanding NVLink,
   cek topologi host (V100S yang diuji di bawah adalah PCIe). Rencana 2x GPU lama (V100S $406 /
  4090 $442 / 5090 $527 untuk 50B) memakai asumsi 92% → kemungkinan optimis.
- 50B token @ 9.0K global = 64 hari: **$373** jika dapat 2x3090 $0.242
   (skenario tarif offer lain, host belum diuji) atau **$600** pada host Serbia
   ($0.389). 1x V100S $0.193: $394 / 85 hari.

Bug DDP yang ketemu & diperbaiki (commit terpisah):

1. Mode multi-proses (launch_ddp.sh) TIDAK PERNAH bisa jalan — rank != 0
   memakai ncclUniqueId tak terinisialisasi (pertukaran ID via file cuma ada
   di path fork). Fix: file-exchange ID di semua mode multi-rank, spin-wait
   20 dtk + unlink file lama.
2. Mode fork (`--ddp` tanpa env RANK) rusak di host ini: child gagal
   "Cuda failure 'initialization error'" (CUDA context parent teracuni).
   Jalur multi-proses (launch_ddp.sh) = satu-satunya yang benar untuk
   multi-GPU.

## Verifikasi ulang Tesla V100S (27 Nov 2026)

- **1× V100S PCIe 32 GB, Jerman** (instance `62827277`): grad_check lulus;
  **6,8–6,9K tok/s** pada run diagnosis batch 8 × seq 512 dengan target 1M token,
  selaras dengan 6,8K tok/s historis. Tarif offer $0,176/jam.
- **2× V100S PCIe** (Polandia `62824929`, Inggris `62828183`): grad_check
  lulus, tetapi tidak ada pembacaan throughput rank 0 yang bisa dipakai.
  Log Inggris menunjukkan rank 1 melaporkan 301 step; log ter-buffer dan
  rank 1 tidak mencetak statistik throughput. Tidak ada angka efisiensi DDP
  atau biaya 50B dari dua percobaan ini.
- Diagnosis lanjutan (`62830960`) mendapat OOM saat proses kedua mencoba
  inisialisasi optimizer, sementara proses sebelumnya masih memakai GPU 0.
  Itu tidak menjelaskan gejala awal. Pool dibuat setelah state optimizer,
  bukan sebelumnya. Semua instance ini sudah dihapus.

## NVIDIA L40S — single dan 2× DDP (27 Nov 2026)

Instance `62834401` (offer `59272067`, Israel, 2× L40S, $1,0681/jam).
Build `sm_89`, grad_check lulus. Sampel 240 detik per mode. Topologi PCIe
`SYS` (lintas NUMA, tanpa NVLink); NCCL memakai konfigurasi P2P default,
tetapi transport yang terpilih tidak direkam. Instance sudah dihapus.

| Mode | Rentang 8 pembacaan terakhir | Median | Batch global |
|---|---:|---:|---:|
| 1× L40S | 12,6–12,7K tok/s | 12,6K tok/s | 8 |
| 2× L40S DDP | 18,3–18,5K tok/s global | 18,4K tok/s | 16 |

Speedup **1,46×**; efisiensi DDP **73,0%** (`18,4 ÷ (2 × 12,6)`).
Tidak ada penawaran 1× L40S yang lolos filter saat scan ini, jadi tidak ada
angka biaya single berdasarkan tarif sewa 1×.

### Percobaan lain tanpa hasil throughput baru

- **2× RTX 5090**: Oregon (`62833730`, $1,3237/jam) dan Illinois (`62835934`,
  $1,3363/jam) sama-sama tidak mencapai SSH dalam batas 300 detik. Keduanya
  sudah dihapus. Build dan grad_check tidak sempat dijalankan, jadi ini
  bukan bukti masalah pada trainer atau scaling RTX 5090.
- **1× RTX A6000** dan **2× RTX 4090**: tidak disewa karena tidak ada offer
  yang memenuhi filter dan batas tarif masing-masing $0,90 dan $1,10/jam.

## Ekstrapolasi 50B token — throughput terukur

`jam = 50.000.000.000 ÷ tok/s ÷ 3600`; `biaya = jam × tarif per jam`.
Biaya komputasi kontinu, belum termasuk jeda, transfer atau biaya tambahan.
Tarif bukan jaminan ketersediaan saat menyewa.

| Konfigurasi | tok/s | $/jam | Hari | Biaya 50B |
|---|---:|---:|---:|---:|
| 1× V100S PCIe (verifikasi Jerman) | 6.800 | 0,176 | 85,1 | $359 |
| 1× RTX 4090 (historis) | 12.631 | 0,366 | 45,8 | $402 |
| 1× RTX 5090 (historis) | 16.400 | 0,613 | 35,3 | $519 |
| 2× RTX 3090 DDP (Serbia) | 9.000 | 0,389 | 64,3 | $600 |
| 2× L40S DDP (Israel) | 18.400 | 1,0681 | 31,5 | $806 |

## Pelajaran

1. Kernel custom CUDA ini efisien di GPU **consumer high-clock** (Ada: 4090/5090),
   BURUK di datacenter Ampere (A100 cuma 9,3K padahal TFLOPS besar).
2. Harga mengalahkan kecepatan: 5090 1,3x lebih cepat dari 4090 tapi menghasilkan
   lebih sedikit token per dolar pada tarif di atas. Untuk tok/$ fisik, hitung
   `tok/s × 3600 ÷ ($/jam)`; kolom tabel lama menampilkan rasio tanpa faktor 3600.
3. Batch 8 > batch 14 di 4090 (26% lebih cepat). Default BATCH_SIZE=8 sudah benar.
4. L40S sekarang terukur. Untuk perluasan berikutnya, cari RTX A6000 dan
   2× RTX 4090 dengan harga masuk akal; jangan samakan hasil antarhost.
5. Estimasi spek-sheet meleset 40-63% (A100 diprediksi 15-25K, nyatanya 9,3K).
   HANYA angka terukur yang boleh dipakai untuk keputusan uang.
6. DDP 2x GPU consumer terukur cuma 64-70% (bukan 92%): AllReduce ~1 GB
   lewat SHM menambah 50% waktu step. Rencana multi-GPU harus dihitung
   ulang dengan pengukuran pada host yang akan digunakan. Catat topologi
   PCIe/NVLink; tok/s pada log DDP adalah angka GLOBAL semua rank.

## Biaya kampanye pengukuran 25 Nov 2026

3 trial (A100 + 3090 Ti + 5090) = **$0,18 total** — masing-masing ~10 menit,
destroy segera setelah 3 pembacaan. Protokol: `rent -> check -> source -> build ->
corpus subset (383MB) -> ukur 2 menit -> vonis -> destroy`.

## Catatan untuk run sungguhan berikutnya

- Korpus: subset 383MB (8 shard, 4 domain) cukup untuk UKUR. Untuk run ~530 juta
  token, upload ~3-4GB campuran shard (jangan sampai banyak epoch pada subset kecil).
- Checkpoint terakhir aman di Drive: step 667.039, 4.765.400.064 token (47,65%
  dari target 10B) — folder 1tVYqa2W5lxloMEE6yXQhc_vpj6q6O18Y.
- Fix permanen di tooling: patch cuDNN symlink dev polos (remote.sh), rsync Mac
  menyarangkan path dengan -R (upload per-file, bukan -R).
