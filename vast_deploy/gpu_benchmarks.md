# Benchmark GPU Terukur — kode trainer 492M (train_pretrain_cuda)

> Semua angka di bawah DIUKUR langsung di Vast.ai (bukan estimasi): sewa instance,
> build arch GPU bersangkutan (grad_check harus lulus), upload korpus subset 383MB,
> jalankan trainer batch 8 x seq 512 dengan bobot scratch, ambil 3 pembacaan [S1].
> Metode identik untuk semua GPU = apples-to-apples. Jangan percaya angka lain.

## Papan peringkat (tanggal 2026-11-25)

| GPU | tok/s (batch 8) | $/jam (eff, +disk 70GB) | tok/$ | catatan |
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

## Pelajaran (dari 5 pengukuran + riwayat 4090)

1. Kernel custom CUDA ini efisien di GPU **consumer high-clock** (Ada: 4090/5090),
   BURUK di datacenter Ampere (A100 cuma 9,3K padahal TFLOPS besar).
2. Harga mengalahkan kecepatan: 5090 1,3x lebih cepat dari 4090 tapi 24% lebih miskin
   token per dolar. Keputusan sewa selalu hitung tok/$ = tok/s ÷ harga.
3. Batch 8 > batch 14 di 4090 (26% lebih cepat). Default BATCH_SIZE=8 sudah benar.
4. Yang BELUM ada di pasar dengan offer valid saat scan: L4, RTX 4080/4080S, L40S
   (keluarga Ada murah — kandidat paling menjanjikan kalau muncul, ukur dulu!).
5. Estimasi spek-sheet meleset 40-63% (A100 diprediksi 15-25K, nyatanya 9,3K).
   HANYA angka terukur yang boleh dipakai untuk keputusan uang.

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
