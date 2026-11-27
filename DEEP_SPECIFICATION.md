# DEEP_SPECIFICATION — transformer_100m_cuda

Spesifikasi mendalam proyek ini, tersusun **dari unit algoritma terkecil sampai orkestrasi terbesar**.
Setiap bagian menyertakan matematika eksak, pemetaan thread/block, penanganan presisi, dan referensi `file:baris`.
Semua nomor baris merujuk pada snapshot repo saat dokumen ditulis (commit terakhir + working tree).

Model: **491,582,720 parameter (492M)** — GQA + RoPE + SwiGLU + RMSNorm, FP16 mixed precision, AdamW FP32 master.
Korpus: EN/JA/code/math/mixed. Vocab: BPE v9, 49.152 token.

---

## Daftar isi

1. [L0 — Utilitas reduksi paralel](#l0)
2. [L1 — Kernel elementwise & konversi](#l1)
3. [L2 — Kernel matematika lapisan (RMSNorm, RoPE, SwiGLU)](#l2)
4. [L3 — Kernel besar (GQA attention, cross-entropy, embedding grad)](#l3)
5. [L4 — AdamW fused](#l4)
6. [L5 — Operasi host & infrastruktur CUDA (cuBLAS, arena, stream, monitor, DDP)](#l5)
7. [L6 — Graf forward/backward (train_step.cuh)](#l6)
8. [L7 — Loop pelatihan pretrain (loss scaling, keamanan, checkpoint)](#l7)
9. [L8 — Konfigurasi arsitektur runtime](#l8)
10. [L9 — Tokenizer BPE](#l9)
11. [L10 — Dataset streaming](#l10)
12. [L11 — Trainer SFT / DPO / RL](#l11)
13. [L12 — Format file (safetensors, train_state)](#l12)
14. [L13 — Orkestrasi & alat bantu](#l13)
15. [Ringkasan presisi, bug historis, performa terukur](#l14)

---

<a name="l0"></a>
## L0 — Utilitas reduksi paralel (unit terkecil)

### `warp_reduce_sum` — cuda/cuda_kernels.cu:10-14
```
val: lanjutkan offset 16→8→4→2→1 dengan __shfl_down_sync(0xffffffff, ...)
```
5 langkah butterfly warp-shuffle; hanya lane 0 yang memegang hasil akhir (tidak di-broadcast).
Dipakai oleh jalur yang tidak butuh nilai penuh di semua lane.

### `warp_sum` — cuda/attn_kernels.cuh:24-30
Identik dengan di atas, tetapi **ditutup `__shfl_sync(..., v, 0)`** sehingga semua lane menerima hasil.
Alasan: kernel attention membutuhkan dot-product penuh di setiap lane (setiap lane memegang 2 dimensi Q/K).

### `block_reduce_sum` — cuda/cuda_kernels.cu:26-44
Kontrak: `blockDim.x` = pangkat 2 ≥ 64 (semua pemanggil memakai 256).
Algoritma: `shared[tid]=val` → `__syncthreads` → pohon shared-memory `for (s = blockDim.x/2; s >= 32; s >>= 1)`
(sinkronisasi tiap level) → lipat warp-shuffle 32 sisanya → lane 0 tulis `shared[0]` → semua thread baca.

**Bug historis terdokumentasi (cuda_kernels.cu:21-25):** loop harus berhenti di `s == 32` *inklusif*.
Varian `s > 32` menyisakan 64 parsial padahal tahap warp hanya melipat `shared[0..31]` — setengah reduksi hilang diam-diam.
"Bug itu membuat RMSNorm menjumlahkan hanya 384 dari 768 dimensi sambil tetap membagi dengan D penuh."
Ditangkap oleh grad_check TEST 1 (lihat [L13](#l13)).

### `atomicMin_f32` / `atomicMax_f32` — cuda/ce_embed_kernels.cuh:267-286
Loop CAS (`atomicCAS` + reinterpretasi `__int_as_float`/`__float_as_int`) karena sm_75 tidak punya atomicMin/Max FP32 native.
Hanya dipakai kernel diagnostik `min_max_fp16_kernel` (ce_embed_kernels.cuh:288-326).

---

<a name="l1"></a>
## L1 — Kernel elementwise & konversi

Semua 1 thread per elemen, 256 thread/block, matematika FP32 setelah promosi `__half2float` (kecuali disebut lain).

| Kernel | Lokasi | Rumus / perilaku |
|---|---|---|
| `fill_fp16_kernel` | cuda_ops.cu:11-14 | isi konstanta `__half` |
| `add_fp16` | cuda_kernels.cu:584-587 | `out = __hadd(a, b)` — aritmetika half native, TANPA promosi FP32 |
| `mul_fp16` / `sqrt_fp16` | cuda_kernels.cu:595-609 | `__hmul` / `hsqrt` native |
| `fp16_to_fp32` / `fp32_to_fp16` | cuda_kernels.cu:559-573 | cast 1:1 |
| `scale_inplace_fp16_kernel` | ce_embed_kernels.cuh:88-100 | `d = isfinite(v) ? v*s : 0` — **nilai non-finite DINOLAKAN menjadi 0** saat skala |
| `clamp_inplace_fp16_kernel` | ce_embed_kernels.cuh:119-132 | utilitas, tidak dipakai jalur utama |
| `scale_fp16` (namespace cuda::) | cuda_ops.cu:399-413 | `out = float(in)*s` — **tanpa guard isfinite** (nama sama, semantik beda dengan `scale_inplace_*`) |
| `l2_norm_sq_fp16_kernel` | ce_embed_kernels.cuh:209-231 | grid-stride loop persisten (grid maks 1024 blok), `val += v²` FP32 per thread, pohon shared, `atomicAdd` ke skalar global. **Dipanggil per-tensor; semua panggilan mengakumulasi ke skalar yang sama** → norma-L2 global seluruh 218 tensor gradien |
| `count_nan_inf_kernel` | ce_embed_kernels.cuh:234-263 | `atomicAdd` penghitung NaN/Inf; dua varian: `launch_count_nan_inf` (memset dulu) dan `launch_count_nan_inf_accum` (tanpa memset → satu pasang penghitung untuk semua tensor per step) |
| `embedding_gather` | cuda_kernels.cu:620-631 | 1 thread per token, **loop serial D=1280 per thread** menyalin baris embedding; id di luar [0,V) di-clamp ke 0. Tidak terkoalesensi (warp membaca baris berbeda) |
| `f16_to_f32_seed_kernel` | ce_embed_kernels.cuh:182-192 | `accum[i] = f32(src[i])` — benih akumulator FP32 gradien embedding |
| `f32_to_f16_add_kernel` (= `launch_cast_f32_f16`) | ce_embed_kernels.cuh:164-179 | cast FP32→FP16 murni (nama "add" menyesatkan); dipakai untuk dQ/dK/dV |
| `f32_add_kernel` | ce_embed_kernels.cuh:195-204 | `dst += src` FP32; dipakai DPO untuk menggabungkan gradien chosen/rejected |
| `arange_i32_kernel` | cuda_ops.cu:16-19 | `d[i] = i` |

**Konsekuensi penting `scale_inplace` yang men-nol-kan non-finite (train_cuda.cu:1621-1626):**
bukti overflow (NaN/Inf) hilang setelah un-scale berjalan, maka **penghitungan NaN/Inf wajib dilakukan SEBELUM un-scale**.
Urutan pipeline di [L7](#l7) menegakkan ini.

---

<a name="l2"></a>
## L2 — Kernel matematika lapisan

Konstanta model yang memparameterisasi semua kernel (model_config.h:18-36):
`D=1280, L=24, HQ=20, HKV=5, DH=64, FFN=3584, V=49152, RoPE θ=500000, eps RMS=1e-6, skala attention=1/√64=0.125`.

### 2.1 RMSNorm forward — cuda_kernels.cu:49-98
```
rsqrt = rsqrtf( mean(x²) + eps )      // eps = 1e-6
normed = x · rsqrt
y = normed · gamma
```
- Grid = `B*S` blok (1 blok per baris), block = 256, shared dinamis `256*sizeof(float)`.
- Loop: (1) akumulasi `sum_sq` FP32 per-thread dengan stride `i += blockDim.x`; (2) `block_reduce_sum`;
  (3) hitung `rsqrt`; (4) lane 0 simpan `saved_rsqrt[row]` (FP16); (5) loop kedua tulis `y` dan `saved_normed`.
- Tersimpan untuk backward: `saved_rsqrt` [B,S] FP16 dan `saved_normed` [B,S,D] FP16 (x·rsqrt sebelum gamma).
- Presisi: akumulasi FP32 penuh; keluaran FP16.

### 2.2 RMSNorm backward — cuda_kernels.cu:100-185
```
dgamma = Σ_baris (dy · normed)                       // diakumulasi FP32 via atomicAdd
dx     = rsqrt · (dy·gamma − normed · mean(dy·gamma·normed))
```
- Grid `dim3(B*S, 2)` — **dua blok per baris**:
  - `col==0`: dua loop berjalan — reduksi `Σ dy·gamma·normed` → `total = Σ/D`; lalu tulis `dx = (dyg − normed·total)·rsqrt`.
  - `col==1`: `atomicAdd(&dgamma_f32[i], dy[i]·normed[i])` ke **scratch FP32** (cudaMallocAsync → memset 0 → kernel →
    `rmsnorm_dgamma_f32_to_f16` → cudaFreeAsync, satu stream).
- **Kenapa FP32 atomic (cuda_kernels.cu:149-150):** sm_75 tidak punya `atomicAdd(__half*, float)` dan akumulasi FP32 jauh lebih akurat.
- **Penempatan gamma (cuda_kernels.cu:106-108):** gamma harus DI DALAM suku pengurangan rata-rata.
  Varian "gamma di luar" hanya ekuivalen untuk gamma seragam — bug historis, ditangkap grad_check TEST 2.

### 2.3 RoPE forward (in-place) — cuda_kernels.cu:190-234
Pasangan interleaved (2j, 2j+1) — bukan split-half:
```
x'_{2j}   = x_{2j}·cos(pos, 2j)   − x_{2j+1}·sin(pos, 2j)
x'_{2j+1} = x_{2j}·sin(pos, 2j)   + x_{2j+1}·cos(pos, 2j)
```
- 1 thread per (b, s, h, j); grid 1D `ceil(B*S*H*32/256)`.
- Tabel cos/sin dihitung host (train_cuda.cu:631-663): `freq = pos / powf(θ, 2j/d_head)` (θ=500000),
  tabel `[max_seq=2048, 32]` FP32 di-upload ke device. `offset` mendukung kelanjutan KV-cache (training selalu 0).
- Presisi: rotasi di FP32, simpan FP16.

### 2.4 RoPE backward — cuda_kernels.cu:237-280
Forward = rotasi 2×2 R(θ) per pasangan, maka backward = **R(θ)ᵀ = R(−θ)**: kernel identik dengan
**sin dinegasikan** (`sin_val = -sin_table[...]`, baris 259), diterapkan in-place pada dx (Q: 20 kepala; K: 5 kepala).

### 2.5 SwiGLU forward — cuda_kernels.cu:283-318
```
silu(u) = u · sigmoid(u) = u / (1 + e^(−u))
y = gate · silu(up)
```
Elementwise murni (N = B·S·FFN); menyimpan `saved_silu = silu(up)` FP16 untuk backward.

### 2.6 SwiGLU backward — cuda_kernels.cu:323-362
```
dgate = dy · silu(up)                     = dy · saved_silu
dup   = dy · gate · silu'(up),   silu'(u) = silu(u) + sigmoid(u)·(1 − silu(u))
```

---

<a name="l3"></a>
## L3 — Kernel besar

### 3.1 GQA attention forward — cuda/attn_kernels.cuh:33-96
- **Tiling: satu WARP (blockDim=32) per (b, hq, query_i)**; grid datar `B·HQ·S`.
- **Pemetaan GQA:** `hkv = hq / (HQ/HKV)` = `hq/4` — 4 kepala query berurutan berbagi satu kepala KV.
- Layout: Q `[B,S,20,64]`, K/V `[B,S,5,64]`, O `[B,S,20,64]`, **LSE `[B,20,S]` FP32**.
- Residensi register: DH=64 → tiap lane memegang dimensi `{lane, lane+32}`.
- **Softmax online (gaya flash) atas kunci kausal `j ≤ i`**, dengan penskalaan ulang tiap langkah:
  ```
  m_baru = max(m, score);  corr = e^(m − m_baru);  p = e^(score − m_baru)
  l = l·corr + p;  a = a·corr + p·V_j;  m = m_baru
  ```
- Epilog: `o = a/l` (guard `l>0`); lane 0 menulis `LSE = m + log(l)`.
- Dot product hanya warp-shuffle (tanpa shared memory, tanpa `__syncthreads`) — "hot path harus murah" (komentar :6-8).
- Intrinsics cepat `__expf`/`__logf`; seluruh skor/softmax/akumulasi di FP32.
- Hanya LSE yang di-cache; Q/K/V tetap disimpan pemanggil untuk backward (train_step.cuh:64).

### 3.2 GQA attention backward — attn_kernels.cuh:102-172
Tiling dan pemetaan GQA sama. Probabilitas **dihitung ulang** dari LSE:
```
p_j = e^(scale·(q·k_j) − LSE_i)
D   = Σ_j p_j · (dO_i · V_j)                      // pass 1
dscore_ij = p_j · (dp_j − D) · scale              // pass 2, dp = dO_i·V_j
dV_j += p_j · dO_i     (atomicAdd, FP32)           // banyak i (dan 4 q-head) menulis slot KV sama
dK_j += dscore_ij·Q_i  (atomicAdd, FP32)
dQ_i += dscore_ij·K_j  (register; ditulis sekali, non-atomik)
```
- dQ/dK/dV berupa **scratch FP32** (dinolkan pemanggil, train_step.cuh:273-275), lalu
  `launch_cast_f32_f16` → **RoPE backward diterapkan pada dQ dan dK** (karena forward merotasi Q/K in-place) — train_step.cuh:285-291.

### 3.3 Cross-entropy forward+backward FUSED — cuda/ce_embed_kernels.cuh:17-85
Ini kernel yang dipakai loop training (train_cuda.cu:1472-1475). Satu blok 256 thread per baris token (grid = B·S),
shared 256 float. Tiga pass atas baris logits V=49.152:
1. **Max** (log-sum-exp trick): reduksi pohon shared.
2. **Sum-exp**: `Σ e^(x−max)` → `log_sum_exp`.
3. **Loss + gradien**: `loss = −(x_tgt − max − log_sum_exp)·mask` (FP32, keluaran per-token);
   `d_logits[j] = f16((softmax[j] − 𝟙[j==tgt])·mask)`.

Mask meng-nol-kan loss DAN gradien baris ber-mask. **Tidak ada pembagian mean dan tidak ada loss_scale di kernel** —
keduanya diterapkan luar: `launch_scale_fp16(d_logits, loss_scale/tok_count)` (train_cuda.cu:1576-1579).
`tok_count` = jumlah token tak ber-mask, dihitung host (train_cuda.cu:1479-1483).

Varian lama (`cross_entropy_forward_kernel`/`backward`, cuda_kernels.cu:365-504) dipertahankan hanya untuk referensi;
versi lama pernah menulis gradien ilegal ke buffer FP16 (`((float*)row)[i]=grad`) yang meluap — lihat komentar :365-371.

### 3.4 Gradien embedding — `embed_scatter_add_kernel` ce_embed_kernels.cuh:139-161
1 thread per (baris, d); `atomicAdd(&accum[id_baris·D + d], f32(d_x))` ke akumulator `[V,D]` FP32.
Akumulator **pra-di-seed** dengan gradien proyeksi output (weight tying) → lihat [L6](#l6).

### 3.5 `masked_log_probs` (untuk DPO) — cuda_ops.cu:330-379
Satu blok 256 thread; tiap thread memproses baris token utuh secara serial: max → sum-exp →
`log_prob = x_tgt − max − log(sum_exp)`; keluaran = **rata-rata log-prob atas token tak ber-mask**.
Sengaja sederhana (lambat, O(V) per thread) — hanya untuk skor model referensi DPO.

---

<a name="l4"></a>
## L4 — AdamW fused — cuda/cuda_kernels.cu:509-554
Per elemen (1 thread/elemen), `step` berbasis-1 (`step+1` dari pemanggil):
```
g  = f32(grad);  jika !isfinite(g) → LEWATI elemen (m, v, param tidak disentuh)
m  = β1·m + (1−β1)·g                          // β1 = 0.9
v  = β2·v + (1−β2)·g²                         // β2 = 0.95
m̂ = m/(1−β1^step);  v̂ = v/(1−β2^step)         // koreksi bias (powf)
master = master − lr·m̂/(√v̂ + eps) − lr·wd·master   // eps=1e-8 DI LUAR √; wd=0.1 (decoupled, pada master FP32)
param  = f16(master)                          // mirror FP16
```
Peran tensor: `param` FP16 (kerja), `master` FP32 (otoritatif), `m`/`v` FP32, `grad` FP16.
Norm gain (rms_attn/rms_ffn/final_rms) mendapat `wd=0` (train_cuda.cu:1817-1832) — norm dimulai dari 1.0
dan tidak punya tekanan skala natural.

Pembungkus `transformer/adamw_cuda.cuh` memuat `AdamWState` (m, v FP32) dan `apply_adamw` yang **tidak dipakai
produksi** (klip di dalamnya tidak berfungsi — menjumlah nilai gradien, bukan kuadratnya, dan `scale` diabaikan;
produksi memanggil `cuda::adamw_step` per-tensor, cuda_ops.cu:292-303 → train_cuda.cu:1806-1832).

---

<a name="l5"></a>
## L5 — Operasi host & infrastruktur CUDA

### 5.1 cuBLAS matmul — cuda/cuda_ops.cu:115-141
Menghitung row-major `C[m,n] = A·B` via **trik kolom-major**: panggil `cublasGemmEx` dengan operan
ditukar (B dulu, transposisi dibalik) dan dimensi `(n, m, k)`. alpha=1, beta=0, algo `CUBLAS_GEMM_DEFAULT`.
Presisi: `cu_compute()` → `CUBLAS_COMPUTE_32F` untuk F32, **`CUBLAS_COMPUTE_32F_FAST_16F` untuk FP16**
(FP16 masuk/keluar, **akumulasi FP32** di tensor core). Ini satu-satunya matmul untuk semua proyeksi dan gradien bobot.

### 5.2 Arena allocator — cuda/cuda_allocator.h
Bump-pointer pool O(1) menghilangkan ~600 `cudaMalloc/cudaFree` per step.
`init_auto(fraksi)` mengambil fraksi VRAM bebas (default kode 0.75; **di-instance Vast dipatch ke 0.85** —
0.93 terbukti menyebabkan `cudaMalloc failed: out of memory` setelah 53 menit di RTX 4090), 1 MB-aligned,
menyisakan ruang workspace cuBLAS. `alloc` 256-byte aligned, gagal → fallback `cudaMalloc`.
`reset()` dipanggil tiap awal step (train_cuda.cu:1336). `PinnedBuffer<T>` (h:121-135) = RAII pinned host biasa.

### 5.3 CudaTensor — cuda/cuda_tensor.h
Kontainer ringan: `alloc_data` coba arena dulu (`from_pool=true`, tanpa `cudaFree`), gagal → `cudaMalloc`
(dengan abort fatal yang jelas, bukan tensor null — komentar :78-86 menjelaskan mengapa tensor null lebih berbahaya:
kernel meluncur ke nullptr → `cudaErrorIllegalAddress` yang lengket).
`transpose()` (cuda_ops.cu:267-278) adalah **relabeling bentuk tanpa memindah data** (view, `owns=false`).

### 5.4 Konteks & stream — cuda/cuda_context.cu
Tiga stream dibuat: `compute_stream` (satu-satunya yang dipakai kernel/cuBLAS), `copy_stream` (dibuat tapi tak
pernah menerima salinan), `loss_stream` (mati total). Handle cuBLAS + cuDNN terikat ke `compute_stream`.
**Tidak ada `cublasSetWorkspace`** dan tidak ada `cublasSetMathMode` global — mode matematika per-GEMM (5.1).

### 5.5 Monitor GPU — cuda/cuda_monitor.cu
`USE_NVML` ditentukan saat build (CMake mencari `libnvidia-ml`). Jalur NVML: util%, mem, suhu, daya.
Jalur fallback: memori via `cudaMemGetInfo`, **util via `popen("nvidia-smi ...")` sekali per tick log**,
suhu/daya = 0 (tampil sebagai 0 di log Kaggle: "[NVML] disabled").

### 5.6 DDP NCCL — cuda/cuda_ddp.h
- **Peluncuran proses**: Kaggle = `fork()` internal (parent rank 0, child rank 1; CUDA belum boleh tersentuh
  sebelum fork). Multi-proses = `scripts/launch_ddp.sh` dengan env RANK/WORLD_SIZE/LOCAL_RANK — jalur ini punya
  **bug laten**: pertukaran `ncclUniqueId` via file hanya terjadi di mode fork; mode env melewatkan id tak terinisialisasi.
- NCCL di **stream terpisah**; `sync()` = sinkron stream NCCL.
- `allreduce_grads`: satu `ncclGroupStart/End` membungkus `ncclAllReduce(SUM)` per tensor FP16,
  urutan: `d_embed, d_final_rms, lalu per lapisan dwq..drms_ffn` (gather_grads, train_cuda_unified.cu:583-591).
- **Rata-rata = SUM lalu skala 1/world** per tensor (`average_grad`).
- `allreduce_scalar_max`: konsensus flag antar rank. **Motivasi (cuda_ddp.h:191-196):** tiap rank membaca data
  berbeda (seed di-offset rank) → keputusan skip-step lokal bisa berbeda → satu rank masuk kolektif yang tidak
  dimasuki rank lain → NCCL tanpa watchdog hang selamanya. Dipakai untuk konsensus skip-step (train_cuda.cu:1517)
  dan konsensus overflow loss-scale (:1653).
- `NCCL_P2P_DISABLE=1` disuntik train_pipeline.py **hanya di Kaggle** (pasangan T4 di sana tidak punya P2P usable).

---

<a name="l6"></a>
## L6 — Graf forward/backward (transformer/train_step.cuh)

`ts::` menggantikan `cuda_autograd.cu` yang lama (yang kini berisi stub `abort()` — lihat [L13](#l13)).
Semua aktivasi **disimpan, tidak dihitung ulang** (kecuali probabilitas attention dari LSE).

### Cache per lapisan (`LayerCache`, train_step.cuh:60-73)
Bentuk untuk B=14, S=512 (BS=7.168):
`x_in` [B,S,D]·FP16 (snapshot, tidak dikonsumsi backward), `rsqrt_a`/`normed_a` (RMSNorm attn),
`n1` [B,S,D] (untuk dwq/dwk/dwv), **`Q` [B,S,20,64] & `K` [B,S,5,64] pasca-RoPE**, `V`,
`attn` [BS,D] (pra-Wo), **`LSE` [B,20,S] F32**, `x_mid`, `rsqrt_f`/`normed_f`/`n2`,
`gate`/`up`/`silu` [B,S,FFN], `ff` [B,S,FFN].
`FwdCache` menambah `embed_out`, vektor lapisan, `rsqrt_final`/`normed_final`/`x_final`
(kepemilikan `x_final` sengaja tetap di cache karena gradien proyeksi output membutuhkannya setelah forward kembali).

### Forward per lapisan (train_step.cuh:94-184)
```
x = embedding_gather(params[0], tokens)                       // [14,512,1280] FP16
per lapisan l (base = 1 + l·9):
  n1   = rms_norm(x, rms_attn, eps=1e-6)                       // simpan rsqrt_a, normed_a
  Q,K,V= matmul(n1_2d, wq|wk|wv)                               // [BS,1280], [BS,320], [BS,320]
  rope_inplace(Q, 20 kepala); rope_inplace(K, 5 kepala)        // pasangan interleaved
  attn + LSE = gqa_attn_fwd(Q,K,V, skala=0.125)                // warp per (b,hq,i), softmax online
  o    = matmul(attn_2d, wo)
  x_mid= x + o                                                 // add_fp16 (__hadd)
  n2   = rms_norm(x_mid, rms_ffn)
  gate = matmul(n2_2d, w1); up = matmul(n2_2d, w3)             // [BS,3584]
  ff   = gate · silu(up)                                       // simpan silu
  down = matmul(ff_2d, w2)                                     // [BS,1280]
  x    = x_mid + down
x_final = rms_norm(x, final_rms)
```
Di main setelahnya: `logits = matmul_transb(x_final, embed)` → **[7.168, 49.152] FP16 (weight tying)**,
lalu CE fused (L3.3).

### Backward (train_step.cuh:196-329, urutan terbalik)
```
dx_final = matmul(d_logits, embed)                             // [BS,49152]×[49152,1280]
seed embed_accum (F32) = f32( d_logitsᵀ @ x_final )            // benih gradien weight-tie (menimpa)
rms_norm_backward(final)  → dx, d_final_rms
per lapisan (23→0):
  dw2 = ffᵀ@d_down;  d_ff = d_down@w2ᵀ
  swiglu_backward  → d_gate, d_up
  dw1 = n2ᵀ@d_gate; dw3 = n2ᵀ@d_up;  dn2 = d_gate@w1ᵀ + d_up@w3ᵀ
  rms_norm_backward(ffn) → d_xmid(+norm), d_rms_ffn
  dwo = attnᵀ@d_o;  d_attn = d_o@woᵀ
  gqa_attn_bwd → dQ/dK/dV (FP32) → cast F16 → rope_backward(dQ, dK)
  dwq = n1ᵀ@dQ; dwk = n1ᵀ@dK; dwv = n1ᵀ@dV;  dn1 = dQ@wqᵀ + dK@wkᵀ + dV@wvᵀ
  rms_norm_backward(attn) → d_x(+norm), d_rms_attn
  dx = d_o + d_x_from_norm                                    // gabungan cabang residual + cabang norm
embed_scatter_add(dx, tokens → embed_accum)                    // gradien sisi input ditambahkan di atas benih
d_embed = f16(embed_accum)
```
**Weight tying — kontrak urutan (train_step.cuh:323-328):** `d_embed = f16( f32(d_logitsᵀ @ x_final) + scatter_add(dx, tokens) )`.

### Grads (train_step.cuh:188-194; alokasi train_cuda.cu:1299-1316)
Semua FP16 dan mengikuti bentuk parameter: `d_embed` [49152,1280], `d_final_rms` [1280],
per lapisan `dwq` [1280,1280], `dwk/dwv` [1280,320], `dwo` [1280,1280], `dw1/dw3` [1280,3584],
`dw2` [3584,1280], `drms_attn/drms_ffn` [1280]; plus **`embed_accum` [49152,1280] F32** (scratch).

---

<a name="l7"></a>
## L7 — Loop pelatihan pretrain (transformer/train_cuda.cu, 1975 baris)

### 7.1 Pipeline FP16 loss-scaling per step (urutan kritis)
```
d_logits *= loss_scale / tok_count        (4096; tok_count = token tak ber-mask)
backward → (DDP allreduce + average)
hitung NaN/Inf gradien (SEBELUM un-scale — scale_inplace men-nol-kan non-finite!)
  → overflow: skip step, loss_scale × 0.5 (min 1); 2000 step bersih: ×2 (maks 65536)
un-scale semua tensor gradien × 1/loss_scale
l2_norm_sq semua tensor → norma global → jika > 0.5: skala semua gradien × 0.5/(norm+1e-6)
adamw_step per tensor (FP32 master; wd=0 untuk norm gain)
```
**Matematika loss_scale (train_cuda.cu:128-135):** entri d_logits FP16 bertahan hanya jika
`p > 5.96e-8 · tok_count / loss_scale` (5.96e-8 = subnormal minimum FP16).
Dengan B=35,S=256 (tok_count 8.960) dan skala 1 → ambang `p > 5.3e-4`, padahal probabilitas softmax median
model vocab-65536 ~4.5e-6 → **"99.6% gradien hilang diam-diam menjadi nol"**. Skala 4096 menurunkan ambang ke ~1.3e-7.

### 7.2 Keamanan berlapis (mode `light` default)
- **Deteksi spike (1494-1507):** `loss > ema·1.4 && loss > 5.5` → skip step optimizer (ema baseline 0.95;
  spike menetes masuk ema dengan beta 0.999 agar batch buruk tunggal tidak menyeret baseline).
- **NaN/Inf loss:** skip step.
- **GLALR / auto-decay:** hanya mode `adaptive`/`legacy` — mode `light` sengaja mematikannya
  (komentar 1531-1535: ratchet auto-decay adalah penyebab utama degradasi LR lintas sesi).
- **OOM check (1431-1462):** semua error CUDA setelah forward = fatal → upload darurat checkpoint terakhir baik
  → **lewatkan penyimpanan** → exit 2. Alasan: error lengket (mis. `cudaErrorIllegalAddress`) akan membuat
  penyimpanan berikutnya menulis tensor ber-nol menimpa checkpoint baik.
- **Sinyal:** SIGTERM/SIGINT hanya set `volatile bool`; loop keluar di batas iterasi, simpan `_final`, upload akhir.

### 7.3 Jadwal LR — token-based, invarian perangkat (248-280)
```
warmup: lr = peak · tokens/warmup_tok            (warmup = min(max_tokens/100, 20M))
cos  : lr = min + ½(peak−min)(1 + cos(π·progress)),  progress = (tokens−warmup)/(total−warmup)
progress DI-CLAMP [0,1]  (tanpa clamp, cos periodik menaikkan LR lagi saat resume dgn max_tokens lebih kecil)
min = 0.1·peak
```
Progres dihitung dalam **token** (bukan langkah) supaya bentuk kurva tidak berubah saat pindah mesin
(ganti jumlah GPU/batch tidak mengubah posisi kurva — komentar 248-269 mendokumentasikan insiden nyata:
di 763M/2B, run 2×T4 sudah meluruh ke lr 9.4e-5 sementara 1×T4 masih di 1.35e-4 = LR 43% lebih tinggi di titik data sama).

### 7.4 Sumber data & angka (diambil dari konfigurasi hidup saat ini)
`batch 14 × seq 512 = 7.168 token/langkah` (setara Kaggle 2 GPU × batch 7), `lr peak 1e-4`, `clip 0.5`,
`loss_scale 4096` (state menyimpan 8192 hasil auto-scale), `safety light`, `save_every 400`,
`upload_every 3200` (dibulatkan ke kelipatan save_every), `eval tiap 200M token`.

### 7.5 Checkpoint — 503-629
- `ckpt_pretrain.safetensors` — 218 tensor bobot FP16: `embed.weight`, `layer.{l}.{wq..rms_ffn}.weight`, `final_rms.weight`.
- `ckpt_optim.safetensors` — 436 tensor momen m/v FP32.
- `train_state.txt` ditulis **TERAKHIR sebagai penanda commit** (dua safetensors sudah ter-rename atomik).
- Penulisan: header JSON `data_offsets`, padding hingga `(8+header)%64==0`, tulis ke `.tmp` lalu
  `rename()` POSIX atomik (safetensors.h:338-393) — mencegah file 983 MB / 3.9 GB rusak saat crash di tengah tulis.

### 7.6 run_eval — 292-500
Simpan checkpoint sementara + `config_eval.json`; jalankan infer CPU per prompt lewat `popen`
(`--max_tokens 8 --temperature 0.8 --top_p 0.9 --rep_penalty 1.3 --show_probs --show_probs_topk 3`);
parse baris probabilitas `    [` (harus dicek SEBELUM filter metadata, komentar 404-407);
laporkan ke stdout + `eval_log.txt` (perplexity `e^loss`, 5 prompt EN + 5 JA faktual);
upload eval_log ke Drive best-effort (tidak boleh mengganggu training). 10 prompt tetap:
"The capital of Japan is", "the chemical formula of water is", "the largest planet in the solar system is",
"the currency of the United States is", "the color of the sky is", "日本の首都は", "地球の衛星は",
"水の化学式は", "日本の通貨は", "人間の足の数は".

### 7.7 Auto-skala batch/VRAM (1338-1370) — **dorman**
Kode penskalaan `batch = cli·256/seq·0.85` pada pergantian seq_len hanya aktif jika kurikulum multi-fase
dihidupkan; `compute_phases()` (230-236) mengembalikan **satu fase** (`{g_max_seq, total_tokens}`)
→ jalur ini tidak pernah menyala di build ini.

---

<a name="l8"></a>
## L8 — Konfigurasi arsitektur runtime (transformer/arch_runtime.h, model_config.{h,cpp})

- Konstanta arsitektur dahulu `constexpr` di .cu (ubah ukuran = rebuild). Sekarang: `static ModelConfig g_arch`
  + global placeholder `D_MODEL/.../MODEL_VOCAB/MAX_SEQ_LEN/ROPE_THETA` yang **selalu ditimpa `sync_arch()`**
  dari `g_arch` setelah parse CLI, sebelum alokasi CUDA (train_cuda.cu:71-81, 789-790).
- `--arch_config <json>`: muat + validasi (`n_q_heads·d_head == d_model`, `n_q_heads % n_kv_heads == 0`,
  `d_head` genap — "RoPE merotasi pasangan").
- `config_492m.json` ≡ `config_pretrain.json` (isinya identik; hanya format). `config_204m.json`: V=65536,
  D=768, L=18, QH=12, KVH=3, FFN=3072, clip 0.5, bahasa +id.
- `param_count()` (model_config.h:64-77): `embed + L·(wq+wk+wv+wo+w1+w2+w3+2·rms) + final_rms`,
  weight-tying dihitung sekali → **491.582.720**.
- Alokasi parameter (train_cuda.cu:666-716, `mt19937(42)`): embed std 0.02; matriks `s=1/√dim`;
  `w2` `sf=1/√3584`; norm gain = 1. Urutan indeks: `params[0]`=embed; `params[1+l·9+i]` =
  `wq,wk,wv,wo,w1,w2,w3,rms_attn,rms_ffn`; `params[217]`=final_rms. Total **218 tensor**; optimizer 436 (m+v).

---

<a name="l9"></a>
## L9 — Tokenizer BPE (bpe/bpe_tokenizer.cpp, 1.703 baris)

- **Byte-level GPT-2**: tabel `bytes_to_unicode` (0x00-0x20, 0x7F-0xA0, 0xAD → Latin Extended-A;
  `\n`→`Ċ`, spasi→`Ġ`). Bukan level karakter.
- **16 token spesial id 0-15** di depan 256 token byte (basis 272): ``(BOS, id 0), ``/``/`` (id 1-3),
  `<|end|>` (EOS, id 4), `<|code|>`, `<|/code|>`, `<|translate|>`, `<|summary|>`, `<|think|>`(9), `<|/think|>`(10),
  `<|lang_id|>`, `<|lang_en/ja/ko|>`, `<|pad|>`(15). Slot 9/10 dahulu `<|phone_event|>`/`<|app_usage|>`
  — dipakai ulang agar "thinking" berbiaya 1 token per sisi.
- **Encoding (encode, cpp:1615-1674):** (1) BOS opsional; (2) split token spesial dengan lookahead `|>` terbatas
  128 karakter (menghindari O(n²) pada LaTeX `<|`); (3) pretokenize — **run whitespace dipertahankan verbatim**
  + aturan fusi satu-spasi GPT-2; (4) per kata: split substring ≥ 512 byte (backup ke lead-byte UTF-8,
  **tidak pernah truncate** — truncation lama membuang 43% karakter wiki_ja); (5) `encode_chunk` =
  split longest-match awal (8→1) lalu **loop merge fixpoint kiri-ke-kanan** (pasangan pertama yang cocok dimerge,
  diulang sampai tak berubah — bukan BPE rank-strict).
- **Decode**: sambung string, byte-decode lookahead 4 karakter.
- **Pelatihan BPE**: pemilihan pasangan ter-frekuentasi via heap maksimum dengan **invalidasi malas**;
  ambang frekuensi adaptif per kelas (CJK 10, Latin 25, kode 20, default 15); **seed merge paksa**;
  pretokenisasi dokumen utuh (akumulasi 8192 byte) supaya `ĊĊ`/indentasi teramati
  (memberi makan baris tunggal = "kenapa vocab_7 hanya punya 1 token ber-newline"); split kelas aksara JP
  (hiragana/katakana/kanji/CJK-punct) karena kalimat JP utuh = 1 "kata" 80 B vs 5 B untuk EN.
- **Format `bpe_v9.bin`**: `"BP2\0"`, int32 vocab=49.152, token (len+bytes UTF-8), int32 merges=48.880,
  merge = (left_id, right_id). `vocab.txt`: satu token per baris = id. `merges.txt`: header + 48.880 aturan
  (**hanya baris `#version:` yang komentar** — `#` adalah token nyata; parser lama membuang 27 merge nyata
  `#!`/shebang/CJK-comment dan bikin id train/infer meleset).
- **Fingerprint (train_cuda.cu:903-920)**: FNV-1a-64 atas 46 token hasil encode kalimat probe
  (EN+JA+kode+LaTeX) — vocab_7 dan vocab_8 sama-sama 49.152 sehingga ukuran tak bisa membedakan.
  Disimpan di `train_state.txt` dan **dicek keras saat resume**. Nilai v9 teramati: `ef8ebb772c03e8ae`.
- Sep feed tokenizer `encode(text, add_bos=true, add_eos=true)` → dokumen `[BOS … <|end|>]`.

---

<a name="l10"></a>
## L10 — Dataset streaming (transformer/pretrain_dataset.cpp)

- **Deteksi bucket** → bobot: quad (code+math tanpa indonesian) = **en 0.27 / ja 0.30 / code 0.19 / math 0.15 /
  mixed 0.09** (env `TW_W_*` bisa menimpa, tapi hanya di cabang quad); enam-bucket; trilingual; bilingual EN/JA 55/45.
- **Bobot file = bobot bahasa × (ukuran file / total bahasa)** → rasio pilihan file = rasio token terlatih
  (tiap shard menghasilkan jumlah token sama).
- **load_shard**: pilih file terbobot; **seek byte acak** (`rng % (size−512)` lalu buang baris parsial);
  target sampel = `shard_tokens/seq_len` (shard 1M token); baca mode-dokumen (pemisah `` U+0004,
  baris kosong dipertahankan — itu konten kode); setelah seek acak, **lompat ke pemisah berikutnya**
  agar tidak pernah berlatih di setengah dokumen; dokumen > 8192 byte **di-split, tidak di-truncate**
  (potong lama membuang ~88% baris fineweb_edu_extra); packing antar-dokumen back-to-back
  **tanpa EOS ekstra** (kode lama menyisipkan id 0 di tiap batas → `END,0,0,DOK_BARU`); shuffle antar-sekuens di shard.
- **Prefetch: satu thread latar** mengisi antrean 16 shard (bukan pool; 16×1M×4B = 64 MB RAM host).
- **Pencampuran batch: 12 shard aktif (kMixShards) round-robin per sampel** — membunuh osilasi loss
  satu-bahasa-per-batch. **Pensiun bertingkat**: shard diterima di `start_pos` tak-nol (fill pertama menyebar
  merata, lalu jitter acak 25% pertama) supaya shard tidak habis serentak — pensiun serentak menimbulkan
  anak tangga EMA ~0.5 nat tiap 683 langkah (komposisi data, bukan divergensi).
- **Dua RNG terpisah**: `rng_(seed)` (thread prefetch) dan `mix_rng_(seed ^ 0x9E37...)` (thread training) —
  berbagi satu = data race.
- **Seed dataset harus maju**: `seed = 42 + (tokens/1M)·1000 (+rank)` — seed tetap membuat tiap resume
  memutar ulang aliran file/offset identik ("loss turun 200× lebih cepat dari budget — memorisasi, bukan belajar").
- `set_seq_len` membuang shard basi (stride lama) dan restart admission bertingkat.
- `dataset_check` (tools/dataset_check.cpp) menguji dua kegagalan senyap itu: ekor batch tak tertulis
  (buffer pra-isi 0xFFFFFFFF) dan rapat BOS > 25% (indikasi satu-baris-satu-dokumen).

---

<a name="l11"></a>
## L11 — Trainer SFT / DPO / Phone / RL (transformer/train_cuda_unified.cu, `--stage 2..5`)

Semua tahap memakai mesin yang sama seperti pretrain: `ts::forward/backward`, CE fused, weight-tie, klip L2 global.

### SFT (stage 2, `train_step_ce` 596-650)
- Data JSONL `{"text": ...}` (unescape manual agar kode multi-baris selamat); teks utuh satu giliran chat
  berakhir `<|end|>`; `encode(text, bos=true, end=!sudah_ada_end)`.
- **Mask (build_assistant_mask, 212-236):** `mask[i]` memboboti prediksi `tokens[i+1]`.
  `mask=1` **di token `` itu sendiri** — targetnya token jawaban pertama (jika 0, token jawaban pertama
  tidak pernah dilatih — fatal untuk terjemahan); 1 di seluruh jawaban; **0 di/	setelah `<|end|>`**
  (kapan berhenti dipelajari di token *sebelum* `<|end|>` yang targetnya memang `<|end|>`); 0 untuk system/user.
- Loss = mean atas token tak ber-mask; `d_logits × 1/tok_count`; weight decay 0.01.

### DPO (stage 3, 1294-1524)
```
raw_margin = (lp_c − rlp_c) − (lp_r − rlp_r)        // lp = log-prob per-token rata-rata (masked_log_probs)
margin     = clamp(raw_margin, −5, +5)              // dpo_max_margin
loss       = log(1 + e^(−β·margin))                 // = −log σ(β·Δ)
gradien    : koefisien chosen = +β·σ(−β·margin);  rejected = −β·σ(−β·margin)
             → dua backward CE penuh (chosen ke st.grads, rejected ke r_grads lalu dijumlah elemen-demi-elemen)
```
Model referensi = checkpoint yang sama dengan inisialisasi policy (pipeline default), beku.
β default 0.1. wd 0.005.

### Phone (stage 4)
Identik dengan loop SFT (data `sft_data/phone`), hanya `lr × 0.3` ("konservatif"). Full-model, bukan LoRA
(lihat [L13](#l13) — LoRA mati).

### RL "GRPO-lite" (stage 5, 1526-1865)
- P prompt × N=6 completions; generasi inkremental (prefiks di-upload ulang, forward penuh, baris logit
  posisi-terakhir dikumpulkan `gather_logits_rows_kernel`), sampling host top-k 50, temp 0.8.
- **Verifier eksternal**: tulis `rl_work/req.jsonl` {id, lang, code, input, expected} → `system()`
  `scripts/rl_verifier.py` → baca `reward.jsonl`. Verifier: reward biner; mode assert/code/math;
  kode diekstrak dari fence ```python, dijalankan dengan stdin, **plus 3 instance uji segar** dari bank tugas
  (anti lucky-pass); matematika pakai jangkar CoT "the answer is|jawabannya adalah|答えは"; timeout 5-10 s;
  blok `<|think|>` dibuang sebelum evaluasi.
- **Keunggulan**: group-normalized `A = (r − mean)/(std + 1e-8)` per prompt atas N completions-nya.
- **Penalti KL (estimator kl-1, komentar 1795-1799):** `r_t = loss_pol − loss_ref` per token;
  `scale_t = (A − β·(1 − e^{r_t})) / N_t` diterapkan per baris (`launch_scale_rows_fp16`) sebelum backward —
  gradien estimators `e^r − r − 1` menyumbang `−β(1−e^r)`.
- wd 0.005; hanya upload final (tidak periodik).

### `sched_lr` unified (672-686)
Warmup linear `(step+1)/warmup`; cosine dengan progress **di-clamp [0,1]**; lantai `max(lr, 0.1·peak)`.
`total_steps` dihitung dari **batch global** (B×world) sehingga bentuk cosine invarian jumlah GPU (1204-1213).

---

<a name="l12"></a>
## L12 — Format file

### safetensors (transformer/safetensors.h)
- 8 byte panjang-header (LE) → JSON `{"nama":{"dtype","shape","data_offsets"}}` → data rapat.
- Header dipad spasi hingga `(8+len) % 64 == 0`.
- Writer: tulis `.tmp` → `rename()` atomik. Konversi BF16 round-to-nearest-even.
- Loader `MiniParser`: parser JSON tulisan tangan khusus subset safetensors (string, int, array int,
  `__metadata__` dilewati). `read_into_f32`: BF16 → shift-kiri-16; F16 → konversi manual IEEE-754 lengkap
  dengan subnormal. `read_into_fp16`: F32 → konversi manual (rebias eksponen −127+15, underflow→0 bertanda,
  overflow→inf).

### train_state.txt (pretrain)
```
tokens_processed, step, phase, ema_loss, ema_local_min, lr_decay_factor, safety_mode,
lr, grad_clip, gnorm_ema, lr_adaptive_factor, loss_scale, tok_fingerprint
```
Resume memuat semuanya + optimizer + membangun ulang master FP32 + cek fingerprint tokenizer (mismatch = gagal keras).
Varian unified: `train_state_<stage>.txt` + `last_ckpt=`.

---

<a name="l13"></a>
## L13 — Orkestrasi & alat bantu

### train_pipeline.py (417 baris)
- Tahap: `pretrain|sft|dpo` → bangun perintah biner `build/train_pretrain_cuda` / `build/train_sft_cuda --stage 2|3`,
  dijalankan dengan `stdbuf -oL`, `CUDA_VISIBLE_DEVICES` dibuang dari env anak.
- `--resume` otomatis jika `train_state*.txt` ada. Penanda selesai: `ckpt_*_final.safetensors`
  (`skip_done` aktif default).
- Deteksi Kaggle: checkpoint dir → `/kaggle/working/checkpoints`, **DDP otomatis 2 GPU**, batch 24→35,
  env anak `NCCL_P2P_DISABLE=1`. Di luar Kaggle: single GPU, tanpa DDP.
- `auto_continue` efektifnya knob mati (cabang 392 tidak pernah tercapai: pipeline selalu lanjut saat sukses,
  dan berhenti saat gagal terlepas dari flag).
- Pencarian base model DPO glob `ckpt_sft_step*` tidak pernah cocok (penamaan aktual `ckpt_sft.safetensors`)
  — jatuh ke jalur `--ref_model`/base eksplisit.

### grad_check (cuda/grad_check.cu, target opt-in)
Empat uji, ambang cos > 0.99 & relerr < 0.02 (reduksi: eksak):
1. **Reducer eksak**: RMSNorm x=ones → `D/rsqrt²` harus bulat D (D∈{64..3072}) — menangkap bug 384/768.
2. **RMSNorm backward**: gamma **tak-seragam** Uniform(0.6,3.2), dy dikorelasikan dengan normed
   (dy iid membuat galat bug turun ke 0.015 — lolos ambang; dikorelasikan → 0.34 — ketahuan);
   referensi = beda hingga (finite-difference) FP64; varian "gamma di luar" dihitung dan dicetak sebagai peringatan.
3. **SwiGLU backward**: analitik vs referensi elemen-wise.
4. **RoPE round-trip**: `backward(forward(v)) == v` (Rᵀ·R = I) — bebas toleransi tanda/pasangan.
   Motivasi header: dua bug nyata bertahan ~37 ribu langkah sebelum harness ini dibuat.

### infer CPU (tools/infer.cpp)
REPL/`--prompt`/`--stdin`. Sampling: **penalti repetisi** jendela 64 (`p = 1.3 + (count−1)·0.2`, logit positif
dibagi, negatif dikali) → greedy (default temp 0 — komentar: 0.8 menghasilkan sampah interaktif) atau
softmax-temperatur → **top-k** (partial_sort + renormalisasi) → **top-p** (mempertahankan token pemotong ambang —
varian lama men-nol-kan semuanya saat p[0]>top_p) → multinomial; fallback greedy bila distribusi ≤ 1e-8.
Baris probabilitas ditulis ke **stderr** dengan format `    [tok 45.4%, ...]` (diikuti endl stdout agar tak
menempel ke teks yang mengalir — dan inilah yang diparse `run_eval`). BOS ditambahkan kalau prompt belum
berawalan `` (konsisten training yang membungkus `[BOS …]`).

### transformer.cpp (referensi CPU)
Decode inkremental per token dengan KV-cache. **Sengaja meniru GPU secara eksak**: RoPE `freq = pos/θ^(2j/dh)`
+ pasangan interleaved identik `rope_forward_kernel`; eps 1e-6; GQA `hkv = hq/4`; swiglu sama; logit weight-tied.
Bobot host FP32 (konversi otomatis dari F16/BF16 saat load).

### LoRA — **kode mati**
`lora.cpp/h` tidak masuk build (CMakeLists transformer: komentar eksplisit), `train` stub, `merge_into`
dideklarasikan tanpa definisi (link error bila dipakai). Stage 4 phone = full fine-tune.

### cuda_autograd.cu — **kuburan**
`init_grads` tanpa pemanggil; `training_backward` dan `flash_attention` kini stub `abort()`
(gradien lama SALAH — identity/outer-product; attention lama menyalin Q ke keluaran). Digantikan `ts::`/`gqa_attn_*`.

### upload_gdrive.py
OAuth2 refresh-token; unggah/UPDATE file bernama sama di folder Drive (`checkpoint_latest.zip` untuk pretrain);
mode `--zip`: snapshot (hard-link — 0 byte tambahan dan konsisten terhadap rename atomik trainer) →
zip (di disk via `CKPT_UPLOAD_TMP`) → unggah resumable 25 MB/chunk, retry 2×.
Folder Drive dan client ID di-hardcode; token di `gdrive_token.json`.

### vast_deploy/ (alat deploy GPU sewaan)
`.env` (koneksi SSH, GPU_MODEL, parameter) → `deploy.sh all` = kirim source+korpus+checkpoint → build dengan
**arsitektur deteksi-otomatis dari nvidia-smi, divalidasi terhadap tabel 88 GPU Vast** (`tools/gpu_info.py`,
kolom sm = field `compute_cap` penawaran Vast) → grad_check wajib lulus → start tmux →
`monitor` (log gaya Kaggle) / `panel` / `pull` (ringkasan tok/s ke `speed_monitoring_gpu/`).
Patch source saat `package`: arena 0.85, snapshot hard-link upload, banner GPU asli.

---

<a name="l14"></a>
## Ringkasan lintas-lapis

### Peta presisi
| Tensor | Penyimpanan |
|---|---|
| Aktivasi, parameter, gradien | FP16 |
| Master Adam, m/v, scratch dQ/dK/dV, scratch dgamma, embed_accum, LSE, loss per-token, penghitung | FP32 |
| Aritmetika kernel | FP32 setelah promosi (kecuali add/mul/sqrt half native) |
| cuBLAS | FP16 masuk/keluar, akumulasi FP32 (`COMPUTE_32F_FAST_16F`) |
| Attention hot path | `__expf`/`__logf`; skor/softmax/akumulasi FP32 |

### Bug historis yang membentuk desain (semua terdokumentasi di komentar kode)
1. `block_reduce_sum` `s > 32` — separuh reduksi hilang (grad_check TEST 1).
2. RMSNorm backward gamma di luar mean — hanya terlihat dengan gamma tak-seragam (TEST 2).
3. CE lama menulis gradien ke buffer FP16 secara ilegal → buffer overflow (kernel fused baru).
4. Pensiun shard serentak → anak tangga EMA 0.5 nat periodik (staggered retirement).
5. Seed dataset tetap → resume memutar ulang data identik → "memorisasi, bukan belajar".
6. Truncation dokumen → membuang 19-88% karakter JA (split + backup UTF-8).
7. Parser merges membaca `#` sebagai komentar → 27 merge nyata hilang → id train/infer meleset.
8. Keputusan skip-step terbelah antar-rank → NCCL hang tanpa watchdog (konsensus `allreduce_scalar_max`).
9. Loss-scale rendah → 99.6% gradien FP16 flush ke nol (matematika ambang di L7.1).
10. Jadwal LR berbasis langkah → bentuk kurva berubah antar mesin (insiden 43% perbedaan LR → jadwal token).
11. Varian top-p lama men-nol-kan distribusi saat p[0] > top_p (token 65535).
12. `scale_inplace` men-nol-kan non-finite → bukti overflow musnah → penghitungan NaN wajib sebelum un-scale.
13. Error CUDA lengket → penyimpanan berikutnya menulis nol menimpa checkpoint baik (exit 2 + upload darurat).
14. Arena 0.93 VRAM → `cudaMalloc` OOM setelah 53 menit di RTX 4090 (instance; dipatch 0.85).

### Ukuran terukur (sesi Vast RTX 4090, sm_89, batch 14×512)
- 9.4–9.9K tok/s stabil (2×T4 Kaggle: 2.9K; batch 8 di 4090: 12.8K tapi gnorm naik 6.2-6.7).
- Profil nsys: GPU kernel-busy ~64%; **`gqa_attn_bwd` = 51% waktu GPU**; papan kedua `cudaFree` (70% waktu API CUDA)
  akibat luapan arena → diperbaiki dengan arena 0.93→lalu 0.85.
- VRAM kerja 21–22.8 GB / 24 GB. Checkpoint 4.49 GB ter-zip; upload Drive terverifikasi md5 byte-per-byte.

### Batas yang diketahui
- Attention: 1 warp per query, tanpa staging KV di shared memory, tanpa vektorisasi — kandidat optimasi utama
  (51% waktu GPU); embedding gather loop serial per thread.
- `copy_stream`/`loss_stream` dibuat tapi tak terpakai; `transpose()` view-only (jangan diindeks dengan asumsi layout lama).
- Jalur DDP multi-proses (`launch_ddp.sh`) punya bug pertukaran ncclUniqueId; jalur fork (Kaggle) yang dipakai pipeline.
- `apply_adamw`/`AdamWState`, `lora.*`, `cuda_autograd` = kode mati; varian CE lama hanya referensi.
- Kurikulum multi-fase & auto-skala batch: ada kodenya, dorman (single-phase build).
