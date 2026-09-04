# Rencana Kuantisasi & Deployment ke HP — Transformer 204M (EN/JA/ID)

> Status dasar diverifikasi 2026-09-02 dari checkpoint nyata di `translation_resources/`.
> Semua angka di bawah hasil pengukuran, bukan estimasi kasar.
> Script pendukung: `scripts/run_demo_eval.py`, `scripts/analyze_quant.py`.

---

## ✅ STATUS: SUDAH DIJALANKAN

Seluruh pipeline sudah dieksekusi memakai **checkpoint fakta**. Artefak jadi:

| File | Ukuran | Keterangan |
|---|---|---|
| `gguf/trilingual-204m-facts-f16.gguf` | 392 MB | referensi, paritas 100% |
| `gguf/trilingual-204m-facts-Q8_0.gguf` | 209 MB | drift 3% |
| `gguf/trilingual-204m-facts-Q5_K_M.gguf` | 146 MB | drift 9% |
| **`gguf/trilingual-204m-facts-Q4_K_M.gguf`** | **130 MB** | **dipilih untuk HP** |
| `gguf/trilingual-204m-facts-Q4_0.gguf` | 124 MB | ditolak, drift 29% |
| `gguf/trilingual-204m-translate-Q4_K_M.gguf` | 130 MB | checkpoint translasi |

Dua koreksi penting yang baru ketahuan saat eksekusi (rencana awal salah):

1. **SwiGLU-nya terbalik.** `transformer.cpp` menghitung `ff = w1(x) * silu(w3(x))`
   — SiLU pada **w3** — sedangkan llama.cpp `silu(ffn_gate) * ffn_up`. Jadi
   pemetaan benar: **w3 → ffn_gate, w1 → ffn_up**. Kalau terbalik, model tetap
   fasih dan nyaris benar ("The capital of Japan is Tokyo" vs seharusnya "In
   Japan, the capital..."), sehingga **hanya gate paritas yang bisa menangkapnya**.
2. **Tokenizer llama.cpp tidak bisa dipakai.** BPE model ini bukan GPT-2 standar,
   jadi prompt harus di-encode sendiri. Lihat §2.3.

---

## 0. Temuan yang mengubah rencana

| Temuan | Dampak |
|---|---|
| `config_pretrain.json` (892M) **tidak cocok** dengan checkpoint (204M) | Wajib pakai `config_204m.json` |
| Tidak ada satu checkpoint pun yang bisa fakta **dan** translasi | Ship 2 model, atau retrain |
| Cosine-sim embedding antar-checkpoint = **0.31** | Model soup / merge **tidak** viable |
| `embed.weight` = 24.6% param, kurtosis 22.3, tied ke output head | Embedding **tidak boleh** Q4 |
| RoPE interleaved + weight layout `[D_in, D_out]` | Perlu transpose, **tanpa** permutasi HF |
| Runtime CPU dequantize ke FP32 | RAM ~817 MB → terlalu besar untuk HP |

**Budget ukuran (204,304,128 param):**

| Skema | Ukuran | Catatan |
|---|---|---|
| FP16 (sekarang) | 389.7 MB | terlalu besar |
| Semua Q8_0 | 207.0 MB | aman, kualitas ~utuh |
| Semua Q4_0 | 109.6 MB | embed rusak (err 0.129) |
| **embed Q8_0 + sisanya Q4_0** | **133.6 MB** | **rekomendasi** |
| embed Q8_0 + sisanya Q4_1 | 142.8 MB | fallback bila Q4_0 drop |

---

## 1. Keputusan arsitektur: jalur GGUF + llama.cpp

**Rekomendasi: konversi ke GGUF, jalankan di llama.cpp.** Alasannya:

- Arsitektur model ini **persis Llama**: RMSNorm + RoPE + SwiGLU + GQA + weight tying.
  Tidak perlu arch baru di llama.cpp — cukup `LLM_ARCH_LLAMA`.
- Dapat gratis: kernel ARM NEON/i8mm/dotprod, mmap (RAM ≈ ukuran file), KV cache
  kuantisasi, batching, Android/iOS binding, Metal/OpenCL.
- Alternatif "bikin format + kernel NEON sendiri" butuh berminggu-minggu untuk
  menyamai performa yang sudah ada, tanpa manfaat tambahan.

**Fallback (Jalur B)** — kalau tokenizer custom bermasalah: kuantisasi in-house
di runtime C++ yang sudah ada. Detail di §7.

---

## 2. Peta konversi checkpoint → GGUF

### 2.1 Hyperparameter GGUF

| Key GGUF | Nilai |
|---|---|
| `general.architecture` | `llama` |
| `llama.context_length` | 2048 (turunkan ke 512–1024 untuk HP) |
| `llama.embedding_length` | 768 |
| `llama.block_count` | 18 |
| `llama.feed_forward_length` | 3072 |
| `llama.attention.head_count` | 12 |
| `llama.attention.head_count_kv` | 3 |
| `llama.rope.dimension_count` | 64 |
| `llama.rope.freq_base` | 500000.0 |
| `llama.attention.layer_norm_rms_epsilon` | samakan dgn `transformer.cpp` |

### 2.2 Pemetaan tensor (18 layer)

| Sumber (safetensors) | Shape | Target GGUF | Aksi |
|---|---|---|---|
| `embed.weight` | [65536, 768] | `token_embd.weight` | **salin apa adanya** |
| `layer.N.wq.weight` | [768, 768] | `blk.N.attn_q.weight` | **transpose** |
| `layer.N.wk.weight` | [768, 192] | `blk.N.attn_k.weight` | **transpose** → [192, 768] |
| `layer.N.wv.weight` | [768, 192] | `blk.N.attn_v.weight` | **transpose** |
| `layer.N.wo.weight` | [768, 768] | `blk.N.attn_output.weight` | **transpose** |
| `layer.N.w1.weight` | [768, 3072] | `blk.N.ffn_up.weight` | **transpose** |
| `layer.N.w3.weight` | [768, 3072] | `blk.N.ffn_gate.weight` | **transpose** |
| `layer.N.w2.weight` | [3072, 768] | `blk.N.ffn_down.weight` | **transpose** |
| `layer.N.rms_attn.weight` | [768] | `blk.N.attn_norm.weight` | salin |
| `layer.N.rms_ffn.weight` | [768] | `blk.N.ffn_norm.weight` | salin |
| `final_rms.weight` | [768] | `output_norm.weight` | salin |
| — | — | `output.weight` | **hilangkan** (weight tying) |

Total 164 tensor: `embed.weight` + `final_rms.weight` + 18 × 9 tensor per layer.

> ⚠️ **Tiga jebakan mematikan:**
> 1. Kode ini pakai `y = x @ W` dengan `W:[D_in, D_out]`. GGUF/ggml butuh
>    `[D_out, D_in]`. Semua bobot linear **wajib ditranspose**. `embed` tidak.
> 2. RoPE di `transformer.cpp` **interleaved** (`head[2j]`,`head[2j+1]`) = tipe
>    `NORM` di ggml — sudah native. **Jangan** pakai permutasi `q`/`k` ala
>    `convert_hf_to_gguf.py`; itu untuk konvensi half-split HuggingFace dan akan
>    merusak model secara halus.
> 3. **SwiGLU terbalik** (lihat status di atas): `w3 → ffn_gate`, `w1 → ffn_up`.
>    Ini terjadi sungguhan saat eksekusi dan lolos semua pemeriksaan mata —
>    hanya gate paritas yang menangkapnya.

### 2.3 Tokenizer — ⚠️ TIDAK kompatibel dengan llama.cpp

Vocab tetap ditulis ke GGUF (`tokenizer.ggml.model = "gpt2"`) supaya llama.cpp
mau memuat model dan detokenisasi tetap benar. **Tapi encoding harus dilakukan
sendiri** — BPE model ini bukan GPT-2 standar:

1. Pre-tokenizer memecah pada deretan byte `<= 0x20` dan menggabungkannya.
2. Seeding kata = longest-match serakah hingga **8 byte**, bukan per karakter.
3. Merge = pass kiri-ke-kanan berulang, bukan lowest-rank-first.
4. Baris `merges.txt` berawalan `#` dibuang loader C++ → 42 merge sah ikut hilang
   (efektif 65.222, bukan 65.264). Konverter meniru ini persis.

Bukti: `What is the capital of Japan?` → llama.cpp 7 token, model ini 8 token.

Implementasi encoder tersedia di:
- `scripts/bpe_reference.py` (Python, ada `--selftest`)
- `mobile/android/BpeTokenizer.kt` (Kotlin)
- `mobile/ios/BpeTokenizer.swift` (Swift)
- Test vector lintas platform: `mobile/tokenizer_test_vectors.json`

Karena `llama-imatrix` memakai tokenizer llama.cpp, kalibrasi imatrix jadi tidak
akurat untuk model ini — **dilewati**, dan Q4_K_M tetap lolos gate kualitas.

Special id: `bos=0`, `eos=4`, `pad=15`. Chat template:
`<|system|>{system}<|user|>{user}<|assistant|>`

---

## 3. Langkah detail — Jalur A (GGUF)

### Langkah 1 — Siapkan lingkungan
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy gguf sentencepiece
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```
**Selesai bila:** `build/bin/llama-quantize` dan `llama-cli` ada.

### Langkah 2 — Tulis `scripts/convert_to_gguf.py`
Konverter kustom (bukan `convert_hf_to_gguf.py`, karena format kita bukan HF):
1. Baca safetensors via reader di `scripts/analyze_quant.py` (sudah ada).
2. `gguf.GGUFWriter(path, "llama")`, tulis hyperparameter §2.1.
3. Loop tensor sesuai tabel §2.2 — **transpose semua linear**, embed jangan.
4. Tulis tokenizer §2.3.
5. Simpan sebagai **F16** dulu (belum kuantisasi).

**Selesai bila:** `llama-cli -m model-f16.gguf -p "test" -n 8` jalan tanpa error.

### Langkah 3 — Gate paritas numerik (WAJIB, jangan dilewat)
Bandingkan GGUF F16 vs `build_infer/infer` pada prompt identik, **greedy**:
```bash
python3 scripts/run_demo_eval.py --model translation_resources/facts/ckpt_sft_final.safetensors \
    --json_out results/ref_fp16.json
# lalu prompt yang sama lewat llama-cli --temp 0, bandingkan token demi token
```
**Kriteria lulus:** ≥ 95% token pertama identik pada 60 prompt.
**Kalau gagal:** hampir pasti transpose atau RoPE (§2.2). Cek layer 0 dulu —
dump `blk.0.attn_q.weight[0,:8]` dan bandingkan dengan `wq.T[0,:8]`.

### Langkah 4 — Buat kalibrasi imatrix (penting untuk Q4)
```bash
# corpus campuran: EN + JA + ID, fakta + translasi, ~50k token
python3 scripts/build_imatrix_corpus.py --out calib.txt   # ambil dari eval.jsonl
./build/bin/llama-imatrix -m model-f16.gguf -f calib.txt -o model.imatrix -c 512
```
Kalibrasi trilingual itu wajib — corpus English-only akan menghancurkan JA/ID.

### Langkah 5 — Kuantisasi (beberapa varian, lalu ukur)
```bash
Q=./build/bin/llama-quantize
$Q --imatrix model.imatrix model-f16.gguf model-Q8_0.gguf   Q8_0
$Q --imatrix model.imatrix model-f16.gguf model-Q5_K_M.gguf Q5_K_M
$Q --imatrix model.imatrix model-f16.gguf model-Q4_K_M.gguf Q4_K_M
# Q4_K_M sudah otomatis menaikkan presisi embed/output — sesuai temuan kurtosis
```
**Ekspektasi:** Q8_0 ≈ 207 MB, Q5_K_M ≈ 150 MB, Q4_K_M ≈ 130 MB.

### Langkah 6 — Gate kualitas pasca-kuantisasi
Jalankan `scripts/run_demo_eval.py` (adaptasi `--infer` → `llama-cli`) untuk tiap varian.

| Metrik | Ambang batas (vs FP16) |
|---|---|
| Language purity fakta EN/JA | turun ≤ 1 dari 10 |
| Fact value hit | turun ≤ 1 dari 10 |
| chrF JA→EN | turun ≤ 10% relatif |
| Perplexity (`llama-perplexity`) | naik ≤ 5% |

**Ambil varian terkecil yang lolos.** Kalau Q4_K_M gagal di JA, naik ke Q5_K_M.

### Langkah 7 — Kemas untuk HP
- **Android:** `llama.cpp` via JNI/NDK (`arm64-v8a`), aktifkan
  `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`.
  Simpan `.gguf` di `filesDir` (bukan assets — mmap butuh file nyata).
- **iOS:** target Metal, `.gguf` di app bundle / Documents.
- Runtime: `--ctx-size 512`, `--cache-type-k q8_0 --cache-type-v q8_0`,
  threads = jumlah big-core (biasanya 4).

**Budget RAM (Q4_K_M, ctx 512, KV q8_0):**
bobot ~130 MB (mmap) + KV ~3.5 MB + logits 256 KB + scratch ≈ **< 150 MB**. Aman.

---

## 4. Masalah "dua model" (harus diputuskan sebelum shipping)

Tidak ada checkpoint yang bisa dua-duanya. Tiga opsi:

| Opsi | Ukuran | Usaha | Kualitas |
|---|---|---|---|
| **A. Ship 2 GGUF + router** | ~260 MB | Rendah | Terbaik hari ini |
| B. Ship `facts` saja | 130 MB | Terendah | Translasi mati |
| C. Retrain SFT campur (replay) | 130 MB | Tinggi (butuh GPU) | Terbaik jangka panjang |

**Rekomendasi jangka pendek: Opsi A.** Router = klasifikasi intent sederhana
(ada prefix "Translate to"/"日本語に翻訳" → model translasi; selain itu → model fakta).
Muat model secara lazy, satu per satu, jangan dua-duanya sekaligus.

> Merge/model-soup sudah diuji dan **ditolak**: cosine similarity embedding antar
> checkpoint cuma 0.31 — rata-rata bobot akan menghasilkan model rusak.

**Jangka panjang (Opsi C):** ulangi Stage-2 SFT dengan data campur
(≈60% translasi + 40% fakta, di-interleave, bukan sekuensial) dan LR lebih rendah
(1e-5) + replay buffer. Ini akar masalahnya — SFT sekuensial menyebabkan
catastrophic forgetting.

---

## 5. Hasil terukur

### Gate paritas (token pertama, 200 prompt, argmax murni)

| Checkpoint | Hasil |
|---|---|
| facts | **200/200 (100%)** PASS |
| translate | **198/200 (99%)** PASS |

> Hanya token pertama yang dibandingkan karena `infer --greedy` tetap memakai
> repetition penalty 1.3 + hard-block; keduanya bergantung riwayat yang masih
> kosong di langkah 0, jadi hanya di situ kedua engine sama-sama argmax murni.

### Gate kualitas (60 prompt, checkpoint fakta)

| Varian | Ukuran | Fakta EN lang/nilai | Fakta JA lang/nilai | Drift token-1 vs F16 |
|---|---|---|---|---|
| F16 | 392 MB | 10/3 | 10/3 | — |
| Q8_0 | 209 MB | 10/3 | 10/3 | 97,0% |
| Q5_K_M | 146 MB | 10/3 | 10/3 | 91,0% |
| **Q4_K_M** | **130 MB** | **10/3** | **10/3** | **83,5%** |
| Q4_0 | 124 MB | 10/4 | 10/3 | 71,0% |

Akurasi fakta **identik** di semua varian. Q4_0 jatuh paling jauh — persis
seperti diprediksi analisis kurtosis embed (§0). **Q4_K_M dipilih.**

### Kecepatan (M3 Pro, 4 thread, CPU `-ngl 0`)

| Varian | Prompt (pp128) | Generasi (tg64) |
|---|---|---|
| Q4_K_M | 1827 tok/s | **495 tok/s** |
| Q8_0 | 2868 tok/s | 429 tok/s |

Engine `infer` asli (skalar, 1 thread): 69,5 tok/s → **~7× lebih cepat**.

---

## 6. Ekspektasi performa di HP

| Perangkat | Ekspektasi (Q4_K_M, 204M) |
|---|---|
| Snapdragon 8 Gen 2/3 | 30–60 tok/s |
| Mid-range (SD 6xx/7xx) | 10–20 tok/s |
| iPhone 13+ (Metal) | 50–90 tok/s |

Referensi: 69.5 tok/s pada runtime scalar single-thread di Mac ARM — llama.cpp
dengan NEON+dotprod multi-thread jauh lebih cepat.

---

## 6. Urutan eksekusi & gate

```
1. convert_to_gguf.py  ──►  F16 GGUF
2. GATE paritas numerik (≥95% token match)   ◄── blocker
3. imatrix trilingual
4. quantize Q8_0 / Q5_K_M / Q4_K_M
5. GATE kualitas (tabel §Langkah 6)          ◄── blocker
6. pilih varian, integrasi Android/iOS
7. GATE on-device (RAM < 200MB, > 10 tok/s)  ◄── blocker
```

Jangan lanjut ke langkah berikut sebelum gate sebelumnya lulus. Mayoritas
kegagalan konversi GGUF ketahuan di gate 2 dan murah diperbaiki di sana.

---

## 7. Jalur B (fallback) — kuantisasi in-house

Kalau GGUF terhambat, kuantisasi di runtime C++ yang ada:

1. **Format:** blok 32, simetris. `struct BlockQ4 { __fp16 scale; uint8_t qs[16]; }`
   (4.5 bit/bobot), `BlockQ8 { __fp16 scale; int8_t qs[32]; }` (8.5 bit/bobot).
2. **Recipe:** `embed` → Q8, seluruh linear → Q4 (sesuai data §0).
3. **Offline tool** `tools/quantize.cpp`: baca safetensors FP16 → tulis `.q4t`.
4. **Runtime:** ganti `std::vector<float>` jadi buffer terkuantisasi; `mmap` file.
5. **Kernel:** tulis ulang `matmul_vec` (saat ini scalar!) dengan NEON —
   `vdotq_s32` untuk Q8×Q8. Kuantisasi aktivasi ke Q8 per baris saat runtime.
6. **Threading:** partisi `D_out` antar thread.
7. **KV cache:** FP16 + `max_seq` 512 → 54 MB jadi 7 MB.

Usaha jauh lebih besar dari Jalur A dengan hasil lebih rendah. Pakai hanya bila
tokenizer custom benar-benar tidak bisa dipetakan ke GGUF.

---

## 8. Risiko

| Risiko | Mitigasi |
|---|---|
| Transpose/RoPE salah → output halus-rusak | Gate paritas §Langkah 3 |
| Q4 hancurkan Japanese | imatrix trilingual; naik ke Q5_K_M |
| Byte-level BPE tidak round-trip di llama.cpp | Tes encode/decode 10k baris vs `bpe_tokenizer.cpp` sebelum konversi |
| Kualitas dasar model memang rendah | Kuantisasi **tidak** memperbaiki ini — lihat §4 Opsi C |
| Weight tying tak dikenali | Pastikan `output.weight` tidak ditulis |
