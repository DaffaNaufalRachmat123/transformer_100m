# SESSION HANDOFF & COMPLETE MEMORY — `transformer_500m_cuda`

> **Terakhir diperbarui:** 27 November 2026  
> **Lokasi Proyek:** `/Users/daffanaufalrachmat/Documents/Real Projects/TransformerSample/transformer_500m_cuda`  
> **Tujuan Dokumen:** Ringkasan memori utuh lintas sesi agar sesi baru dapat langsung melanjutkan pekerjaan tanpa kehilangan konteks arsitektur, *patch* C++/CUDA, lokasi dataset `10B` (`v8`) vs `50B` (`v9`), maupun konfigurasi Kaggle & Vast.ai.

---

## 1. Status Proyek & Struktur Direktori

1. **Rename & Pembersihan Workspace:**
   - Folder proyek telah di-*rename* dari `transformer_100m_cuda` menjadi **`transformer_500m_cuda`** (seluruh referensi di `CMakeLists.txt`, `scripts/`, `vast_deploy/`, dan `running_commands/` sudah diperbarui).
   - Di dalam parent `/Users/daffanaufalrachmat/Documents/Real Projects/TransformerSample/`, hanya disisakan 3 folder proyek aktif: `transformer_500m_cuda`, `transformer_audio_module`, dan `transformer_vision_650m` (beserta dokumen referensi top-level).
   - Seluruh artefak usang (`vocab_7`, `vocab_8`, SFT/DPO lama yang bocor/rusak ~8.6 GB, serta *cache* sementara HuggingFace ~11 GB) telah dihapus. Sisa ruang kosong SSD MacBook terjaga aman di **`~23 GB` (`~21 GiB`)**.

2. **Status Live Training Pengguna Saat Ini (Kaggle 2× Tesla T4 `14.6 GB`, DDP ×2):**
   - Sedang menjalankan **Stage 1 Pre-training (`10.00B` token budget)** pada `Batch: 7 | Seq: 512 | LR: 1.0e-4 | grad_clip: 0.5 | safety: light`.
   - Checkpoint terakhir yang dilaporkan pengguna: `step=646800, tokens=4.63B / 10.00B, ema_loss=2.4066`.
   - Tokenizer aktif di Kaggle sudah cocok 100%: `bpe_v9.bin (vocab=49152, merges=48880)`, **`fingerprint=ef8ebb772c03e8ae`**.

---

## 2. Spesifikasi Arsitektur & Kompatibilitas Lintas Stage

Seluruh file konfigurasi (`config_500m.json`, `config_492m.json`, `config_pretrain.json`, dan `transformer/model_config.h`) telah diverifikasi **100% identik**:

| Parameter | Nilai Terkunci | Catatan Kompatibilitas |
| :--- | :--- | :--- |
| **`params` / Tensors** | **`491.58M` (`491.6M`)** / `218` tensor | `983.2 MB` dalam FP16 (`~310 MB` dalam `Q4_K_M`). |
| **`d_model` / `n_layers`** | `1280` / `24` | Arsitektur `GQA + RoPE + RMSNorm + SwiGLU`. |
| **`n_q_heads` / `n_kv_heads` / `d_head`** | `20` / `5` (rasio 4:1) / `64` | `q_dim = 1280`, `kv_dim = 320`. |
| **`ffn_dim`** | `3584` | `2.8 × d_model`. |
| **`vocab_size` & `weight_tying`** | `49,152` & `true` | `vocab_9/` + `bpe_v9.bin` (`fingerprint=ef8ebb772c03e8ae`). |
| **`max_seq_len` & `rope_theta`** | `2048` & `500000.0` | Tabel RoPE di-*precompute* sampai `2048` di semua stage. |
| **Special Tokens (`0..15`)** | `0:<\|endoftext\|>`, `1:<\|system\|>`, `2:<\|user\|>`, `3:<\|assistant\|>`, `4:<\|end\|>`, `7:<\|translate\|>`, `9:<\|think\|>`, `10:<\|/think\|>`, `12:<\|lang_en\|>`, `13:<\|lang_ja\|>`, `15:<\|pad\|>` | Konsisten di Pretrain, SFT, DPO, RL, dan `infer_cpp`. |

> **Kompatibilitas `seq=512` Pretrain ke Post-Training:**  
> Pretrain pada `--seq 512` **100% kompatibel** dengan Stage 2 SFT (`--sft_max_seq 1024`), Stage 3 DPO (`--dpo_max_seq 1024`), dan Stage 5 RL (`--max_seq 768`), karena dimensi tensor bobot tidak bergantung pada `seq_len` dan tabel frekuensi RoPE (`max_seq=2048, theta=500000`) identik di `train_cuda.cu` maupun `train_cuda_unified.cu`.

---

## 3. Daftar Patch C++/CUDA & Skrip Pipeline yang Telah Diterapkan

1. **Unifikasi CLI `--seq` & `--max_seq`:**
   - `transformer/train_cuda.cu`, `transformer/train_cuda_unified.cu`, dan `scripts/train_pipeline.py` kini menerima **kedua flag (`--seq` maupun `--max_seq`)** agar tidak pernah gagal akibat perbedaan argumen CLI.
2. **Perbaikan Mesin Post-Training (`cuda/cuda_ops.{h,cu}` & `transformer/train_cuda_unified.cu`):**
   - **Proteksi Chat Embedding:** Menyalin `BOS -> SYSTEM/USER/ASSISTANT` kini dikunci ketat di `if (g_stage == 2 && !g_resume)` sehingga Stage 3 DPO, Stage 5 RL, dan `--resume` **tidak lagi menimpa** *embedding* chat yang sudah terlatih di SFT.
   - **Kernel DPO Per-Sampel `[B]` & Clipped-DPO:** Mengganti reduksi skalar batch lama dengan `cuda::masked_log_probs_batch` (`[B]`) + `launch_scale_rows_fp16` + turunan *clamp* `--dpo_max_margin 4.0`.
   - **Dynamic Per-Batch `S` (SFT & DPO):** Panjang sekuens per batch dihitung dinamis dari sampel terpanjang pada `global_bs` (dibulatkan ke kelipatan 32, rentang `[64, g_max_seq]`), mempercepat SFT & DPO ~5× lipat tanpa mengubah nilai gradien.
   - **Perbaikan Stage 5 RL / GRPO-Lite:**
     - `load_rl_data()` mendukung penuh kolom `tests` + `setup` (`ASSERT` mode), menghapus bug `add_eos=true` pada prefix prompt RL, dan men-skip otomatis file `*eval*`/`*test*`/`*holdout*`.
     - Saat men-*decode* *rollout* untuk `scripts/rl_verifier.py`, memakai `tok.decode(comp, false)` (dipotong pada `TOK_END`) agar blok `<|think|>...<|/think|>` tetap utuh dan dibersihkan dengan aman oleh `strip_think()` di `rl_verifier.py`.
     - Menaikkan `--rl_max_new` ke `384` (`--max_seq 768`) dan menambahkan `trigger_upload(step, "rl")` berkala.
   - **Aturan Multi-Turn SFT (`build_assistant_mask`):** Mask bernilai `1.0` mulai dari `<|assistant|>` dan berhenti hanya saat bertemu `<|end|>`. Oleh karena itu, setiap giliran asisten pada data *multi-turn* **wajib ditutup `<|end|>`** sebelum `<|user|>` berikutnya.
3. **Auto-Guard VRAM GPU `<= 16 GB` (Tesla T4 `14.6 GB`):**
   - `scripts/train_pipeline.py` mendeteksi VRAM via `nvidia-smi` dan pada GPU `< 18000 MiB` / Kaggle otomatis membatasi:
     - Pretrain: `--batch_size 7` (`@ seq=512`)
     - Stage 2 SFT: `--sft_batch_size 4` (`@ max_seq=1024`)
     - Stage 3 DPO: `--dpo_batch_size 2` (`@ max_seq=1024`)
   - `scripts/run_rl.sh` otomatis memilih `batch_size=1, n_samples=4` pada GPU `< 20000 MiB` (Tesla T4) dan `batch_size=2, n_samples=6` pada GPU `>= 20000 MiB`.
4. **Sinkronisasi Paket Deploy & Dokumentasi Kaggle:**
   - `vast_deploy/source_code.zip` (`3.3 MB`) sudah di-*rebuild* via `(cd vast_deploy && ./deploy.sh package)` mencakup seluruh *patch* terbaru + `rl_data/ceiling_enja`.
   - Seluruh panduan `running_commands/KAGGLE_*.md` (`KAGGLE_PRE-TRAINING.md`, `KAGGLE_SFT.md`, `KAGGLE_DPO.md`, `KAGGLE_SFT_DPO.md`, `KAGGLE_FULL_TRAINING.md`) sudah diseragamkan ke `vocab_9`, `bpe_v9.bin`, `config_500m.json`, dan *batch size* aman Tesla T4.

---

## 4. Inventaris Lengkap Dataset: Paket `10B` (`v8`) vs Paket Tabungan `50B` (`v9`)

> **PENTING:** Paket `v8` (untuk run `10B` sekarang) dan Paket `v9` (tabungan tambahan khusus run `50B`) disimpan **100% terpisah** agar tidak saling mencampur. Seluruh baris/dokumen di `v9` telah dideduplikasi terhadap `v8` (`0%` duplikat) dan didekontaminasi dari seluruh *holdout* evaluasi (`gsm8k/test.jsonl`, `mgsm_ja_test.jsonl`, `humaneval.jsonl`, `mbpp_heldout.jsonl`).

### A. Paket `10B` (`v8` — Murni, Tidak Tersentuh, Siap Jalan di Kaggle Sekarang)

| Stage | Lokasi Folder | Kuantitas & Ukuran | Rincian & Status Audit (`bpe_v9`) |
| :--- | :--- | :--- | :--- |
| **Stage 1: Pretrain + Mid-Train** | `data/corpus_pretrain_v8/` | **`26.81 GB`** (`25.0 GiB`)<br/>**`54` file `.txt`** (`0` symlink)<br/>**`7.51B` token unik** | Sudah mencakup fisik ke-8 shard *mid-training* (`411 MB` dari `data/pretrain_midtrain_addons/`: `GSM8K-JA`, `OpenMathInstruct-JA`, `MetaMath-EN`, `Phi-1 Python`, `OSS Code`, `KFTT`, `SciQ`, `JCommonsenseQA`, `Dolly` EN/JA). `PREFLIGHT OK` (`1.31` epoch @ `10B`). |
| **Stage 2: SFT v4 Ultra-Max** | `sft_data/sft_mixed_enja/` | **`631,727` train** + `6,000` eval<br/>(`25.48M` masked tok/epoch) | 33% Translasi (`273.6K`), 24% Fakta+QA (`286.2K`), 23% Math Opsi B1 `# = nilai` (`50.9K`), 20% Coding (`21.0K`). `99.02% <= 512` tok, `max_len=999 <= 1024`, `audit_sft.py = 0.00%`. |
| **Stage 3: DPO Ceiling** | `sft_data/dpo_ceiling_enja/` | **`22,999` pasang** | 4 domain (`fact_attr_precision`, `trans_anti_qa_collapse`, `math_trace_precision`, `coding_precision`, `translation_precision`). `prefix_trap = 0`, rasio panjang median `1.00`. |
| **Stage 5: GRPO-Lite RL** | `rl_data/ceiling_enja/` | **`12,498` soal** | `7,600` GSM8K EN+JA + `3,998` Python `assert` (`840` MBPP + `3,158` Phi-1 teruji eksekusi) + `900` `code_stdin`. `holdout_leaks = 0`, *anti-cheat* `PASS`. |

---

### B. Paket Tabungan `50B` (`v9` — Terpisah Khusus Eksperimen `50B` Nanti)

| Stage | Lokasi Folder Baru (`v9`) | Tambahan Baru di `v9` | **Total Saat Digabung (`v8 + v9`)** | Catatan Khusus Run `50B` |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Pretrain v9** | `data/corpus_pretrain_v9/`<br/>*(Builder: `scripts/build_corpus_v9_50b.py`)* | **`+15.56 GB`** (`14.49 GiB`)<br/>`17` file `v9_*.txt`<br/>**`+4.31B` tok unik** (`6.07M` dok) | **`42.37 GB`**<br/>**`11.82B` token unik** | Rata-rata `4.23` epoch @ `50B` (`math=3.47`, `mixed=3.51`, `ja=4.27`, `code=4.42`, `en=4.94`) ≈ **`~39.8B` token efektif** (~85–90% potensi penuh 50B). Builder mendukung *auto-resume* jika nanti di Vast/SSD luar ingin tambah ~7 GB lagi. |
| **Stage 2: SFT Hard-Tier v9** | `sft_data/sft_v9_50b/`<br/>*(`v9_sft_hard_train.jsonl`, `163 MB`)* | **`+119,305` train** + `1,488` eval<br/>(`39.01M` tok / `24.12M` masked tok) | **`751,032` train**<br/>(`49.60M` masked tok/ep) | Rata-rata `327` tok/baris (**3.5× lebih dalam** dari `v8`): translasi paragraf 3–4 kalimat `KFTT` & 2-turn (`44.4K`), `Competition MATH` EN/JA (`31.6K`), dialog 2–3 turn `OASST-JA` (`22.6K`), coding kompleks (`21.7K`). **Gunakan `--sft_lr 1.5e-5` pada base `50B`.** |
| **Stage 3: DPO Subtle v9** | `sft_data/dpo_v9_50b/`<br/>*(`v9_dpo_hard_train.jsonl`, `27 MB`)* | **`+18,938` pasang**<br/>(`p50_len_ratio = 1.000`) | **`41,937` pasang** | Fokus jebakan halus: *off-by-one* kode AST-valid (`5K`), selip langkah akhir math (`5K`), urutan klausa paragraf EN↔JA (`5K`), penalaran sains (`4K`). `dropped_bad = 0`, `max_tok = 997`. |
| **Stage 5: GRPO Medium-Hard v9** | `rl_data/rl_v9_50b/`<br/>*(`v9_rl_hard_train.jsonl`, `9.0 MB`)* | **`+15,889` soal**<br/>(JA `~8.0K` · EN `~7.9K`) | **`28,387` soal**<br/>(`~170.3K` rollout @ `G=6`) | Mencegah *zero-advantage trap* di base `50B`: `6,401` `Competition MATH` EN+JA + `5,998` sisa `GSM8K` EN+JA + `3,589` fungsi Python murni ber-*assert* ketat (`0/24 = 0.0%` *anti-cheat* PASS). |

#### Cara Menggabungkan `v8` ke `v9` Saat Siap Menjalankan `50B` (0 Byte Disk Tambahan)
Jangan dijalankan selama masih mengunggah `v8` terpisah. Saat nanti siap melatih `50B`:
```bash
# Menautkan seluruh file v8 ke dalam folder *_v9* memakai hardlink (0 byte ekstra, folder v8 tetap murni):
bash scripts/merge_v8_into_v9.sh --hardlink-v8-into-v9

# Untuk melepas kembali tautan v8 dari folder *_v9* kapan saja:
bash scripts/merge_v8_into_v9.sh --remove-v8-links-from-v9
```
*(Builder C++ filter native juga tersimpan di `tools/filter_audit_v9_posttrain.cpp` dan builder Python di `scripts/build_posttrain_v9_50b.py`).*

---

## 5. Konfigurasi Eksekusi Teruji (Kaggle 2× T4 `14.6 GB` vs Vast.ai)

| Stage | Flag Penting Kaggle `2× Tesla T4 (14.6 GB)` | Flag Penting Vast.ai (`>= 24 GB` VRAM, mis. RTX 4090 / H100) |
| :--- | :--- | :--- |
| **Stage 1: Pretrain** | `--arch_config config_500m.json --vocab_dir vocab_9 --bpe bpe_v9.bin --batch_size 7 --seq 512 --lr 1.0e-4 --grad_clip 0.5 --max_tokens 10000000000 --ddp --num_gpus 2 --resume` | `--batch_size 14` (atau `28–32` di H100), `--max_tokens 50000000000` (untuk run `50B` dengan `--data_dir data/corpus_pretrain_v9`). |
| **Stage 2: SFT** | `--sft_dir sft_data/sft_mixed_enja --sft_batch_size 4 --sft_max_seq 1024 --sft_lr 2e-5 --sft_epochs 2` | `--sft_batch_size 8` (untuk `50B`: `--sft_dir sft_data/sft_v9_50b --sft_lr 1.5e-5`). |
| **Stage 3: DPO** | `--dpo_dir sft_data/dpo_ceiling_enja --dpo_batch_size 2 --dpo_max_seq 1024 --dpo_lr 5e-6 --dpo_epochs 1 --dpo_beta 0.1 --dpo_max_margin 4.0` | `--dpo_batch_size 4` (untuk `50B`: `--dpo_dir sft_data/dpo_v9_50b`). |
| **Stage 5: GRPO / RL** | `bash scripts/run_rl.sh --ddp` *(otomatis `batch_size=1, n_samples=4, max_seq=768, rl_max_new=384`)* | `RL_DIR=rl_data/rl_v9_50b bash scripts/run_rl.sh --ddp` *(otomatis `batch_size=2, n_samples=6`)*. |

---

## 6. Peta Jalan Skala Model & Langkah Selanjutnya

1. **Langkah Terdekat (Run `10B` di Kaggle):**
   - Biarkan sesi Pretrain yang sedang berjalan selesai menyimpan *checkpoint*.
   - Update dataset korpus di Kaggle menggunakan `data/corpus_pretrain_v8` (`54` file) untuk menyelesaikan sisa perjalanan menuju `10.00B` token.
   - Sebelum masuk Stage 2 SFT, update dataset *source code* di Kaggle memakai `vast_deploy/source_code.zip` (atau salinan proyek terbaru) serta unggah `sft_data/sft_mixed_enja`, `sft_data/dpo_ceiling_enja`, dan `rl_data/ceiling_enja`, lalu jalankan panduan di `running_commands/KAGGLE_SFT_DPO.md` atau `running_commands/KAGGLE_FULL_TRAINING.md`.
2. **Langkah Menengah (Overtraining `50B` Token pada Arsitektur `491.6M`):**
   - Direkomendasikan dijalankan di Vast.ai (`deploy.sh`) karena `50B` token hanya memakan ~2.5–4 hari di GPU modern (vs ~680 jam di T4).
   - Aktifkan gabungan `v8 + v9` dengan `bash scripts/merge_v8_into_v9.sh --hardlink-v8-into-v9` dan ubah `--max_tokens 50000000000` sebelum kurva *cosine LR* habis.
3. **Langkah Jangka Panjang (Jika Target Naik ke `200B` Token):**
   - Jangan gunakan arsitektur `491.6M` (`407` tok/param terbentur kapasitas dimensi `1280`).
   - Naikkan spesifikasi SLM ke **`1.36B` (`D=2048, L=28, QH=16, KVH=4, DH=128, FFN=5632`)** atau **`2.0B` (`D=2304, L=32, QH=18, KVH=6, DH=128, FFN=6400, vocab=65536, untied lm_head`)** dengan konteks latih `seq=2048` serta upgrade mesin CUDA ke **`BF16` + `FlashAttention-2` + `QK-Norm` + jadwal LR `WSD` (*Warmup-Stable-Decay*)** dan korpus unik `~58B` token (`~210 GB`).
