# Tabel Paper Pembanding SLM (Translasi, Fakta, Coding, Math)

Subjek: model **492M** di repo ini (`transformer/model_config.h`): 24 layer, d_model 1280, 20 head Q / 5 head KV (GQA 4:1), FFN SwiGLU 3584, vocab 49,152, weight tying, RoPE theta 500k. Parameter dihitung **491.6M**. Data pretraining: EN, JA, code, math, plus paralel EN-JA. Target 10B token, korpus unik sekitar 7.4B (`data/corpus_pretrain_v8/stats.json`).

## Cara membaca dokumen ini

- Semua angka dan spesifikasi di bawah **dibaca dari teks lengkap paper** (arXiv HTML) atau model card Hugging Face yang disebut di kolom. Tidak ada yang diambil dari ingatan.
- **"tidak diekstrak"** artinya paper kemungkinan memuatnya, tetapi saya tidak membaca bagian itu. **"tidak dilaporkan"** artinya saya mencarinya dan tidak menemukannya. Keduanya sengaja tidak saya isi dengan tebakan.
- Skor dari paper berbeda **tidak boleh digabung dalam satu tabel perbandingan**. Buktinya ada di bagian "Peringatan" di bawah.

---

## Peringatan 1: skor model yang sama berbeda antar sumber

| Model | Benchmark | Laporan resmi model tersebut | Sumber lain | Selisih |
|---|---|---|---|---|
| Qwen2.5-0.5B | GSM8K | 41.6 ([Qwen2.5 Tech Report, Tabel 5](https://arxiv.org/abs/2412.15115)) | 33.4, 5-shot ([model card SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)) | 8.2 poin |
| Qwen2.5-0.5B | HellaSwag | 52.1 (Qwen2.5 report) | 51.2 (model card SmolLM2-360M) | 0.9 poin |
| Qwen2.5-1.5B | GSM8K | 68.5 (Qwen2.5 report) | 61.7, 5-shot ([SmolLM2 paper, Tabel 4](https://arxiv.org/abs/2502.02737)) | 6.8 poin |

Kesimpulan: jalankan baseline **dan** model Anda lewat satu harness yang sama (misalnya `lm-evaluation-harness` atau `lighteval`, dengan jumlah shot yang sama). MobileLLM melakukan hal ini sendiri: semua baseline dievaluasi ulang dari checkpoint Hugging Face agar prosedurnya seragam ([MobileLLM, Sec. 3.2](https://arxiv.org/abs/2402.14905)).

## Peringatan 2: budget token model Anda jauh di bawah baseline

Token per parameter, dihitung dari spesifikasi paper di Tabel C:

| Model | Token training | Token / parameter | Kali lipat vs 492M @10B (20.3 tok/param) |
|---|---|---|---|
| SmolLM2-135M | 2T | 14,815 | 728x |
| SmolLM2-360M | 4T | 11,111 | 546x |
| MobileLLM-125M | 1T | 8,000 | 393x |
| MobileLLM-350M | 1T | 2,899 | 142x |
| Qwen2-0.5B | 12T | 24,000 | 1,180x |
| Pythia-160M | ~300B | 1,852 | 91x |
| Pythia-410M | ~300B | 741 | 36x |
| OLMo-1B | 2T | 2,000 | 98x |
| TinyLlama-1.1B | 1T (3 epoch) | 909 | 45x |
| **Model Anda (492M)** | **10B target, 7.4B unik** | **20.3 (15.1 unik)** | 1x |

Catatan: Pythia dihitung dengan 300B, yaitu ukuran korpus The Pile menurut paper (versi dedup 207B). Qwen2.5-0.5B tidak saya masukkan karena paper hanya menyebut korpus 18T secara umum dan tidak memberi jumlah token per ukuran model.

Untuk menafsirkan kesenjangan ini: [Gadre et al. 2024](https://arxiv.org/abs/2403.08540) membangun 104 model (0.011B sampai 6.9B parameter) dengan jumlah token berbeda dan menunjukkan loss over-training dapat diprediksi dari jumlah token, serta terkait dengan skor downstream. Jadi kalau skor Anda lebih rendah dari baseline, jumlah token adalah penjelas pertama yang perlu diuji sebelum menyalahkan arsitektur atau data.

---

## Tabel A. Paper yang membandingkan SLM dengan SLM lain

| # | Paper (link) | Tentang apa | Model yang dibandingkan + spesifikasi | Hasil utama (angka dari paper) | Domain yang diukur |
|---|---|---|---|---|---|
| A1 | [MobileLLM, Liu et al. 2024 (arXiv 2402.14905)](https://arxiv.org/abs/2402.14905) | Desain arsitektur SLM di bawah 1B: SwiGLU, deep-thin, embedding sharing, GQA | **MobileLLM-125M** (30 layer, 125M) dan **-350M** (32 layer, 345M), 1T token, 32xA100, Adam wd 0.1, LR 2e-3 cosine. Dibandingkan dengan OPT-125M/350M (12/24 layer, 125M/331M), GPT-Neo-125M, Pythia-160M (12 layer, 162M), Pythia-410M (24 layer, 405M), RWKV-169M/430M, BLOOM-560M (24 layer, 559M), Cerebras-GPT-111M/256M/590M, Galactica-125M, LaMini-GPT-124M | Rata-rata 8 tugas zero-shot commonsense: MobileLLM-125M 46.3, Pythia-160M 42.5, OPT-125M 42.6. MobileLLM-350M 51.3, Pythia-410M 46.6, OPT-350M 43.9. TriviaQA 1-shot F1: 13.9 vs 2.1 (Pythia-160M); 22.0 vs 12.4 (Pythia-410M). Ablation: SwiGLU 42.6 ke 43.9; GQA 16 head Q dengan 4 head KV hampir tanpa penurunan akurasi | EN commonsense, TriviaQA (fakta), RACE, chat (AlpacaEval, MT-Bench), API calling. **Tidak ada JA, translasi, code, math** |
| A2 | [SmolLM2, Ben Allal et al. 2025 (arXiv 2502.02737)](https://arxiv.org/abs/2502.02737) | Training SLM yang berfokus pada data: FineMath, Stack-Edu, SmolTalk; mixing multi-stage | **SmolLM2-1.7B**: arsitektur Llama2, 11T token, 256xH100, AdamW (0.9, 0.95), WSD, LR 5e-4, vocab 49,152. Dibandingkan dengan **Llama3.2-1B** dan **Qwen2.5-1.5B**. Varian **135M (2T token)** dan **360M (4T token)** memakai GQA, WSD 20% decay, LR 3e-3 | Base 1.7B / Llama3.2-1B / Qwen2.5-1.5B: GSM8K (5-shot) 31.1 / 7.6 / 61.7; MATH 11.6 / 3.3 / 34.3; HumanEval 22.6 / 18.9 / 37.2; TriviaQA 36.7 / 28.1 / 20.9; NQ 8.7 / 6.2 / 10.5. Skor 135M dan 360M **tidak ada di paper**; lihat model card di Tabel C | EN fakta, code, math. Model card menyatakan "primarily English" |
| A3 | [Qwen2.5, Qwen Team 2024 (arXiv 2412.15115)](https://arxiv.org/abs/2412.15115) | Keluarga LLM terbuka 0.5B sampai 72B, termasuk model edge | **Qwen2.5-0.5B** (24 layer, 14/2 head Q/KV, tied) dibandingkan dengan Qwen2-0.5B, Qwen2-1.5B, Gemma2-2.6B, Qwen2.5-1.5B/3B (Tabel 5) | Base 0.5B (Qwen2 vs Qwen2.5): MMLU 44.3 vs 47.5; GSM8K 36.4 vs 41.6; MATH 11.2 vs 19.5; HumanEval 22.6 vs 30.5; MBPP 33.1 vs 39.3; Multi-Translation 14.1 vs 15.3. Paper menyatakan Qwen2.5-0.5B mengungguli Gemma2-2.6B di banyak tugas math dan code | EN, multibahasa, math, code, translasi (agregat "Multi-Translation", definisi metrik tidak saya baca) |
| A4 | [Qwen2, Yang et al. 2024 (arXiv 2407.10671)](https://arxiv.org/abs/2407.10671) | Pendahulu Qwen2.5, spesifikasi arsitektur lengkap | Tabel 1: Qwen2-0.5B (hidden 896, 24 layer, 14/2 head, FFN 4864, vocab 151,646, tied, 12T token) dan Qwen2-1.5B (1536, 28 layer, 12/2 head, FFN 8960, 7T token) | Abstrak: dukungan sekitar 30 bahasa termasuk Jepang. Skor per bahasa JA tidak saya ekstrak | EN, multibahasa, code, math |
| A5 | [Gemma 2, Gemma Team 2024 (arXiv 2408.00118)](https://arxiv.org/abs/2408.00118) | LLM terbuka 2B/9B/27B; distilasi dan attention lokal-global | **Gemma 2 2B**: d_model 2304, 26 layer, 8/4 head Q/KV, FFN 18432, vocab 256,128, tied, GeGLU, 2T token. Dilatih dengan distilasi, bukan next-token murni | Dalam Tabel 5 Qwen2.5, Gemma2 "2.6B" (parameter total): MMLU 52.2, GSM8K 30.3, HumanEval 19.5, MBPP 42.1, Multi-Translation 26.9. Paper menyatakan model "not trained specifically for state-of-the-art multilingual capabilities" | Terutama EN |
| A6 | [Gemma 3, Gemma Team 2025 (arXiv 2503.19786)](https://arxiv.org/abs/2503.19786) | Gemma multimodal, cakupan bahasa lebih luas, long context | **Gemma 3 1B**: 302M embedding + 698M non-embedding, vocab 256K, 2T token, konteks 32K | Menambah data multibahasa (monolingual dan paralel). Skor 1B per domain tidak saya ekstrak | Multibahasa, math, chat |
| A7 | [MobileLLM-R1, Zhao et al. 2025 (arXiv 2509.24945)](https://arxiv.org/abs/2509.24945) | SLM sub-1B untuk reasoning dengan data lebih sedikit | MobileLLM-R1 **140M, 360M, 950M** vs SmolLM2-135M/360M/1.7B-Instruct dan OLMo-2-1.48B-SFT. Pretraining 950M: 4.2T token (11.7% dari 36T Qwen3). Post-training dua tahap (Tulu-3 lalu data reasoning) | MATH / GSM8K / LiveCodeBench v6: R1-140M 4.8 / 3.7 / 1.1 vs SmolLM2-135M-Instruct 3.2 / 1.6 / 0.6. R1-360M 19.2 / 23.8 / 4.0 vs SmolLM2-360M-Instruct 5.2 / 7.4 / 3.4. R1-950M 57.8 / 68.5 / 13.7 vs SmolLM2-1.7B-Instruct 41.4 / 50.5 / 7.4 | Math, code (EN), setelah post-training reasoning |
| A8 | [TinyLlama, Zhang et al. 2024 (arXiv 2401.02385)](https://arxiv.org/abs/2401.02385) | Model 1.1B dengan arsitektur dan tokenizer Llama 2 | TinyLlama v1.0 / v1.1 (1.1B, sekitar 1T token, 3 epoch) vs OPT-1.3B, Pythia-1.0B, Pythia-1.4B | Rata-rata zero-shot commonsense (Tabel 3): v1.1 53.63; v1.0 52.99; OPT-1.3B 51.44; Pythia-1.4B 51.33; Pythia-1.0B 48.30. Varian "Math&code" 53.75 dan "Chinese" 53.41 | EN commonsense, problem-solving (InstructEval) |
| A9 | [Pythia, Biderman et al. 2023 (arXiv 2304.01373)](https://arxiv.org/abs/2304.01373) | 16 model (70M sampai 12B) pada data dan urutan yang sama, 154 checkpoint tiap model | Tabel 1: 160M (12 layer, d 768, 12 head, 85.06M non-embedding, LR 6e-4), 410M (24 layer, d 1024, 16 head, 302.3M non-emb, LR 3e-4). Dilatih pada The Pile (sekitar 300B token) dan Pile-dedup (sekitar 207B) | Paper menyebut padanan: 160M setara GPT-Neo-125M / OPT-125M, 410M setara OPT-350M. Tidak dibuat untuk berebut skor, tetapi untuk studi dinamika training | EN; memorisasi, bias, few-shot |
| A10 | [OLMo, Groeneveld et al. 2024 (arXiv 2402.00838)](https://arxiv.org/abs/2402.00838) | LM terbuka penuh: data, kode, log | **OLMo-1B**: 16 layer, hidden 2048, 16 head, 2T token, weight tying ya, AdamW (0.9, 0.95), LR 4e-4, warmup 2000 langkah, batch sekitar 4M | Perbandingan skor lengkap tidak saya ekstrak | EN |
| A11 | [MiniCPM, Hu et al. 2024 (arXiv 2404.06395)](https://arxiv.org/abs/2404.06395) | SLM dengan "model wind tunnel" dan scheduler WSD | Tabel 2: **MiniCPM-1.2B** (1.247B non-emb, d 1536, FFN 3840, 24 head Q / 8 head KV, 52 layer, 1.1T token), **MiniCPM-2.4B** (2.442B non-emb, d 2304, FFN 5760, 36/36 head, 40 layer, 1.1T token) | Abstrak: klaim setara model 7B sampai 13B. Tabel skor tidak saya ekstrak | EN, Zh, code, math |
| A12 | [Goldfish, Chang et al. 2024 (arXiv 2408.10441)](https://arxiv.org/abs/2408.10441) | Model monolingual kecil untuk 350 bahasa, termasuk Jepang | **Goldfish 124M** (GPT-2, 125M, tokenizer Unigram 50K, data sampai 1GB per bahasa setelah byte-premium; Jepang byte premium 1.32) vs Gemma 3 270M/1B, BLOOM 560M, XGLM 564M/1.7B, Llama 3.2 1B | Rata-rata lintas bahasa (Tabel 3), Goldfish / Gemma3-270M / BLOOM-560M / XGLM-564M / Gemma3-1B / Llama3-1B / XGLM-1.7B: **MultiBLiMP** 78.8 / 72.3 / 64.2 / 66.6 / 78.0 / 77.4 / 62.7 (acak 50). **Belebele** 28.2 / 22.7 / 28.9 / 29.0 / 22.7 / 28.9 / 28.2 (acak 25). Paper: pada benchmark reasoning semua model "around chance" | Multibahasa (perplexity, grammar). Skor khusus JA tidak saya ekstrak |
| A13 | [Swallow, Fujii et al. 2024 (arXiv 2404.17790)](https://arxiv.org/abs/2404.17790) | Continual pretraining Llama 2 ke bahasa Jepang (ekspansi vocab) | **Bukan SLM**: Swallow 7B/13B/70B (dasar Llama 2) vs LLM Jepang from-scratch: calm2-7b (arsitektur Llama, 1.3T token JA+EN), llm-jp-13b-v1.0 (GPT-2, 300B token), PLaMo-13b (180B JA + 1.32T EN) | Rata-rata 8 tugas JA: Swallow lebih tinggi 8.4 sampai 17.4 poin dari ketiga model from-scratch (Tabel 5). Swallow-13b: JCQA 78.4, JEMHopQA 50.6, NIILC 64.0, JSQuAD 90.1, XL-Sum 21.7, MGSM 20.4, WMT'20 en-ja 27.2 BLEU, ja-en 17.7 BLEU, rata-rata 46.3. Skor naik monoton sampai sekitar 100B token | **JA lengkap: fakta (NIILC, JEMHopQA), math (MGSM), translasi (WMT'20 en-ja/ja-en)**. Berguna sebagai daftar benchmark untuk ditiru |

---

## Tabel B. Paper per kemampuan: translasi, fakta, coding, math

Hanya paper yang punya spesifikasi model dan ablation atau perbandingan yang bisa dikutip.

### B1. Translasi

| Paper (link) | Tentang apa | Model + spesifikasi yang dibandingkan | Temuan | Catatan untuk Anda |
|---|---|---|---|---|
| [ALMA, Xu et al. 2023 (2309.11674)](https://arxiv.org/abs/2309.11674) | Fine-tuning LLM untuk MT dua tahap: data monolingual lalu sedikit data paralel berkualitas | **ALMA 7B / 13B** (basis LLaMA-2) vs NLLB-54B, GPT-3.5-D, GPT-3.5-T (zero-shot). Pasangan: en dan {de, cs, is, zh, ru}, WMT'21 dan WMT'22. LLaMA-2-7B dan MPT-7B dibandingkan dengan data paralel 10K, 100K, 1M, 5M, 20M | Rata-rata naik lebih dari 12 BLEU dan 12 COMET dari zero-shot. Cukup 1B token monolingual untuk menyamai NLLB-54B. Untuk LLaMA-2-7B (contoh arah en→ru di paper), 10K sampai 100K paralel sudah baik; 5M sampai 20M menurunkan kualitas ("wash out") | **Tidak ada Jepang.** Model 7B ke atas. Mendukung SFT kecil dan berkualitas; bertentangan dengan korpus Anda yang 9.34M pasangan dengan 69% CCMatrix hasil mining |
| [Swallow (2404.17790)](https://arxiv.org/abs/2404.17790) | Lihat A13 | Lihat A13 | WMT'20 en-ja dan ja-en memakai BLEU, 4-shot | Satu-satunya paper di daftar ini yang mengukur en-ja/ja-en dengan benchmark baku |
| [Qwen2.5 (2412.15115)](https://arxiv.org/abs/2412.15115) | Lihat A3 | Tabel 5, model kecil | Multi-Translation: Qwen2.5-0.5B 15.3, Qwen2.5-1.5B 25.0, Gemma2-2.6B 26.9, Qwen2.5-3B 29.3 | Skor naik tegas dengan ukuran model; bahasa dalam agregat tidak saya baca |
| [NLLB, NLLB Team 2022 (2207.04672)](https://arxiv.org/abs/2207.04672) | Model MT 200 bahasa | Dipakai ALMA sebagai pembanding (NLLB-54B). Anda memakai NLLB-3.3B di `kaggle_nllb33b_jaid.py` | Isi dan spesifikasi tidak saya baca di sesi ini | Hanya referensi untuk pembuatan data |
| [FLORES-101, Goyal et al. 2021 (2106.03193)](https://arxiv.org/abs/2106.03193) | Benchmark MT multibahasa | Benchmark, bukan model | Tidak saya baca isinya | Benchmark translasi yang saya sarankan, sebagai ganti Tatoeba/OpenSubtitles yang ada di training Anda |

### B2. Fakta (pengetahuan)

| Paper (link) | Tentang apa | Model + spesifikasi | Temuan | Catatan untuk Anda |
|---|---|---|---|---|
| [MobileLLM (2402.14905)](https://arxiv.org/abs/2402.14905) | Lihat A1 | Tabel 4: 125M dan 350M vs Pythia, OPT, BLOOM, Cerebras | TriviaQA 1-shot F1: MobileLLM-125M 13.9, OPT-125M 8.7, GPT-Neo-125M 8.0, Pythia-160M 2.1. MobileLLM-350M 22.0, Pythia-410M 12.4, OPT-350M 11.0 | Rentang fakta untuk model kecil: 2 sampai 24 F1 |
| [SmolLM2 (2502.02737)](https://arxiv.org/abs/2502.02737) dan model card [135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) / [360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) | Lihat A2 | 1.7B vs Llama3.2-1B vs Qwen2.5-1.5B; kartu model: 135M dan 360M vs Qwen2.5-0.5B | TriviaQA: 1.7B 36.7; 360M 16.9; 135M 4.1; Qwen2.5-0.5B 4.3 (model card). Paper: kurasi data punya pengaruh besar karena model kecil harus memakai kapasitas untuk pengetahuan inti, bukan fakta insidental | Model 135M dan 0.5B yang dilatih triliunan token masih hanya 4.1 sampai 4.3 pada TriviaQA |
| [Physics of LMs 3.3, Allen-Zhu & Li 2024 (2404.05405)](https://arxiv.org/abs/2404.05405) | Kapasitas pengetahuan model | Dataset sintetis terkontrol; spesifikasi per model tidak saya ekstrak | Abstrak: model menyimpan sekitar 2 bit pengetahuan per parameter, bahkan setelah kuantisasi int8 | Estimasi batas atas: 492M x 2 = sekitar 0.98 Gbit |
| [Physics of LMs 3.1, Allen-Zhu & Li 2023 (2309.14316)](https://arxiv.org/abs/2309.14316) | Penyimpanan dan ekstraksi pengetahuan | Dataset biografi terkontrol | Abstrak: agar fakta dapat diekstrak, harus ada augmentasi (parafrase, pengacakan kalimat, translasi) saat pretraining. Tanpa itu, akurasi bisa 0% walau di-instruction-tune | Mendukung desain fact-injection Anda (banyak template dan alias) |
| [Morris et al. 2025 (2505.24832)](https://arxiv.org/abs/2505.24832) | Kapasitas memorisasi LM | Model GPT-style | Abstrak: sekitar 3.6 bit per parameter | Dipakai bersama 2 bit/param sebagai rentang |
| [Lu et al. 2024 (2406.15720)](https://arxiv.org/abs/2406.15720) | Scaling law memorisasi fakta | Spesifikasi tidak saya ekstrak | Abstrak: kapasitas fakta linear terhadap ukuran model dan eksponensial negatif terhadap jumlah epoch | Relevan untuk fakta yang diulang berkali-kali |
| [Gekhman et al. 2024 (2405.05904)](https://arxiv.org/abs/2405.05904) | Fine-tuning dengan pengetahuan baru dan halusinasi | LLM besar, closed-book QA | Abstrak: fakta baru lewat fine-tuning dipelajari jauh lebih lambat dan dapat mendorong halusinasi | Relevan untuk SFT fakta (`sft_facts_v11`, 500K baris) |

### B3. Coding

| Paper (link) | Tentang apa | Model + spesifikasi | Temuan | Catatan untuk Anda |
|---|---|---|---|---|
| [To Code, or Not To Code?, Aryabumi et al. 2024 (2408.10914)](https://arxiv.org/abs/2408.10914) | Pengaruh proporsi code di pretraining | Model **470M sampai 2.8B**; run 200B token (4,736 TPU-chip jam untuk 470M) | Proporsi code **25%** optimal untuk NL reasoning dan world knowledge; tanpa code, NL reasoning turun 3.4% relatif; kinerja code naik hampir linear dengan proporsi, dari 25% ke 100% naik 2.6x. Rata-rata mulai menurun di 75% code, dan world knowledge turun sampai 86.1% pada rasio code tertinggi | **Skala paling dekat dengan 492M.** Code Anda 19% dari token |
| [Textbooks Are All You Need (phi-1), Gunasekar et al. 2023 (2306.11644)](https://arxiv.org/abs/2306.11644) | Data "kualitas buku teks" untuk code | **phi-1 1.3B** (4 hari, 8 A100; 6B token web terfilter + 1B token sintetis; sekitar 8 pass atas 7B, total sekitar 50B token). **phi-1-small 350M** dengan pipeline sama | HumanEval pass@1 50.6%, MBPP 55.5% (phi-1). phi-1-small (350M) 45% HumanEval. Pembanding kompetitor tidak saya ekstrak | Menunjukkan code kecil bisa bagus **bila datanya terkurasi** |
| [Qwen2.5 (2412.15115)](https://arxiv.org/abs/2412.15115) | Lihat A3 | Tabel 5 | Base: Qwen2.5-0.5B HumanEval 30.5, MBPP 39.3, MultiPL-E 18.9; Qwen2-0.5B 22.6 / 33.1 / 16.3; Gemma2-2.6B 19.5 / 42.1 / 17.6 | Skor code untuk SLM sekitar 0.5B yang dilatih triliunan token |
| [SmolLM2 (2502.02737)](https://arxiv.org/abs/2502.02737) | Lihat A2 | Stack-Edu: sekitar 125B token, 15 bahasa. Skor MultiPL-E Python 20.7 ke 25.6 setelah filter edukasi | Filter edukasi menaikkan MultiPL-E di semua bahasa teratas (Python, C++, JS, Java) | Mendukung filter kualitas pada code. Code Anda memakai filter heuristik, bukan filter edukasi |
| [StarCoder, Li et al. 2023 (2305.06161)](https://arxiv.org/abs/2305.06161) | Code LLM; sumber `starcoderdata` Anda | StarCoder 15.5B, konteks 8K, multi-query attention, 1T token | Sekitar 40% pass@1 HumanEval | Referensi data, bukan SLM |

### B4. Math

| Paper (link) | Tentang apa | Model + spesifikasi | Temuan | Catatan untuk Anda |
|---|---|---|---|---|
| [SmolLM2 (2502.02737)](https://arxiv.org/abs/2502.02737) | FineMath (sumber math Anda) | 1.7B; ablation anneal 60B token | FineMath4+: 2x pada GSM8K dan 6x pada MATH dibanding InfiMM-WebMath. OpenWebMath (OWM) mentok di GSM8K 10%, InfiMM-WebMath 14% (5 epoch OWM) | Anda memakai FineMath dan OpenWebMath, sumber yang sama. Pada 1.7B dengan 11T token, GSM8K base masih 31.1 |
| Model card [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), [-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) | Skor SLM kecil | 135M (2T token), 360M (4T token) | GSM8K 5-shot: **135M 1.4**; **360M 3.2**; Qwen2.5-0.5B 33.4 | **SLM 135M yang dilatih 2T token pun hanya 1.4 pada GSM8K.** Jadi nol di model Anda wajar, dan bukan bukti ada bug |
| [MobileLLM-R1 (2509.24945)](https://arxiv.org/abs/2509.24945) | Lihat A7 | 140M, 360M, 950M | MATH: 4.8 / 19.2 / 57.8; GSM8K: 3.7 / 23.8 / 68.5 | Math pada skala 140M bisa naik, tetapi hanya dengan post-training reasoning besar |
| [Qwen2.5 (2412.15115)](https://arxiv.org/abs/2412.15115) | Lihat A3 | Tabel 5 | Base 0.5B: GSM8K 41.6, MATH 19.5; 1.5B: 68.5 / 35.0 | Sebagai batas atas dari 18T token |
| [OpenWebMath, Paster et al. 2023 (2310.06786)](https://arxiv.org/abs/2310.06786) | Dataset math web 14.7B token | Model **1.4B**, dilatih pada 14.7B token OWM | Melampaui model yang dilatih lebih dari 20x data bahasa umum | Sumber data Anda |
| [Llemma, Azerbayev et al. 2023 (2310.10631)](https://arxiv.org/abs/2310.10631) | Continual pretraining Code Llama untuk math | **7B dan 34B**, Proof-Pile-2 55B token (11B code, 15B OWM, 29B ArXiv) | Mengungguli semua base model terbuka dan Minerva pada MATH untuk parameter setara | Bukan SLM |
| [DeepSeekMath, Shao et al. 2024 (2402.03300)](https://arxiv.org/abs/2402.03300) | Math data 120B token + GRPO | 7B, lanjutan DeepSeek-Coder-Base-v1.5 7B | MATH 51.7% tanpa tool; 60.9% dengan self-consistency 64 sampel | Sumber GRPO Anda; skala 7B |
| [Tokenization Counts, Singh & Strouse 2024 (2402.14903)](https://arxiv.org/abs/2402.14903) | Tokenisasi angka dan aritmetika | GPT-3.5 dan GPT-4 | Tokenisasi kanan-ke-kiri meningkatkan aritmetika | Pretokenizer Anda memisah digit satu per satu |

### B5. Jepang / multibahasa (pendukung translasi dan fakta JA)

| Paper (link) | Model + spesifikasi | Temuan | Catatan untuk Anda |
|---|---|---|---|
| [Goldfish (2408.10441)](https://arxiv.org/abs/2408.10441) | 124M GPT-2, vocab 50K, data sampai 1GB | Model kecil monolingual mengalahkan model multibahasa besar pada perplexity dan grammar; pada reasoning semua mendekati acak | Pembanding skala untuk JA. Bukan pembanding instruksi atau translasi |
| [Scaling Laws for Multilingual LMs, He et al. 2024 (2410.12883)](https://arxiv.org/abs/2410.12883) | Proxy **85M** dipakai untuk memprediksi rasio sampling model sampai **1.2B** | Loss tiap family bahasa ditentukan oleh sampling ratio-nya sendiri; rasio optimal dari model kecil berlaku di model besar | Dasar uji rasio JA 30% dengan proxy kecil |
| [When Is Multilinguality a Curse?, Chang et al. 2023 (2311.09205)](https://arxiv.org/abs/2311.09205) | 10,000+ model, sampai **45M** parameter, 250+ bahasa | Menambah data multibahasa dalam jumlah moderat membantu bahasa low-resource, setara menambah data hingga 33% | Skala jauh di bawah 492M |
| [Swallow (2404.17790)](https://arxiv.org/abs/2404.17790) | 7B/13B/70B | Skor JA naik monoton dengan token continual pretraining sampai sekitar 100B | JA butuh jumlah token besar |
| [Petrov et al. 2023 (2305.15425)](https://arxiv.org/abs/2305.15425) | Berbagai tokenizer | Abstrak: teks yang sama dapat berbeda panjang token sampai 15x antar bahasa | Dasar mengukur fertility tokenizer v9 pada JA |
| [Language Imbalance, Schäfer et al. 2024 (2404.07982)](https://arxiv.org/abs/2404.07982) | Eksperimen bahasa kloning terkontrol | Bahasa dominan meningkatkan generalisasi lintas bahasa dan keselarasan representasi | Dasar mengapa EN dominan bisa membantu JA |

### B6. Data dan metode (ablation dengan spesifikasi model)

| Paper (link) | Model + spesifikasi | Temuan |
|---|---|---|
| [FineWeb, Penedo et al. 2024 (2406.17557)](https://arxiv.org/abs/2406.17557) | Ablation: **1.71B** (arsitektur Llama, seq 2048, batch sekitar 2M token, tokenizer GPT-2). Ablation filter: sekitar 28B token | Memilih filter dan dedup dengan model ablation identik; sumber English Anda (FineWeb-Edu) |
| [DoReMi, Xie et al. 2023 (2305.10429)](https://arxiv.org/abs/2305.10429) | Proxy **280M** menentukan bobot domain untuk model **8B** | Akurasi few-shot rata-rata +6.5 poin, mencapai baseline dengan 2.6x lebih sedikit langkah |
| [Muennighoff et al. 2023 (2305.16264)](https://arxiv.org/abs/2305.16264) | Sampai 900B token, sampai 9B parameter | Sampai 4 epoch hampir sama dengan data unik; lebih dari itu manfaatnya turun menuju nol |
| [Gadre et al. 2024 (2403.08540)](https://arxiv.org/abs/2403.08540) | 104 model, 0.011B sampai 6.9B | Skala over-training dapat diprediksi; terkait skor downstream |
| [Hoffmann et al. 2022 (2203.15556)](https://arxiv.org/abs/2203.15556) | 400+ model, 70M sampai 16B, 5B sampai 500B token | Ukuran model dan jumlah token harus diskalakan seimbang |

---

## Tabel C. Lembar spesifikasi baseline vs model Anda

"tidak diekstrak" berarti paper memuatnya tetapi saya tidak membaca bagian itu. Sel tidak boleh diisi dari ingatan.

| Model | Parameter | Layer | d_model | Head Q / KV | FFN | Vocab | Tied | Token training | Sumber |
|---|---|---|---|---|---|---|---|---|---|
| **Model Anda (492M)** | **491.6M** (embedding 62.9M, 12.8%) | 24 | 1280 | 20 / 5 | 3584 (SwiGLU) | 49,152 | ya | **10B target, 7.4B unik** | `model_config.h` |
| Pythia-160M | 162M total; 85.06M non-emb | 12 | 768 | 12 head (KV tidak diekstrak) | tidak diekstrak | tidak diekstrak | tidak diekstrak | sekitar 300B (The Pile) | [Pythia Tabel 1](https://arxiv.org/abs/2304.01373); total dari MobileLLM Tabel 3 |
| Pythia-410M | 405M total; 302.3M non-emb | 24 | 1024 | 16 head (KV tidak diekstrak) | tidak diekstrak | tidak diekstrak | tidak diekstrak | sekitar 300B | Pythia Tabel 1 |
| MobileLLM-125M | 125M | 30 | tidak diekstrak | tidak diekstrak (GQA) | SwiGLU | tidak diekstrak | ya | 1T | [MobileLLM](https://arxiv.org/abs/2402.14905) |
| MobileLLM-350M | 345M | 32 | tidak diekstrak | tidak diekstrak (GQA) | SwiGLU | tidak diekstrak | ya | 1T | MobileLLM |
| SmolLM2-135M | 135M | tidak diekstrak | tidak diekstrak | GQA | tidak diekstrak | 49,152 (tokenizer paper) | tidak diekstrak | 2T | [SmolLM2](https://arxiv.org/abs/2502.02737), [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) |
| SmolLM2-360M | 360M | tidak diekstrak | tidak diekstrak | GQA | tidak diekstrak | 49,152 | tidak diekstrak | 4T | SmolLM2, [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) |
| Qwen2-0.5B | sekitar 0.5B | 24 | 896 | 14 / 2 | 4,864 | 151,646 | ya | 12T | [Qwen2 Tabel 1](https://arxiv.org/abs/2407.10671) |
| Qwen2.5-0.5B | sekitar 0.5B | 24 | tidak dilaporkan di Tabel 1 | 14 / 2 | tidak dilaporkan di Tabel 1 | tidak dilaporkan di Tabel 1 | ya | tidak dilaporkan per model (korpus 18T) | [Qwen2.5 Tabel 1](https://arxiv.org/abs/2412.15115) |
| Gemma 2 2B | sekitar 2.6B total | 26 | 2304 | 8 / 4 | 18,432 (GeGLU) | 256,128 | ya | 2T | [Gemma 2 Tabel 1](https://arxiv.org/abs/2408.00118) |
| Gemma 3 1B | 302M emb + 698M non-emb | tidak diekstrak | tidak diekstrak | tidak diekstrak | tidak diekstrak | 256K | tidak diekstrak | 2T | [Gemma 3 Tabel 1](https://arxiv.org/abs/2503.19786) |
| OLMo-1B | sekitar 1B | 16 | 2048 | 16 head (KV tidak diekstrak) | tidak diekstrak | tidak diekstrak | ya | 2T | [OLMo Tabel 1](https://arxiv.org/abs/2402.00838) |
| TinyLlama | 1.1B | tidak diekstrak | tidak diekstrak | tidak diekstrak | tidak diekstrak | Llama 2 | tidak diekstrak | sekitar 1T (3 epoch) | [TinyLlama](https://arxiv.org/abs/2401.02385) |
| MiniCPM-1.2B | 1.247B non-emb | 52 | 1536 | 24 / 8 | 3,840 | tidak diekstrak | tidak diekstrak | 1.1T | [MiniCPM Tabel 2](https://arxiv.org/abs/2404.06395) |
| MobileLLM-R1-140M / 360M / 950M | 140M / 359M / 949M | tidak diekstrak | tidak diekstrak | tidak diekstrak | tidak diekstrak | tidak diekstrak | tidak diekstrak | 950M: 4.2T; lainnya tidak diekstrak | [MobileLLM-R1](https://arxiv.org/abs/2509.24945) |
| Goldfish (JA, 1GB) | 125M (124M) | tidak diekstrak | tidak diekstrak | tidak diekstrak | tidak diekstrak | 50K (Unigram) | tidak diekstrak | sampai 1GB teks | [Goldfish](https://arxiv.org/abs/2408.10441) |

Padanan terdekat yang terverifikasi untuk 492M adalah **Qwen2-0.5B / Qwen2.5-0.5B** (24 layer, GQA 7:1, tied, sekitar 0.5B) karena kelas parameter, jumlah layer, GQA, dan tying-nya sama. Perbedaannya: vocab 151,646 (3x lebih besar), 12T token (24,000 token/param), dan banyak bahasa.

---

## Tabel D. Matriks cakupan: baseline mana yang punya skor untuk domain mana

Hanya mencatat apa yang **saya lihat di teks yang dibaca**. "tidak dicek" berarti saya tidak membaca bagian evaluasinya.

| Baseline | EN umum | JA | Translasi | Fakta | Code | Math |
|---|---|---|---|---|---|---|
| MobileLLM 125M/350M | ya | tidak | tidak | ya (TriviaQA) | tidak | tidak |
| SmolLM2 135M/360M/1.7B | ya | tidak ("primarily English") | tidak | ya (TriviaQA, NQ) | ya (HumanEval; 1.7B) | ya (GSM8K; MATH di 1.7B) |
| Qwen2 / Qwen2.5 0.5B | ya | JA termasuk 30 bahasa, skor per bahasa tidak diekstrak | ya, agregat "Multi-Translation" | tidak ada TriviaQA di tabel kecil; ada TruthfulQA | ya (HumanEval, MBPP, MultiPL-E) | ya (GSM8K, MATH, Multi-Mathematics) |
| Gemma 2 2B | ya | tidak ("not trained for multilingual") | ya (agregat via Qwen2.5 Tabel 5) | tidak dicek | ya (via Tabel 5) | ya (via Tabel 5) |
| Gemma 3 1B | ya | multibahasa, tidak diekstrak | tidak dicek | tidak dicek | tidak dicek | tidak dicek |
| Pythia 160M/410M | ya | tidak | tidak | tidak dicek | tidak | tidak |
| OLMo-1B | ya | tidak | tidak | tidak dicek | tidak dicek | tidak dicek |
| TinyLlama 1.1B | ya | tidak | tidak | tidak dicek | tidak dicek (InstructEval) | tidak dicek (InstructEval) |
| MobileLLM-R1 | ya | tidak | tidak | tidak | ya (LiveCodeBench, HumanEval) | ya (MATH, GSM8K) |
| Goldfish 124M | ya | **ya** (perplexity, grammar) | tidak | tidak | tidak | tidak |
| Swallow 7B+ (bukan SLM) | ya | **ya** | **ya (WMT'20 en-ja/ja-en)** | **ya (NIILC, JEMHopQA)** | tidak | **ya (MGSM)** |
| ALMA 7B/13B (bukan SLM) | tidak | **tidak** | ya (bukan JA) | tidak | tidak | tidak |

Yang bisa disimpulkan dari tabel ini, dengan batas pencarian saya:
- **Tidak ada satu baseline SLM pun yang punya skor di keempat domain (translasi, fakta, code, math) dengan JA.** Itu temuan dari teks yang saya baca, bukan hasil pencarian menyeluruh.
- Untuk **code dan math**, Qwen2.5-0.5B adalah baseline kelas ukuran terdekat yang punya skor lengkap. SmolLM2 menambahkan skor 135M dan 360M di model card.
- Untuk **JA dan translasi**, tidak ada SLM sebanding. Pilihan: ambil benchmark dari Swallow (WMT'20 en-ja/ja-en, MGSM, NIILC, JEMHopQA) lalu jalankan sendiri pada baseline multibahasa kecil (Qwen2.5-0.5B, Gemma 3 1B), dan pada model Anda.

---

## Saran protokol perbandingan (berdasarkan temuan di atas)

1. **Satu harness, satu konfigurasi.** Jalankan model Anda, Qwen2.5-0.5B, SmolLM2-360M, Pythia-410M, dan MobileLLM-350M lewat `lm-evaluation-harness` atau `lighteval` dengan shot yang sama. Alasannya ada di Peringatan 1.
2. **Metrik utama: loss atau bits-per-byte per domain** (EN, JA, code, math), karena skor akurasi di skala ini mendekati acak (SmolLM2-135M: GSM8K 1.4). Bits-per-byte sebanding antar tokenizer.
3. **Translasi:** sacreBLEU dan chrF pada WMT'20 en-ja/ja-en (seperti Swallow) atau FLORES, bukan data yang ada di training Anda.
4. **Laporkan rasio token per parameter** (Peringatan 2) di samping setiap skor.
5. **Ablation yang punya preseden di tabel:** proporsi code seperti Aryabumi (B3), rasio bahasa seperti He et al. (B5), kualitas data seperti FineWeb dan SmolLM2 (B6, B3).
6. **Jangan klaim "model kecil jelek"** sebagai penjelasan tunggal: SmolLM2-135M dengan 2T token pun hanya GSM8K 1.4 dan TriviaQA 4.1. Itu batas skala, sedangkan perbedaan dengan model Anda yang perlu dijelaskan adalah jumlah token dan kualitas data.

## Yang belum diverifikasi

- Spesifikasi sel "tidak diekstrak" di Tabel C.
- Skor JA per bahasa di Qwen2/2.5 dan Gemma 3, dan skor Goldfish khusus Jepang.
- Daftar kompetitor phi-1 pada HumanEval dan MBPP.
- Definisi metrik "Multi-Translation" di Qwen2.5.
- Skor MiniCPM, OLMo-1B, dan TinyLlama pada fakta, code, dan math.
- JGLUE dan JMMLU tidak ada di arXiv atau tidak ditemukan ID-nya sesi ini; lihat `SLM_SUPPORTING_PAPERS.md`.
