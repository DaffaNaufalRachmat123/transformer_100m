# Paper Pendukung untuk Perbandingan SLM (EN / JA / Code / Math)

Dokumen ini memetakan **spesifikasi dan data nyata di repo ini** ke paper yang mendukung klaim "spesifikasi + data memengaruhi kualitas translasi, fakta, code, dan math".

Status verifikasi:
- **[V]** judul, tahun, dan arXiv ID dicek lewat arXiv API pada sesi ini. Pencocokan judul-ID dijamin benar.
- **[V+A]** sama dengan [V], dan klaim yang dikutip dicocokkan dengan abstrak.
- **[?]** tidak ditemukan di arXiv, atau tidak bisa dicek di sesi ini. Cek manual sebelum masuk daftar pustaka.

---

## 0. Temuan scan codebase yang memengaruhi cara menulis bab perbandingan

Baca bagian ini dulu. Ada hal yang bisa membuat klaim di tesis tidak konsisten dengan repo.

### 0.1 Ada tiga model berbeda di repo, dan angka eval berasal dari model 204M

| Varian | vocab | d_model | layers | Q/KV heads | FFN | Param (dihitung) | Embedding / total | Sumber |
|---|---|---|---|---|---|---|---|---|
| **204M** | 65,536 | 768 | 18 | 12/3 | 3072 | **204.3M** | **24.6%** | `config_204m.json` (EN/JA/**ID**) |
| **492M** (default kode) | 49,152 | 1280 | 24 | 20/5 | 3584 | **491.6M** | 12.8% | `transformer/model_config.h:18-25`, `config_pretrain.json` |
| 599M | 49,152 | 1280 | 30 | 20/5 | 3584 | 598.7M | 10.5% | `config_599m.json` |
| 892M (README) | 65,536 | 1280 | 34 | 20/5 | 5120 | 891.7M | 9.4% | `README.md:8-21` (sudah usang) |

Jumlah parameter dihitung ulang dengan rumus `ModelConfig::param_count()` (`model_config.h:64-77`).

Akibatnya:
- Spesifikasi di daftar Anda (HQ=20, HKV=5, V=49152, ffn=3584) adalah **492M**, bukan 100M. Nama folder `transformer_100m_cuda` menyesatkan.
- Semua hasil eval yang ada (`results/*.json`, `gguf/*`) memakai **checkpoint 204M** (`gguf/trilingual-204m-*`, `translation_resources/*.safetensors` sekitar 408 MB, setara 204M dalam FP16). Model ini trilingual EN/JA/ID, bukan EN/JA/Code/Math.
- `train_state (18).txt` (382M token, step 24,900, EMA loss 3.81) dan `loss_log.txt` (T4, `seq=256`, ~400M dari target 2.0B token, loss ~3.8) adalah log training lama. `loss_log.txt:31-36` memuat `bpe_v6.bin (vocab=65536)`, jadi **bukan** model 492M (vocab 49,152). Ukuran modelnya tidak tercatat di log: bisa 204M atau 892M (keduanya V=65,536). Jangan diasumsikan.
- Hasil eval untuk model 492M belum ada di repo.

**Tentukan satu model sebagai subjek perbandingan.** Spesifikasi dan hasil eval harus berasal dari model yang sama. Kalau SLM yang dibandingkan adalah 492M, hasil eval harus dari checkpoint 492M. Kalau 204M, spesifikasi di tesis harus diganti ke 768/18/12:3/3072/V=65,536.

### 0.2 Komposisi data pretraining (492M, `scripts/build_pretrain_corpus.py`, `data/corpus_pretrain_v8/stats.json`)

| Bucket | Byte (kept) | Token estimasi | Target share | Sumber utama |
|---|---|---|---|---|
| English | 6.50 GB | 1.69B | 27% | fineweb-edu, Wikipedia EN, OpenWebText, fineweb |
| Japanese | 7.00 GB | 1.90B | 30% | Wikipedia JA, CulturaX-ja, mC4-ja |
| Code | 4.50 GB | 1.47B | 19% | StarCoder (Python), CodeSearchNet, GitHub JS, Kotlin, code_contests |
| Math | 5.00 GB | 1.43B | 15% | FineMath, OpenWebMath, + GSM8K train & SFT math |
| Mixed | 3.40 GB | 0.90B | 9% | fact-injection (Wikidata), paralel EN↔JA (10.8M pasang) |
| **Total unik** | | **~7.4B** | | target 10B, jadi ~1.35 epoch |

Catatan penting:
- Dokumen run (`running_commands/KAGGLE_PRE-TRAINING.md`) masih menulis mix lama **JA 36 / EN 32 / code 20 / math 12**. Skrip v8 memakai **27/30/19/15/9**. Pakai angka dari skrip dan `stats.json`.
- Math dominan berasal dari FineMath dan OpenWebMath (~4.9 GB dari ~5.0 GB). Isi 'math' lainnya sangat kecil.
- Code v8 lebih beragam daripada bucket lama (yang 94.6% Python menurut komentar skrip). Dari `stats.json`: StarCoder Python ~2.25 GB (~50%), GitHub JS 1.06 GB, CodeSearchNet (Java, PHP, JS, Go, Ruby) ~0.87 GB, Kotlin 0.24 GB, code_contests 0.09 GB. Jadi Python masih dominan tapi bukan satu-satunya. Eval Anda (HumanEval, MBPP) hanya Python.
- Tokenizer v9: 49,152 token, byte-level BPE dengan pretokenizer sadar-script (hiragana, katakana, kanji dipisah), digit dipisah satu per token, dilatih dengan sampel JA 40%. Ini sudah ada di `bpe/bpe_tokenizer.cpp` dan `tools/train_bpe.cpp`.

### 0.3 Risiko validitas yang harus ditangani sebelum membandingkan

1. **Fact-injection diulang ~1000x per fakta** (`build_fact_injection_corpus.py`; `fact_injection_stats.json` mencatat exposure=150, 63,066 fakta, ~370M token). Ini jauh di atas ambang ~4 epoch pada Muennighoff et al. Ini sengaja (tujuannya menanamkan fakta), tapi hasilnya adalah memorisasi, bukan pengetahuan umum. Benchmark seperti PopQA tidak akan menangkapnya. Jelaskan ini sebagai keputusan desain.
2. **GSM8K train masuk ke pretraining math dan SFT.** Test split terpisah (1,319 baris), jadi ini bukan kontaminasi test, tetapi **in-distribution training**. Skor GSM8K tidak sebanding dengan model yang tidak melihat GSM8K train. Laporkan ini secara eksplisit.
3. **Eval code**: `eval_data/mbpp_heldout.jsonl` (497) tumpang tindih **5** prompt dengan `rl_data/coding_mbpp.jsonl` (pengecekan awalan 80 karakter). Kecil, tapi perlu dideduplikasi lengkap lalu dilaporkan.
4. **Metrik translasi di `results/translation_eval_final.json` tidak konsisten**: agregat ja-en chrF 0.763 dan BLEU 0.82, padahal sampel di file yang sama punya chrF per kalimat 0.0 sampai 0.15. en-ja chrF 0.013 tapi BLEU 0.63. Kemungkinan besar ada bug skala atau agregasi (saya belum membaca skrip evalnya). **Jangan dikutip di tesis** sebelum dihitung ulang dengan sacreBLEU (Post 2018) atau COMET.
5. Hasil `results/baseline_facts_ckpt_multiskill.json` (coding chrF 0.05, math answer_hit 0.0, translation chrF 0.035) dan `demo_eval_fact_complete.json` menunjukkan model **tidak mengikuti instruksi** (jawaban fakta untuk pertanyaan translasi). Ini menunjukkan kegagalan SFT dan bukan keterbatasan kapasitas. Jangan atribusikan semuanya ke "model kecil".

---

## 1. Spesifikasi model → kualitas

### 1.1 Scaling law, over-training, data berulang

| Paper | ID | Klaim yang dipakai | Status |
|---|---|---|---|
| Kaplan et al. 2020, *Scaling Laws for Neural Language Models* | 2001.08361 | Loss mengikuti power-law terhadap N, D, compute | V |
| Hoffmann et al. 2022, *Training Compute-Optimal LLMs* (Chinchilla) | 2203.15556 | ~20 token/param. Dasar `CHINCHILLA_TOKENS` di kode | V |
| Touvron et al. 2023, *LLaMA* | 2302.13971 | Model kecil yang dilatih melebihi Chinchilla tetap membaik | V |
| Sardana & Frankle 2023, *Beyond Chinchilla-Optimal* | 2401.00448 | Token/param optimal naik bila inference diperhitungkan | V |
| Gadre et al. 2024, *LMs scale reliably with over-training and on downstream tasks* | 2403.08540 | Loss over-training bisa diprediksi dan terkait skor downstream (104 model, 0.011B-6.9B) | V+A |
| Muennighoff et al. 2023, *Scaling Data-Constrained LMs* | 2305.16264 | Sampai ~4 epoch hampir sama dengan data unik, lebih dari itu manfaatnya turun | V+A |
| Hernandez et al. 2022, *Scaling Laws and Interpretability of Learning from Repeated Data* | 2205.10487 | Data berulang menurunkan kinerja secara tidak proporsional | V |
| Kumar et al. 2024, *Scaling Laws for Precision* | 2411.04330 | Dampak presisi (training/inference) pada skala | V |
| Geiping & Goldstein 2022, *Cramming* | 2212.14034 | Resep training budget kecil | V |

Pakai untuk: 492M x 20 = 9.8B token adalah target, tapi korpus unik ~7.4B (~1.35 epoch). Itu masih di bawah 4 epoch. Run 204M (~2B token) = ~10 token/param, **di bawah** Chinchilla. Laporkan angka ini.

### 1.2 Desain SLM: arsitektur sub-1B dan pembanding

| Paper | ID | Dipakai untuk | Status |
|---|---|---|---|
| Liu et al. 2024, *MobileLLM* | 2402.14905 | Deep-thin, embedding sharing, GQA untuk <1B (klaim cocok dengan abstrak) | V+A |
| Zhao et al. 2025, *MobileLLM-R1* | 2509.24945 | SLM sub-1B untuk reasoning (pembanding math) | V |
| Zhang et al. 2024, *TinyLlama* | 2401.02385 | 1.1B, ~1T token selama ~3 epoch (pembanding upper-bound). Koreksi: versi awal dokumen ini salah menulis ~3T | V+A |
| Ben Allal et al. 2025, *SmolLM2* | 2502.02737 | 1.7B, ~11T token, data-centric, FineMath/Stack-Edu/SmolTalk | V+A |
| Hu et al. 2024, *MiniCPM* | 2404.06395 | SLM dengan strategi training terukur | V |
| Biderman et al. 2023, *Pythia* | 2304.01373 | Suite 70M-12B dengan data yang sama (pembanding bersih) | V |
| Groeneveld et al. 2024, *OLMo* / Team OLMo 2025, *OLMo 2* | 2402.00838 / 2501.00656 | Model terbuka dengan data dan log | V |
| Gemma Team 2024/2025, *Gemma 2*, *Gemma 3* | 2408.00118 / 2503.19786 | Pembanding kecil modern | V |
| Yang et al. 2024/2025, *Qwen2*, *Qwen3* | 2407.10671 / 2505.09388 | Pembanding multibahasa (termasuk JA) | V |
| Qwen 2024, *Qwen2.5* | 2412.15115 | Qwen2.5-0.5B untuk pembanding | V |
| Abdin et al. 2024, *Phi-3* / *Phi-4* | 2404.14219 / 2412.08905 | SLM dengan data sintetis dan terkurasi | V |
| Narang et al. 2021, *Do Transformer Modifications Transfer...?* | 2102.11972 | Hanya sebagian modifikasi yang konsisten membantu | V |
| Tay et al. 2022, *Scaling Laws vs Model Architectures* | 2207.10551 | Pengaruh inductive bias pada scaling | V |
| Xiong et al. 2020, *On Layer Normalization in the Transformer Architecture* | 2002.04745 | Pre-LN lebih stabil | V |

**Catatan keterbatasan.** MobileLLM mengklaim arsitektur lebih penting daripada data/parameter di sub-1B. Itu bertentangan dengan klaim data-centric SmolLM2/Phi. Ini bisa dibahas sebagai dua sisi argumen.

Kaitan arsitektur dan ukuran di repo ini:
- Pada 204M, embedding = 24.6% parameter (V=65,536). Pada 492M, 12.8%. Weight tying (Press & Wolf, 1608.05859 [V]) menghemat porsi yang besar pada model kecil.
- Tao et al. 2024, *Scaling Laws with Vocabulary* (2407.13623) [V+A]: vocab optimal naik seiring compute. Takase et al. 2024 (2406.16508) [V]: vocab besar membantu.

---

## 2. Data → kualitas

### 2.1 Kualitas, filtering, dan dedup

| Paper | ID | Klaim | Status |
|---|---|---|---|
| Gunasekar et al. 2023, *Textbooks Are All You Need* (phi-1) | 2306.11644 | Data berkualitas membuat model kecil kuat untuk code | V |
| Li et al. 2023, *phi-1.5* | 2309.05463 | Data sintetis berkualitas untuk reasoning | V |
| Eldan & Li 2023, *TinyStories* | 2305.07759 | Model sangat kecil bisa koheren dengan data sederhana | V |
| Penedo et al. 2024, *FineWeb* | 2406.17557 | FineWeb-Edu, filter kualitas (sumber English Anda) | V |
| Penedo et al. 2023, *RefinedWeb* | 2306.01116 | Web yang difilter dan dededup menyaingi korpus kurasi | V |
| Li et al. 2024, *DataComp-LM* | 2406.11794 | Filter model-based meningkatkan downstream | V |
| Lee et al. 2022, *Deduplicating Training Data Makes LMs Better* | 2107.06499 | Dedup mengurangi memorisasi, perplexity tetap baik | V |
| Longpre et al. 2023, *A Pretrainer's Guide to Training Data* | 2305.13169 | Pengaruh kualitas, domain, dan toksisitas | V |
| Soldaini et al. 2024, *Dolma* | 2402.00159 | Korpus terbuka + dokumentasi filter | V |
| Gao et al. 2020, *The Pile* | 2101.00027 | Korpus multi-domain | V |
| Thuat Nguyen et al. 2023, *CulturaX* | 2309.09400 | Sumber JA/ID Anda (`fetch_culturax.py`) | V |
| Lin et al. 2024, *Rho-1: Not All Tokens Are What You Need* | 2404.07965 | Tidak semua token sama bermanfaat | V |
| Cheng et al. 2024, *Instruction Pre-Training* | 2406.14491 | Data instruksi di pretraining | V |

Pipeline Anda sudah punya filter (adult, gambling, URL spam, mojibake, repetitive, low lexical diversity, dedup). Statistik drop per alasan ada di `stats.json`. Pakai itu sebagai ablation data cleaning, tidak hanya sitasi.

### 2.2 Komposisi data (rasio EN/JA/Code/Math)

| Paper | ID | Klaim | Status |
|---|---|---|---|
| Xie et al. 2023, *DoReMi* | 2305.10429 | Proporsi domain sangat memengaruhi LM, model proxy 280M bisa menentukan bobot | V+A |
| Ye et al. 2024, *Data Mixing Laws* | 2403.16952 | Kinerja mixture bisa diprediksi | V |
| Liu et al. 2024, *RegMix* | 2407.01492 | Regresi dengan proxy kecil | V |
| Yadlowsky et al. 2023, *Pretraining Data Mixtures Enable Narrow Model Selection* | 2311.00871 | Kemampuan terbatas pada distribusi training | V |
| Gururangan et al. 2020, *Don't Stop Pretraining* | 2004.10964 | Adaptasi domain | V |
| Luo et al. 2023, *Catastrophic Forgetting in LLMs During Continual Fine-tuning* | 2308.08747 | Relevan untuk pipeline SFT/DPO/RL berurutan | V |

---

## 3. Per domain

### 3.1 Code

| Paper | ID | Klaim | Status |
|---|---|---|---|
| Li et al. 2023, *StarCoder* | 2305.06161 | Data (`starcoderdata`) yang Anda pakai | V |
| Lozhkov et al. 2024, *StarCoder 2 / The Stack v2* | 2402.19173 | | V |
| Kocetkov et al. 2022, *The Stack* | 2211.15533 | | V |
| Rozière et al. 2023, *Code Llama* | 2308.12950 | | V |
| Guo et al. 2024, *DeepSeek-Coder* | 2401.14196 | | V |
| Aryabumi et al. 2024, *To Code, or Not To Code?* | 2408.10914 | Code di pretraining membantu task non-code, ablation 470M-2.8B (skala dekat model Anda) | V+A |
| Chen et al. 2021, *Codex / HumanEval* | 2107.03374 | Metrik pass@k | V |
| Austin et al. 2021, *Program Synthesis with LLMs / MBPP* | 2108.07732 | | V |
| Husain et al. 2019, *CodeSearchNet* | 1909.09436 | Sumber `csn_*` Anda | [?] (ID dari ingatan, belum dicek) |
| Chen et al. 2022 *PoT*; Gao et al. 2022 *PAL* | 2211.12588 / 2211.10435 | Program-of-thought, cocok dengan `sft_math_pot` | V |

Keterbatasan data code Anda: hampir seluruhnya Python, jadi jangan klaim multi-language.

### 3.2 Math

| Paper | ID | Klaim | Status |
|---|---|---|---|
| Lewkowycz et al. 2022, *Minerva* | 2206.14858 | Data math khusus | V |
| Paster et al. 2023, *OpenWebMath* | 2310.06786 | Sumber Anda | V |
| Azerbayev et al. 2023, *Llemma* | 2310.10631 | | V |
| Shao et al. 2024, *DeepSeekMath* | 2402.03300 | Data math + GRPO (sumber GRPO Anda) | V |
| Allal et al. 2025, *SmolLM2* (FineMath) | 2502.02737 | FineMath, sumber Anda | V+A |
| Cobbe et al. 2021, *GSM8K* | 2110.14168 | | V |
| Hendrycks et al. 2021, *MATH* | 2103.03874 | | V |
| Singh & Strouse 2024, *Tokenization counts* | 2402.14903 | Skema tokenisasi angka memengaruhi aritmetika. Pretokenizer Anda memisah digit | V+A |
| Nogueira et al. 2021, *Limitations of Transformers with Simple Arithmetic* | 2102.13019 | | V |
| Yuan et al. 2023, *Scaling Relationship on Learning Mathematical Reasoning* | 2308.01825 | Skala vs kemampuan math | V |

Hasil Anda: `answer_hit = 0.0` pada math (`baseline_facts_ckpt_multiskill.json`). Sebelum menyimpulkan "kapasitas terlalu kecil", periksa dulu apakah model mengikuti format instruksi (lihat 0.3 poin 5).

### 3.3 Japanese dan multibahasa

| Paper | ID | Klaim | Status |
|---|---|---|---|
| Conneau et al. 2020, *XLM-R* | 1911.02116 | Curse of multilinguality | V |
| Xue et al. 2021, *mT5* | 2010.11934 | | V |
| Chang et al. 2023, *When Is Multilinguality a Curse?* | 2311.09205 | Efek menambah bahasa pada LM per bahasa, model sampai 45M | V+A |
| Chang et al. 2024, *Goldfish* | 2408.10441 | Model monolingual 125M dengan data ≤1GB mengalahkan model multibahasa besar pada perplexity. Pembanding untuk skala Anda | V+A |
| He et al. 2024, *Scaling Laws for Multilingual LMs* | 2410.12883 | Loss tiap family ditentukan sampling ratio-nya. Dasar rasio JA 30% | V+A |
| Schäfer et al. 2024, *Role of Language Imbalance in Cross-lingual Generalisation* | 2404.07982 | Bahasa dominan membantu bahasa lain | V+A |
| Fernandes et al. 2023, *Scaling Laws for Multilingual NMT* | 2302.09650 | | V |
| Fujii et al. 2024, *Swallow: Continual Pre-Training for Cross-Lingual LLM Adaptation (Japanese)* | 2404.17790 | Pembanding data JA | V |
| Okazaki et al. 2024, *Building a Large Japanese Web Corpus for LLMs* | 2404.17733 | Korpus web JA (Swallow Corpus) | V |
| LLM-jp 2024, *LLM-jp* | 2407.03963 | LLM JA terbuka | V |
| Marchisio et al. 2024, *Understanding and Mitigating Language Confusion* | 2406.20052 | Model menjawab di bahasa yang salah. Dalam hasil Anda, jawaban fakta EN muncul dalam bahasa Jepang | V |
| Shi et al. 2022, *MGSM* | 2210.03057 | Math multibahasa (JA termasuk) | V |
| Bandarkar et al. 2023, *Belebele* | 2308.16884 | Reading comprehension 122 bahasa (JA termasuk) | V |
| Kurihara et al. 2022, *JGLUE* | – | Bukan di arXiv (LREC 2022) | [?] |
| JMMLU | – | ID arXiv tidak ditemukan sesi ini. ID `2409.07413` yang sempat saya sebut **salah** | [?] |

### 3.4 Tokenizer dan JA

| Paper | ID | Klaim | Status |
|---|---|---|---|
| Petrov et al. 2023, *LM Tokenizers Introduce Unfairness Between Languages* | 2305.15425 | Panjang token berbeda sampai 15x antar bahasa | V+A |
| Ahia et al. 2023, *Do All Languages Cost the Same?* | 2305.13707 | Biaya token per bahasa | V |
| Rust et al. 2021, *How Good is Your Tokenizer?* | 2012.15613 | Kualitas tokenizer memengaruhi performa monolingual | V |
| Ali et al. 2023, *Tokenizer Choice For LLM Training* | 2310.08754 | Pilihan tokenizer berdampak signifikan | V |
| Huang et al. 2024, *Compression Represents Intelligence Linearly* | 2404.09937 | Kompresi korelasi dengan skor benchmark. Dasar memakai bits-per-byte | V |
| Radford et al. 2019 (GPT-2), Sennrich et al. 2016 BPE | 1508.07909 | | V (BPE) |

Fertility JA: skrip mengukur 1.16 karakter/token untuk vocab_7 dan perbaikan +22% di vocab_8 (komentar di `tools/train_bpe.cpp`). Hitung ulang untuk vocab_9 dengan `scripts/measure_bytes_per_token.py --vocab_dir vocab_9`.

---

## 4. Fakta dan pengetahuan

| Paper | ID | Klaim | Status |
|---|---|---|---|
| Allen-Zhu & Li 2024, *Physics of LMs 3.3: Knowledge Capacity Scaling Laws* | 2404.05405 | ~2 bit pengetahuan per parameter | V+A |
| Allen-Zhu & Li 2023, *Physics of LMs 3.1: Knowledge Storage and Extraction* | 2309.14316 | Pengetahuan harus diaugmentasi (parafrase, translasi) agar bisa diekstrak. Cocok dengan desain fact-injection Anda | V+A |
| Allen-Zhu & Li 2023, *Physics of LMs 3.2: Knowledge Manipulation* | 2309.14402 | | V |
| Morris et al. 2025, *How much do LMs memorize?* | 2505.24832 | Estimasi ~3.6 bit/parameter | V+A |
| Lu et al. 2024, *Scaling Laws for Fact Memorization* | 2406.15720 | Kapasitas fakta linear terhadap ukuran model, eksponensial negatif terhadap epoch | V+A |
| Roberts et al. 2020, *How Much Knowledge Can You Pack...?* | 2002.08910 | | V |
| Petroni et al. 2019, *LAMA* | 1909.01066 | | V |
| Kandpal et al. 2023, *LLMs Struggle to Learn Long-Tail Knowledge* | 2211.08411 | | V |
| Mallen et al. 2023, *When Not to Trust LMs* | 2212.10511 | PopQA | V |
| Lin et al. 2022, *TruthfulQA* | 2109.07958 | | V |
| Berglund et al. 2023, *The Reversal Curse* | 2309.12288 | Relevan untuk pertanyaan fakta bentuk berbeda | V |
| Gekhman et al. 2024, *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?* | 2405.05904 | Fakta baru di SFT dipelajari lambat dan bisa memicu halusinasi | V+A |
| Ovadia et al. 2023, *Fine-Tuning or Retrieval?* | 2312.05934 | | V |
| Zhou et al. 2023, *LIMA* | 2305.11206 | | V |
| Tirumala et al. 2022; Carlini et al. 2022; Biderman et al. 2023 | 2205.10770; 2202.07646; 2304.11158 | Memorisasi | V |

Perhitungan kapasitas untuk bab ini (pakai bit/param dari paper di atas, bukan klaim Anda):
- 204M x 2 bit = ~0.41 Gbit. 492M x 2 bit = ~0.98 Gbit.
- Fact set Anda: 63,066 fakta (`fact_injection_stats.json`). Fakta ini jauh di bawah batas kapasitas. Jadi kegagalan fakta (value_hit 20-40% di `facts_eval_v8_honest.json`, n=20) **tidak otomatis karena kapasitas**. Kandidat penyebab yang lebih kuat: ekstraksi dan format (Allen-Zhu 3.1), ketidaksesuaian atribut-subjek yang tercatat di docstring `build_fact_injection_corpus.py`, dan SFT yang tidak stabil.
- Sampel eval facts hanya n=20. Terlalu kecil untuk kesimpulan statistik. Perbesar.

---

## 5. Translasi

| Paper | ID | Klaim | Status |
|---|---|---|---|
| Brown et al. 2020, *GPT-3* | 2005.14165 | Translasi muncul dari data multibahasa | V |
| Briakou et al. 2023, *Incidental Bilingualism in PaLM's Translation Capability* | 2305.10266 | Translasi muncul dari data bilingual yang tidak disengaja | V |
| Xu et al. 2023, *ALMA* | 2309.11674 | Finetuning data paralel sedikit tapi berkualitas | V |
| Xu et al. 2024, *CPO* | 2401.08417 | Preference optimization untuk MT. Cocok dengan DPO translasi Anda | V |
| Zhu et al. 2023, *Multilingual MT with LLMs* | 2304.04675 | | V |
| NLLB Team 2022 | 2207.04672 | Model NLLB-3.3B yang Anda pakai di `kaggle_nllb33b_jaid.py` | V |
| Goyal et al. 2022, *FLORES-101* | 2106.03193 | Benchmark translasi yang disarankan | V |
| Schwenk et al. 2019, *WikiMatrix* | 1907.05791 | Sumber paralel Anda | V |
| Schwenk et al. 2020, *CCMatrix* | 1911.04944 | Sumber paralel Anda | V |
| Feng et al. 2020, *LaBSE* | 2007.01852 | Filter paralel (`filter_parallel_labse.py`) | V |
| Rei et al. 2020, *COMET* | 2009.09025 | Metrik | V |
| Post 2018, *A Call for Clarity in Reporting BLEU* | 1804.08771 | sacreBLEU. **Gunakan ini untuk memperbaiki metrik di 0.3 poin 4** | V |

Data translasi Anda (`logs/mixed_bucket_stats.json`): **9.34M pasang EN↔JA** (~470M token) dalam bucket `mixed`. Komposisi: CCMatrix yang difilter LaBSE 6.47M (69%), `parallel_en_ja.tsv` 1.75M, TED talks 0.35M, OPUS-100 0.22M, WikiMatrix 0.09M, CCMatrix kandidat 0.44M, Tatoeba 0.02M. Plus SFT translasi (`sft_translation_tri`, 2.0M baris: 800K EN↔JA, sisanya EN↔ID). Angka "10.8M" di `KAGGLE_PRE-TRAINING.md` tidak cocok dengan statistik ini. Pakai 9.34M.

Caveat: 69% data paralel berasal dari CCMatrix hasil mining otomatis, yang berisik (komentar `fetch_corpus.py` mencatat pasangan yang salah). Filter LaBSE (Feng et al. 2020) membantu tapi tidak menghilangkan noise. Ini penjelasan yang masuk akal untuk kualitas en-ja yang rendah, dan bisa diuji dengan ablation (buang CCMatrix, bandingkan).

---

## 6. Alignment (SFT / DPO / GRPO / LoRA) dan kuantisasi

| Paper | ID | Dipakai | Status |
|---|---|---|---|
| Ouyang et al. 2022 | 2203.02155 | SFT/RLHF | V |
| Rafailov et al. 2023 | 2305.18290 | DPO | V |
| Shao et al. 2024 (DeepSeekMath) | 2402.03300 | GRPO | V |
| DeepSeek-AI 2025, *DeepSeek-R1* | 2501.12948 | RL dengan verifier berbasis aturan | V |
| Hu et al. 2021, *LoRA* | 2106.09685 | | V |
| Dettmers et al. 2023, *QLoRA* | 2305.14314 | | V |
| Lambert et al. 2024, *Tulu 3* | 2411.15124 | Resep post-training terbuka | V |
| Dettmers & Zettlemoyer 2022, *k-bit Inference Scaling Laws* | 2212.09720 | Kuantisasi 4-bit (Q4_K_M) | V |
| Marchisio et al. 2024, *How Does Quantization Affect Multilingual LLMs?* | 2407.03211 | Kuantisasi dan multibahasa. Relevan untuk `results/q_*.json` | V |
| Frantar et al. 2022, *GPTQ* | 2210.17323 | | V |

Catatan: `results/q_f16.json` vs `q_Q4_K_M.json` memberi bukti langsung. Keduanya sama-sama 3/10 value_ok pada FACTS EN. Jadi kuantisasi tidak menjadi penyebab utama (n=10, sangat kecil).

---

## 7. Evaluasi dan kontaminasi

| Paper | ID | Klaim | Status |
|---|---|---|---|
| Hendrycks et al. 2021, *MMLU* | 2009.03300 | | V |
| Zellers et al. 2019, *HellaSwag* | 1905.07830 | | V |
| Clark et al. 2018, *ARC* | 1803.05457 | | V |
| Liang et al. 2022, *HELM* | 2211.09110 | | V |
| Wei et al. 2022, *Emergent Abilities* | 2206.07682 | | V |
| Schaeffer et al. 2023, *Mirage?* | 2304.15004 | Metrik kontinu lebih informatif di skala kecil | V |
| Sainz et al. 2023, *NLP Evaluation in Trouble* | 2310.18018 | Kontaminasi | V |
| Wortsman et al. 2023, *Small-scale proxies for large-scale Transformer training instabilities* | 2309.14322 | Instabilitas training di skala kecil | V |

---

## 8. Paper yang ID-nya dari ingatan saya ternyata salah (jangan dipakai)

Daftar ini penting karena saya sempat menyebut paper ini di jawaban sebelumnya.

| Yang saya sebut | Masalah |
|---|---|
| JMMLU `2409.07413` | ID itu paper lain (SPRING, imaging). ID JMMLU belum ditemukan |
| `2410.00322` (saya kira kuantisasi) | Paper lain (ekonomi/teori informasi) |
| Gemma Team 2024 "Gemma dan Gemma 2" | Benar, tapi dua paper dengan ID terpisah (2403.08295 dan 2408.00118) |
| "He et al. 2024 (cek)" dan "Takase et al. 2024 (cek)" | Sudah terverifikasi: 2410.12883 dan 2406.16508 |

Belum terverifikasi atau tidak ada di arXiv (cek manual): JGLUE (Kurihara et al., LREC 2022), JMMLU, Ma et al. 2023 *At Which Training Stage Does Code Data Help LLMs Reasoning?*, Bengio et al. 2003 (JMLR), Ba et al. 2016 *Layer Normalization* (arXiv 1607.06450 dari ingatan), Radford et al. 2019 *GPT-2* (laporan OpenAI, bukan arXiv), Husain et al. *CodeSearchNet* (1909.09436 dari ingatan).

---

## 9. Rencana bab perbandingan yang bisa dipertanggungjawabkan

1. **Pilih satu model** (lihat 0.1) dan sesuaikan spesifikasi dan eval.
2. **Tabel spesifikasi pembanding** (parameter, token training, vocab, komposisi data) untuk: Pythia-160M/410M, SmolLM2-135M/360M, OLMo-2 kecil, Qwen2.5-0.5B, MobileLLM-125M/350M, Goldfish-125M (JA). Ambil angka dari paper masing-masing.
3. **Metrik utama: loss atau bits-per-byte per domain** (Huang 2404.09937; Schaeffer 2304.15004). Hitung pada held-out EN, JA, code, math. Ini sebanding antar tokenizer bila memakai bits-per-byte.
4. **Benchmark akurasi** hanya bila model > ~500M token-ekuivalen cukup. Di skala ini MMLU/GSM8K mendekati acak. Gunakan HellaSwag, ARC-easy, dan HumanEval pass@1 sebagai skor kasar, dan laporkan baseline acak.
5. **Translasi**: sacreBLEU + chrF + COMET pada FLORES-200 dev (JA↔EN), bukan pada Tatoeba/OpenSubtitles yang ada di training.
6. **Fakta**: perbesar eval (n≥200), pisahkan *seen* vs *holdout* (`data/eval/facts_holdout_*.jsonl` sudah ada) agar bisa membedakan memorisasi dari generalisasi.
7. **Ablation yang paling murah dan paling meyakinkan**: ganti satu faktor pada model 204M (rasio JA, ada/tidaknya fact-injection, MHA vs GQA), ukur loss per domain.
8. **Laporkan limitasi**: GSM8K train di pretraining, code Python-only, fact-injection berulang ~1000x, run 204M ~10 token/param.
