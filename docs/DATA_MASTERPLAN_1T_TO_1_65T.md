# Masterplan Korpus `1.0T` → `1.65T` Token (`2.0T` Efektif)
**Cakupan:** Quadrilingual Penuh (`ID`, `EN`, `JA`, `KO`) · Code (`Golang`, `JS`, `Java`, `C++`, `C`, `Assembly`) · Translasi 4-Arah (`12` Pasang Terarah) · Deep `Math` · Deep `Physics` · Deep `Biology` · Deep `Medical`

> **Kesimpulan Evaluasi Kapasitas `1.0T` Token:**  
> Memaksakan **4 bahasa alami penuh + 6 bahasa sistem/kode (termasuk Assembly) + translasi 12-arah + 4 domain sains/klinis tingkat lanjut (Math, Physics, Biology, Medical) yang kompatibel di ke-4 bahasa** ke dalam **`1.00T` (`1.000B`) token** sebenarnya **masih sempit (*tight*)** — rata-rata tiap domain STEM/Medis per bahasa non-Inggris hanya kebagian `~9B–15B` token.  
> **Rekomendasi Utama:** Naikkan target korpus unik ke **`1.65T` (`1.650B`) token unik**, yang saat dikombinasikan dengan *multi-epoch upsampling* (`1.5×–2.0×`) pada subset emas (Assembly terverifikasi kompilator, bukti teorema Math/Physics, pedoman klinis Medical, dan translasi level-dokumen) menghasilkan **`2.00T` (`2.000B`) token latih efektif** tanpa biaya storage tambahan di atas `1.65T`.

---

## 1. Alokasi Anggaran Token: Baseline `1.00T` vs Rekomendasi `1.65T` (`2.00T` Efektif)

| Pilar Utama | Sub-Komponen & Distribusi Bahasa | Tier A: `1.00T` Unik | Tier B: `1.65T` Unik *(Rekomendasi)* | Upsampling ke `2.00T` Efektif |
| :--- | :--- | :---: | :---: | :---: |
| **1. Bahasa Umum, Ensiklopedis & Akademik (`ID, EN, JA, KO`)** | Tier B: `EN 150B` · `JA 130B` · `KO 120B` · `ID 120B`<br/>*(Tier A: `EN 100B` · `JA 85B` · `KO 80B` · `ID 75B`)* | **`340B`** *(34.0%)* | **`520B`** *(31.5%)* | `1.00×` (`520B`) |
| **2. Code (`Go, JS, Java, C++, C, Assembly`) + Kompatibel 4 Bahasa** | Tier B: `C++ 80B` · `C 65B` · `Java 70B` · `JS 70B` · `Go 60B` · `Assembly/IR 35B` · `Multilingual Docs/CoT (ID/EN/JA/KO) 50B` | **`260B`** *(26.0%)* | **`430B`** *(26.1%)* | `1.19×` (`510B`) |
| **3. Translasi 4-Arah (`ID ↔ EN ↔ JA ↔ KO` = 12 Arah)** | Tier B: `ID↔EN 26B` · `EN↔JA 26B` · `EN↔KO 26B` · `JA↔KO 22B` · `ID↔JA 20B` · `ID↔KO 20B` *(kalimat + dokumen + istilah STEM/Code)* | **`80B`** *(8.0%)* | **`140B`** *(8.5%)* | `1.43×` (`200B`) |
| **4. Math Deep Knowledge (`ID, EN, JA, KO`)** | Tier B: `EN 96B` · `JA 30B` · `KO 28B` · `ID 26B` *(arXiv, Proof-Pile-2, Lean4, Olimpiade, S1–S3 CoT)* | **`105B`** *(10.5%)* | **`180B`** *(10.9%)* | `1.39×` (`250B`) |
| **5. Physics Deep Knowledge (`ID, EN, JA, KO`)** | Tier B: `EN 65B` · `JA 20B` · `KO 20B` · `ID 20B` *(Mekanika, Elektrodinamika, Kuantum/QFT, StatMech, Relativitas, Astro)* | **`70B`** *(7.0%)* | **`125B`** *(7.6%)* | `1.36×` (`170B`) |
| **6. Biology Deep Knowledge (`ID, EN, JA, KO`)** | Tier B: `EN 61B` · `JA 18B` · `KO 18B` · `ID 18B` *(Molekuler, Sel, Genetika/Genomik, Biokimia, Neurosains, Ekologi)* | **`65B`** *(6.5%)* | **`115B`** *(7.0%)* | `1.39×` (`160B`) |
| **7. Medical Deep Knowledge (`ID, EN, JA, KO`)** | Tier B: `EN 74B` · `JA 22B` · `KO 22B` · `ID 22B` *(PMC-OA Klinis, Pedoman WHO/PNPK/Minds, Farmakologi, USMLE/IgakuQA/KMLE/UKMPPD CoT)* | **`80B`** *(8.0%)* | **`140B`** *(8.5%)* | `1.36×` (`190B`) |
| **TOTAL KORPUS** | **Seluruh 7 Pilar × 4 Bahasa** | **`1,000B` (`1.00T`)** | **`1,650B` (`1.65T`)** | **`2,000B` (`2.00T`)** |

---

## 2. Estimasi Presisi Kebutuhan Storage (`1.00T` vs `1.65T` Token)

Berdasarkan kepadatan byte per token pada proyek ini (`stats.json`: `EN = 3.84 B/tok`, `JA = 3.69 B/tok`, `Code = 3.06 B/tok`, `Math = 3.49 B/tok`), dengan penambahan BPE merge penuh untuk `ID` (`~4.10 B/tok`) dan `KO` Hangul (`~3.45 B/tok`) serta literatur `Bio/Med` (`~4.05 B/tok`), rata-rata global tertimbang korpus ini adalah **`3.62 Byte / token`** dalam teks UTF-8 mentah.

> **PERINGATAN TOKENIZER (`bpe_v10_quad_65k.bin`):**  
> `bpe_v9.bin` (`vocab=49,152`) membuang merge `ID` dan `KO`. Sebelum men-tokenisasi `1T+` token, latih dulu **`bpe_v10_quad_65k.bin` (`vocab_size = 65,536`)** agar Hangul & Bahasa Indonesia tidak mengalami pembengkakan token 2.5x lipat sekaligus menjaga ID token tetap muat di **`uint16_t` (`2 Byte/token`)**.

| Lapisan Penyimpanan | Kepadatan Rata-Rata | Target `1.00T` Token | Target `1.65T` Token *(Rekomendasi)* | Fungsi & Rekomendasi Media |
| :--- | :---: | :---: | :---: | :--- |
| **1. Master Archive (`Parquet + Zstd-6` / `.jsonl.zst`)** | `~1.27 B/tok` | **`1.27 TB`** *(`1.15 TiB`)* | **`2.10 TB`** *(`1.91 TiB`)* | **Wajib disimpan permanen** di Object Storage (S3 / Cloudflare R2 / Backblaze B2) atau NAS ZFS. |
| **2. Teks UTF-8 Mentah Tak Terkompresi (`.txt` / `.jsonl`)** | `~3.62 B/tok` | **`3.62 TB`** *(`3.29 TiB`)* | **`5.97 TB`** *(`5.43 TiB`)* | **Jangan disimpan utuh di SSD!** Stream langsung dari ekstraksi ke `Parquet Zstd` + `.bin`. |
| **3A. Binary Shards Siap Latih — `uint16_t` (`vocab <= 65,536`)** | `2.015 B/tok` *(+ `.idx`)* | **`2.02 TB`** *(`1.83 TiB`)* | **`3.32 TB`** *(`3.02 TiB`)* | **Format Utama NVMe Training (`mmap`)**. |
| **3B. Binary Shards Alternatif — `uint32_t` (`vocab > 65,536`)** | `4.015 B/tok` *(+ `.idx`)* | **`4.02 TB`** *(`3.65 TiB`)* | **`6.62 TB`** *(`6.03 TiB`)* | Hanya jika memakai `vocab > 65,536` (mis. `131,072`). |
| **4. Scratch & Dedup Workspace *(Streaming per-Bucket)*** | Staging + MinHash/XXH3 | **`1.50 TB`** *(`1.36 TiB`)* | **`2.20 TB`** *(`2.00 TiB`)* | Buffer unduhan sementara Parquet mentah + indeks MinHash LSH (`~120–180 GB`). |

- **Total Rekomendasi Disk (`1.00T`, `uint16`):** **`8 TB` NVMe** + `~1.3 TB` Cloud Cold Backup.
- **Total Rekomendasi Disk (`1.65T`, `uint16`):** **`12 TB` NVMe** (atau `8 TB` NVMe + `8–12 TB` Enterprise HDD/Cloud Object Storage untuk arsip `Parquet Zstd` `2.10 TB`). Jika memilih `uint32` (`vocab=131K`), sediakan **`16 TB` NVMe**.

---

## 3. Katalog Lengkap Link Sumber Data Terverifikasi (7 Pilar × 4 Bahasa)

### Pilar 1 — Korpus Umum, Edukasi, Ensiklopedis & Akademik (`ID`, `EN`, `JA`, `KO`)
- **`ID, JA, KO` (FineWeb-2):** https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 (`ind_Latn`, `jpn_Jpan`, `kor_Hang`)
- **`EN` (FineWeb-Edu):** https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **`EN` (DCLM-Edu & Baseline):** https://huggingface.co/datasets/HuggingFaceTB/dclm-edu | https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
- **`EN` (Zyda-2 & Dolma):** https://huggingface.co/datasets/Zyphra/Zyda-2 | https://huggingface.co/datasets/allenai/dolma
- **`ID, EN, JA, KO` (CulturaX `[Gated: Auto]` & MADLAD-400):** https://huggingface.co/datasets/uonlp/CulturaX | https://huggingface.co/datasets/allenai/MADLAD-400
- **`ID, EN, JA, KO` (Wikimedia / Wikipedia):** https://huggingface.co/datasets/wikimedia/wikipedia | https://dumps.wikimedia.org
- **`KO` (Korean-WebText, Edu, Synthetic & CoT):** https://huggingface.co/datasets/HAERAE-HUB/KOREAN-WEBTEXT | https://huggingface.co/datasets/eliceai/korean-webtext-edu | https://huggingface.co/datasets/HAERAE-HUB/KOREAN-SyntheticText-1.5B | https://huggingface.co/datasets/HAERAE-HUB/HAE-RAE-COT-1.5M
- **`JA` (Swallow, LLM-JP & J-STAGE OA):** https://huggingface.co/datasets/tokyotech-llm/Swallow-Nemotron-Post-Training-Dataset-v1 | https://huggingface.co/datasets/tokyotech-llm/Swallow-Instruct-v0.1 | https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0 | https://www.jstage.jst.go.jp
- **`ID` (Indo4B, SEACrowd, Aya, Garuda & BSE):** https://huggingface.co/datasets/SEACrowd/indo4b | https://huggingface.co/SEACrowd | https://huggingface.co/datasets/CohereForAI/aya_collection | https://garuda.kemdikbud.go.id | https://buku.kemdikbud.go.id

### Pilar 2 — Code (`Golang`, `JS`, `Java`, `C++`, `C`, `Assembly`) + Kompatibilitas 4 Bahasa
- **The Stack v2 Dedup (`Go, JavaScript, Java, C++, C, Assembly`) `[Gated: Auto]`:** https://huggingface.co/datasets/bigcode/the-stack-v2-dedup | https://huggingface.co/datasets/bigcode/the-stack-v2
- **Common-Pile Permissive StackV2 & GitHub Archive:** https://huggingface.co/datasets/common-pile/stackv2 | https://huggingface.co/datasets/common-pile/github_archive
- **Stack-Edu (`C, Cpp, Go, Java, JavaScript` `score>=3`):** https://huggingface.co/datasets/HuggingFaceTB/stack-edu
- **StarCoderData & GitHub-Code:** https://huggingface.co/datasets/bigcode/starcoderdata | https://huggingface.co/datasets/codeparrot/github-code
- **ExeBench (`4.5M` Compilable C Functions -> Cross-Compile `x86_64/ARM64/RISC-V`):** https://huggingface.co/datasets/jordiae/exebench
- **LLM4Binary Decompilation (`Assembly <-> C`):** https://huggingface.co/datasets/LLM4Binary/decompile-ghidra-100k | https://huggingface.co/datasets/LLM4Binary/decompile-bench | https://huggingface.co/datasets/LLM4Binary/decompile-bench-raw *(Exclude `LLM4Binary/decompile-eval` for eval!)*
- **ComPile (`2.7 TB` LLVM IR from C/C++):** https://huggingface.co/datasets/llvm-ml/ComPile
- **Code Reasoning & Multilingual Adaptation:** https://huggingface.co/datasets/nvidia/OpenCodeReasoning | https://huggingface.co/datasets/nvidia/OpenCodeInstruct | https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2 | https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction | https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K | https://huggingface.co/datasets/common-pile/stackexchange

### Pilar 3 — Translasi 4-Arah (`ID ↔ EN ↔ JA ↔ KO` = 12 Arah Terarah)
- **NLLB Direct 6-Pairs (`ind_Latn`, `eng_Latn`, `jpn_Jpan`, `kor_Hang`):** https://huggingface.co/datasets/allenai/nllb *(Filter `LASER3/LaBSE >= 0.82`)*
- **OPUS Direct Bitext (`id-ja, id-ko, ja-ko, en-id, en-ja, en-ko`) & OPUS-100:** https://opus.nlpl.eu | https://huggingface.co/datasets/Helsinki-NLP/opus-100
- **Multi-Way Parallel (`TED Talks, WikiMatrix, CCMatrix, GlobalVoices, Tatoeba`):** https://huggingface.co/datasets/sentence-transformers/parallel-sentences-talks | https://huggingface.co/datasets/sentence-transformers/parallel-sentences-wikimatrix | https://huggingface.co/datasets/sentence-transformers/parallel-sentences-ccmatrix | https://huggingface.co/datasets/sentence-transformers/parallel-sentences-global-voices | https://huggingface.co/datasets/sentence-transformers/parallel-sentences-tatoeba
- **Asian Language Treebank (`ID-JA-EN` Human Parallel) & TED EN-ID:** https://huggingface.co/datasets/SEACrowd/parallel_asian_treebank | https://huggingface.co/datasets/SEACrowd/ted_en_id
- **JParaCrawl, KFTT & ASPEC (`EN-JA` Web & STEM):** https://huggingface.co/datasets/Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus | http://www.phontron.com/kftt/ | https://lotus.kuee.kyoto-u.ac.jp/ASPEC/
- **Korean Parallel Corpora (`EN-KO`):** https://huggingface.co/datasets/lemon-mint/korean_parallel_sentences_v1.1 | https://huggingface.co/datasets/Moo/korean-parallel-corpora
- **Benchmark Holdout *(Dilarang Masuk Training)*:** https://huggingface.co/datasets/openlanguagedata/flores_plus

### Pilar 4 — Math Deep Knowledge (`ID, EN, JA, KO`)
- **MegaMath (`371B` tok):** https://huggingface.co/datasets/LLM360/MegaMath
- **FineMath (`4plus` & `InfiWebMath-4plus`):** https://huggingface.co/datasets/HuggingFaceTB/finemath
- **Proof-Pile-2 & OpenWebMath:** https://huggingface.co/datasets/EleutherAI/proof-pile-2 | https://huggingface.co/datasets/open-web-math/open-web-math
- **Lean-Workbook (`Lean 4` Formal Proofs):** https://huggingface.co/datasets/internlm/Lean-Workbook
- **Olympiad & Deep Math CoT:** https://huggingface.co/datasets/nvidia/OpenMathReasoning | https://huggingface.co/datasets/nvidia/OpenMathInstruct-2 | https://huggingface.co/datasets/AI-MO/NuminaMath-1.5 | https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
- **Clean Structured & Regional (`JA / KO`):** https://huggingface.co/datasets/tokyotech-llm/swallow-math-v2 | https://huggingface.co/datasets/kunishou/OpenMathInstruct-1-1.8m-ja | https://huggingface.co/datasets/HAERAE-HUB/HR-Instruct-Math-v0.1

### Pilar 5 — Physics Deep Knowledge (`ID, EN, JA, KO`)
- **arXiv Full-Text Physics (`quant-ph, hep-*, cond-mat, gr-qc, astro-ph, physics.*`):** https://huggingface.co/datasets/common-pile/arxiv_papers | https://huggingface.co/datasets/common-pile/arxiv_abstracts
- **peS2o (`Physics` Subset):** https://huggingface.co/datasets/allenai/peS2o | https://huggingface.co/datasets/common-pile/peS2o
- **Textbooks (`Physics LibreTexts`, `OpenStax University Physics`, `DOAB`):** https://huggingface.co/datasets/common-pile/libretexts | https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs | https://huggingface.co/datasets/common-pile/doab
- **Physics StackExchange & Olympiad/University Derivations:** https://huggingface.co/datasets/common-pile/stackexchange | https://huggingface.co/datasets/MegaScience/MegaScience | https://huggingface.co/datasets/facebook/natural_reasoning | https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k | https://huggingface.co/datasets/TIGER-Lab/TheoremQA | https://huggingface.co/datasets/camel-ai/physics

### Pilar 6 — Biology Deep Knowledge (`ID, EN, JA, KO`)
- **PMC Open Access (`Life Sciences / Molecular / Genomics / Ecology` Subset):** https://huggingface.co/datasets/common-pile/pubmed | https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/
- **bioRxiv Preprints:** https://huggingface.co/datasets/mteb/raw_biorxiv | https://www.biorxiv.org/tdm
- **peS2o (`Biology`) & Biodiversity Heritage Library:** https://huggingface.co/datasets/allenai/peS2o | https://huggingface.co/datasets/common-pile/biodiversity_heritage_library_filtered
- **Textbooks (`Biology LibreTexts` & `OpenStax Biology 2e / Microbiology`):** https://huggingface.co/datasets/common-pile/libretexts | https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs
- **Biology/Bioinformatics StackExchange & Reasoning:** https://huggingface.co/datasets/common-pile/stackexchange | https://huggingface.co/datasets/MegaScience/MegaScience | https://huggingface.co/datasets/camel-ai/biology

### Pilar 7 — Medical Deep Knowledge (`ID, EN, JA, KO`)
- **PMC-OA Clinical Full-Text & 36M PubMed Abstracts:** https://huggingface.co/datasets/common-pile/pubmed | https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/ | https://huggingface.co/datasets/ncbi/pubmed | https://huggingface.co/datasets/MedRAG/pubmed
- **Clinical Practice Guidelines (`WHO, CDC, NICE, WikiDoc`):** https://huggingface.co/datasets/epfl-llm/guidelines
- **Multilingual Medical Books & Encyclopedias:** https://huggingface.co/datasets/FreedomIntelligence/ApolloCorpus | https://huggingface.co/datasets/MedRAG/textbooks | https://huggingface.co/datasets/common-pile/libretexts
- **Clinical Differential Diagnosis & CoT:** https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data | https://huggingface.co/datasets/TsinghuaC3I/UltraMedical | https://huggingface.co/datasets/openlifescienceai/medmcqa | https://huggingface.co/datasets/bigbio/med_qa | https://huggingface.co/datasets/qiaojin/PubMedQA
- **Regional Clinical & Licensing (`JA, KO, ID`):**
  - `JA`: https://huggingface.co/datasets/japan-ai-official/igakuqa-subset-curated | https://www.jstage.jst.go.jp | https://minds.jcqhc.or.jp
  - `KO`: https://huggingface.co/datasets/sean0042/KorMedMCQA | https://synapse.koreamed.org
  - `ID`: https://huggingface.co/datasets/IndoHealth-NLP/NCD_Instruct-Tuning_Medical_QA_Indonesian | https://garuda.kemdikbud.go.id | https://kemkes.go.id

---

## 4. Ringkasan Tahapan Eksekusi Teknis (6 Fase)

1. **Fase 1 (`Tokenizer v10`):** Latih `bpe_v10_quad_65k.bin` (`vocab_size = 65,536`, single-digit splitting `0-9`, kelas aksara Hangul + CJK + Latin + Code + LaTeX, mempertahankan 16 token spesial `0..15` yang sudah ada).
2. **Fase 2 (`Streaming Ingestion`):** Normalisasi `NFC` (bukan `NFKC` agar simbol LaTeX & isotop aman), filter bahasa `fastText lid.176`, simpan ke `Parquet Zstd-6`.
3. **Fase 3 (`Deduplikasi & Dekontaminasi`):** Gunakan `datatrove` / C++ terpartisi disk (`XXH3-128` exact + `MinHash LSH 14×8` fuzzy + filter AST/auto-generated untuk kode) serta daftar hitam 13-gram seluruh benchmark 4 bahasa (`FLORES+`, `HumanEval/MultiPL-E`, `decompile-eval`, `MGSM`, `GPQA`, `OlympiadBench`, `USMLE/IgakuQA/KorMedMCQA/IndoMMLU`).
4. **Fase 4 (`Assembly Factory & Multilingual STEM Bridging`):** Cross-compile jutaan fungsi `ExeBench` + `Stack-Edu` ke `x86_64`, `ARM64`, dan `RISC-V` (`-O0/-O2/-O3`), serta jalankan *Protected-Span Transcreation* (mengunci LaTeX & blok kode) dari top-tier `FineMath-4+`, `LibreTexts`, `PMC-OA`, dan `Stack-Edu` ke `ID, JA, KO`.
5. **Fase 5 (`Quality Tiering`):** Pisahkan shard ke dalam 3 tier kurikulum (`Stage A Foundation`, `Stage B Mid-Train STEM/Code/12-Way Doc Translation`, `Stage C Gold Annealing`).
6. **Fase 6 (`Binary Pre-Tokenization`):** Konversi `Parquet Zstd` (`2.10 TB`) menjadi shard biner `uint16_t` `.bin` + `.idx` (`3.32 TB` @ `1.65T`) siap `mmap` zero-copy.
