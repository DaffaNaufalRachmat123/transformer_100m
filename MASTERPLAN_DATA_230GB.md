# MASTERPLAN DATA `235B` TOKEN (EDISI MENTOK `230.00 GB` DARI DISK `250 GB`)
**Mencakup 100% 6 Rumpun Ilmu Pengetahuan Dunia (OECD/UNESCO) + Deep Code (`Go, JS, Java, C++, C, Assembly`) + Seluruh `7.170+` Bahasa Dunia (Ethnologue Ed. 29 / Glottolog 5.3)**

> **Terakhir diperbarui:** 27 November 2026  
> **Lokasi File:** `/Users/daffanaufalrachmat/Documents/Real Projects/TransformerSample/transformer_500m_cuda/MASTERPLAN_DATA_230GB.md`  
> **Target Utama:** Memaksimalkan korpus hingga batas **`230.00 GB`** (`225.00 GB` blok terkompresi `.tok.zst` + `5.00 GB` indeks cepat `.zstidx`), menghasilkan **`235,00 Miliar` (`235.00B` / `0.235T`) token unik** berkualitas puncak (**naik `+60.00B` token** dibanding edisi `175B`), sambil tetap menyisakan **`20.00 GB`** ruang kerja aman di dalam batas disk **`250.00 GB`**.

---

## 1. PETA ANGGARAN RUANG KETAT `230.00 GB` DI DALAM DISK `250.00 GB`

Agar korpus bisa dimentokkan ke **`230.00 GB`** tanpa membuat proses *download* maupun *training* gagal karena kehabisan ruang di sisa **`20.00 GB`**, pemakaian ruang dibagi ke dalam **2 siklus terpisah** (file dedup & temp dihapus otomatis sebelum training dimulai):

| Komponen Disk | Saat Siklus Unduh & Build (`GB`) | Saat Siklus Training GPU (`GB`) | Strategi Anti-Penuh (`Zero Overflow`) |
| :--- | :---: | :---: | :--- |
| **1. Shards Korpus `.tok.zst` (`235.00B` Token, `Zstd-15`)** | **`225.00 GB`** | **`225.00 GB`** | Rata-rata empiris **`0.957 Byte/token`**. Dekompresi ke RAM `>1.8 GB/s` per core CPU (`0%` beban BPE live). |
| **2. Indeks Cepat Blok `1M` Token (`.zstidx` + Manifest)** | **`5.00 GB`** | **`5.00 GB`** | Memungkinkan *random seek* instan O(1) ke blok mana pun. **Total Korpus = `230.00 GB` tepat.** |
| **3. Bloom Filter Anti-Duplikat (`mmap`) + Rolling Temp (`1 Shard`)** | `4.00 GB` *(sementara)* | `0.00 GB` *(dihapus)* | `2.5 GB` Bloom Filter + maks `1.5 GB` 1-shard Parquet sementara; keduanya di-`unlink` begitu korpus selesai. |
| **4. Checkpoints Training (`ckpt_pretrain` + `ckpt_optim` + `.tmp`)** | `0.00 GB` *(belum ada)* | `8.84 GB` | Bobot FP16 (`0.98 GB`) + AdamW FP32 (`3.93 GB`) + buffer tulis atomik `.tmp` (`3.93 GB`). |
| **5. Sisa Ruang Bebas Murni (*Headroom OS & Log*)** | **`16.00 GB`** | **`11.16 GB`** | Tetap aman dari `Disk Full`. *(Catatan: Jika `250 GB` Anda adalah label SSD pabrikan yang di OS terbaca `232.8 GiB` = `250.0 GB` desimal, angka GB di tabel ini sudah memakai basis desimal `10^9` byte yang sama persis!)* |
| **TOTAL DISK** | **`250.00 GB`** | **`250.00 GB`** | **Mentok Maksimal & 100% Terkendali.** |

---

## 2. MATRIKS ALOKASI `235.00B` TOKEN (`+60B` EKSPANSI DARI EDISI `175B`)

Tambahan **`+60.00B` token** disuntikkan langsung ke cabang-cabang paling haus kedalaman penalaran (`+38.00B` ke 6 Rumpun Ilmu & Deep Code/Assembly, serta `+22.00B` ke 7.170+ Bahasa & Translasi 12-Arah):

| Bagian & Rumpun Utama | Edisi `175B` | **Edisi Mentok `230GB` (`235.00B`)** | Rasio `Zstd-15` (`B/tok`) | Ukuran `.tok.zst` (`GB`) |
| :--- | :---: | :---: | :---: | :---: |
| **Bagian 1A — Ilmu Formal & Deep Code (`Go, JS, Java, C++, C, ASM`)** | `46.00B` | **`63.00B`** *(+17.0B)* | `0.700` | **`44.10 GB`** |
| **Bagian 1B — Ilmu Alam (`Fisika, Kimia, Astronomi, Bumi, Hayati, Neuro`)** | `22.00B` | **`30.00B`** *(+8.0B)* | `1.045` | **`31.35 GB`** |
| **Bagian 1C — Ilmu Sosial (`7 Cabang Lengkap`)** | `9.00B` | **`12.00B`** *(+3.0B)* | `1.150` | **`13.80 GB`** |
| **Bagian 1D — Humaniora (`6 Cabang Lengkap`)** | `9.00B` | **`12.00B`** *(+3.0B)* | `1.140` | **`13.68 GB`** |
| **Bagian 1E — Ilmu Terapan & Profesi (`Medis, Teknik, Hukum, Bisnis, Agro, dll.`)** | `17.00B` | **`22.50B`** *(+5.5B)* | `1.112` | **`25.02 GB`** |
| **Bagian 1F — Ilmu Interdisipliner (`AI, Bioinformatika, Kognitif, Iklim, Nano`)** | `5.00B` | **`6.50B`** *(+1.5B)* | `1.077` | **`7.00 GB`** |
| **Bagian 2A — Ensiklopedia Struktur & Leksikon `7.674` Bahasa (`CLDF + UD`)** | `1.00B` | **`1.50B`** *(+0.5B)* | `0.900` | **`1.35 GB`** |
| **Bagian 2B — Fondasi Puncak 4 Bahasa Inti (`ID, EN, JA, KO`)** | `33.00B` | **`43.50B`** *(+10.5B)* | `1.070` | **`46.55 GB`** |
| **Bagian 2C — 9 Bahasa Makro Dunia Lainnya (`ZH, ES, AR, HI, FR, PT, BN, RU, DE`)** | `12.00B` | **`15.50B`** *(+3.5B)* | `1.040` | **`16.12 GB`** |
| **Bagian 2D — `1.800+` Bahasa Lintas 19 Rumpun + Isolat + Isyarat + Kreol + Buatan + Klasik** | `11.00B` | **`15.00B`** *(+4.0B)* | `1.010` | **`15.15 GB`** |
| **Bagian 2E — Translasi Multi-Arah (`12-Arah ID↔EN↔JA↔KO` + Jangkar `1.000` Bahasa)** | `10.00B` | **`13.50B`** *(+3.5B)* | `0.806` | **`10.88 GB`** |
| **SUBTOTAL SHARDS `.tok.zst`** | `175.00B` | **`235.00B`** | **`0.957`** | **`225.00 GB`** |
| **INDEKS BLOK & MANIFEST (`.zstidx`)** | — | *(Offset `235.000` blok)* | — | **`5.00 GB`** |
| **GRAND TOTAL KORPUS DI DISK** | `174.5 GB` | **`235.00B` Token** | — | **`230.00 GB`** |

---

# BAGIAN 1: RINCIAN KUOTA `146.00B` TOKEN & LINK AKURAT — 6 RUMPUN ILMU (OECD/UNESCO)

## A. ILMU FORMAL & REKAYASA KODE (`63.00B` Token | `44.10 GB`)

| Sub-Cabang | Kuota Baru | Rincian Isi & Link Sumber Terverifikasi |
| :--- | :---: | :--- |
| **1. Matematika Lengkap** *(Aljabar, Geometri, Kalkulus, Analisis, Teori Bilangan, Topologi, Kombinatorika)* | **`16.50B`** | `finemath-4plus` + `infiwebmath-4plus` + `Swallow-Math-v2` + *Mathematics LibreTexts* + `Proof-Pile-2` (`arXiv math.*`) + `NuminaMath-1.5` + `OpenMathReasoning` + `OpenMathInstruct-JA` & transkreasi pembuktian ke `ID, JA, KO`.<br/>• https://huggingface.co/datasets/HuggingFaceTB/finemath<br/>• https://huggingface.co/datasets/tokyotech-llm/swallow-math-v2<br/>• https://huggingface.co/datasets/EleutherAI/proof-pile-2<br/>• https://huggingface.co/datasets/AI-MO/NuminaMath-1.5<br/>• https://huggingface.co/datasets/nvidia/OpenMathReasoning<br/>• https://huggingface.co/datasets/kunishou/OpenMathInstruct-1-1.8m-ja<br/>• https://huggingface.co/datasets/LLM360/MegaMath |
| **2. Logika, Statistika, Stokastik & Sistem/Keputusan** *(Lean4/Coq/Isabelle, Peluang, Teori Permainan, Riset Operasi, Sibernetika)* | **`5.50B`** | `Lean-Workbook` + `AlgebraicStack` (`Lean/Coq/Isabelle`) + *Statistics LibreTexts* + `stats.stackexchange` & `or.stackexchange` + paper `arXiv` (`math.LO, cs.LO, stat.*, math.PR, cs.GT, math.OC, eess.SY`).<br/>• https://huggingface.co/datasets/internlm/Lean-Workbook<br/>• https://huggingface.co/datasets/EleutherAI/proof-pile-2<br/>• https://huggingface.co/datasets/common-pile/arxiv_papers<br/>• https://huggingface.co/datasets/common-pile/libretexts<br/>• https://huggingface.co/datasets/common-pile/stackexchange |
| **3. Komp. Teoretis, Algoritma, Kriptografi & Deep Code (`Go, JS, Java, C++, C, Assembly`)** | **`41.00B`** | **Alokasi diperbesar:** `C++ (7.5B)` · `C (6.0B)` · `Java (6.0B)` · `JavaScript (6.0B)` · `Golang (5.5B)` · `Assembly x86_64/ARM64/RISC-V + LLVM IR pairs (5.0B)` · `Algoritma, Kriptografi & Multilingual Code CoT ID/EN/JA/KO (5.0B)`.<br/>• https://huggingface.co/datasets/HuggingFaceTB/stack-edu<br/>• https://huggingface.co/datasets/common-pile/stackv2<br/>• https://huggingface.co/datasets/bigcode/the-stack-v2-dedup<br/>• https://huggingface.co/datasets/jordiae/exebench *(cross-compile `-O0/-O2/-O3` ke `x86_64, ARM64, RISC-V`)*<br/>• https://huggingface.co/datasets/LLM4Binary/decompile-ghidra-100k<br/>• https://huggingface.co/datasets/LLM4Binary/decompile-bench<br/>• https://huggingface.co/datasets/llvm-ml/ComPile<br/>• https://huggingface.co/datasets/nvidia/OpenCodeReasoning<br/>• https://huggingface.co/datasets/deepmind/code_contests<br/>• https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2 |

---

## B. ILMU ALAM: FISIKA & HAYATI (`30.00B` Token | `31.35 GB`)

| Sub-Cabang | Kuota Baru | Rincian Isi & Link Sumber Terverifikasi |
| :--- | :---: | :--- |
| **1. Fisika Lengkap** *(Mekanika, Termodinamika, Elektromagnetisme, Optika, Kuantum, Relativitas, Partikel, Zat Padat)* | **`9.00B`** | *Physics LibreTexts* + *OpenStax Vol 1–3* + `physics.stackexchange` + `arXiv` (`quant-ph, hep-*, cond-mat.*, gr-qc, nucl-*, physics.*`) + `MegaScience / NaturalReasoning / TheoremQA / CAMEL-Physics` + transkreasi `ID/JA/KO`.<br/>• https://huggingface.co/datasets/common-pile/libretexts · https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs · https://huggingface.co/datasets/common-pile/arxiv_papers · https://huggingface.co/datasets/common-pile/stackexchange · https://huggingface.co/datasets/MegaScience/MegaScience · https://huggingface.co/datasets/facebook/natural_reasoning · https://huggingface.co/datasets/TIGER-Lab/TheoremQA · https://huggingface.co/datasets/camel-ai/physics |
| **2. Kimia Lengkap** *(Organik, Anorganik, Fisik, Analitik, Biokimia, Kimia Material)* | **`4.80B`** | https://huggingface.co/datasets/jablonkagroup/chempile-education · https://huggingface.co/datasets/jablonkagroup/chempile-reasoning · https://huggingface.co/datasets/jablonkagroup/chempile-paper · https://huggingface.co/datasets/osunlp/SMolInstruct · https://huggingface.co/datasets/zjunlp/Mol-Instructions · https://huggingface.co/datasets/camel-ai/chemistry · https://pubchem.ncbi.nlm.nih.gov |
| **3. Astronomi (`1.70B`) & Ilmu Bumi (`2.50B`)** *(Astrofisika, Kosmologi, Keplanetan; Geologi, Geofisika, Meteorologi, Klimatologi, Oseanografi, Hidrologi, Geografi Fisik)* | **`4.20B`** | https://huggingface.co/datasets/common-pile/arxiv_papers *(`astro-ph.*, physics.ao-ph, physics.geo-ph`)* · https://ui.adsabs.harvard.edu · https://eartharxiv.org · https://huggingface.co/datasets/mlfoundations-dev/stackexchange_earthscience · https://huggingface.co/datasets/common-pile/libretexts · https://huggingface.co/datasets/allenai/peS2o |
| **4. Biologi Dasar (`5.40B`), Organisme (`4.00B`) & Neurosains (`2.60B`)** *(Sel/Molekuler, Genetika, Evolusi, Anatomi, Fisiologi; Botani, Zoologi, Mikrobiologi, Virologi, Mikologi, Ekologi, Kelautan; Sistem Saraf & Otak)* | **`12.00B`** | https://huggingface.co/datasets/common-pile/pubmed · https://huggingface.co/datasets/mteb/raw_biorxiv · https://www.biorxiv.org/tdm · https://huggingface.co/datasets/common-pile/biodiversity_heritage_library_filtered · https://huggingface.co/datasets/common-pile/libretexts · https://huggingface.co/datasets/camel-ai/biology · https://www.ncbi.nlm.nih.gov/genbank/ · https://huggingface.co/datasets/allenai/peS2o |

---

## C. ILMU SOSIAL LENGKAP (`12.00B` Token | `13.80 GB`)
- **Sosiologi (`1.70B`):** https://osf.io/preprints/socarxiv · https://huggingface.co/datasets/common-pile/libretexts · https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs · https://huggingface.co/datasets/allenai/peS2o
- **Psikologi (`2.40B`):** https://osf.io/preprints/psyarxiv · https://huggingface.co/datasets/Psychotherapy-LLM/PsychoCounsel-Preference · https://huggingface.co/datasets/jkhedri/psychology-dataset · https://huggingface.co/datasets/common-pile/pubmed
- **Ilmu Ekonomi (`3.00B`):** https://www.econstor.eu · https://huggingface.co/datasets/common-pile/arxiv_papers *(`econ.*, q-fin.*`)* · https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k · https://huggingface.co/datasets/AdaptLLM/finance-tasks
- **Ilmu Politik, Kebijakan Publik & HI (`2.00B`):** https://huggingface.co/datasets/common-pile/usgpo · https://huggingface.co/datasets/Helsinki-NLP/europarl · https://opus.nlpl.eu *(`MultiUN`)* · https://huggingface.co/datasets/allenai/peS2o
- **Antropologi & Arkeologi (`1.20B`):** https://huggingface.co/datasets/WithinUsAI/linguistic_anthropology_25k · https://huggingface.co/datasets/common-pile/doab · https://core.ac.uk
- **Geografi Manusia & Demografi (`0.80B`):** https://huggingface.co/datasets/allenai/peS2o *(`Geography`)* · https://huggingface.co/datasets/common-pile/doab · https://huggingface.co/datasets/common-pile/libretexts
- **Komunikasi, Jurnalisme & Kriminologi (`0.90B`):** https://huggingface.co/datasets/common-pile/news · https://huggingface.co/datasets/PleIAs/YouTube-Commons · https://huggingface.co/datasets/common-pile/caselaw_access_project

---

## D. HUMANIORA LENGKAP (`12.00B` Token | `13.68 GB`)
- **Filsafat (`2.00B`):** https://plato.stanford.edu · https://huggingface.co/datasets/LisaMegaWatts/philosophy-corpus · https://huggingface.co/datasets/common-pile/stackexchange *(`philosophy`)* · https://huggingface.co/datasets/common-pile/project_gutenberg
- **Sejarah (`3.00B`):** https://huggingface.co/datasets/common-pile/public_domain_review · https://huggingface.co/datasets/common-pile/library_of_congress · https://huggingface.co/datasets/common-pile/pre_1929_books · https://huggingface.co/datasets/common-pile/doab
- **Linguistik & Filologi (`2.00B`):** https://huggingface.co/datasets/universal-dependencies/universal_dependencies · https://huggingface.co/datasets/mlfoundations-dev/stackexchange_linguistics · https://huggingface.co/datasets/common-pile/doab
- **Sastra Kanonik Dunia (`2.10B`):** https://huggingface.co/datasets/common-pile/project_gutenberg · https://huggingface.co/datasets/manu/project_gutenberg · https://dumps.wikimedia.org · https://huggingface.co/datasets/common-pile/doab
- **Seni Rupa, Musik, Tari, Teater, Film & Desain (`1.30B`):** https://huggingface.co/datasets/sander-wood/irishman · https://huggingface.co/datasets/huggan/wikiart · https://huggingface.co/datasets/common-pile/libretexts · https://huggingface.co/datasets/common-pile/doab
- **Agama & Kajian Budaya (`1.60B`):** https://github.com/Sefaria/Sefaria-Export · https://cbeta.org · https://tanzil.net · https://gretil.sub.uni-goettingen.de · https://huggingface.co/datasets/common-pile/doab

---

## E. ILMU TERAPAN & PROFESI LENGKAP (`22.50B` Token | `25.02 GB`)
- **Kesehatan & Kedokteran Mendalam (`10.00B` — Kedokteran, Farmasi, Keperawatan, Gigi, Hewan, Gizi, Kesmas lintas `ID/EN/JA/KO`):** https://huggingface.co/datasets/epfl-llm/guidelines · https://huggingface.co/datasets/MedRAG/textbooks · https://huggingface.co/datasets/FreedomIntelligence/ApolloCorpus · https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data · https://huggingface.co/datasets/TsinghuaC3I/UltraMedical · https://huggingface.co/datasets/common-pile/pubmed · https://huggingface.co/datasets/openlifescienceai/medmcqa · https://huggingface.co/datasets/japan-ai-official/igakuqa-subset-curated · https://huggingface.co/datasets/sean0042/KorMedMCQA · https://huggingface.co/datasets/IndoHealth-NLP/NCD_Instruct-Tuning_Medical_QA_Indonesian
- **Teknik / Rekayasa Semua Jurusan (`4.60B` — Sipil, Mesin, Elektro, Kimia, Industri, Dirgantara, Material, Nuklir, Biomedis, Lingkungan):** https://huggingface.co/datasets/common-pile/libretexts · https://huggingface.co/datasets/HUPD/hupd · https://huggingface.co/datasets/ccdv/patent-classification · https://huggingface.co/datasets/common-pile/arxiv_papers *(`eess.*`)*
- **Hukum (`3.00B` — Pidana, Perdata, Tata Negara, Bisnis, Internasional):** https://huggingface.co/datasets/pile-of-law/pile-of-law · https://huggingface.co/datasets/common-pile/caselaw_access_project · https://huggingface.co/datasets/coastalcph/multi_eurlex · https://huggingface.co/datasets/common-pile/regulations
- **Bisnis, Manajemen, Akuntansi & Keuangan (`2.10B`):** https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs · https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k · https://huggingface.co/datasets/TeraflopAI/SEC-EDGAR · https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train
- **Agro & Pangan (`1.30B`):** https://agris.fao.org · https://huggingface.co/datasets/allenai/peS2o · https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only · https://huggingface.co/datasets/common-pile/foodista
- **Pendidikan, Arsitektur & PWK (`0.90B`):** https://huggingface.co/datasets/common-pile/oercommons · https://huggingface.co/datasets/common-pile/pressbooks · https://huggingface.co/datasets/common-pile/doab · https://core.ac.uk
- **Perpustakaan, Kearsipan & Strategi Pertahanan/Militer (`0.60B`):** https://discover.dtic.mil · https://huggingface.co/datasets/common-pile/library_of_congress · https://huggingface.co/datasets/common-pile/usgpo

---

## F. ILMU INTERDISIPLINER (`6.50B` Token | `7.00 GB`)
- **AI & Ilmu Data (`2.30B`):** https://huggingface.co/datasets/common-pile/arxiv_papers *(`cs.LG, cs.AI, cs.CL, cs.CV, stat.ML`)* · https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k
- **Bioinformatika (`1.30B`):** https://www.uniprot.org · https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks · https://huggingface.co/datasets/common-pile/stackexchange *(`bioinformatics`)*
- **Ilmu Kognitif (`0.90B`):** https://huggingface.co/datasets/common-pile/stackexchange *(`cogsci`)* · https://osf.io/preprints/psyarxiv · https://huggingface.co/datasets/allenai/peS2o
- **Ilmu Lingkungan & Iklim (`1.20B`):** https://eartharxiv.org · https://huggingface.co/datasets/allenai/peS2o *(`Environmental Science`)* · https://huggingface.co/datasets/common-pile/regulations
- **Nanoteknologi & Ekonomi Perilaku (`0.80B`):** https://huggingface.co/datasets/common-pile/arxiv_papers *(`cond-mat.mes-hall, cond-mat.mtrl-sci`)* · https://huggingface.co/datasets/jablonkagroup/chempile-paper · https://www.econstor.eu

---

# BAGIAN 2: RINCIAN KUOTA `89.00B` TOKEN & LINK AKURAT — SELURUH `7.170+` BAHASA DUNIA (`90.05 GB`)

1. **2A — Katalog Struktur, Tipologi, Fonem & Leksikon Penuh `7.674` Bahasa (`1.50B` Token | `1.35 GB`):**
   - https://glottolog.org · https://github.com/glottolog/glottolog · https://github.com/lexibank/asjp · https://github.com/grambank/grambank · https://github.com/cldf-datasets/wals · https://github.com/phoible/dev · https://github.com/cldf-datasets/apics · https://github.com/unimorph · https://huggingface.co/datasets/universal-dependencies/universal_dependencies
2. **2B — Fondasi Puncak 4 Bahasa Inti (`43.50B` Token | `46.55 GB` — `EN 12.0B`, `ID 10.5B`, `JA 10.5B`, `KO 10.5B`):**
   - https://huggingface.co/datasets/wikimedia/wikipedia · https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu · https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 · https://huggingface.co/datasets/SEACrowd/indo4b · https://huggingface.co/datasets/CohereForAI/aya_collection · https://huggingface.co/datasets/tokyotech-llm/Swallow-Nemotron-Post-Training-Dataset-v1 · https://huggingface.co/datasets/eliceai/korean-webtext-edu · https://huggingface.co/datasets/HAERAE-HUB/KOREAN-WEBTEXT · https://huggingface.co/datasets/HAERAE-HUB/HAE-RAE-COT-1.5M
3. **2C — 9 Bahasa Makro Dunia Lainnya (`15.50B` Token | `16.12 GB` — `Mandarin 2.8B`, `Spanyol 2.1B`, `Arab 2.1B`, `Hindi 2.0B`, `Prancis 1.9B`, `Portugis 1.5B`, `Bengali 1.5B`, `Rusia + Jerman 1.6B`):**
   - https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1 · https://huggingface.co/datasets/ClusterlabAi/101_billion_arabic_words_dataset · https://huggingface.co/datasets/ai4bharat/sangraha · https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 · https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned
4. **2D — `1.800+` Bahasa Lintas 19 Rumpun + 5 Kelompok Non-Genetik (`15.00B` Token | `15.15 GB`):**
   - *Niger-Kongo, Afro-Asiatik Non-AR, Nilo-Sahara & Khoisan (`2.80B`):* https://huggingface.co/datasets/castorini/wura · https://huggingface.co/datasets/lelapa/Inkuba-Mono · https://huggingface.co/datasets/masakhane/masakhanews · https://repo.sadilar.org · https://github.com/BibleNLP/ebible
   - *Austronesia Nusantara/Pasifik, Trans-Nugini (`Dani, Ekari, Enga, Huli, Melpa`), Austroasiatik & Kra-Dai (`3.00B`):* https://huggingface.co/SEACrowd · https://github.com/BibleNLP/ebible · https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
   - *Dravida (`Tamil, Telugu, Kannada, Malayalam`) & Indo-Arya Regional (`1.90B`):* https://huggingface.co/datasets/ai4bharat/sangraha
   - *Indo-Eropa Lainnya, Uralik, Turkik, Kaukasus & Mongolik/Tungusik (`2.70B`):* https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 · https://huggingface.co/datasets/turkic-interlingua/turkic_xwmt · https://huggingface.co/datasets/joanitolopo/flores-plus-uralic · https://huggingface.co/datasets/cis-lmu/GlotCC-V1
   - *Sino-Tibet Non-Mandarin (`Kanton, Wu, Min, Hakka, Tibet, Burma`), Ryukyu & Jeju (`1.30B`):* https://openpecha.org · https://huggingface.co/datasets/shajiu/TibetanGeneral_corpus · https://huggingface.co/datasets/Mireu-Lab/Korean-jeju-dialect-speech · https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
   - *Pribumi Amerika (`Quechua, Guarani, Nahuatl, Maya, Navajo, Cherokee`) & Pribumi Australia (`Warlpiri, Pitjantjatjara, Tiwi, Yolŋu`) (`1.20B`):* https://github.com/AmericasNLP · https://huggingface.co/datasets/Llamacha/monolingual-quechua-iic · https://huggingface.co/datasets/somosnlp/dataset-cultura-guarani_corpus-it · https://huggingface.co/datasets/somosnlp-hackathon-2022/Axolotl-Spanish-Nahuatl · https://github.com/BibleNLP/ebible
   - *Isolat (`Basque, Ainu, Sumeria`), Bahasa Isyarat (`ASL, BSL, JSL, BISINDO, SignWriting`), Pijin/Kreol (`Tok Pisin, Haiti, Papiamento, Melayu Ambon/Manado`), Bahasa Buatan (`Esperanto, Interlingua, Lojban, Toki Pona, Klingon`) & Klasik/Punah (`Latin, Sanskerta, Yunani Kuno, Akkadia/Sumeria, Kawi/Jawa Kuno`) (`2.10B`):* https://huggingface.co/datasets/HiTZ/latxa-corpus-v1.1 · https://huggingface.co/datasets/wav2gloss/NINJAL-Ainu-Folklore · https://github.com/sign-language-processing/datasets · https://huggingface.co/datasets/saillab/alpaca-esperanto-cleaned · https://huggingface.co/datasets/NetherQuartz/tatoeba-tokipona · https://huggingface.co/datasets/Fece228/latin-literature-dataset-170M · https://gretil.sub.uni-goettingen.de · https://github.com/OpenGreekAndLatin/First1KGreek · https://cdli.mpiwg-berlin.mpg.de · https://huggingface.co/datasets/veezbo/akkadian_english_corpus · https://sastra.org
5. **2E — Jembatan Translasi Multi-Arah (`13.50B` Token | `10.88 GB` — `8.50B` Translasi 12-Arah `ID↔EN↔JA↔KO` + `5.00B` Jangkar Paralel `1.000+` Bahasa):**
   - https://huggingface.co/datasets/allenai/nllb · https://opus.nlpl.eu · https://huggingface.co/datasets/SEACrowd/parallel_asian_treebank · https://huggingface.co/datasets/Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus · https://huggingface.co/datasets/lemon-mint/korean_parallel_sentences_v1.1 · https://huggingface.co/datasets/sentence-transformers/parallel-sentences-talks · https://huggingface.co/datasets/sentence-transformers/parallel-sentences-tatoeba · https://github.com/BibleNLP/ebible
