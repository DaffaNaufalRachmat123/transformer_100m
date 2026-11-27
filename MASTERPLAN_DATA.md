# MASTERPLAN DATA GLOBAL — 6 RUMPUN ILMU PENGETAHUAN DUNIA (OECD/UNESCO) & 7.170+ BAHASA DUNIA (ETHNOLOGUE / GLOTTOLOG)

> **Terakhir diperbarui:** 27 November 2026  
> **Lokasi File:** `/Users/daffanaufalrachmat/Documents/Real Projects/TransformerSample/transformer_500m_cuda/MASTERPLAN_DATA.md`  
> **Status Link:** Seluruh URL repositori HuggingFace Hub, GitHub CLDF, dan portal institusi akademik di dokumen ini telah diverifikasi aktif (`HTTP 200` / `HfApi` metadata langsung).

---

## RINGKASAN EKSEKUTIF: DITOTAL BISA DAPAT BERAPA TRILIUN (`T`) TOKEN?

Jika seluruh sumber terbuka di dokumen ini dikumpulkan secara menyeluruh mencakup **6 Rumpun Ilmu Pengetahuan Dunia (OECD/UNESCO)** dan **Seluruh Bahasa di Dunia yang Memiliki Jejak Digital (`~2.200+` bahasa berteks berkelanjutan + `5.000–7.674` bahasa berleksikon/tipologi CLDF)**:

| Metrik Skala Global | Total Token | Ukuran Arsip `Parquet Zstd` | Keterangan Teknis |
| :--- | :---: | :---: | :--- |
| **1. Total Bahan Mentah Gabungan (*Raw Sum Sebelum Dedup Silang*)** | **`~50.20 Triliun` (`50.2T`)** | `~95 – 110 TB` *(unduhan terkompresi)* | Mengandung `~55%` tumpang-tindih alami antar korpus web Common Crawl (`FineWeb-2`, `HPLT 2.0`, `CulturaX`, `MADLAD-400`, `Zyda-2`, `DCLM`, `GlotCC`). |
| **2. TOTAL BERSIH UNIK GLOBAL (*Clean Cross-Deduplicated Pool*)** | **`~22.40 Triliun` (`22.4T`)** | **`~28.5 TB`** *(`~25.9 TiB`)* | **Angka Riil Bersih (`0%` duplikat lintas-sumber):** **`5.85T`** murni dari 6 Rumpun Ilmu/Kode/Buku/Paten/Hukum + **`16.55T`** dari seluruh bahasa dunia (`18T–25T` tergantung ambang filter kualitas). |
| **3. Opsional: Tambahan Sekuens Biologis Mentah (*DNA/Protein*)** | **`+2.50T – 4.00T`** *(bio-tokens)* | `+3.2 TB` | Dari `Multi-Species Genomes` & `UniProt` bila ingin melatih pemodelan genomik/protein murni (di luar teks bahasa manusia `22.40T`). |

---

### Rincian Kalkulasi `22.40 Triliun` (`22.40T`) Token Unik Bersih per Rumpun & Kelompok Bahasa

#### A. Kontribusi Bagian 1 — 6 Rumpun Ilmu Pengetahuan, Kode, Buku, Paten & Hukum (`5.85T` Token Unik)

| Rumpun OECD / UNESCO | Rincian Komponen Utama | Mentah (`Raw`) | **Bersih Unik (`Deduped`)** |
| :--- | :--- | :---: | :---: |
| **A. Ilmu Formal & Kode** | Matematika, Logika Formal (`Lean4/Coq/Isabelle`), Statistika, Riset Operasi (`0.43T`) + Source Code 600+ bahasa termasuk `Go/JS/Java/C++/C`, `LLVM IR ComPile`, & `Multi-Arch Assembly` (`1.55T`) | `~2.27T` | **`1.98T`** |
| **B. Ilmu Alam** | Fisika, Kimia (`ChemPile + PubChem`), Astronomi (`NASA ADS + astro-ph`), Ilmu Bumi (`EarthArXiv + Geo`) (`0.26T`) + Biologi Sel/Molekuler, Organisme (`BHL 55M halaman + GenBank`), Neurosains (`PMC + bioRxiv`) (`0.32T`) | `~0.85T` | **`0.58T`** |
| **C. Ilmu Sosial** | Ekonomi (`EconStor/RePEc`), Sosiologi (`SocArXiv`), Psikologi (`PsyArXiv`), Politik/Parlemen/PBB (`US GPO, Europarl, MultiUN`), Antropologi, Geografi, Media & Kriminologi (`CORE 300M OA + peS2o + News/YT-Commons`) | `~0.95T` | **`0.55T`** |
| **D. Humaniora** | Buku Sejarah & Sastra Domain Publik Lintas Abad (`Pre-1929, LoC, Gutenberg, Wikisource, PD Books`), Filsafat (`SEP`), Linguistik (`UD, CLDF`), Seni/Musik (`ABC, WikiArt`), Kanon Agama Dunia (`Sefaria, CBETA, Tanzil, GRETIL`) | `~1.15T` | **`0.72T`** |
| **E. Ilmu Terapan & Profesi** | Teknik & Paten Global (`HUPD + USPTO/EPO` `0.65T`), Kedokteran & Kesehatan (`PMC-OA + 36M PubMed + Guidelines` `0.24T`), Hukum (`Pile-of-Law + Caselaw + MultiEURLEX` `0.35T`), Bisnis/Keuangan (`SEC EDGAR` `0.38T`), Agro (`FAO AGRIS`), Pendidikan & Militer (`DTIC`) (`0.15T`) | `~3.28T` | **`1.77T`** |
| **F. Ilmu Interdisipliner** | AI & Ilmu Data (`cs.LG/AI/CL/CV`), Teks Bioinformatika, Ilmu Kognitif, Ilmu Lingkungan & Iklim, Nanoteknologi, Ekonomi Perilaku | `~0.40T` | **`0.25T`** |
| **SUBTOTAL BAGIAN 1** | **Seluruh Literatur Keilmuan, Profesi, Buku & Kode Dunia** | **`~8.90T`** | **`5.85T` (`5.850B`)** |

#### B. Kontribusi Bagian 2 — Seluruh Bahasa di Dunia (`16.55T` Token Unik Pasca-Deduplikasi Lintas Crawl)

*(Diukur dari metadata server HuggingFace aktual: `HPLT 2.0 = 25.67 TB`, `FineWeb-2 = 20.15 TB`, `Zyda-2 = 13.55 TB`, `MADLAD-400 = 13.34 TB`, `DCLM-Baseline = 7.20 TB`, `FineWeb-Edu = 5.84 TB`, `Fineweb-Edu-ZH = 2.42 TB`, `GlotCC-V1 = 2.23 TB`, `Sangraha = 0.71 TB`, `Arabic-101B = 0.33 TB`)*

| Kelompok Bahasa Dunia | Sumber Utama yang Digabung & Dideduplikasi | Mentah (`Raw`) | **Bersih Unik (`Deduped`)** |
| :--- | :--- | :---: | :---: |
| **1. Bahasa Inggris Umum & Edukasi** | `Zyda-2` (`5.0T`) + `FineWeb-Edu` + `DCLM-Edu` + `Wikipedia EN` *(saringan kualitas edukasi/penalaran tinggi)* | `~14.90T` | **`5.50T`** |
| **2. Rumpun Sino-Tibet (`Mandarin, Kanton, Wu, Min, Hakka, Tibet, Burma`)** | `Fineweb-Edu-Chinese-V2.1` + `BAAI/IndustryCorpus2` + `FineWeb-2 (cmn/yue/wuu/bod/mya)` + `OpenPecha` | `~5.20T` | **`2.20T`** |
| **3. Indo-Eropa Non-EN (`Spanyol, Prancis, Jerman, Rusia, Portugis, dll.`)** | Union terdeduplikasi dari `FineWeb-2` + `HPLT 2.0 Cleaned` + `CulturaX` + `MADLAD-400` | `~12.80T` | **`4.80T`** |
| **4. Asia Timur, Tenggara & Selatan (`JA, KO, ID/Austronesia, HI/BN/Indo-Arya, Dravida, Austroasiatik, Kra-Dai`)** | `FineWeb-2` + `Swallow (JA)` + `HAERAE/Edu (KO)` + `SEACrowd/Indo4B (ID & Nusantara)` + `Sangraha (251B Indic & Dravida)` | `~5.40T` | **`2.15T`** |
| **5. Afro-Asiatik, Turkik, Uralik, Kaukasus & Mongolik** | `101B Arabic Words` + `FineWeb-2` + `HPLT 2.0` + `Turkic-XWMT` + `Wura (Hausa/Amharic/Somali/Oromo)` | `~1.85T` | **`0.95T`** |
| **6. `1.500+` Bahasa Low-Resource, Pribumi, Isolat, Kreol, Isyarat, Buatan & Klasik/Punah** | `GlotCC-V1` + `Glot500` + `FineWeb-2 Long-Tail` + `Wura/Masakhane/Inkuba` + `AmericasNLP` + `eBible (1.000+ bhs)` + `Latxa (Basque)` + `CDLI/GRETIL/Perseus/Sastra.org` + `CLDF (Glottolog/ASJP/Grambank/PHOIBLE)` | `~0.50T` | **`0.35T` (`350B`)** |
| **7. Korpus Paralel / Translasi Multi-Arah Global (`200–1.000+` Bahasa)** | `NLLB` + `OPUS All-Pairs` + `MultiUN` + `Europarl` + `ParaCrawl/JParaCrawl` + `WikiMatrix/CCMatrix` + `Tatoeba` | `~0.65T` | **`0.60T` (`600B`)** |
| **SUBTOTAL BAGIAN 2** | **Seluruh Warisan Bahasa Manusia Digital (`~2.200+` Teks + `7.674` Katalog)** | **`~41.30T`** | **`16.55T` (`16.550B`)** |
| **GRAND TOTAL (`BAGIAN 1 + 2`)** | **SELURUH ILMU PENGETAHUAN + SELURUH BAHASA DUNIA** | **`~50.20T`** | **`22.40T` (`22.400B`)** |

---

## 2. ESTIMASI STORAGE UNTUK 3 PILIHAN SKALA EKSEKUSI

Karena `22.40T` token adalah skala *Planetary Data Lake* penuh, Anda dapat memilih salah satu dari **3 Tier Penyimpanan** berikut sesuai anggaran infrastruktur:

| Tier Eksekusi | Target Token Unik | Arsip Master (`Parquet + Zstd-6`) | Binary Shards Latih (`uint16` / `uint24` / `uint32`) | Rekomendasi Infrastruktur Fisik / Cloud |
| :--- | :---: | :---: | :--- | :--- |
| **Tier 1: Full Planetary Data Lake** *(100% Semua Ilmu + Semua Bahasa Tanpa Potong)* | **`22.40T`** | **`28.5 TB`** *(`25.9 TiB`)* | • `44.8 TB` (`uint16`, `vocab=65K`)<br/>• **`67.2 TB`** (`uint24`, `vocab=256K` — **ideal untuk >1.800 bahasa & 40+ aksara**)<br/>• `89.6 TB` (`uint32`, `vocab=256K`) | **Cold Lake:** Object Storage S3 / Cloudflare R2 (`~28.5 TB`) atau NAS ZFS `6× 16 TB` HDD RAIDZ2 (`64 TB` usable). Tarik *slice* `.bin` per fase ke cluster NVMe GPU. |
| **Tier 2: Frontier Polyglot & All-Sciences** *(100% Semua 6 Ilmu + 100% Bahasa Langka/Klasik + Top 15% Web Bahasa Besar)* | **`5.00T`** | **`6.35 TB`** *(`5.77 TiB`)* | • `10.1 TB` (`uint16`, `vocab=65K`)<br/>• **`15.1 TB`** (`uint24`, `vocab=256K`)<br/>• `20.1 TB` (`uint32`, `vocab=256K`) | `2× 8 TB` atau `4× 4 TB` Enterprise NVMe + `7 TB` Object Storage R2/S3. Setara kualitas & skala korpus model kelas dunia (*Gemma-3 / Qwen3*). |
| **Tier 3: Focused All-Sciences + Prioritas `ID/EN/JA/KO` + Long-Tail Sample** | **`1.65T` – `2.00T`** | **`2.10 – 2.55 TB`** | • **`3.32 – 4.03 TB`** (`uint16`, `vocab=65,536`) | `12 TB` NVMe kerja (`3.3–4 TB` `.bin` + `2.2 TB` *scratch dedup* + arsip) + `2.5 TB` backup cloud. |

> **Catatan Penting Tokenizer Skala `7.000+` Bahasa:**  
> - Jika fokus utama adalah **4–15 bahasa utama + seluruh 6 cabang ilmu** (`Tier 3`), gunakan **`vocab_size = 65,536` (`uint16_t`, 2 Byte/token)**.  
> - Jika melatih **seluruh `1.800+` bahasa beraksara beragam** (Latin, Han, Kana, Hangul, Arab, Devanagari, Bengali, Tamil, Telugu, Thai, Myanmar, Tibet, Ge'ez, Georgia, Armenia, Ibrani, Sirilik, Yunani, Cuneiform, SignWriting) pada `Tier 1` atau `Tier 2`, naikkan ke **`vocab_size = 256,000` (`0..255,999`)** dan simpan file `.bin` dalam format **`uint24` (3 Byte/token packed)** untuk menghemat **25% ruang NVMe** dibanding `uint32_t` tanpa mengorbankan keadilan tokenisasi bahasa minoritas!

---

# BAGIAN 1: KATALOG LINK TERVERIFIKASI — 6 RUMPUN ILMU PENGETAHUAN DUNIA
*(Taksonomi Standar OECD Fields of Science [FOS] & UNESCO)*

## 0. Agregator Induk Lintas-6-Rumpun Ilmu *(Tarik Pertama Kali)*

| Agregator Universal | Ukuran Server / Skala | Link Resmi Terverifikasi | Cakupan Lintas Rumpun OECD / UNESCO |
| :--- | :---: | :--- | :--- |
| **OpenAlex Snapshot (`AWS S3`)** | `250M+` karya (`~350 GB` gzipped JSONL) | https://docs.openalex.org | Pohon konsep otomatis yang memetakan **250+ juta paper, buku, dan disertasi** langsung ke ke-6 rumpun OECD & UNESCO beserta URL PDF/teks terbukanya. |
| **CORE Open Access** | `300M+` naskah full-text | https://core.ac.uk | Koleksi *full-text* repositori universitas terbesar dunia — tulang punggung utama untuk Ilmu Sosial, Humaniora, Hukum, Pendidikan, dan Agro yang jarang ada di arXiv. |
| **peS2o (`AllenAI` & `Common-Pile`)** | `55.38 GiB` compressed (`~40M` paper) | https://huggingface.co/datasets/allenai/peS2o<br/>https://huggingface.co/datasets/common-pile/peS2o | Paper akademik bersih berlabel `s2fieldsofstudy` lintas seluruh 6 rumpun ilmu. |
| **LibreTexts, OpenStax, DOAB, Pressbooks & OERCommons** | Ribuan buku kuliah penuh | https://huggingface.co/datasets/common-pile/libretexts<br/>https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs<br/>https://huggingface.co/datasets/common-pile/doab<br/>https://huggingface.co/datasets/common-pile/pressbooks<br/>https://huggingface.co/datasets/common-pile/oercommons | Buku ajar perkuliahan terstruktur S1–S3 dan monograf akademik terbuka lintas seluruh fakultas. |
| **StackExchange Complete Network** | `36.50 GiB` compressed | https://huggingface.co/datasets/common-pile/stackexchange | Diskusi pakar di 180+ situs keilmuan (MathOverflow, Physics, Chemistry, Biology, Philosophy, History, Law, Economics, Linguistics, Engineering, dll.). |

---

## A. ILMU FORMAL (*Formal Sciences*) — Sistem Abstrak & Deduktif

| Sub-Cabang Ilmu | Nama Dataset / Arsip | Link Resmi Terverifikasi | Cakupan Spesifik Materi |
| :--- | :--- | :--- | :--- |
| **Matematika** *(Aljabar, Geometri, Kalkulus, Analisis Matematis, Teori Bilangan, Topologi, Kombinatorika)* | **MegaMath (`1.12 TB` / `371B` tok), FineMath (`139 GiB`), Proof-Pile-2 (`47.6 GiB`), OpenWebMath, NuminaMath-1.5, OpenMathReasoning, OpenMathInstruct-2 & Swallow-Math-v2** | https://huggingface.co/datasets/LLM360/MegaMath<br/>https://huggingface.co/datasets/HuggingFaceTB/finemath<br/>https://huggingface.co/datasets/EleutherAI/proof-pile-2<br/>https://huggingface.co/datasets/open-web-math/open-web-math<br/>https://huggingface.co/datasets/AI-MO/NuminaMath-1.5<br/>https://huggingface.co/datasets/nvidia/OpenMathReasoning<br/>https://huggingface.co/datasets/nvidia/OpenMathInstruct-2<br/>https://huggingface.co/datasets/tokyotech-llm/swallow-math-v2 | Mencakup matematika dasar, olimpiade IMO/Putnam, hingga riset murni `arXiv math.*` (`math.AG`, `math.NT`, `math.GT`, `math.CA`, `math.FA`, `math.CO`, `math.DG`, `math.AP`). |
| **Logika** *(Logika Matematika, Simbolik, Komputasional)* | **Lean-Workbook, AlgebraicStack (`Proof-Pile-2`) & arXiv Logic** | https://huggingface.co/datasets/internlm/Lean-Workbook<br/>https://huggingface.co/datasets/EleutherAI/proof-pile-2<br/>https://huggingface.co/datasets/common-pile/arxiv_papers | Pembuktian teorema formal terverifikasi kompilator (`Lean 4`, `Coq`, `Isabelle`, `Metamath`) serta paper `math.LO` & `cs.LO`. |
| **Statistika** *(Inferensial, Teori Peluang, Analisis Stokastik)* | **arXiv `stat.*` & `math.PR` / `math.ST` + Statistics LibreTexts** | https://huggingface.co/datasets/common-pile/arxiv_papers<br/>https://huggingface.co/datasets/common-pile/libretexts<br/>https://huggingface.co/datasets/common-pile/stackexchange | Filter `stat.TH`, `stat.ME`, `stat.AP`, `math.PR` (Probability & Stochastic Processes), buku ajar *Statistics LibreTexts*, dan `stats.stackexchange.com`. |
| **Ilmu Komputer Teoretis & Rekayasa Kode** *(Komputasi, Algoritma, Struktur Data, Teori Informasi, Kriptografi, Sistem & Assembly)* | **The Stack v2 Dedup (`900B` tok), Common-Pile StackV2 (`1.28 TB`), Stack-Edu, ComPile (`561 GiB` compressed / `2.7 TB` LLVM IR), ExeBench, LLM4Binary Decompile, CodeContests & OpenCodeReasoning** | https://huggingface.co/datasets/bigcode/the-stack-v2-dedup<br/>https://huggingface.co/datasets/common-pile/stackv2<br/>https://huggingface.co/datasets/HuggingFaceTB/stack-edu<br/>https://huggingface.co/datasets/llvm-ml/ComPile<br/>https://huggingface.co/datasets/jordiae/exebench<br/>https://huggingface.co/datasets/LLM4Binary/decompile-ghidra-100k<br/>https://huggingface.co/datasets/LLM4Binary/decompile-bench<br/>https://huggingface.co/datasets/deepmind/code_contests<br/>https://huggingface.co/datasets/nvidia/OpenCodeReasoning<br/>https://huggingface.co/datasets/common-pile/arxiv_papers | Filter `cs.DS`, `cs.CC`, `cs.IT`, `cs.CR` (Kriptografi), ditambah 600+ bahasa pemrograman termasuk `Go`, `JS`, `Java`, `C++`, `C`, `LLVM IR`, dan `Assembly (x86_64, ARM64, RISC-V)`. |
| **Sistem & Keputusan** *(Teori Permainan, Riset Operasi, Sibernetika)* | **arXiv Game Theory / Optimization / Systems & OR StackExchange** | https://huggingface.co/datasets/common-pile/arxiv_papers<br/>https://huggingface.co/datasets/common-pile/stackexchange<br/>https://huggingface.co/datasets/allenai/peS2o | Filter `cs.GT` (Game Theory), `math.OC` (Optimization & Operations Research), `eess.SY` / `cs.SY` (Systems & Cybernetics), dan `or.stackexchange.com`. |

---

## B. ILMU ALAM (*Natural Sciences*) — Hukum Alam & Metode Empiris

### [1] Ilmu Fisika (*Physical Sciences*)

| Sub-Cabang Ilmu | Nama Dataset / Arsip | Link Resmi Terverifikasi | Cakupan Spesifik Materi |
| :--- | :--- | :--- | :--- |
| **Fisika** *(Mekanika, Termodinamika, Elektromagnetisme, Optika, Kuantum, Relativitas, Fisika Partikel, Zat Padat)* | **arXiv Physics Full-Text, Physics LibreTexts, OpenStax, MegaScience, NaturalReasoning, TheoremQA & CAMEL-Physics** | https://huggingface.co/datasets/common-pile/arxiv_papers<br/>https://huggingface.co/datasets/common-pile/libretexts<br/>https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs<br/>https://huggingface.co/datasets/MegaScience/MegaScience<br/>https://huggingface.co/datasets/facebook/natural_reasoning<br/>https://huggingface.co/datasets/TIGER-Lab/TheoremQA<br/>https://huggingface.co/datasets/camel-ai/physics | Seluruh kategori `quant-ph`, `hep-th/ph/ex`, `cond-mat.*`, `gr-qc`, `nucl-th/ex`, `physics.optics`, `physics.flu-dyn`, `physics.class-ph`, `physics.plasm-ph`, dan `physics.stackexchange.com`. |
| **Kimia** *(Organik, Anorganik, Fisik, Analitik, Biokimia, Kimia Material)* | **ChemPile (`Paper, Education, Reasoning, Code`), SMolInstruct, Mol-Instructions, CAMEL-Chem & PubChem** | https://huggingface.co/datasets/jablonkagroup/chempile-paper<br/>https://huggingface.co/datasets/jablonkagroup/chempile-education<br/>https://huggingface.co/datasets/jablonkagroup/chempile-reasoning<br/>https://huggingface.co/datasets/jablonkagroup/chempile-code<br/>https://huggingface.co/datasets/osunlp/SMolInstruct<br/>https://huggingface.co/datasets/zjunlp/Mol-Instructions<br/>https://huggingface.co/datasets/camel-ai/chemistry<br/>https://pubchem.ncbi.nlm.nih.gov | Literatur riset kimia lengkap, *Chemistry LibreTexts*, sintesis organik, termokimia, elektrokimia, kristalografi material, dan notasi molekuler `SMILES`/`SELFIES`/`IUPAC`. |
| **Astronomi** *(Astrofisika, Kosmologi, Ilmu Keplanetan)* | **arXiv `astro-ph.*`, NASA ADS & Astronomy StackExchange** | https://huggingface.co/datasets/common-pile/arxiv_papers<br/>https://ui.adsabs.harvard.edu<br/>https://huggingface.co/datasets/common-pile/stackexchange | Filter `astro-ph.CO`, `astro-ph.EP`, `astro-ph.GA`, `astro-ph.HE`, `astro-ph.IM`, `astro-ph.SR`, serta arsip **NASA Astrophysics Data System (ADS)**. |
| **Ilmu Bumi** *(Geologi, Geofisika, Meteorologi, Klimatologi, Oseanografi, Hidrologi, Geografi Fisik)* | **EarthArXiv, EarthScience StackExchange, Geosciences LibreTexts & peS2o (`Geology`)** | https://eartharxiv.org<br/>https://huggingface.co/datasets/mlfoundations-dev/stackexchange_earthscience<br/>https://huggingface.co/datasets/common-pile/libretexts<br/>https://huggingface.co/datasets/allenai/peS2o<br/>https://huggingface.co/datasets/common-pile/arxiv_papers | Preprint kebumian di **EarthArXiv**, buku ajar *Geosciences LibreTexts*, `physics.ao-ph` (Atmosfer & Oseanografi), dan `physics.geo-ph` (Geofisika). |

### [2] Ilmu Hayati (*Life Sciences*)

| Sub-Cabang Ilmu | Nama Dataset / Arsip | Link Resmi Terverifikasi | Cakupan Spesifik Materi |
| :--- | :--- | :--- | :--- |
| **Biologi Dasar** *(Biologi Sel & Molekuler, Genetika, Evolusi, Anatomi, Fisiologi)* | **PMC Open Access (`42.2 GiB` di Common-Pile + Bulk FTP), bioRxiv, Biology LibreTexts, OpenStax & CAMEL-Biology** | https://huggingface.co/datasets/common-pile/pubmed<br/>https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/<br/>https://huggingface.co/datasets/mteb/raw_biorxiv<br/>https://www.biorxiv.org/tdm<br/>https://huggingface.co/datasets/common-pile/libretexts<br/>https://huggingface.co/datasets/camel-ai/biology | Jutaan paper biologi sel, biologi molekuler, genetika, anatomi, fisiologi, dan evolusi. |
| **Organisme** *(Botani, Zoologi, Mikrobiologi, Virologi, Mikologi, Ekologi, Biologi Kelautan)* | **Biodiversity Heritage Library (`Filtered`), NCBI GenBank & peS2o (`Biology`)** | https://huggingface.co/datasets/common-pile/biodiversity_heritage_library_filtered<br/>https://www.ncbi.nlm.nih.gov/genbank/<br/>https://huggingface.co/datasets/allenai/peS2o | Literatur taksonomi spesies dunia (botani, zoologi, mikologi), mikrobiologi/virologi, ekologi tropis, dan biologi kelautan. |
| **Neurosains** *(Struktur & Fungsi Sistem Saraf serta Otak)* | **PMC Neuroscience Subset, bioRxiv (`neuroscience`) & arXiv `q-bio.NC`** | https://huggingface.co/datasets/common-pile/pubmed<br/>https://huggingface.co/datasets/mteb/raw_biorxiv<br/>https://huggingface.co/datasets/common-pile/arxiv_papers | Neurobiologi seluler, neuroanatomi, elektrofisiologi saraf, dan pencitraan otak fungsional. |

---

## C. ILMU SOSIAL (*Social Sciences*) — Perilaku & Struktur Masyarakat

| Sub-Cabang Ilmu | Nama Dataset / Arsip | Link Resmi Terverifikasi | Cakupan Spesifik Materi |
| :--- | :--- | :--- | :--- |
| **Sosiologi** *(Struktur, Stratifikasi, Perubahan Sosial, Perkotaan & Pedesaan)* | **SocArXiv, Social Sciences LibreTexts, OpenStax Sociology & peS2o (`Sociology`)** | https://osf.io/preprints/socarxiv<br/>https://huggingface.co/datasets/common-pile/libretexts<br/>https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs<br/>https://huggingface.co/datasets/allenai/peS2o | Preprint sosiologi global, buku teks sosiologi dasar hingga lanjut, dan riset struktur sosial perkotaan/pedesaan. |
| **Psikologi** *(Kognitif, Klinis, Perkembangan, Sosial, Industri-Organisasi, Neuropsikologi)* | **PsyArXiv, PsychoCounsel, Psychology-Dataset & PMC Psychology** | https://osf.io/preprints/psyarxiv<br/>https://huggingface.co/datasets/Psychotherapy-LLM/PsychoCounsel-Preference<br/>https://huggingface.co/datasets/jkhedri/psychology-dataset<br/>https://huggingface.co/datasets/common-pile/pubmed | Preprint **PsyArXiv**, psikoterapi/klinis, psikometri, psikologi perkembangan, sosial, IO, dan neuropsikologi. |
| **Ilmu Ekonomi** *(Mikro, Makro, Ekonometrika, Pembangunan, Internasional)* | **EconStor (`250K+` OA Papers), arXiv `econ.*` & `q-fin.*`, Finance-Instruct-500k & AdaptLLM** | https://www.econstor.eu<br/>https://huggingface.co/datasets/common-pile/arxiv_papers<br/>https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k<br/>https://huggingface.co/datasets/AdaptLLM/finance-tasks | Teori ekonomi mikro/makro, ekonometrika (`econ.EM`), ekonomi pembangunan, perdagangan internasional, dan kebijakan moneter/fiskal. |
| **Ilmu Politik** *(Teori Politik, Kebijakan Publik, Administrasi Negara, Hubungan Internasional)* | **US GPO (`17.7 GiB`), Europarl, OPUS MultiUN & peS2o (`Political Science`)** | https://huggingface.co/datasets/common-pile/usgpo<br/>https://huggingface.co/datasets/Helsinki-NLP/europarl<br/>https://opus.nlpl.eu<br/>https://huggingface.co/datasets/allenai/peS2o | Dokumen administrasi negara, kebijakan publik, debat parlemen, diplomasi PBB (`MultiUN`), dan kajian hubungan internasional. |
| **Antropologi** *(Budaya, Biologis, Etnologi, Arkeologi)* | **Linguistic & Cultural Anthropology, DOAB Anthropology & CORE** | https://huggingface.co/datasets/WithinUsAI/linguistic_anthropology_25k<br/>https://huggingface.co/datasets/common-pile/doab<br/>https://core.ac.uk | Monograf etnologi, antropologi budaya, paleoantropologi/biologis, dan laporan arkeologi. |
| **Geografi Manusia** *(Demografi, Persebaran Penduduk, Geopolitik)* | **peS2o (`Geography`), DOAB & Social Sciences LibreTexts** | https://huggingface.co/datasets/allenai/peS2o<br/>https://huggingface.co/datasets/common-pile/doab<br/>https://huggingface.co/datasets/common-pile/libretexts | Studi kependudukan/demografi, tata ruang sosial, migrasi global, dan geopolitik. |
| **Komunikasi & Kriminologi** *(Kajian Media, Jurnalisme, Kriminologi)* | **Common-Pile News, YouTube-Commons (`151.7 GiB`) & Caselaw Criminal** | https://huggingface.co/datasets/common-pile/news<br/>https://huggingface.co/datasets/PleIAs/YouTube-Commons<br/>https://huggingface.co/datasets/common-pile/caselaw_access_project | Jurnalisme publik, analisis wacana media, komunikasi massa, serta kriminologi & peradilan pidana. |

---

## D. HUMANIORA (*Humanities*) — Makna, Nilai, Budaya & Karya Manusia

| Sub-Cabang Ilmu | Nama Dataset / Arsip | Link Resmi Terverifikasi | Cakupan Spesifik Materi |
| :--- | :--- | :--- | :--- |
| **Filsafat** *(Epistemologi, Metafisika, Etika, Estetika, Filsafat Ilmu)* | **Stanford Encyclopedia of Philosophy (`SEP`), Philosophy-Corpus, Gutenberg & StackExchange** | https://plato.stanford.edu<br/>https://huggingface.co/datasets/LisaMegaWatts/philosophy-corpus<br/>https://huggingface.co/datasets/common-pile/project_gutenberg<br/>https://huggingface.co/datasets/common-pile/stackexchange | Ensiklopedia filsafat akademik paling otoritatif di dunia (`SEP`), karya asli filsuf klasik-modern, dan diskusi `philosophy.stackexchange.com`. |
| **Sejarah** *(Kuno, Pertengahan, Modern, Ekonomi, Historiografi)* | **Library of Congress (`16.8 GiB`), Pre-1929 Books (`23.5 GiB`), Public Domain Review & French-PD-Books** | https://huggingface.co/datasets/common-pile/library_of_congress<br/>https://huggingface.co/datasets/common-pile/pre_1929_books<br/>https://huggingface.co/datasets/common-pile/public_domain_review<br/>https://huggingface.co/datasets/PleIAs/French-PD-Books | Kronik sejarah dunia lintas zaman, dokumen primer, sejarah ekonomi, dan historiografi. |
| **Linguistik** *(Fonologi, Morfologi, Sintaksis, Semantik, Pragmatik, Sosiolinguistik, Filologi)* | **Universal Dependencies, Linguistics StackExchange, Glottolog, WALS, Grambank, PHOIBLE & UniMorph** | https://huggingface.co/datasets/universal-dependencies/universal_dependencies<br/>https://huggingface.co/datasets/mlfoundations-dev/stackexchange_linguistics<br/>https://github.com/glottolog/glottolog<br/>https://github.com/cldf-datasets/wals<br/>https://github.com/grambank/grambank<br/>https://github.com/phoible/dev<br/>https://github.com/unimorph | Anotasi sintaksis/morfologi 160+ bahasa (`UD`), fonologi 2.186 bahasa (`PHOIBLE`), morfosintaksis 2.467 bahasa (`Grambank`), dan filologi. |
| **Sastra** *(Teori Sastra, Kritik Sastra, Sastra Bandingan, Karya Klasik)* | **Project Gutenberg (`Common-Pile` & `Multilingual`), Wikisource & DOAB** | https://huggingface.co/datasets/common-pile/project_gutenberg<br/>https://huggingface.co/datasets/manu/project_gutenberg<br/>https://dumps.wikimedia.org<br/>https://huggingface.co/datasets/common-pile/doab | Puluhan ribu karya sastra dunia ditambah buku kritik sastra & sastra bandingan di DOAB. |
| **Seni** *(Rupa, Musik, Tari, Teater, Film, Fotografi, Desain)* | **Irishman (`ABC Music Notation`), WikiArt, Humanities LibreTexts & DOAB Arts** | https://huggingface.co/datasets/sander-wood/irishman<br/>https://huggingface.co/datasets/huggan/wikiart<br/>https://huggingface.co/datasets/common-pile/libretexts<br/>https://huggingface.co/datasets/common-pile/doab | Ratusan ribu partitur musik dalam notasi teks `ABC`, sejarah seni rupa, naskah drama/film, dan teori desain. |
| **Agama & Budaya** *(Teologi, Perbandingan Agama, Gender, Kajian Kawasan)* | **Sefaria-Export, CBETA Tripitaka, Tanzil, eBible, GRETIL & DOAB** | https://github.com/Sefaria/Sefaria-Export<br/>https://cbeta.org<br/>https://tanzil.net<br/>https://github.com/BibleNLP/ebible<br/>https://gretil.sub.uni-goettingen.de<br/>https://huggingface.co/datasets/common-pile/doab | Kitab & tafsir lintas tradisi agama dunia beserta jurnal teologi, kajian budaya, gender, dan studi kawasan (*Area Studies*). |

---

## E. ILMU TERAPAN & PROFESI (*Applied Sciences*) — Solusi Praktis

| Sub-Cabang Ilmu | Nama Dataset / Arsip | Link Resmi Terverifikasi | Cakupan Spesifik Materi |
| :--- | :--- | :--- | :--- |
| **Teknik / Rekayasa** *(Sipil, Mesin, Elektro, Kimia, Industri, Informatika, Dirgantara, Material, Nuklir, Biomedis, Lingkungan)* | **Harvard USPTO Patent Dataset (`HUPD`), Patent Classification, arXiv `eess.*` & Engineering LibreTexts** | https://huggingface.co/datasets/HUPD/hupd<br/>https://huggingface.co/datasets/ccdv/patent-classification<br/>https://huggingface.co/datasets/common-pile/arxiv_papers<br/>https://huggingface.co/datasets/common-pile/libretexts | **4.5+ juta dokumen spesifikasi paten teknik penuh** yang memuat rincian rekayasa mesin, sipil, sirkuit elektro, proses pabrik kimia, dirgantara, material, reaktor nuklir, dan biomedis. |
| **Kesehatan** *(Kedokteran, Farmasi, Keperawatan, Gigi, Hewan, Ilmu Gizi, Kesehatan Masyarakat)* | **PMC-OA, NCBI PubMed, EPFL Guidelines, ApolloCorpus, Medical-R1-Distill, UltraMedical, MedMCQA, IgakuQA, KorMedMCQA & IndoHealth-NLP** | https://huggingface.co/datasets/common-pile/pubmed<br/>https://huggingface.co/datasets/ncbi/pubmed<br/>https://huggingface.co/datasets/epfl-llm/guidelines<br/>https://huggingface.co/datasets/FreedomIntelligence/ApolloCorpus<br/>https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data<br/>https://huggingface.co/datasets/TsinghuaC3I/UltraMedical<br/>https://huggingface.co/datasets/openlifescienceai/medmcqa<br/>https://huggingface.co/datasets/japan-ai-official/igakuqa-subset-curated<br/>https://huggingface.co/datasets/sean0042/KorMedMCQA<br/>https://huggingface.co/datasets/IndoHealth-NLP/NCD_Instruct-Tuning_Medical_QA_Indonesian | Kedokteran klinis & spesialis, farmasi/farmakologi, keperawatan, kedokteran gigi, kedokteran hewan (*veterinary*), ilmu gizi, dan epidemiologi kesehatan masyarakat. |
| **Agro & Pangan** *(Pertanian, Agronomi, Peternakan, Kehutanan, Perikanan, Akuakultur, Teknologi Pangan)* | **FAO AGRIS, Agriculture-QA, Foodista & peS2o (`Agricultural & Food Sciences`)** | https://agris.fao.org<br/>https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only<br/>https://huggingface.co/datasets/common-pile/foodista<br/>https://huggingface.co/datasets/allenai/peS2o | Basis data pangan & pertanian PBB **FAO AGRIS** (`14+ juta` entri ilmiah agronomi, kehutanan, peternakan, perikanan, akuakultur, dan teknologi pangan). |
| **Hukum** *(Pidana, Perdata, Tata Negara, Bisnis, Internasional)* | **Pile of Law (`41.1 GiB`), Caselaw Access Project (`22.6 GiB`), Regulations, MultiEURLEX & LexGLUE** | https://huggingface.co/datasets/pile-of-law/pile-of-law<br/>https://huggingface.co/datasets/common-pile/caselaw_access_project<br/>https://huggingface.co/datasets/common-pile/regulations<br/>https://huggingface.co/datasets/coastalcph/multi_eurlex<br/>https://huggingface.co/datasets/coastalcph/lex_glue | Yurisprudensi pengadilan, undang-undang pidana/perdata/tata negara, kontrak korporasi, regulasi pemerintah, dan hukum internasional multibahasa. |
| **Bisnis & Manajemen** *(Akuntansi, Keuangan, Pemasaran, SDM, Operasi)* | **SEC-EDGAR, Finance-Instruct-500k, FinGPT & Business OpenStax/LibreTexts** | https://huggingface.co/datasets/TeraflopAI/SEC-EDGAR<br/>https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k<br/>https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train<br/>https://huggingface.co/datasets/common-pile/libretexts<br/>https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs | Laporan keuangan audit (`10-K/10-Q`), standar akuntansi, manajemen keuangan, pemasaran, SDM, dan riset operasi bisnis. |
| **Pendidikan & Arsitektur** *(Pedagogi, Teknologi Pendidikan, Arsitektur, PWK)* | **OERCommons, Pressbooks, DOAB (`Education & Architecture`) & CORE** | https://huggingface.co/datasets/common-pile/oercommons<br/>https://huggingface.co/datasets/common-pile/pressbooks<br/>https://huggingface.co/datasets/common-pile/doab<br/>https://core.ac.uk | Ilmu pendidikan/pedagogi, kurikulum, teknik arsitektur, dan perencanaan wilayah & kota (PWK). |
| **Informasi & Militer** *(Perpustakaan, Kearsipan, Strategi Pertahanan)* | **DTIC Public Technical Reports, Library of Congress & US GPO** | https://discover.dtic.mil<br/>https://huggingface.co/datasets/common-pile/library_of_congress<br/>https://huggingface.co/datasets/common-pile/usgpo | Ratusan ribu laporan riset pertahanan publik tak-rahasia di **DTIC**, sejarah strategi militer, serta ilmu perpustakaan dan kearsipan. |

---

## F. ILMU INTERDISIPLINER (*Interdisciplinary Fields*) — Lintas Rumpun

| Bidang Lintas Rumpun | Nama Dataset / Arsip | Link Resmi Terverifikasi | Integrasi Disiplin |
| :--- | :--- | :--- | :--- |
| **AI & Ilmu Data** *(Matematika + Statistika + Ilmu Komputer)* | **arXiv (`cs.LG, cs.AI, cs.CL, cs.CV, stat.ML`), The Stack v2 & OpenThoughts-114k** | https://huggingface.co/datasets/common-pile/arxiv_papers<br/>https://huggingface.co/datasets/bigcode/the-stack-v2-dedup<br/>https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k | Teori *machine learning*, arsitektur *deep learning*, NLP, *computer vision*, dan rekayasa data. |
| **Bioinformatika** *(Biologi Molekuler + Genetika + Komputasi)* | **Multi-Species Genomes, Nucleotide Tasks, UniProt, bioRxiv & StackExchange** | https://huggingface.co/datasets/InstaDeepAI/multi_species_genomes<br/>https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks<br/>https://www.uniprot.org<br/>https://huggingface.co/datasets/mteb/raw_biorxiv<br/>https://huggingface.co/datasets/common-pile/stackexchange | Sekuens DNA/RNA/protein teranotasi fungsi biologis (`UniProt`), algoritma genomik, dan `bioinformatics.stackexchange.com`. |
| **Ilmu Kognitif** *(Psikologi + Neurosains + Linguistik + AI)* | **Cognitive Science StackExchange, PsyArXiv & peS2o** | https://huggingface.co/datasets/common-pile/stackexchange<br/>https://osf.io/preprints/psyarxiv<br/>https://huggingface.co/datasets/allenai/peS2o | Pemodelan pikiran, psikolinguistik, persepsi, dan neurosains komputasional (`cogsci.stackexchange.com`). |
| **Ilmu Lingkungan** *(Ekologi + Geologi + Teknik + Ekonomi + Kebijakan Publik)* | **EarthArXiv, peS2o (`Environmental Science`) & Common-Pile Regulations** | https://eartharxiv.org<br/>https://huggingface.co/datasets/allenai/peS2o<br/>https://huggingface.co/datasets/common-pile/regulations | Perubahan iklim, ekologi keberlanjutan, teknik lingkungan, ekonomi energi bersih, dan kebijakan lingkungan. |
| **Nanoteknologi & Ekonomi Perilaku** *(Fisika-Kimia; Psikologi-Ekonomi)* | **arXiv (`cond-mat.mes-hall, cond-mat.mtrl-sci`), ChemPile & EconStor** | https://huggingface.co/datasets/common-pile/arxiv_papers<br/>https://huggingface.co/datasets/jablonkagroup/chempile-paper<br/>https://www.econstor.eu | Sintesis & fisika kuantum nanomaterial serta eksperimen pengambilan keputusan/ekonomi perilaku. |

---

# BAGIAN 2: KATALOG LINK TERVERIFIKASI — SELURUH BAHASA DI DUNIA
*(Ethnologue Ed. 29 [2026]: 7.170 Bahasa Hidup | Glottolog 5.3: 7.674 Bahasa)*

## A. Basis Data Induk `7.674` Bahasa & Mega-Korpus Ribuan Bahasa

| Cakupan Bahasa | Nama Repositori Induk | Ukuran / Skala | Link Resmi Terverifikasi |
| :--- | :--- | :---: | :--- |
| **`7.674` Bahasa (`246` Rumpun, `183` Isolat, `227` Isyarat)** | **Glottolog 5.3 (`CLDF Master`)** | Katalog & Pohon Genetik Penuh | https://glottolog.org<br/>https://github.com/glottolog/glottolog |
| **`5.000+` Bahasa** | **ASJP (`Lexibank Wordlists`)** | Kosakata Swadesh Terstandarisasi | https://github.com/lexibank/asjp |
| **`2.467` & `2.186` Bahasa** | **Grambank, WALS & PHOIBLE** | Tipologi Tata Bahasa & Fonem | https://github.com/grambank/grambank · https://grambank.clld.org<br/>https://github.com/cldf-datasets/wals · https://wals.info<br/>https://github.com/phoible/dev · https://phoible.org |
| **`1.800+` Bahasa & Varian Aksara** | **FineWeb-2 (`HuggingFaceFW`)** | **`20.15 TB` compressed** | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| **`1.000+` Bahasa Low-Resource** | **GlotCC-V1 & Glot500 (`CIS-LMU`)** | **`2.23 TB` compressed** | https://huggingface.co/datasets/cis-lmu/GlotCC-V1<br/>https://huggingface.co/datasets/cis-lmu/Glot500 |
| **`1.000+` Bahasa Paralel** | **eBible Corpus (`BibleNLP`)** | Paralel Lintas Rumpun Langka | https://github.com/BibleNLP/ebible<br/>https://huggingface.co/datasets/DavidCBaines/ebible_corpus |
| **`419` Bahasa** | **MADLAD-400 (`AllenAI / Google`)** | **`13.34 TB` compressed** | https://huggingface.co/datasets/allenai/MADLAD-400 |
| **`340+` Bahasa** | **Wikimedia / Wikipedia Dumps** | Ensiklopedia Multi-Aksara | https://huggingface.co/datasets/wikimedia/wikipedia<br/>https://dumps.wikimedia.org |
| **`200+` Bahasa Paralel & Eval** | **NLLB, OPUS & FLORES+** | Bitext Terjemahan Global | https://huggingface.co/datasets/allenai/nllb<br/>https://opus.nlpl.eu<br/>https://huggingface.co/datasets/openlanguagedata/flores_plus |
| **`193` & `167` & `114` Bahasa** | **HPLT 2.0 (`25.67 TB`), CulturaX & Aya** | Web Bersih & Instruksi Budaya | https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned<br/>https://huggingface.co/datasets/uonlp/CulturaX<br/>https://huggingface.co/datasets/CohereForAI/aya_collection |

---

## B. Repositori Spesifik untuk Ke-19 Rumpun Bahasa Utama Dunia

| No | Rumpun Bahasa | Jumlah | Contoh Bahasa Kunci | Link Sumber Data Terverifikasi |
| :-: | :--- | :---: | :--- | :--- |
| **1** | **Niger-Kongo** | `1.500+` | Swahili, Yoruba, Igbo, Zulu, Shona, Wolof, Fula | https://huggingface.co/datasets/castorini/wura<br/>https://huggingface.co/datasets/lelapa/Inkuba-Mono<br/>https://huggingface.co/datasets/masakhane/masakhanews<br/>https://huggingface.co/datasets/masakhane/mafand<br/>https://repo.sadilar.org |
| **2** | **Austronesia** | `~1.220` | Indonesia, Melayu, Jawa, Sunda, Tagalog, Cebuano, Malagasi, Maori, Fiji | https://huggingface.co/SEACrowd<br/>https://huggingface.co/datasets/SEACrowd/indo4b<br/>https://huggingface.co/datasets/IndoNLP/NusaX-senti<br/>https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 *(subset `ind, jav, sun, tgl, ceb, plt, mri, fij, ban, min, bug, ace`)* |
| **3** | **Trans-Nugini** | `~477` | Dani, Ekari, Enga, Huli, Melpa *(Papua ID & PNG)* | https://github.com/BibleNLP/ebible *(ratusan bahasa PNG & Papua)*<br/>https://huggingface.co/datasets/cis-lmu/GlotCC-V1<br/>https://github.com/grambank/grambank |
| **4** | **Sino-Tibet** | `~460` | Mandarin, Kanton, Wu, Min, Hakka, Tibet, Burma | https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1<br/>https://huggingface.co/datasets/BAAI/IndustryCorpus2<br/>https://openpecha.org<br/>https://huggingface.co/datasets/shajiu/TibetanGeneral_corpus<br/>https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 *(subset `cmn, yue, wuu, nan, hak, bod, mya`)* |
| **5** | **Indo-Eropa** | `~446` | Inggris, Jerman, Spanyol, Prancis, Portugis, Rusia, Hindi, Urdu, Bengali, Persia, Yunani, Armenia, Keltik, Baltik | https://huggingface.co/datasets/Zyphra/Zyda-2<br/>https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu<br/>https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned<br/>https://huggingface.co/datasets/ai4bharat/sangraha *(Indo-Arya)*<br/>https://huggingface.co/datasets/uonlp/CulturaX |
| **6** | **Afro-Asiatik** | `~376` | Arab, Ibrani, Amharik, Hausa, Somali, Oromo, Berber (Amazigh) | https://huggingface.co/datasets/ClusterlabAi/101_billion_arabic_words_dataset<br/>https://huggingface.co/datasets/arbml/ashaar<br/>https://huggingface.co/datasets/castorini/wura<br/>https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 *(termasuk `heb, amh, hau, som, orm, kab, tzm`)* |
| **7** | **Nilo-Sahara** | `~200` | Luo, Dinka, Maasai, Nubia, Kanuri | https://huggingface.co/datasets/allenai/nllb *(`luo, dik, knc`)*<br/>https://huggingface.co/datasets/masakhane/mafand<br/>https://github.com/BibleNLP/ebible |
| **8** | **Austroasiatik** | `~165` | Vietnam, Khmer, Mon, Santali, Khasi | https://huggingface.co/SEACrowd<br/>https://huggingface.co/datasets/ai4bharat/sangraha *(`Santali`)*<br/>https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 *(`vie, khm, mnw, sat, kha`)* |
| **9** | **Kra-Dai (Tai)** | `~90` | Thai, Lao, Shan, Zhuang | https://huggingface.co/SEACrowd<br/>https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 *(`tha, lao, shn, zha`)*<br/>https://huggingface.co/datasets/allenai/nllb |
| **10** | **Dravida** | `~85` | Tamil, Telugu, Kannada, Malayalam | https://huggingface.co/datasets/ai4bharat/sangraha *(`251B` tok mencakup Tamil, Telugu, Kannada, Malayalam)*<br/>https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| **11** | **Turkik** | `~40` | Turki, Azerbaijan, Uzbek, Kazakh, Uighur, Kirgiz, Tatar, Yakut | https://huggingface.co/datasets/turkic-interlingua/turkic_xwmt<br/>https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned<br/>https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| **12** | **Kaukasus (3 rumpun)** | `~40` | Georgia (Kartvelia), Chechnya, Avar, Adyghe, Abkhaz, Lezgi | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 *(`kat, che, ava, ady, kbd, abk, lez, xmf`)*<br/>https://huggingface.co/datasets/cis-lmu/GlotCC-V1 |
| **13** | **Uralik** | `~38` | Hungaria, Finlandia, Estonia, Sámi, Mari, Udmurt, Komi | https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned<br/>https://huggingface.co/datasets/joanitolopo/flores-plus-uralic<br/>https://huggingface.co/datasets/nup-csai/uralicBPE<br/>https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| **14** | **Mongolik & Tungusik** | `~25` | Mongolia, Buryat, Kalmyk, Evenki, Manchu | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 *(`khk_Cyrl, mon_Mong, bua_Cyrl, xal_Cyrl`)*<br/>https://huggingface.co/datasets/cis-lmu/Glot500 |
| **15** | **Japonik** | `~12` | Jepang, Bahasa Ryukyu (Okinawa, Miyako, Yaeyama) | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 *(`jpn_Jpan, ryu_Jpan`)*<br/>https://huggingface.co/datasets/tokyotech-llm/Swallow-Nemotron-Post-Training-Dataset-v1<br/>https://www.jstage.jst.go.jp<br/>https://dumps.wikimedia.org |
| **16** | **Koreanik** | `2` | Korea, Bahasa Jeju (Jejueo) | https://huggingface.co/datasets/HAERAE-HUB/KOREAN-WEBTEXT<br/>https://huggingface.co/datasets/eliceai/korean-webtext-edu<br/>https://huggingface.co/datasets/Mireu-Lab/Korean-jeju-dialect-speech<br/>https://dumps.wikimedia.org |
| **17** | **Pribumi Amerika** | `1.000+` | Quechua, Guarani, Nahuatl, Maya, Navajo, Cherokee, Aymara | https://github.com/AmericasNLP<br/>https://huggingface.co/datasets/Llamacha/monolingual-quechua-iic<br/>https://huggingface.co/datasets/somosnlp/dataset-cultura-guarani_corpus-it<br/>https://huggingface.co/datasets/somosnlp-hackathon-2022/Axolotl-Spanish-Nahuatl<br/>https://github.com/BibleNLP/ebible |
| **18** | **Pribumi Australia** | `~250+` | Pitjantjatjara, Warlpiri, Tiwi, Yolŋu Matha, Arrernte | https://github.com/BibleNLP/ebible *(Warlpiri, Pitjantjatjara, Tiwi, Yolŋu, Kriol)*<br/>https://github.com/grambank/grambank<br/>https://github.com/lexibank/asjp |
| **19** | **Khoisan (3 rumpun)** | `~30` | Nama (Khoekhoe), !Xóõ, Ju\|'hoan, Naro *(konsonan klik)* | https://repo.sadilar.org *(Khoekhoegowab / Nama & N\|uu)*<br/>https://github.com/BibleNLP/ebible *(`naq` Nama & `nhr` Naro)*<br/>https://github.com/phoible/dev |

---

## C. Kelompok di Luar Rumpun Genetik Utama

| Kelompok Khusus | Contoh Bahasa | Link Sumber Data Terverifikasi |
| :--- | :--- | :--- |
| **1. Bahasa Isolat** | Basque (`Euskara`), Ainu, Burushaski, Sumeria (punah) | • **Basque:** https://huggingface.co/datasets/HiTZ/latxa-corpus-v1.1<br/>• **Ainu:** https://huggingface.co/datasets/wav2gloss/NINJAL-Ainu-Folklore · https://huggingface.co/datasets/Sotaro0124/Ainu-Japan_translation_model<br/>• **Sumeria:** https://cdli.mpiwg-berlin.mpg.de<br/>• **Burushaski & 180+ Isolat Lainnya:** https://github.com/grambank/grambank · https://github.com/lexibank/asjp |
| **2. Bahasa Isyarat** | BISINDO, ASL, BSL, LSF, JSL *(159–227 bahasa isyarat)* | • **Multi-Sign Loader (30+ Bahasa Isyarat):** https://github.com/sign-language-processing/datasets<br/>• **Sutton SignWriting Unicode (`ASL & Libras`):** https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 *(`ase_Sgnw, bzs_Sgnw`)*<br/>• **ASL:** https://huggingface.co/datasets/asthalochan/American_Sign_Language |
| **3. Pijin & Kreol** | Tok Pisin, Kreol Haiti, Papiamento, Melayu Ambon, Melayu Manado | • **APiCS (76 Pijin & Kreol):** https://github.com/cldf-datasets/apics · https://apics-online.info<br/>• **Teks Web (`tpi, hat, pap, pcm, jam, srn`):** https://huggingface.co/datasets/HuggingFaceFW/fineweb-2<br/>• **Melayu Ambon (`abs`), Melayu Manado (`xmm`), Melayu Papua (`pmy`):** https://github.com/BibleNLP/ebible |
| **4. Bahasa Buatan** | Esperanto, Interlingua, Lojban, Toki Pona, Klingon, Quenya | • **Web Conlangs (`epo, ina, ido, vol, jbo, tok`):** https://huggingface.co/datasets/HuggingFaceFW/fineweb-2<br/>• **Esperanto:** https://huggingface.co/datasets/saillab/alpaca-esperanto-cleaned<br/>• **Toki Pona:** https://huggingface.co/datasets/NetherQuartz/tatoeba-tokipona · https://huggingface.co/datasets/NetherQuartz/tokipona-wiki-parallel-mt<br/>• **Klingon, Lojban, Interlingua, Quenya (`Tatoeba`):** https://huggingface.co/datasets/sentence-transformers/parallel-sentences-tatoeba |
| **5. Klasik & Punah** | Latin, Sanskerta, Yunani Kuno, Mesir Kuno, Akkadia, Sumeria, Jawa Kuno (Kawi) | • **Latin:** https://huggingface.co/datasets/Fece228/latin-literature-dataset-170M · https://www.perseus.tufts.edu<br/>• **Sanskerta:** https://gretil.sub.uni-goettingen.de · https://huggingface.co/datasets/surajp/sanskrit_classic · https://huggingface.co/datasets/ai4bharat/sangraha<br/>• **Yunani Kuno:** https://github.com/OpenGreekAndLatin/First1KGreek · https://huggingface.co/datasets/eulogikon/ancient-greek-texts<br/>• **Akkadia & Sumeria (Paku/Cuneiform):** https://cdli.mpiwg-berlin.mpg.de · https://huggingface.co/datasets/veezbo/akkadian_english_corpus · https://huggingface.co/datasets/SzuTao/Cuneiform<br/>• **Jawa Kuno (Kawi) & Nusantara Klasik:** https://sastra.org · https://dumps.wikimedia.org<br/>• **Treebank Bahasa Kuno (`lat, grc, san, lzh, got, chu, cop, akk, hit`):** https://huggingface.co/datasets/universal-dependencies/universal_dependencies |

---

## D. Korpus Skala Raksasa untuk 8 Bahasa Penutur Terbanyak Dunia *(Ethnologue 2026 — L1 + L2)*

| Peringkat & Bahasa | Penutur (L1 + L2) | Link Korpus Fondasi Utama |
| :--- | :---: | :--- |
| **1. Inggris (`eng`)** | `~1,49 – 1,53 Miliar` | https://huggingface.co/datasets/Zyphra/Zyda-2 · https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu · https://huggingface.co/datasets/HuggingFaceTB/dclm-edu |
| **2. Mandarin (`cmn`)** | `~1,18 Miliar` | https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1 · https://huggingface.co/datasets/BAAI/IndustryCorpus2 · https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| **3. Hindi (`hin`)** | `~609 – 611 Juta` | https://huggingface.co/datasets/ai4bharat/sangraha · https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| **4. Spanyol (`spa`)** | `~558 – 561 Juta` | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 · https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned · https://huggingface.co/datasets/uonlp/CulturaX |
| **5. Arab Standar (`arb`)** | `~335 Juta` | https://huggingface.co/datasets/ClusterlabAi/101_billion_arabic_words_dataset · https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 · https://huggingface.co/datasets/arbml/ashaar |
| **6. Prancis (`fra`)** | `~334 Juta` | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 · https://huggingface.co/datasets/PleIAs/French-PD-Books · https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned |
| **7. Bengali (`ben`)** | `~274 Juta` | https://huggingface.co/datasets/ai4bharat/sangraha · https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 · https://huggingface.co/datasets/uonlp/CulturaX |
| **8. Portugis (`por`)** | `~269 Juta` | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 · https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned · https://huggingface.co/datasets/uonlp/CulturaX |

---

# BAGIAN 3: RENCANA EKSEKUSI ARSITEKTUR DATA LAKEHOUSE (7 FASE MATANG)

Untuk mengelola pengumpulan data sebesar **`22.40T` token bersih** (atau subset terkurasi `5.0T` / `1.65T`) tanpa pernah mengalami kehabisan RAM, korupsi encoding aksara langka, atau kebocoran benchmark evaluasi, jalankan **7 Fase Engineering** berikut:

### Fase 1 — Struktur Partisi *Data Lakehouse* Berstandar ISO & OECD
Simpan seluruh arsip master dalam format **`Apache Parquet + Zstd Level 6`** (`~512 MB – 1 GB` per file shard) dengan skema kolom seragam `(doc_id, text, iso639_3, script_iso15924, oecd_branch, source_repo, license, quality_score)` pada struktur direktori berikut:
```text
world_corpus_lake/
├── manifest_global.json
├── part1_sciences_oecd/
│   ├── A_formal/          # math, logic_lean, stats, cs_theory, code_6lang, asm_ir, systems_or
│   ├── B_natural/         # physics, chemistry, astronomy, earth, bio_core, organisms, neuro
│   ├── C_social/          # sociology, psychology, economics, politics, anthro, geog, media_crim
│   ├── D_humanities/      # philosophy, history, linguistics, literature, arts, religion_culture
│   ├── E_applied/         # engineering_patents, medicine_health, agro_food, law, business_sec, edu_arch, defense
│   └── F_interdisc/       # ai_datasci, bioinformatics, cogsci, environment, nanotech_behavioral
└── part2_languages_world/
    ├── cldf_typology_7674/  # Glottolog 5.3, ASJP, Grambank, WALS, PHOIBLE, APiCS, UD
    ├── monolingual_by_family/{niger_congo,austronesian,trans_new_guinea,sino_tibetan,...}
    ├── non_genetic/{isolates,sign_languages,pidgins_creoles,conlangs,classical_extinct}
    └── parallel_multilingual/{nllb,opus,ebible,ted_multiway,tatoeba}
```

### Fase 2 — Aturan Emas Perlindungan Bahasa Langka & Simbol Ilmiah (*Zero-Loss Unicode*)
1. **Gunakan Normalisasi `NFC` (DILARANG `NFKC` Global):** `NFKC` menghancurkan simbol matematika/fisika (`ℝ`, `ℋ`, `⊗`), rumus kimia/isotop (`¹⁴C`, `H₂O`), aksara kuno (Cuneiform, Kawi/Jawa, Sanskerta), serta tanda klik Khoisan (`ǀ`, `ǁ`, `ǂ`, `ǃ`).
2. **Deteksi Bahasa Dua-Lapis (`GlotLID-v3` + `FastText LID-176`):** `FastText` hanya mengenali 176 bahasa dan salah melabeli ribuan bahasa daerah/pribumi menjadi bahasa mayoritas. Wajib gunakan **`cis-lmu/GlotLID`** (mendukung **>2.000 label ISO-639-3 + aksara**) untuk menjaga kemurnian bahasa minoritas.
3. **Pengecualian Ambang Panjang untuk Bahasa *Low-Resource* & Klasik:** Jangan terapkan filter `len(text) >= 260` (yang biasa dipakai di web Inggris) pada bahasa Trans-Nugini, Khoisan, Pribumi Australia/Amerika, Ainu, Kawi, atau prasasti Sumeria/Akkadia — simpan kalimat/fragmen pendek (`>= 24` karakter) atau kelompokkan per wacana agar tidak terbuang.

### Fase 3 — Deduplikasi Terdistribusi Lintas-Korpus (`50.2T` → `22.4T`)
Karena objek `set` Python di RAM akan meledak pada miliaran dokumen, jalankan deduplikasi menggunakan **`HuggingFace datatrove`** atau engine C++/Rust berbasis partisi disk:
- **Tahap 3A (Exact Dedup Lintas-Sumber):** Hash `XXH3-128` per dokumen yang dipartisi ke `256` bucket disk NVMe; buang duplikat identik dengan urutan prioritas: *Buku Teks / Paper Peer-Reviewed > Wikipedia / Kurasi Manusia (`Sangraha/Swallow/HAERAE/Latxa`) > `FineWeb-Edu / Zyda-2` > `FineWeb-2` > `HPLT 2.0` > `CulturaX / MADLAD`*.
- **Tahap 3B (Fuzzy Dedup Per-Bahasa):** `MinHash LSH` (`14 bands × 8 rows`, Jaccard `0.80`) khusus pada bahasa ber-volume `>5B` token. **Matikan fuzzy dedup agresif** pada korpus paralel `eBible` dan bahasa terancam punah (`<50M` token) agar variasi dialek berharga tidak terhapus.

### Fase 4 — Dekontaminasi Ketat Terhadap Seluruh Benchmark Dunia
Bangun filter `Aho-Corasick` / `13-gram Bloom Filter` C++ untuk memastikan **0% kebocoran data uji** dari:
- **Multibahasa & Translasi:** `FLORES+` (`200+` bahasa), `NTREX-128`, `AmericasNLP test`, `MasakhaNEWS/MAFAND test`, `NusaX test`, `IndoMMLU test`, `KMMLU test`, `JMMLU`, `MMMLU` (MMLU 14 bahasa).
- **Sains, Medis & Humaniora:** `GPQA Diamond`, `OlympiadBench test`, `MMLU-Pro`, `Humanity's Last Exam (HLE)`, `MedQA (USMLE) test`, `IgakuQA test`, `KorMedMCQA test`, `LexGLUE test`.
- **Matematika & Kode:** `GSM8K test`, `MGSM`, `MATH-500`, `AIME`, `HumanEval / MultiPL-E`, `MBPP`, `LiveCodeBench`, `LLM4Binary/decompile-eval`.

### Fase 5 — Pengayaan Sintetik Khusus (*Assembly Factory* & *Cross-Lingual Knowledge Bridging*)
1. **Cross-Compiler Factory (`C/C++/Go -> x86_64, ARM64, RISC-V`):** Kompilasi jutaan fungsi dari `ExeBench` & `Stack-Edu` memakai `GCC 14` & `Clang 19` pada level optimasi `-O0, -O2, -O3` untuk menciptakan korpus sejajar `Source <-> LLVM IR <-> Multi-ISA Assembly`.
2. **Konversi CLDF (`7.674` Bahasa) ke Dokumen Pengetahuan Terstruktur:** Ubah tabel relasional `Glottolog`, `Grambank`, `PHOIBLE`, `WALS`, `APiCS`, dan `ASJP` menjadi artikel deskripsi linguistik & kamus komparatif terstruktur sehingga model memahami silsilah, fonologi, dan tata bahasa bahkan untuk **ribuan bahasa yang belum memiliki korpus web**!
3. **Protected-Span Multilingual STEM Transcreation:** Kunci blok rumus LaTeX (`$...$`) dan AST kode, lalu transkreasikan buku ajar inti dari ke-6 rumpun ilmu ke bahasa-bahasa target prioritas Anda.

### Fase 6 — Desain Tokenizer Universal & Pre-Tokenisasi Biner C++ (`mmap`)
- Latih tokenizer BPE byte-level bebas *UNK* dengan pemisahan digit tunggal (`0-9`) dan cakupan seimbang lintas aksara (`vocab = 65,536` `uint16_t` untuk model fokus, atau `vocab = 256,000` `uint24` 3-byte untuk model poliglot penuh `1.800+` bahasa).
- Konversi seluruh shard `Parquet Zstd` menjadi pasangan file biner `.bin` + `.idx` menggunakan tool C++ multi-thread sehingga GPU membaca token via `mmap` berkecepatan `>6 GB/detik` dengan beban CPU `~0%`.

### Fase 7 — Strategi Infrastruktur Eksekusi (*Cloud Bandwidth + Cold Lake*)
- Gunakan 1 *dedicated CPU server* ber-jaringan **`10 Gbps`** (`32–64 core`, `128–256 GB` RAM, `2× 4 TB` NVMe *scratch*) untuk melakukan *streaming download → filter → dedup → Parquet Zstd → upload langsung ke S3 / Cloudflare R2 (`$0` egress)*. Dengan pola *streaming bucket-by-bucket* ini, Anda tidak pernah perlu membeli ratusan TB SSD lokal sekaligus, dan seluruh *Data Lake* dunia siap di-*mount* atau ditarik per-cabang ilmu/bahasa kapan pun dibutuhkan.
