# MASTERPLAN DATA `175B` TOKEN (EDISI PADAT PENGETAHUAN DI DALAM `250 GB`)
**Mencakup 100% 6 Rumpun Ilmu Pengetahuan Dunia (OECD/UNESCO) + Deep Code (`Go, JS, Java, C++, C, Assembly`) + Seluruh `7.170+` Bahasa Dunia (Ethnologue Ed. 29 / Glottolog 5.3)**

> **Terakhir diperbarui:** 27 November 2026  
> **Lokasi File:** `/Users/daffanaufalrachmat/Documents/Real Projects/TransformerSample/transformer_500m_cuda/MASTERPLAN_DATA_175B.md`  
> **Target Utama:** Memadatkan intisari pengetahuan peradaban manusia (tanpa membuang satu pun cabang ilmu atau rumpun bahasa) ke dalam **`175,00 Miliar` (`175.00B`) token unik** yang muat dengan sangat aman di dalam batas kapasitas total **`250 GB`**.

---

## 1. MENGAPA SELURUH ILMU & BAHASA DUNIA BISA MUAT DI `175B` TOKEN (`<= 250 GB`)?

Korpus mentah internet berukuran puluhan Triliun token (`22.4T–50.2T`) bukan karena ilmu manusia sebanyak itu, melainkan karena **`>98.5%` isi web adalah pengulangan masif** (blog SEO berulang, berita serupa, boilerplate kode CRUD, dan percakapan forum dangkal). Sebagai perbandingan kepadatan informasi murni:
- **Seluruh `6,9 juta` artikel Wikipedia Bahasa Inggris** (semua bidang ilmu) = hanya **`~4.6B` token**.
- **Seluruh `10.000` buku teks perkuliahan tebal lintas fakultas** (`@ 400K` tok/buku) = hanya **`~4.0B` token**.
- **Seluruh profil tipologi, fonologi, & kosakata `7.674` bahasa dunia di Glottolog + Grambank + WALS + PHOIBLE + ASJP** = hanya **`~0.05B` (`50 Juta`) token**!

Dengan menerapkan paradigma **"Pure Knowledge Crystallization"** (mengambil **100% buku teks universitas, ensiklopedia, pedoman resmi, *survey/review papers* ber-sitasi tinggi, pembuktian formal, kode skor edukasi `>= 4`, dan sampel murni tiap bahasa dunia**, sambil membuang 99% sampah web repetitif), **`175.00B` token** justru memiliki kepadatan kecerdasan per-token berkali-kali lipat lebih tinggi dibanding korpus web mentah `2T` token.

---

## 2. ANGGARAN PRESISI DISK `250 GB` (DIUKUR EMPIRIS PADA KORPUS PROYEK)

Berdasarkan pengujian langsung kompresi `Zstd` pada korpus proyek ini (`english = 1.22–1.37 B/tok`, `japanese/CJK = 0.96–1.13 B/tok`, `code = 0.54–0.65 B/tok`, `math/LaTeX = 0.93–1.04 B/tok`), berikut adalah peta pemakaian disk agar **tidak pernah menyentuh batas `250 GB`** (bahkan pada SSD ukuran pabrikan `250 GB` desimal = `232.8 GiB` riil di OS):

| Komponen Penyimpanan di Disk `250 GB` | Ukuran Aktual (`GB` Desimal) | Ukuran Aktual (`GiB` Biner) | Keterangan Teknis |
| :--- | :---: | :---: | :--- |
| **1. Korpus Inti `175.00B` Token (`Seekable Zstd-15` / `.tok.zst`)** | **`170.0 GB`** | `158.3 GiB` | Rata-rata tertimbang **`0.971 Byte/token`**. Kecepatan dekompresi `Zstd-15` identik dengan `Zstd-1` (`>1.8 GB/detik` per core CPU)! |
| **2. Indeks Offset & Filter Deduplikasi Ringkas (`.zstidx` + Bloom)** | **`4.5 GB`** | `4.2 GiB` | Tabel alamat blok `1M` token + filter anti-duplikat berbasis disk hemat ruang. |
| **3. Rolling Download Staging Buffer *(Maks 1 Shard Sementara)*** | **`8.0 GB`** | `7.5 GiB` | Setiap shard Parquet mentah yang diunduh langsung difilter → dikompres → **dihapus seketika** (`unlink`), menjaga *cache* sementara selalu `< 8 GB`. |
| **4. Checkpoints Training (`ckpt` + `optim` + Tokenizer)** | **`12.5 GB`** | `11.6 GiB` | Ruang kerja aman untuk bobot model, *optimizer state*, dan snapshot upload. |
| **5. Sisa Ruang Bebas Aman (*OS / Swap / SSD Wear-Leveling*)** | **`55.0 GB`** | `51.2 GiB` | Margin bebas `22%` agar kecepatan tulis NVMe tidak pernah drop. |
| **TOTAL KAPASITAS** | **`250.0 GB`** | **`232.8 GiB`** | **100% Aman, Zero Disk Overflow.** |

> **Trik Format Penyimpanan Terbaik (`Pre-Tokenized Zstd Blocks` — `.tok.zst`):**  
> Jangan simpan `.txt` mentah (`633 GB` — tidak muat) ataupun `.bin` tak terkompresi (`350 GB` — tidak muat). Pre-tokenisasi teks satu kali menjadi ID token biner, potong per blok **`1.048.576` token (`1M` token)**, lalu kompres tiap blok dengan **`Zstd -15 --train` (kamus Zstd khusus token)**. Ukuran akhirnya tetap **`~170–178 GB`**, tetapi saat training GPU berjalan, C++ `PretrainDataset` cukup memanggil `ZSTD_decompress()` langsung ke RAM dalam `< 1.2 ms` tanpa beban CPU BPE `encode()` sama sekali!

---

## 3. MATRIKS ALOKASI KUOTA `175.00B` TOKEN & UKURAN DISK PER CABANG

### Ringkasan Eksekutif Pembagian `175.00B` Token (`170.0 GB`)

| Bagian & Rumpun Utama | Kuota Token (`B`) | Porsi (`%`) | Rasio `Zstd-15` (`B/tok`) | Ukuran di Disk (`GB`) |
| :--- | :---: | :---: | :---: | :---: |
| **Bagian 1A — Ilmu Formal & Deep Code (`Go, JS, Java, C++, C, ASM`)** | **`46.00B`** | `26.29%` | `0.709` | **`32.60 GB`** |
| **Bagian 1B — Ilmu Alam (`Fisika, Kimia, Astronomi, Bumi, Hayati, Neuro`)** | **`22.00B`** | `12.57%` | `1.045` | **`22.99 GB`** |
| **Bagian 1C — Ilmu Sosial (`7 Cabang Lengkap`)** | **`9.00B`** | `5.14%` | `1.160` | **`10.44 GB`** |
| **Bagian 1D — Humaniora (`6 Cabang Lengkap`)** | **`9.00B`** | `5.14%` | `1.150` | **`10.35 GB`** |
| **Bagian 1E — Ilmu Terapan & Profesi (`Medis, Teknik, Hukum, Bisnis, Agro, dll.`)** | **`17.00B`** | `9.71%` | `1.120` | **`19.04 GB`** |
| **Bagian 1F — Ilmu Interdisipliner (`AI, Bioinformatika, Kognitif, Iklim, Nano`)** | **`5.00B`** | `2.86%` | `1.080` | **`5.40 GB`** |
| **Bagian 2A — Basis Data Struktur & Leksikon Seluruh `7.674` Bahasa (`CLDF + UD`)** | **`1.00B`** | `0.57%` | `0.900` | **`0.90 GB`** |
| **Bagian 2B — Fondasi Puncak 4 Bahasa Inti (`ID, EN, JA, KO`)** | **`33.00B`** | `18.86%` | `1.080` | **`35.64 GB`** |
| **Bagian 2C — 9 Bahasa Makro Dunia Lainnya (`ZH, ES, AR, HI, FR, PT, BN, RU, DE`)** | **`12.00B`** | `6.86%` | `1.050` | **`12.60 GB`** |
| **Bagian 2D — `1.800+` Bahasa Lintas 19 Rumpun + Isolat + Isyarat + Kreol + Buatan + Klasik** | **`11.00B`** | `6.29%` | `1.020` | **`11.22 GB`** |
| **Bagian 2E — Translasi Multi-Arah (`12-Arah ID↔EN↔JA↔KO` + Jangkar `1.000` Bahasa)** | **`10.00B`** | `5.71%` | `0.880` | **`8.80 GB`** |
| **GRAND TOTAL KORPUS** | **`175.00B`** | **`100.0%`** | **`0.971`** | **`169.98 GB`** |

---

# BAGIAN 1: RINCIAN KUOTA, FILTER INTISARI & LINK AKURAT — 6 RUMPUN ILMU (OECD/UNESCO)

> **Aturan Seleksi Ketat Edisi `175B` ("Hanya Ambil Puncak Piramida"):**  
> Karena anggaran seluruh 6 rumpun ilmu + kode adalah **`108.00B` token**, **JANGAN** menarik seluruh paper mentah secara acak. Gunakan filter prioritas berikut saat streaming:  
> **Prioritas 1 (Ambil 100%):** Buku teks kuliah (`LibreTexts`, `OpenStax`, `DOAB`, `Pressbooks`), pedoman resmi (`EPFL Guidelines`), & ensiklopedia spesialis (`SEP`).  
> **Prioritas 2:** Artikel *Review / Survey / Tutorial* & paper sitasi tertinggi (`top 5–10%` di `OpenAlex` / `peS2o` / `PMC-OA` / `arXiv`).  
> **Prioritas 3:** Jejak penalaran langkah-demi-langkah (*Chain-of-Thought* / pembuktian formal / *case vignettes*) lintas bahasa.

## A. ILMU FORMAL & REKAYASA KODE (`46.00B` Token | `32.60 GB`)

| Sub-Cabang | Kuota | Filter Kurasi Padat & Link Sumber Terverifikasi |
| :--- | :---: | :--- |
| **1. Matematika Lengkap** *(Aljabar, Geometri, Kalkulus, Analisis, Teori Bilangan, Topologi, Kombinatorika)* | **`12.00B`** | Ambil murni `finemath-4plus` + `infiwebmath-4plus` + `Swallow-Math-v2` + buku teks `Mathematics LibreTexts` + soal olimpiade/pembuktian `NuminaMath-1.5` & `OpenMathReasoning` (termasuk porsi `ID/JA/KO` dari `OpenMathInstruct-JA` & transkreasi).<br/>• https://huggingface.co/datasets/HuggingFaceTB/finemath<br/>• https://huggingface.co/datasets/tokyotech-llm/swallow-math-v2<br/>• https://huggingface.co/datasets/EleutherAI/proof-pile-2<br/>• https://huggingface.co/datasets/AI-MO/NuminaMath-1.5<br/>• https://huggingface.co/datasets/nvidia/OpenMathReasoning<br/>• https://huggingface.co/datasets/kunishou/OpenMathInstruct-1-1.8m-ja<br/>• https://huggingface.co/datasets/LLM360/MegaMath *(ambil hanya subset QA & buku/kode math bersih)* |
| **2. Logika, Statistika, Stokastik & Sistem/Keputusan** *(Lean4/Coq/Isabelle, Peluang, Teori Permainan, Riset Operasi, Sibernetika)* | **`4.00B`** | Ambil penuh `Lean-Workbook`, subset `AlgebraicStack` (`Lean/Coq/Isabelle`), *Statistics LibreTexts*, `stats.stackexchange.com` & `or.stackexchange.com`, serta paper terpilih `arXiv` kategori `math.LO, cs.LO, stat.TH, stat.ME, math.PR, cs.GT, math.OC, eess.SY`.<br/>• https://huggingface.co/datasets/internlm/Lean-Workbook<br/>• https://huggingface.co/datasets/EleutherAI/proof-pile-2<br/>• https://huggingface.co/datasets/common-pile/arxiv_papers<br/>• https://huggingface.co/datasets/common-pile/libretexts<br/>• https://huggingface.co/datasets/common-pile/stackexchange |
| **3. Komp. Teoretis, Algoritma, Kriptografi & Deep Code (`Go, JS, Java, C++, C, Assembly`)** | **`30.00B`** | **Rincian ketat:** `C++ (5.5B)` · `C (4.5B)` · `Java (4.5B)` · `JavaScript (4.5B)` · `Golang (4.0B)` · `Assembly x86_64/ARM64/RISC-V + LLVM IR pairs (3.5B)` · `Algoritma/Kriptografi/Multilingual Code CoT ID-EN-JA-KO (3.5B)`. Hanya ambil kode `Stack-Edu (score >= 4)` + repo berbintang tinggi bebas boilerplate!<br/>• https://huggingface.co/datasets/HuggingFaceTB/stack-edu<br/>• https://huggingface.co/datasets/common-pile/stackv2<br/>• https://huggingface.co/datasets/bigcode/the-stack-v2-dedup<br/>• https://huggingface.co/datasets/jordiae/exebench *(kompilasi silang lokal GCC/Clang `-O0/-O2/-O3` ke `x86_64, ARM64, RISC-V`)*<br/>• https://huggingface.co/datasets/LLM4Binary/decompile-ghidra-100k<br/>• https://huggingface.co/datasets/LLM4Binary/decompile-bench<br/>• https://huggingface.co/datasets/llvm-ml/ComPile *(ambil sampel `1.0B` IR terbersih yang dipasangkan dengan C/C++)*<br/>• https://huggingface.co/datasets/nvidia/OpenCodeReasoning<br/>• https://huggingface.co/datasets/deepmind/code_contests<br/>• https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2 |

---

## B. ILMU ALAM: FISIKA & HAYATI (`22.00B` Token | `22.99 GB`)

| Sub-Cabang | Kuota | Filter Kurasi Padat & Link Sumber Terverifikasi |
| :--- | :---: | :--- |
| **1. Fisika Lengkap** *(Mekanika, Termodinamika, Elektromagnetisme, Optika, Kuantum, Relativitas, Partikel, Zat Padat)* | **`6.50B`** | 100% *Physics LibreTexts* + *OpenStax University Physics Vol 1–3* + `physics.stackexchange.com` + soal derivasi `MegaScience / NaturalReasoning / TheoremQA / CAMEL-Physics` + paper inti `arXiv` (`quant-ph, hep-*, cond-mat.*, gr-qc, nucl-*, physics.*`) + porsi `ID/JA/KO`.<br/>• https://huggingface.co/datasets/common-pile/libretexts<br/>• https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs<br/>• https://huggingface.co/datasets/common-pile/arxiv_papers<br/>• https://huggingface.co/datasets/common-pile/stackexchange<br/>• https://huggingface.co/datasets/MegaScience/MegaScience<br/>• https://huggingface.co/datasets/facebook/natural_reasoning<br/>• https://huggingface.co/datasets/TIGER-Lab/TheoremQA<br/>• https://huggingface.co/datasets/camel-ai/physics |
| **2. Kimia Lengkap** *(Organik, Anorganik, Fisik, Analitik, Biokimia, Kimia Material)* | **`3.50B`** | Ambil penuh `chempile-education` + `chempile-reasoning` + top-cited `chempile-paper` + `SMolInstruct` & `Mol-Instructions` + deskripsi sifat senyawa inti `PubChem`.<br/>• https://huggingface.co/datasets/jablonkagroup/chempile-education<br/>• https://huggingface.co/datasets/jablonkagroup/chempile-reasoning<br/>• https://huggingface.co/datasets/jablonkagroup/chempile-paper<br/>• https://huggingface.co/datasets/osunlp/SMolInstruct<br/>• https://huggingface.co/datasets/zjunlp/Mol-Instructions<br/>• https://huggingface.co/datasets/camel-ai/chemistry<br/>• https://pubchem.ncbi.nlm.nih.gov |
| **3. Astronomi & Ilmu Bumi** *(Astrofisika, Kosmologi, Keplanetan; Geologi, Geofisika, Meteorologi, Klimatologi, Oseanografi, Hidrologi, Geografi Fisik)* | **`3.00B`** | `1.2B` Astronomi (`arXiv astro-ph.*` review/core + *OpenStax Astronomy* + `astronomy.stackexchange` + **NASA ADS**) + `1.8B` Ilmu Bumi (*Geosciences LibreTexts* + `stackexchange_earthscience` + **EarthArXiv** + `peS2o Geology`).<br/>• https://huggingface.co/datasets/common-pile/arxiv_papers<br/>• https://ui.adsabs.harvard.edu<br/>• https://eartharxiv.org<br/>• https://huggingface.co/datasets/mlfoundations-dev/stackexchange_earthscience<br/>• https://huggingface.co/datasets/common-pile/libretexts<br/>• https://huggingface.co/datasets/allenai/peS2o |
| **4. Biologi Dasar, Organisme & Neurosains** *(Sel/Molekuler, Genetika, Evolusi, Anatomi, Fisiologi; Botani, Zoologi, Mikrobiologi, Virologi, Mikologi, Ekologi, Kelautan; Otak & Sistem Saraf)* | **`9.00B`** | `4.0B` Biologi Dasar (*Biology LibreTexts*, *OpenStax Biology 2e*, review papers *eLife / PLOS Biology / NAR* di `PMC-OA` & `bioRxiv`) + `3.0B` Organisme & Ekologi (`biodiversity_heritage_library_filtered` kurasi taksonomi + mikrobiologi/virologi/kelautan `peS2o` + `GenBank` summaries) + `2.0B` Neurosains (`PMC Neuroscience` + `arXiv q-bio.NC` + `bioRxiv neuro`).<br/>• https://huggingface.co/datasets/common-pile/pubmed<br/>• https://huggingface.co/datasets/mteb/raw_biorxiv<br/>• https://www.biorxiv.org/tdm<br/>• https://huggingface.co/datasets/common-pile/biodiversity_heritage_library_filtered<br/>• https://huggingface.co/datasets/common-pile/libretexts<br/>• https://huggingface.co/datasets/camel-ai/biology<br/>• https://www.ncbi.nlm.nih.gov/genbank/<br/>• https://huggingface.co/datasets/allenai/peS2o |

---

## C. ILMU SOSIAL LENGKAP (`9.00B` Token | `10.44 GB`)

| Sub-Cabang Ilmu Sosial | Kuota | Link Sumber Intisari Terverifikasi |
| :--- | :---: | :--- |
| **1. Sosiologi** *(Struktur, Stratifikasi, Perubahan Sosial, Perkotaan/Pedesaan)* | **`1.30B`** | https://osf.io/preprints/socarxiv · https://huggingface.co/datasets/common-pile/libretexts · https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs · https://huggingface.co/datasets/allenai/peS2o |
| **2. Psikologi** *(Kognitif, Klinis, Perkembangan, Sosial, IO, Neuropsikologi)* | **`1.80B`** | https://osf.io/preprints/psyarxiv · https://huggingface.co/datasets/Psychotherapy-LLM/PsychoCounsel-Preference · https://huggingface.co/datasets/jkhedri/psychology-dataset · https://huggingface.co/datasets/common-pile/pubmed |
| **3. Ilmu Ekonomi** *(Mikro, Makro, Ekonometrika, Pembangunan, Internasional)* | **`2.20B`** | https://www.econstor.eu · https://huggingface.co/datasets/common-pile/arxiv_papers *(`econ.*, q-fin.*`)* · https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k · https://huggingface.co/datasets/AdaptLLM/finance-tasks |
| **4. Ilmu Politik & HI** *(Teori Politik, Kebijakan Publik, Adm. Negara, HI)* | **`1.50B`** | https://huggingface.co/datasets/common-pile/usgpo · https://huggingface.co/datasets/Helsinki-NLP/europarl · https://opus.nlpl.eu *(`MultiUN`)* · https://huggingface.co/datasets/allenai/peS2o |
| **5. Antropologi & Arkeologi** *(Budaya, Biologis, Etnologi, Arkeologi)* | **`0.90B`** | https://huggingface.co/datasets/WithinUsAI/linguistic_anthropology_25k · https://huggingface.co/datasets/common-pile/doab · https://core.ac.uk |
| **6. Geografi Manusia** *(Demografi, Persebaran Penduduk, Geopolitik)* | **`0.60B`** | https://huggingface.co/datasets/allenai/peS2o *(`Geography`)* · https://huggingface.co/datasets/common-pile/doab · https://huggingface.co/datasets/common-pile/libretexts |
| **7. Komunikasi & Kriminologi** *(Kajian Media, Jurnalisme, Kriminologi)* | **`0.70B`** | https://huggingface.co/datasets/common-pile/news *(sampel kurasi analitis)* · https://huggingface.co/datasets/PleIAs/YouTube-Commons *(kuliah akademik)* · https://huggingface.co/datasets/common-pile/caselaw_access_project |

---

## D. HUMANIORA LENGKAP (`9.00B` Token | `10.35 GB`)

| Sub-Cabang Humaniora | Kuota | Link Sumber Intisari Terverifikasi |
| :--- | :---: | :--- |
| **1. Filsafat** *(Epistemologi, Metafisika, Etika, Estetika, Filsafat Ilmu)* | **`1.50B`** | https://plato.stanford.edu · https://huggingface.co/datasets/LisaMegaWatts/philosophy-corpus · https://huggingface.co/datasets/common-pile/stackexchange *(`philosophy`)* · https://huggingface.co/datasets/common-pile/project_gutenberg |
| **2. Sejarah** *(Kuno, Pertengahan, Modern, Ekonomi, Historiografi)* | **`2.20B`** | https://huggingface.co/datasets/common-pile/public_domain_review · https://huggingface.co/datasets/common-pile/library_of_congress · https://huggingface.co/datasets/common-pile/pre_1929_books *(kurasi buku sejarah)* · https://huggingface.co/datasets/common-pile/doab |
| **3. Linguistik** *(Fonologi, Morfologi, Sintaksis, Semantik, Pragmatik, Sosiolinguistik, Filologi)* | **`1.50B`** | https://huggingface.co/datasets/universal-dependencies/universal_dependencies · https://huggingface.co/datasets/mlfoundations-dev/stackexchange_linguistics · https://huggingface.co/datasets/common-pile/doab *(`Linguistics`)* |
| **4. Sastra** *(Teori Sastra, Kritik Sastra, Sastra Bandingan, Kanon Dunia)* | **`1.60B`** | https://huggingface.co/datasets/common-pile/project_gutenberg *(1.500 karya sastra kanonik dunia)* · https://huggingface.co/datasets/manu/project_gutenberg · https://huggingface.co/datasets/common-pile/doab |
| **5. Seni** *(Rupa, Musik, Tari, Teater, Film, Fotografi, Desain)* | **`1.00B`** | https://huggingface.co/datasets/sander-wood/irishman *(`ABC Music`)* · https://huggingface.co/datasets/huggan/wikiart · https://huggingface.co/datasets/common-pile/libretexts *(`Humanities/Arts`)* · https://huggingface.co/datasets/common-pile/doab |
| **6. Agama & Budaya** *(Teologi, Perbandingan Agama, Gender, Kajian Kawasan)* | **`1.20B`** | https://github.com/Sefaria/Sefaria-Export · https://cbeta.org · https://tanzil.net · https://gretil.sub.uni-goettingen.de · https://huggingface.co/datasets/common-pile/doab |

---

## E. ILMU TERAPAN & PROFESI LENGKAP (`17.00B` Token | `19.04 GB`)

| Sub-Cabang Ilmu Terapan | Kuota | Link Sumber Intisari Terverifikasi |
| :--- | :---: | :--- |
| **1. Kesehatan & Kedokteran Mendalam** *(Kedokteran, Farmasi, Keperawatan, Gigi, Hewan, Gizi, Kesmas — Kompatibel `ID/EN/JA/KO`)* | **`7.50B`** | Ambil 100% `epfl-llm/guidelines` + `MedRAG/textbooks` + *Medicine LibreTexts* + `ApolloCorpus` + `Medical-R1-Distill` + `UltraMedical` + review klinis & *case reports* terbaik `PMC-OA` + regional `IgakuQA / KorMedMCQA / IndoHealth-NLP / Garuda-PNPK`.<br/>• https://huggingface.co/datasets/epfl-llm/guidelines<br/>• https://huggingface.co/datasets/MedRAG/textbooks<br/>• https://huggingface.co/datasets/FreedomIntelligence/ApolloCorpus<br/>• https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data<br/>• https://huggingface.co/datasets/TsinghuaC3I/UltraMedical<br/>• https://huggingface.co/datasets/common-pile/pubmed<br/>• https://huggingface.co/datasets/openlifescienceai/medmcqa<br/>• https://huggingface.co/datasets/japan-ai-official/igakuqa-subset-curated<br/>• https://huggingface.co/datasets/sean0042/KorMedMCQA<br/>• https://huggingface.co/datasets/IndoHealth-NLP/NCD_Instruct-Tuning_Medical_QA_Indonesian |
| **2. Teknik / Rekayasa** *(Sipil, Mesin, Elektro, Kimia, Industri, Dirgantara, Material, Nuklir, Biomedis, Lingkungan)* | **`3.50B`** | 100% *Engineering LibreTexts* + `arXiv eess.*` + sampel merata lintas kelas IPC `A–H` dari paten `HUPD` (klaim & deskripsi teknis inti).<br/>• https://huggingface.co/datasets/common-pile/libretexts<br/>• https://huggingface.co/datasets/HUPD/hupd<br/>• https://huggingface.co/datasets/ccdv/patent-classification<br/>• https://huggingface.co/datasets/common-pile/arxiv_papers |
| **3. Hukum** *(Pidana, Perdata, Tata Negara, Bisnis, Internasional)* | **`2.20B`** | Putusan *landmark* & perundang-undangan terstratifikasi dari `Pile of Law`, `Caselaw Access Project`, `MultiEURLEX`, dan `US Regulations`.<br/>• https://huggingface.co/datasets/pile-of-law/pile-of-law<br/>• https://huggingface.co/datasets/common-pile/caselaw_access_project<br/>• https://huggingface.co/datasets/coastalcph/multi_eurlex<br/>• https://huggingface.co/datasets/common-pile/regulations |
| **4. Bisnis & Manajemen** *(Akuntansi, Keuangan, Pemasaran, SDM, Operasi)* | **`1.60B`** | Buku ajar lengkap *OpenStax Business/Accounting/Finance* + `Finance-Instruct-500k` + analisis MD&A terpilih dari `SEC-EDGAR`.<br/>• https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs<br/>• https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k<br/>• https://huggingface.co/datasets/TeraflopAI/SEC-EDGAR<br/>• https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train |
| **5. Agro & Pangan** *(Pertanian, Kehutanan, Peternakan, Perikanan, Akuakultur, Tek. Pangan)* | **`1.00B`** | https://agris.fao.org · https://huggingface.co/datasets/allenai/peS2o *(`Agricultural & Food Sciences`)* · https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only · https://huggingface.co/datasets/common-pile/foodista |
| **6. Pendidikan, Arsitektur & PWK** | **`0.70B`** | https://huggingface.co/datasets/common-pile/oercommons · https://huggingface.co/datasets/common-pile/pressbooks · https://huggingface.co/datasets/common-pile/doab · https://core.ac.uk |
| **7. Perpustakaan, Kearsipan & Strategi Pertahanan/Militer** | **`0.50B`** | https://discover.dtic.mil · https://huggingface.co/datasets/common-pile/library_of_congress · https://huggingface.co/datasets/common-pile/usgpo |

---

## F. ILMU INTERDISIPLINER (`5.00B` Token | `5.40 GB`)

| Bidang Interdisipliner | Kuota | Link Sumber Intisari Terverifikasi |
| :--- | :---: | :--- |
| **1. AI & Ilmu Data** | **`1.80B`** | https://huggingface.co/datasets/common-pile/arxiv_papers *(`cs.LG, cs.AI, cs.CL, cs.CV, stat.ML` top papers)* · https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k |
| **2. Bioinformatika** *(Teks Fungsi Gen/Protein + Kode Genomik)* | **`1.00B`** | https://www.uniprot.org *(Swiss-Prot curated functional annotations)* · https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks · https://huggingface.co/datasets/common-pile/stackexchange *(`bioinformatics`)* |
| **3. Ilmu Kognitif** | **`0.70B`** | https://huggingface.co/datasets/common-pile/stackexchange *(`cogsci`)* · https://osf.io/preprints/psyarxiv · https://huggingface.co/datasets/allenai/peS2o |
| **4. Ilmu Lingkungan & Iklim** | **`0.90B`** | https://eartharxiv.org · https://huggingface.co/datasets/allenai/peS2o *(`Environmental Science`)* · https://huggingface.co/datasets/common-pile/regulations |
| **5. Nanoteknologi & Ekonomi Perilaku** | **`0.60B`** | https://huggingface.co/datasets/common-pile/arxiv_papers *(`cond-mat.mes-hall, cond-mat.mtrl-sci`)* · https://huggingface.co/datasets/jablonkagroup/chempile-paper · https://www.econstor.eu |

---

# BAGIAN 2: RINCIAN KUOTA & LINK AKURAT — SELURUH `7.170+` BAHASA DUNIA (`67.00B` Token | `69.16 GB`)

## A. Pengetahuan Eksplisit Seluruh `7.674` Bahasa (`1.00B` Token | `0.90 GB`)
Konversikan 100% tabel relasional CLDF menjadi artikel deskripsi bahasa, fonologi, morfosintaksis, dan kosakata terstruktur agar **tak satu pun dari 7.674 bahasa terlewat**:
- **Glottolog 5.3 (`7.674` bahasa, `246` rumpun, `183` isolat, `227` isyarat):** https://glottolog.org | https://github.com/glottolog/glottolog
- **ASJP (`5.000+` daftar kosakata Swadesh):** https://github.com/lexibank/asjp
- **Grambank (`2.467` bahasa), WALS & PHOIBLE (`2.186` bahasa fonem):** https://github.com/grambank/grambank | https://github.com/cldf-datasets/wals | https://github.com/phoible/dev
- **APiCS (`76` Pijin/Kreol), UniMorph & Universal Dependencies (`160+` treebank):** https://github.com/cldf-datasets/apics | https://github.com/unimorph | https://huggingface.co/datasets/universal-dependencies/universal_dependencies

## B. 4 Bahasa Prioritas Utama Kualitas Tertinggi (`33.00B` Token | `35.64 GB`)
- **Inggris (`9.00B` — melengkapi `~70B` teks ilmu EN di Bagian 1):** `Wikipedia EN` penuh + `FineWeb-Edu (score >= 4.0)` → https://huggingface.co/datasets/wikimedia/wikipedia | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **Indonesia (`8.00B`):** `Wikipedia ID` penuh + `FineWeb-2 (ind_Latn)` top clean + `Indo4B/SEACrowd` + `Aya Indonesian` → https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | https://huggingface.co/datasets/SEACrowd/indo4b | https://huggingface.co/datasets/CohereForAI/aya_collection
- **Jepang (`8.00B`):** `Wikipedia JA` penuh + `FineWeb-2 (jpn_Jpan)` top clean + `Swallow-Nemotron` → https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | https://huggingface.co/datasets/tokyotech-llm/Swallow-Nemotron-Post-Training-Dataset-v1
- **Korea (`8.00B`):** `Wikipedia KO` penuh + `korean-webtext-edu` + `HAERAE KOREAN-WEBTEXT` + `HAE-RAE-COT-1.5M` → https://huggingface.co/datasets/eliceai/korean-webtext-edu | https://huggingface.co/datasets/HAERAE-HUB/KOREAN-WEBTEXT | https://huggingface.co/datasets/HAERAE-HUB/HAE-RAE-COT-1.5M

## C. 9 Bahasa Makro Dunia Lainnya (`12.00B` Token | `12.60 GB`)
Ambil **Wikipedia penuh + subset edukasi tertinggi** untuk masing-masing bahasa besar dunia:
- **Mandarin (`2.20B`):** https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1
- **Spanyol (`1.60B`), Prancis (`1.50B`), Portugis (`1.20B`), Rusia + Jerman (`1.20B`):** https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned
- **Arab Standar (`1.60B`):** https://huggingface.co/datasets/ClusterlabAi/101_billion_arabic_words_dataset | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- **Hindi (`1.50B`) & Bengali (`1.20B`):** https://huggingface.co/datasets/ai4bharat/sangraha

## D. Cakupan Lengkap `1.800+` Bahasa Lintas 19 Rumpun + 5 Kelompok Non-Genetik (`11.00B` Token | `11.22 GB`)
> **Aturan Kuota Adil (*Cap-per-Language*):** Batasi bahasa menengah maks `50M` token, tetapi **ambil `100%` tanpa pemotongan** untuk seluruh bahasa terancam punah, pribumi, isolat, isyarat, kreol, buatan, dan klasik yang volumenya `< 10M` token!

| Kelompok / Rumpun | Kuota | Sumber Spesifik Terverifikasi |
| :--- | :---: | :--- |
| **Niger-Kongo (`1.500+`), Afro-Asiatik Non-AR, Nilo-Sahara (`~200`) & Khoisan (`~30`)** | **`2.00B`** | https://huggingface.co/datasets/castorini/wura · https://huggingface.co/datasets/lelapa/Inkuba-Mono · https://huggingface.co/datasets/masakhane/masakhanews · https://repo.sadilar.org · https://github.com/BibleNLP/ebible |
| **Austronesia Nusantara/Pasifik (`~1.220`), Trans-Nugini (`~477`), Austroasiatik (`~165`) & Kra-Dai (`~90`)** | **`2.20B`** | https://huggingface.co/SEACrowd *(Jawa, Sunda, Bali, Bugis, Minang, Aceh, Vietnam, Khmer, Thai, Lao)* · https://github.com/BibleNLP/ebible *(Dani, Ekari, Enga, Huli, Melpa & ratusan bahasa PNG/Papua)* · https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 |
| **Dravida (`~85`) & Indo-Arya Regional** | **`1.40B`** | https://huggingface.co/datasets/ai4bharat/sangraha *(Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi, Santali)* |
| **Indo-Eropa Lainnya, Uralik (`~38`), Turkik (`~40`), Kaukasus (`~40`) & Mongolik/Tungusik (`~25`)** | **`2.00B`** | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 · https://huggingface.co/datasets/turkic-interlingua/turkic_xwmt · https://huggingface.co/datasets/joanitolopo/flores-plus-uralic · https://huggingface.co/datasets/cis-lmu/GlotCC-V1 |
| **Sino-Tibet Non-Mandarin (`Kanton, Wu, Min, Hakka, Tibet, Burma`), Japonik (`Ryukyu`) & Koreanik (`Jeju`)** | **`1.00B`** | https://openpecha.org · https://huggingface.co/datasets/shajiu/TibetanGeneral_corpus · https://huggingface.co/datasets/Mireu-Lab/Korean-jeju-dialect-speech · https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 *(`yue, wuu, nan, bod, mya, ryu`)* |
| **Pribumi Amerika (`1.000+`) & Pribumi Australia (`~250+`)** | **`0.90B`** | https://github.com/AmericasNLP · https://huggingface.co/datasets/Llamacha/monolingual-quechua-iic · https://huggingface.co/datasets/somosnlp/dataset-cultura-guarani_corpus-it · https://huggingface.co/datasets/somosnlp-hackathon-2022/Axolotl-Spanish-Nahuatl · https://github.com/BibleNLP/ebible *(Warlpiri, Pitjantjatjara, Tiwi, Yolŋu, Navajo, Cherokee, Maya)* |
| **5 Kelompok Khusus: Isolat, Bahasa Isyarat, Pijin/Kreol, Bahasa Buatan & Klasik/Punah (`Latin, Sanskerta, Yunani Kuno, Sumeria/Akkadia, Mesir Kuno, Kawi`)** | **`1.50B`** | • **Isolat:** https://huggingface.co/datasets/HiTZ/latxa-corpus-v1.1 *(Basque)* · https://huggingface.co/datasets/wav2gloss/NINJAL-Ainu-Folklore *(Ainu)*<br/>• **Isyarat:** https://github.com/sign-language-processing/datasets · `FineWeb-2 (ase_Sgnw, bzs_Sgnw)`<br/>• **Pijin/Kreol:** `FineWeb-2 (tpi, hat, pap, pcm)` · `eBible (Melayu Ambon abs, Melayu Manado xmm)`<br/>• **Buatan:** https://huggingface.co/datasets/saillab/alpaca-esperanto-cleaned · https://huggingface.co/datasets/NetherQuartz/tatoeba-tokipona · `Tatoeba (Lojban, Interlingua, Klingon)`<br/>• **Klasik & Punah:** https://huggingface.co/datasets/Fece228/latin-literature-dataset-170M · https://gretil.sub.uni-goettingen.de · https://github.com/OpenGreekAndLatin/First1KGreek · https://cdli.mpiwg-berlin.mpg.de · https://huggingface.co/datasets/veezbo/akkadian_english_corpus · https://sastra.org *(Jawa Kuno / Kawi)* |

## E. Jembatan Translasi Multi-Arah (`10.00B` Token | `8.80 GB`)
- **`6.00B` Token Translasi 12-Arah Presisi Tinggi (`ID ↔ EN ↔ JA ↔ KO`):** Gabungan kalimat, paragraf (`ALT`, `TED`, `KFTT`, `ASPEC`, `JParaCrawl`), dan istilah STEM/Medis/Kode dari `NLLB (score >= 0.85)`, `OPUS`, dan `Korean Parallel v1.1`.
  - https://huggingface.co/datasets/allenai/nllb | https://huggingface.co/datasets/SEACrowd/parallel_asian_treebank | https://huggingface.co/datasets/Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus | https://huggingface.co/datasets/lemon-mint/korean_parallel_sentences_v1.1
- **`4.00B` Token Jangkar Paralel Lintas `200–1.000+` Bahasa Dunia:** Menautkan seluruh bahasa minoritas/pribumi ke bahasa utama melalui `TED Talks Multi-Way`, `MultiUN`, `Tatoeba`, dan `eBible`.
  - https://huggingface.co/datasets/sentence-transformers/parallel-sentences-talks | https://huggingface.co/datasets/sentence-transformers/parallel-sentences-tatoeba | https://github.com/BibleNLP/ebible

---

## 4. RENCANA EKSEKUSI AMAN DI RUANG TERBATAS `250 GB` (5 LANGKAH ANTI-PENUH)

Agar proses pengumpulan **`175.00B` token** berjalan mulus di dalam mesin berkapasitas **`250 GB`** tanpa pernah kehabisan ruang disk sedikit pun:

1. **Langkah 1 — Latih Tokenizer Poliglot + STEM + Kode (`vocab_size = 65,536` `uint16_t`) pada Sampel `4 GB`:**
   - Tarik sampel mini terstratifikasi `4 GB` lintas aksara utama + LaTeX + 6 bahasa kode, latih tokenizer BPE byte-level (bebas `<unk>` untuk aksara apa pun di Unicode berkat *byte-level fallback* 256 byte dasar, ditambah merge penuh untuk Latin/ID/EN, Kana/Kanji/JA, Hangul/KO, Sirilik, Arab, Devanagari, Han, LaTeX, dan simbol kode), lalu hapus sampel sementara `4 GB` tersebut.
2. **Langkah 2 — Gunakan Pipeline *Zero-Cache Rolling Stream* (`Maks 8 GB Temp`):**
   - Set environment `HF_HUB_DISABLE_SYMLINKS_WARNING=1` dan **jangan pernah memakai `datasets.load_dataset()` tanpa `streaming=True`** (karena cache `~/.cache/huggingface` bisa diam-diam membengkak puluhan GB!).
   - Gunakan pola `iter_prefetched_parquet` (seperti di `scripts/build_corpus_v9_50b.py`): unduh **1 file Parquet** (`~300–800 MB`) → saring kualitas + cek dekontaminasi benchmark (`FLORES+`, `HumanEval`, `GSM8K/MGSM`, `GPQA`, `USMLE/IgakuQA/KorMedMCQA/IndoMMLU`) → cek hash duplikat pada *Memory-Mapped Bloom Filter* (`2.5 GB` untuk 1 miliar dokumen dengan *false-positive* `< 0.1%`!) → tokenisasi langsung ke buffer `uint16_t` → kompres blok `1M` token memakai `Zstd -15` → tulis ke `.tok.zst` → **`unlink()` file Parquet sumber detik itu juga**!
3. **Langkah 3 — Hard Disk-Floor Guard Otomatis (`STOP_IF_FREE_BELOW = 40 GB`):**
   - Pasang penjaga `shutil.disk_usage()` setiap penulisan blok `1M` token zodat kuota tiap sub-cabang berhenti presisi sesuai tabel di atas (`Total = 170.0 GB`) dan menyisakan minimal `55 GB` ruang bebas untuk checkpoint dan OS.
4. **Langkah 4 — Patch Ringan `PretrainDataset` C++ (`ZSTD_decompress` In-Memory):**
   - Tambahkan dukungan membaca blok `.tok.zst` + `.zstidx` di `transformer/pretrain_dataset.cpp` menggunakan pustaka standar `libzstd` (`#include <zstd.h>`). Thread `prefetch_loop()` cukup memilih shard `.tok.zst` sesuai bobot cabang ilmu/bahasa, melakukan `fseek` ke offset blok `1M` token acak, dan men-dekompresinya langsung ke vektor RAM `Shard::tokens` dalam `< 1.2 milidetik`!
5. **Langkah 5 — Kurikulum Pelatihan 3-Fase di Atas Korpus `175B`:**
   - **Fase 1 (`0% → 60%`):** Campuran seimbang seluruh bahasa dunia + pengetahuan dasar 6 rumpun ilmu + kode umum.
   - **Fase 2 (`60% → 88%`):** Tingkatkan bobot buku teks kuliah (`LibreTexts/OpenStax`), kode sistem & `Assembly`, serta translasi 12-arah level dokumen.
   - **Fase 3 (`88% → 100%` *Decay*):** Fokus pada bukti teorema formal (`Lean4` / `FineMath-4+`), penurunan fisika/kimia, pedoman klinis kedokteran, dan akurasi lintas-bahasa tingkat tinggi.
