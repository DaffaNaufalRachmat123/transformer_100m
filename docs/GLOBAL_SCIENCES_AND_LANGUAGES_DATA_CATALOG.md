# Katalog Induk Sumber Data Dunia: 6 Rumpun Ilmu (OECD/UNESCO) & 7.170+ Bahasa (Ethnologue / Glottolog)

> **Catatan Verifikasi & Realitas Ketersediaan Digital:**  
> 1. Seluruh link di dokumen ini telah diverifikasi aktif (`HTTP 200` / API HuggingFace Hub & GitHub CLDF).  
> 2. Untuk mengunduh seluruh **6 Rumpun Ilmu OECD/UNESCO** secara efisien, tarik terlebih dahulu **Agregator Akademik Universal** (`OpenAlex`, `CORE`, `peS2o`, `Common-Pile`, `LibreTexts/DOAB/OpenStax`) yang memetakan 250+ juta karya ilmiah langsung ke taksonomi keilmuan global.  
> 3. Dari **7.170 bahasa hidup (Ethnologue Ed. 29) / 7.674 bahasa (Glottolog 5.3)**, sekitar **~5.000+ bahasa** memiliki basis data leksikal/fonologi/tipologi (`ASJP`, `Glottolog`, `Grambank`, `PHOIBLE`), **~1.800–2.200 bahasa** memiliki korpus teks berkelanjutan (`FineWeb-2`, `GlotCC-V1`, `eBible`), dan **~420 bahasa** memiliki korpus web skala besar (`MADLAD-400`, `CulturaX`, `HPLT 2.0`, `Sangraha`).

---

# BAGIAN 1: CABANG ILMU PENGETAHUAN DI SELURUH DUNIA
*(Taksonomi Standar OECD FOS & UNESCO: 6 Rumpun Lengkap)*

## 0. Agregator Induk Lintas-6-Rumpun Ilmu
- **OpenAlex (`250M+` Karya Ilmiah Berlabel Taksonomi OECD/UNESCO):** https://docs.openalex.org
- **CORE (`300M+` Full-Text Open Access Lintas Fakultas):** https://core.ac.uk
- **peS2o (`AllenAI` & `Common-Pile`):** https://huggingface.co/datasets/allenai/peS2o | https://huggingface.co/datasets/common-pile/peS2o
- **Buku Teks & Monograf Universitas (`LibreTexts, OpenStax, DOAB, Pressbooks, OERCommons`):** https://huggingface.co/datasets/common-pile/libretexts | https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs | https://huggingface.co/datasets/common-pile/doab | https://huggingface.co/datasets/common-pile/pressbooks | https://huggingface.co/datasets/common-pile/oercommons
- **StackExchange (`180+` Komunitas Keilmuan):** https://huggingface.co/datasets/common-pile/stackexchange

## A. ILMU FORMAL (*Formal Sciences*)
- **Matematika *(Aljabar, Geometri, Kalkulus, Analisis, Teori Bilangan, Topologi, Kombinatorika)*:** https://huggingface.co/datasets/LLM360/MegaMath | https://huggingface.co/datasets/HuggingFaceTB/finemath | https://huggingface.co/datasets/EleutherAI/proof-pile-2 | https://huggingface.co/datasets/open-web-math/open-web-math | https://huggingface.co/datasets/AI-MO/NuminaMath-1.5 | https://huggingface.co/datasets/nvidia/OpenMathReasoning
- **Logika *(Matematika, Simbolik, Komputasional: Lean4, Coq, Isabelle)*:** https://huggingface.co/datasets/internlm/Lean-Workbook | https://huggingface.co/datasets/EleutherAI/proof-pile-2 | https://huggingface.co/datasets/common-pile/arxiv_papers *(`math.LO`, `cs.LO`)*
- **Statistika *(Inferensial, Peluang, Stokastik)*:** https://huggingface.co/datasets/common-pile/arxiv_papers *(`stat.*`, `math.PR`, `math.ST`)* | https://huggingface.co/datasets/common-pile/libretexts *(`Statistics`)* | https://huggingface.co/datasets/common-pile/stackexchange *(`stats`)*
- **Ilmu Komputer Teoretis *(Komputasi, Algoritma, Struktur Data, Teori Informasi, Kriptografi)*:** https://huggingface.co/datasets/common-pile/arxiv_papers *(`cs.DS, cs.CC, cs.IT, cs.CR`)* | https://huggingface.co/datasets/deepmind/code_contests | https://huggingface.co/datasets/nvidia/OpenCodeReasoning | https://huggingface.co/datasets/HuggingFaceTB/stack-edu | https://huggingface.co/datasets/bigcode/the-stack-v2-dedup
- **Sistem & Keputusan *(Teori Permainan, Riset Operasi, Sibernetika)*:** https://huggingface.co/datasets/common-pile/arxiv_papers *(`cs.GT, math.OC, cs.SY, eess.SY`)* | https://huggingface.co/datasets/common-pile/stackexchange *(`or`)*

## B. ILMU ALAM (*Natural Sciences*)
### [1] Ilmu Fisika (*Physical Sciences*)
- **Fisika *(Mekanika, Termodinamika, Elektromagnetisme, Optika, Kuantum, Relativitas, Partikel, Zat Padat)*:** https://huggingface.co/datasets/common-pile/arxiv_papers *(`quant-ph, hep-*, cond-mat, gr-qc, nucl-*, physics.*`)* | https://huggingface.co/datasets/common-pile/libretexts | https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs | https://huggingface.co/datasets/MegaScience/MegaScience | https://huggingface.co/datasets/camel-ai/physics
- **Kimia *(Organik, Anorganik, Fisik, Analitik, Biokimia, Kimia Material)*:** https://huggingface.co/datasets/jablonkagroup/chempile-paper | https://huggingface.co/datasets/jablonkagroup/chempile-education | https://huggingface.co/datasets/jablonkagroup/chempile-reasoning | https://huggingface.co/datasets/jablonkagroup/chempile-code | https://huggingface.co/datasets/osunlp/SMolInstruct | https://huggingface.co/datasets/zjunlp/Mol-Instructions | https://huggingface.co/datasets/camel-ai/chemistry | https://pubchem.ncbi.nlm.nih.gov
- **Astronomi *(Astrofisika, Kosmologi, Ilmu Keplanetan)*:** https://huggingface.co/datasets/common-pile/arxiv_papers *(`astro-ph.*`)* | https://ui.adsabs.harvard.edu | https://huggingface.co/datasets/common-pile/stackexchange *(`astronomy`)*
- **Ilmu Bumi *(Geologi, Geofisika, Meteorologi, Klimatologi, Oseanografi, Hidrologi, Geografi Fisik)*:** https://eartharxiv.org | https://huggingface.co/datasets/mlfoundations-dev/stackexchange_earthscience | https://huggingface.co/datasets/common-pile/libretexts *(`Geosciences`)* | https://huggingface.co/datasets/allenai/peS2o *(`Geology`)* | https://huggingface.co/datasets/common-pile/arxiv_papers *(`physics.ao-ph, physics.geo-ph`)*

### [2] Ilmu Hayati (*Life Sciences*)
- **Biologi Dasar *(Sel & Molekuler, Genetika, Evolusi, Anatomi, Fisiologi)*:** https://huggingface.co/datasets/common-pile/pubmed | https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/ | https://huggingface.co/datasets/mteb/raw_biorxiv | https://www.biorxiv.org/tdm | https://huggingface.co/datasets/common-pile/libretexts | https://huggingface.co/datasets/camel-ai/biology
- **Organisme *(Botani, Zoologi, Mikrobiologi, Virologi, Mikologi, Ekologi, Biologi Kelautan)*:** https://huggingface.co/datasets/common-pile/biodiversity_heritage_library_filtered | https://www.ncbi.nlm.nih.gov/genbank/ | https://huggingface.co/datasets/allenai/peS2o
- **Neurosains *(Sistem Saraf & Otak)*:** https://huggingface.co/datasets/common-pile/pubmed *(`Neurosciences MeSH`)* | https://huggingface.co/datasets/mteb/raw_biorxiv *(`neuroscience`)* | https://huggingface.co/datasets/common-pile/arxiv_papers *(`q-bio.NC`)*

## C. ILMU SOSIAL (*Social Sciences*)
- **Sosiologi:** https://osf.io/preprints/socarxiv | https://huggingface.co/datasets/common-pile/libretexts | https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs | https://huggingface.co/datasets/allenai/peS2o
- **Psikologi:** https://osf.io/preprints/psyarxiv | https://huggingface.co/datasets/Psychotherapy-LLM/PsychoCounsel-Preference | https://huggingface.co/datasets/jkhedri/psychology-dataset | https://huggingface.co/datasets/common-pile/pubmed
- **Ilmu Ekonomi:** https://www.econstor.eu | https://huggingface.co/datasets/common-pile/arxiv_papers *(`econ.*, q-fin.*`)* | https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k | https://huggingface.co/datasets/AdaptLLM/finance-tasks
- **Ilmu Politik, Kebijakan Publik & HI:** https://huggingface.co/datasets/common-pile/usgpo | https://huggingface.co/datasets/Helsinki-NLP/europarl | https://opus.nlpl.eu | https://huggingface.co/datasets/allenai/peS2o
- **Antropologi & Arkeologi:** https://huggingface.co/datasets/WithinUsAI/linguistic_anthropology_25k | https://huggingface.co/datasets/common-pile/doab | https://core.ac.uk
- **Geografi Manusia & Demografi:** https://huggingface.co/datasets/allenai/peS2o *(`Geography`)* | https://huggingface.co/datasets/common-pile/doab | https://huggingface.co/datasets/common-pile/libretexts
- **Komunikasi, Jurnalisme & Kriminologi:** https://huggingface.co/datasets/common-pile/news | https://huggingface.co/datasets/PleIAs/YouTube-Commons | https://huggingface.co/datasets/common-pile/caselaw_access_project

## D. HUMANIORA (*Humanities*)
- **Filsafat:** https://plato.stanford.edu | https://huggingface.co/datasets/LisaMegaWatts/philosophy-corpus | https://huggingface.co/datasets/common-pile/project_gutenberg | https://huggingface.co/datasets/common-pile/stackexchange *(`philosophy`)*
- **Sejarah:** https://huggingface.co/datasets/common-pile/library_of_congress | https://huggingface.co/datasets/common-pile/pre_1929_books | https://huggingface.co/datasets/common-pile/public_domain_review | https://huggingface.co/datasets/PleIAs/French-PD-Books
- **Linguistik:** https://huggingface.co/datasets/universal-dependencies/universal_dependencies | https://huggingface.co/datasets/mlfoundations-dev/stackexchange_linguistics | https://github.com/glottolog/glottolog | https://github.com/cldf-datasets/wals | https://github.com/grambank/grambank | https://github.com/phoible/dev | https://github.com/unimorph
- **Sastra:** https://huggingface.co/datasets/common-pile/project_gutenberg | https://huggingface.co/datasets/manu/project_gutenberg | https://dumps.wikimedia.org | https://huggingface.co/datasets/common-pile/doab
- **Seni *(Rupa, Musik, Tari, Teater, Film, Foto, Desain)*:** https://huggingface.co/datasets/sander-wood/irishman | https://huggingface.co/datasets/huggan/wikiart | https://huggingface.co/datasets/common-pile/libretexts | https://huggingface.co/datasets/common-pile/doab
- **Agama & Budaya:** https://github.com/Sefaria/Sefaria-Export | https://cbeta.org | https://tanzil.net | https://github.com/BibleNLP/ebible | https://huggingface.co/datasets/common-pile/doab

## E. ILMU TERAPAN & PROFESI (*Applied Sciences*)
- **Teknik / Rekayasa *(Sipil, Mesin, Elektro, Kimia, Industri, Informatika, Dirgantara, Material, Nuklir, Biomedis, Lingkungan)*:** https://huggingface.co/datasets/HUPD/hupd | https://huggingface.co/datasets/ccdv/patent-classification | https://huggingface.co/datasets/common-pile/arxiv_papers *(`eess.*`)* | https://huggingface.co/datasets/common-pile/libretexts
- **Kesehatan *(Kedokteran, Farmasi, Keperawatan, Gigi, Hewan, Gizi, Kesmas)*:** https://huggingface.co/datasets/common-pile/pubmed | https://huggingface.co/datasets/ncbi/pubmed | https://huggingface.co/datasets/epfl-llm/guidelines | https://huggingface.co/datasets/FreedomIntelligence/ApolloCorpus | https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data | https://huggingface.co/datasets/TsinghuaC3I/UltraMedical
- **Agro & Pangan *(Pertanian, Agronomi, Peternakan, Kehutanan, Perikanan, Akuakultur, Tek. Pangan)*:** https://agris.fao.org | https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only | https://huggingface.co/datasets/common-pile/foodista | https://huggingface.co/datasets/allenai/peS2o
- **Hukum *(Pidana, Perdata, Tata Negara, Bisnis, Internasional)*:** https://huggingface.co/datasets/pile-of-law/pile-of-law | https://huggingface.co/datasets/common-pile/caselaw_access_project | https://huggingface.co/datasets/common-pile/regulations | https://huggingface.co/datasets/coastalcph/multi_eurlex | https://huggingface.co/datasets/coastalcph/lex_glue
- **Bisnis & Manajemen *(Akuntansi, Keuangan, Pemasaran, SDM, Operasi)*:** https://huggingface.co/datasets/TeraflopAI/SEC-EDGAR | https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k | https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train | https://huggingface.co/datasets/common-pile/libretexts | https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs
- **Pendidikan, Arsitektur & PWK:** https://huggingface.co/datasets/common-pile/oercommons | https://huggingface.co/datasets/common-pile/pressbooks | https://huggingface.co/datasets/common-pile/doab | https://core.ac.uk
- **Perpustakaan, Kearsipan & Strategi Pertahanan/Militer:** https://discover.dtic.mil | https://huggingface.co/datasets/common-pile/library_of_congress | https://huggingface.co/datasets/common-pile/usgpo

## F. ILMU INTERDISIPLINER (*Interdisciplinary Fields*)
- **AI & Ilmu Data:** https://huggingface.co/datasets/common-pile/arxiv_papers *(`cs.LG, cs.AI, cs.CL, cs.CV, stat.ML`)* | https://huggingface.co/datasets/bigcode/the-stack-v2-dedup | https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k
- **Bioinformatika:** https://huggingface.co/datasets/InstaDeepAI/multi_species_genomes | https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks | https://www.uniprot.org | https://huggingface.co/datasets/mteb/raw_biorxiv | https://huggingface.co/datasets/common-pile/stackexchange *(`bioinformatics`)*
- **Ilmu Kognitif:** https://huggingface.co/datasets/common-pile/stackexchange *(`cogsci`)* | https://osf.io/preprints/psyarxiv | https://huggingface.co/datasets/allenai/peS2o
- **Ilmu Lingkungan:** https://eartharxiv.org | https://huggingface.co/datasets/allenai/peS2o *(`Environmental Science`)* | https://huggingface.co/datasets/common-pile/regulations
- **Nanoteknologi & Ekonomi Perilaku:** https://huggingface.co/datasets/common-pile/arxiv_papers *(`cond-mat.mes-hall, cond-mat.mtrl-sci`)* | https://huggingface.co/datasets/jablonkagroup/chempile-paper | https://www.econstor.eu

---

# BAGIAN 2: SELURUH BAHASA DI DUNIA (`7.170` Ethnologue / `7.674` Glottolog)

## A. Katalog Induk Global & Mega-Korpus Ribuan Bahasa
- **Glottolog 5.3 (`7.674` Bahasa, `246` Rumpun, `183` Isolat, `227` Isyarat):** https://glottolog.org | https://github.com/glottolog/glottolog
- **ASJP (`5.000+` Bahasa Leksikon Dasar):** https://github.com/lexibank/asjp
- **Grambank (`2.467` Bahasa), WALS & PHOIBLE (`2.186` Bahasa Fonologi):** https://github.com/grambank/grambank | https://grambank.clld.org | https://github.com/cldf-datasets/wals | https://wals.info | https://github.com/phoible/dev | https://phoible.org
- **FineWeb-2 (`1.800+` Bahasa & Aksara Web Bersih):** https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- **GlotCC-V1 & Glot500 (`1.000+` Bahasa Minoritas/Low-Resource):** https://huggingface.co/datasets/cis-lmu/GlotCC-V1 | https://huggingface.co/datasets/cis-lmu/Glot500
- **eBible Parallel (`1.000+` Bahasa Terjajar):** https://github.com/BibleNLP/ebible | https://huggingface.co/datasets/DavidCBaines/ebible_corpus
- **MADLAD-400 (`419`), Wikipedia (`340+`), NLLB (`200+`), CulturaX (`167`) & Aya (`114`):** https://huggingface.co/datasets/allenai/MADLAD-400 | https://huggingface.co/datasets/wikimedia/wikipedia | https://huggingface.co/datasets/allenai/nllb | https://huggingface.co/datasets/openlanguagedata/flores_plus | https://huggingface.co/datasets/uonlp/CulturaX | https://huggingface.co/datasets/CohereForAI/aya_collection

## B. Sumber Spesifik per 19 Rumpun Bahasa Utama Dunia
1. **Niger-Kongo (`1.500+`):** https://huggingface.co/datasets/castorini/wura | https://huggingface.co/datasets/lelapa/Inkuba-Mono | https://huggingface.co/datasets/masakhane/masakhanews | https://huggingface.co/datasets/masakhane/mafand | https://repo.sadilar.org
2. **Austronesia (`~1.220`):** https://huggingface.co/SEACrowd | https://huggingface.co/datasets/SEACrowd/indo4b | https://huggingface.co/datasets/IndoNLP/NusaX-senti | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
3. **Trans-Nugini (`~477`):** https://github.com/BibleNLP/ebible | https://huggingface.co/datasets/cis-lmu/GlotCC-V1 | https://github.com/grambank/grambank
4. **Sino-Tibet (`~460`):** https://huggingface.co/datasets/BAAI/IndustryCorpus2 | https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1 | https://openpecha.org | https://huggingface.co/datasets/shajiu/TibetanGeneral_corpus | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
5. **Indo-Eropa (`~446`):** https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned | https://huggingface.co/datasets/ai4bharat/sangraha | https://huggingface.co/datasets/uonlp/CulturaX
6. **Afro-Asiatik (`~376`):** https://huggingface.co/datasets/ClusterlabAi/101_billion_arabic_words_dataset | https://huggingface.co/datasets/arbml/ashaar | https://huggingface.co/datasets/castorini/wura | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
7. **Nilo-Sahara (`~200`):** https://huggingface.co/datasets/allenai/nllb | https://huggingface.co/datasets/masakhane/mafand | https://github.com/BibleNLP/ebible
8. **Austroasiatik (`~165`):** https://huggingface.co/SEACrowd | https://huggingface.co/datasets/ai4bharat/sangraha *(`Santali`)* | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
9. **Kra-Dai (`~90`):** https://huggingface.co/SEACrowd | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | https://huggingface.co/datasets/allenai/nllb
10. **Dravida (`~85`):** https://huggingface.co/datasets/ai4bharat/sangraha *(`Tamil, Telugu, Kannada, Malayalam`)* | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
11. **Turkik (`~40`):** https://huggingface.co/datasets/turkic-interlingua/turkic_xwmt | https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
12. **Kaukasus (`~40`):** https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 *(`kat, che, ava, ady, kbd, abk, lez, xmf`)* | https://huggingface.co/datasets/cis-lmu/GlotCC-V1
13. **Uralik (`~38`):** https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned | https://huggingface.co/datasets/joanitolopo/flores-plus-uralic | https://huggingface.co/datasets/nup-csai/uralicBPE | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
14. **Mongolik & Tungusik (`~25`):** https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 *(`khk_Cyrl, mon_Mong, bua_Cyrl, xal_Cyrl`)* | https://huggingface.co/datasets/cis-lmu/Glot500
15. **Japonik (`~12`):** https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 *(`jpn_Jpan, ryu_Jpan`)* | https://huggingface.co/datasets/tokyotech-llm/Swallow-Nemotron-Post-Training-Dataset-v1 | https://dumps.wikimedia.org
16. **Koreanik (`2`):** https://huggingface.co/datasets/HAERAE-HUB/KOREAN-WEBTEXT | https://huggingface.co/datasets/eliceai/korean-webtext-edu | https://huggingface.co/datasets/Mireu-Lab/Korean-jeju-dialect-speech | https://dumps.wikimedia.org
17. **Pribumi Amerika (`1.000+`):** https://github.com/AmericasNLP | https://huggingface.co/datasets/Llamacha/monolingual-quechua-iic | https://huggingface.co/datasets/somosnlp/dataset-cultura-guarani_corpus-it | https://huggingface.co/datasets/somosnlp-hackathon-2022/Axolotl-Spanish-Nahuatl | https://github.com/BibleNLP/ebible
18. **Pribumi Australia (`~250+`):** https://github.com/BibleNLP/ebible | https://github.com/grambank/grambank | https://github.com/lexibank/asjp
19. **Khoisan (`~30`):** https://repo.sadilar.org | https://github.com/BibleNLP/ebible | https://github.com/phoible/dev

## C. Kelompok di Luar Rumpun Genetik Utama
- **Bahasa Isolat *(Basque, Ainu, Burushaski, Sumeria)*:** https://huggingface.co/datasets/HiTZ/latxa-corpus-v1.1 | https://huggingface.co/datasets/wav2gloss/NINJAL-Ainu-Folklore | https://huggingface.co/datasets/Sotaro0124/Ainu-Japan_translation_model | https://cdli.mpiwg-berlin.mpg.de | https://github.com/grambank/grambank
- **Bahasa Isyarat *(BISINDO, ASL, BSL, LSF, JSL, SignWriting)*:** https://github.com/sign-language-processing/datasets | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 *(`ase_Sgnw, bzs_Sgnw`)* | https://huggingface.co/datasets/asthalochan/American_Sign_Language
- **Pijin & Kreol *(Tok Pisin, Kreol Haiti, Papiamento, Melayu Ambon, Melayu Manado)*:** https://github.com/cldf-datasets/apics | https://apics-online.info | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | https://github.com/BibleNLP/ebible
- **Bahasa Buatan *(Esperanto, Interlingua, Lojban, Toki Pona, Klingon, Quenya)*:** https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | https://huggingface.co/datasets/saillab/alpaca-esperanto-cleaned | https://huggingface.co/datasets/NetherQuartz/tatoeba-tokipona | https://huggingface.co/datasets/NetherQuartz/tokipona-wiki-parallel-mt | https://huggingface.co/datasets/sentence-transformers/parallel-sentences-tatoeba
- **Bahasa Klasik & Punah *(Latin, Sanskerta, Yunani Kuno, Akkadia, Sumeria, Mesir Kuno, Jawa Kuno / Kawi)*:**
  - Latin: https://huggingface.co/datasets/Fece228/latin-literature-dataset-170M | https://www.perseus.tufts.edu
  - Sanskerta: https://gretil.sub.uni-goettingen.de | https://huggingface.co/datasets/surajp/sanskrit_classic | https://huggingface.co/datasets/ai4bharat/sangraha
  - Yunani Kuno: https://github.com/OpenGreekAndLatin/First1KGreek | https://huggingface.co/datasets/eulogikon/ancient-greek-texts
  - Sumeria & Akkadia: https://cdli.mpiwg-berlin.mpg.de | https://huggingface.co/datasets/veezbo/akkadian_english_corpus | https://huggingface.co/datasets/SzuTao/Cuneiform
  - Jawa Kuno (Kawi) & Nusantara Klasik: https://sastra.org | https://dumps.wikimedia.org
  - Treebank Bahasa Kuno (`lat, grc, san, lzh, got, chu, cop, akk, hit`): https://huggingface.co/datasets/universal-dependencies/universal_dependencies

## D. 8 Bahasa Terbesar Dunia *(Ethnologue 2026 — L1 + L2)*
1. **Inggris:** https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu | https://huggingface.co/datasets/HuggingFaceTB/dclm-edu | https://huggingface.co/datasets/Zyphra/Zyda-2
2. **Mandarin:** https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1 | https://huggingface.co/datasets/BAAI/IndustryCorpus2 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
3. **Hindi:** https://huggingface.co/datasets/ai4bharat/sangraha | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
4. **Spanyol:** https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned | https://huggingface.co/datasets/uonlp/CulturaX
5. **Arab Standar:** https://huggingface.co/datasets/ClusterlabAi/101_billion_arabic_words_dataset | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | https://huggingface.co/datasets/arbml/ashaar
6. **Prancis:** https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | https://huggingface.co/datasets/PleIAs/French-PD-Books | https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned
7. **Bengali:** https://huggingface.co/datasets/ai4bharat/sangraha | https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | https://huggingface.co/datasets/uonlp/CulturaX
8. **Portugis:** https://huggingface.co/datasets/HuggingFaceFW/fineweb-2 | https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned | https://huggingface.co/datasets/uonlp/CulturaX
