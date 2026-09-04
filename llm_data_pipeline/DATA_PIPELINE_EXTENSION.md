# llm_data_pipeline/ — Extension Plan for 1B Multimodal

> Extension: Vision + Voice + Updated LLM Text (100B tokens)
> Date: 2026-07-18
> Target: Cover all data preparation for transformer_1b_multimodal

---

## 1. NEW DIRECTORY STRUCTURE

```
llm_data_pipeline/
│
├── (=== EXISTING MODULES — LLM TEXT PRETRAINING ===)
│   ├── pipeline_runner.py          # UPDATE: dukung mode multimodal
│   ├── configs/
│   │   └── pretrain.yaml           # UPDATE: budget 77.8B → 100B
│   ├── acquisition/                # (existing)
│   ├── extraction/                 # (existing)
│   ├── language_id/               # (existing)
│   ├── cleaning/                  # (existing)
│   ├── quality_scoring/           # (existing)
│   ├── safety/                    # (existing)
│   ├── deduplication/             # (existing)
│   ├── contamination/             # (existing)
│   ├── filtering/                 # (existing)
│   ├── mixing/                    # UPDATE: budget 100B, 4 bahasa
│   ├── tokenization/              # (existing)
│   ├── binary_dataset/            # (existing)
│   ├── provenance/                # (existing)
│   ├── monitoring/                # (existing)
│   └── inspection/                # (existing)
│
├── (=== NEW: VISION DATA PIPELINE ===)
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── siglip_connector.py    # Download LAION, COCO, Flickr
│   │   ├── detection_connector.py # Download COCO det, Objects365
│   │   ├── ocr_connector.py       # Download OCR datasets
│   │   └── synthtext_cjk.py       # Generate synthetic CJK OCR data
│   ├── configs/
│   │   ├── vision_siglip.yaml
│   │   ├── vision_detection.yaml
│   │   └── vision_ocr.yaml
│
├── (=== NEW: VOICE DATA PIPELINE ===)
│   ├── voice/
│   │   ├── __init__.py
│   │   ├── ssl_connector.py       # Download SSL audio data
│   │   ├── asr_connector.py       # Download ASR datasets
│   │   ├── speaker_connector.py   # Download speaker datasets
│   │   └── youtube_scraper.py     # Scrape YouTube auto-captions
│   ├── configs/
│   │   ├── voice_ssl.yaml
│   │   ├── voice_asr.yaml
│   │   └── voice_speaker.yaml
│
├── (=== NEW: SFT/DPO DATA PIPELINE ===)
│   ├── sft/
│   │   ├── __init__.py
│   │   └── sft_builder.py         # Build 800K SFT examples
│   ├── dpo/
│   │   ├── __init__.py
│   │   └── dpo_builder.py         # Build 200K DPO pairs
│   └── configs/
│       ├── sft.yaml
│       └── dpo.yaml
│
├── (=== NEW: CROSS-CUTTING ===)
│   ├── data_registry.py           # Centralized data tracking
│   ├── multimodal_runner.py       # Orchestrator for all pipelines
│   └── DATA_PIPELINE_EXTENSION.md # This document
```

---

## 2. DATA STORAGE STRUCTURE

Semua data masuk ke folder terpusat `data/` seperti ini:

```
data/
├── raw/                            # ═══ RAW DOWNLOADS ═══
│   ├── llm/                         # LLM text pretraining
│   │   ├── en/                      # English raw texts
│   │   ├── id/                      # Indonesian raw texts
│   │   ├── ja/                      # Japanese raw texts
│   │   ├── ko/                      # Korean raw texts
│   │   └── math/                    # Math datasets
│   │
│   ├── vision/                      # Vision raw data
│   │   ├── siglip/                  # Image-text pairs
│   │   │   ├── laion/              # LAION-400M downloads
│   │   │   ├── coco/               # COCO Captions
│   │   │   └── flickr30k/          # Flickr30K
│   │   ├── detection/              # Detection datasets
│   │   │   ├── coco_det/           # COCO 2017 detection
│   │   │   └── objects365/         # Objects365
│   │   └── ocr/                    # OCR datasets
│   │       ├── synthtiger_output/  # Generated synthetic data
│   │       ├── coco_text/          # COCO-Text
│   │       ├── textocr/            # TextOCR
│   │       ├── icdar/              # ICDAR datasets
│   │       └── uber_text/          # Uber-Text
│   │
│   └── voice/                       # Voice raw data
│       ├── en/                      # English audio
│       ├── ja/                      # Japanese audio
│       ├── ko/                      # Korean audio
│       └── id/                      # Indonesian audio
│
├── processed/                       # ═══ PROCESSED / CLEANED ═══
│   ├── llm/                         # LLM: cleaned JSONL
│   ├── vision/                      # Vision: processed tensors
│   │   ├── siglip/                  # image, caption pairs
│   │   ├── detection/               # image, bbox, class pairs
│   │   └── ocr/                     # image, bbox, text tuples
│   └── voice/                       # Voice: processed audio
│       ├── ssl/                     # mel spectrograms
│       ├── asr/                     # audio, transcript pairs
│       └── speaker/                 # audio, speaker_id pairs
│
└── binary/                          # ═══ BINARY TRAINING FILES ═══
    ├── llm/                         # Tokenized binary shards
    │   ├── en/                      # train_en_0001.bin ... train_en_XXXX.bin
    │   ├── id/                      # train_id_0001.bin ...
    │   ├── ja/                      # train_ja_0001.bin ...
    │   ├── ko/                      # train_ko_0001.bin ...
    │   └── math/                    # train_math_0001.bin ...
    ├── vision/                      # Vision binary datasets
    └── voice/                       # Voice binary datasets
```

---

## 3. configs/pretrain.yaml — UPDATES NEEDED

File existing perlu diupdate untuk 100B token budget:

```yaml
# ─── Pipeline Globals ───
pipeline:
  name: "pretrain_1b_multimodal"
  seed: 42
  data_root: "../data"
  output_root: "../data/processed/llm"
  binary_root: "../data/binary/llm"
  workers: 16
  batch_size: 10000

# ─── Acquisition Budget (UPDATED: 100B) ───
acquisition:
  target_total_tokens: 100000000000  # 100B (was 77.8B)
  sources:
    # English
    fineweb_edu:
      enabled: true
      token_budget: 18000000000      # 18B (was ~14B)
      languages: ["en"]
    dclm_en:
      enabled: true
      token_budget: 5000000000       # 5B
      languages: ["en"]
    wikipedia_en:
      enabled: true
      token_budget: 3000000000       # 3B (papers+wiki)
      languages: ["en"]
    scientific_en:
      enabled: true
      token_budget: 2000000000       # 2B (arXiv, PubMed OA)
      languages: ["en"]
    
    # Indonesian (MASSIVE INCREASE from ~6B → 18B)
    cc100_id:
      enabled: true
      token_budget: 5000000000       # 5B (was ~2B)
      languages: ["id"]
    oscar_id:
      enabled: true
      token_budget: 5000000000       # 5B (was ~2B)
      languages: ["id"]
    wikipedia_id:
      enabled: true
      token_budget: 2000000000       # 2B
      languages: ["id"]
    sealang_id:
      enabled: true
      token_budget: 4000000000       # 4B (news, legal, govt)
      languages: ["id"]
    
    # Japanese
    mc4_ja:
      enabled: true
      token_budget: 8000000000       # 8B
      languages: ["ja"]
    cc100_ja:
      enabled: true
      token_budget: 4000000000       # 4B
      languages: ["ja"]
    wikipedia_ja:
      enabled: true
      token_budget: 2000000000       # 2B
      languages: ["ja"]
    aozora_ja:
      enabled: true
      token_budget: 2000000000       # 2B (literary)
      languages: ["ja"]
    
    # Korean
    culturax_ko:
      enabled: true
      token_budget: 8000000000       # 8B
      languages: ["ko"]
    mc4_ko:
      enabled: true
      token_budget: 4000000000       # 4B
      languages: ["ko"]
    wikipedia_ko:
      enabled: true
      token_budget: 2000000000       # 2B
      languages: ["ko"]
    korean_books:
      enabled: true
      token_budget: 2000000000       # 2B
      languages: ["ko"]
    
    # Math (INCREASED from ~10B → 12B)
    metamath:
      enabled: true
      token_budget: 4000000000       # 4B
    gsm8k_math:
      enabled: true
      token_budget: 3000000000       # 3B
    orca_math:
      enabled: true
      token_budget: 3000000000       # 3B
    code_math:
      enabled: true
      token_budget: 2000000000       # 2B interleaved
    
    # Code generation: DISABLED (removed from plan)
    the_stack_v2:
      enabled: false
    stackoverflow:
      enabled: false

# ─── Mixing Proportions (UPDATED) ───
mixing:
  domain_proportions:
    web_text: 0.55
    math: 0.12
    wiki: 0.08
    science: 0.05
    books: 0.05
    news: 0.03
    educational: 0.07
    legal_medical: 0.05
  language_proportions:
    en: 0.28
    id: 0.18
    ja: 0.18
    ko: 0.18
    math: 0.12
  curriculum_phases:
    - {pct: 0.35, seq_len: 256}
    - {pct: 0.35, seq_len: 512}
    - {pct: 0.30, seq_len: 1024}
```

---

## 4. NEW MODULE: vision/ — Vision Data Pipeline

### 4a. vision/siglip_connector.py

```
PURPOSE: Download and prepare image-text pair datasets for SigLIP training.

SOURCES:
  1. LAION-400M subset (45M pairs)
     - Download from HuggingFace: laion/laion400m
     - Filter: aesthetic_score > 4.5, safety=sfw, resolution > 256px
     - Output: {image_path, caption, aesthetic_score, resolution}
  
  2. COCO Captions (600K)
     - Download from: https://cocodataset.org
     - Format: JSON annotations + JPEG images
     - 5 captions per image, take all
  
  3. Flickr30K (30K)
     - Download from: https://github.com/BryanPlummer/flickr30k_entities
     - Format: TSV + JPEG
  
  4. TextCaps (140K)
     - Download from: HuggingFace datasets (lmms-lab/TextCaps)
     - Images with OCR-aware captions
  
  5. Multilingual COCO (4M)
     - Download COCO captions in EN
     - Translate 600K captions to ID via Google Translate API
     - Translate 600K captions to JA via Google Translate API / DeepL
     - Output: {image, caption_en, caption_id, caption_ja}

USAGE:
  python llm_data_pipeline/vision/siglip_connector.py \
    --config llm_data_pipeline/configs/vision_siglip.yaml \
    --output ../data/processed/vision/siglip/
```

### 4b. vision/detection_connector.py

```
PURPOSE: Download detection datasets with bbox annotations.

SOURCES:
  1. COCO 2017 Detection (118K images, 80 classes)
     - Download: cocodataset.org/#download
     - Files: train2017.zip, val2017.zip, annotations_trainval2017.zip
     - Format: {image, bboxes: [{x,y,w,h,class_id}], image_id}
  
  2. Objects365 (1.7M images, 365 classes)
     - Download: objects365.org
     - Filter: exclude objects < 16×16 pixels
     - Format: {image, bboxes: [{x,y,w,h,class_id}]}

OUTPUT FORMAT (per image):
  {
    "image_path": "coco/train2017/000000391895.jpg",
    "image_size": [640, 480],
    "bboxes": [
      {"cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.4, "class_id": 1, "class_name": "person"},
      {"cx": 0.2, "cy": 0.7, "w": 0.1, "h": 0.1, "class_id": 3, "class_name": "car"}
    ]
  }

USAGE:
  python llm_data_pipeline/vision/detection_connector.py \
    --config llm_data_pipeline/configs/vision_detection.yaml \
    --output ../data/processed/vision/detection/
```

### 4c. vision/ocr_connector.py

```
PURPOSE: Download real-world OCR datasets.

SOURCES:
  1. COCO-Text (63K)
     - Download from: https://vision.cornell.edu/se3/coco-text-2/
     - Format: COCO-style JSON + images
  
  2. TextOCR (28K)
     - Download: HuggingFace (textvqa/textocr)
     - Format: {image, text, bbox}
  
  3. ICDAR (30K)
     - Download: rrc.cvc.uab.es (ICDAR 2015, 2017, 2019, 2023)
     - Format: per-competition, various
  
  4. Uber-Text (100K)
     - Download: https://s3.amazonaws.com/uber-text/
     - Format: street-level text images
  
  5. MLT-2017/2019 (Multi-Lingual Text)
     - Download: rrc.cvc.uab.es
     - 9 scripts including Latin, Japanese, Korean
     - Filter subsets per language
  
  6. AI Hub Korean OCR (50K)
     - Download: aihub.or.kr (requires virtual Korean number)
     - Format: image + UTF-8 text annotations
  
  7. ID-Specific OCR (20K)
     - Self-collected: Indonesian street signs, restaurant menus, KTP cards
     - Method: manual photography + annotation
     - Target: 5000 images, 4 photos each
  
  OUTPUT FORMAT (per region):
  {
    "image_path": "coco_text/...",
    "regions": [
      {
        "bbox": {"x": 100, "y": 50, "w": 200, "h": 30},
        "text": "Jalan Malioboro No. 123",
        "lang": "id",
        "script": "latin",
        "orientation": "horizontal"
      }
    ]
  }

USAGE:
  python llm_data_pipeline/vision/ocr_connector.py \
    --config llm_data_pipeline/configs/vision_ocr.yaml \
    --output ../data/processed/vision/ocr/
```

### 4d. vision/synthtext_cjk.py (SYNTHETIC GENERATOR — 7 HARI ENGINEERING)

```
PURPOSE: Generate synthetic OCR training data for Japanese and Korean.

JAPANESE GENERATOR:
  Font families (5):
    - Gothic (Noto Sans JP) — modern, clean
    - Mincho (Noto Serif JP) — traditional, newspapers
    - Kyokasho (UD Digi Kyokasho) — textbook style
    - Maru (Rounded M+) — informal, signs
    - Handwriting (Klee One) — semi-cursive
  
  Features:
    - Horizontal text: 70% of data
    - Vertical text (tategaki): 30% of data
    - Furigana: 20% with ruby above kanji
    - Mixed-script: kanji + hiragana + katakana + Latin + numbers
    - Character spacing variation
  
  Text sources:
    - Japanese Wikipedia paragraphs
    - Aozora Bunko (public domain literature)
    - News headlines
    - Common phrases corpus
  
  Backgrounds:
    - Solid colors (30%)
    - Natural images from COCO (40%)
    - Document textures (20%)
    - Noise/gradient (10%)

KOREAN GENERATOR:
  Font families (4):
    - Gothic (Noto Sans KR)
    - Myeongjo (Noto Serif KR)
    - Handwriting (Hi Melody)
    - Display (Black Han Sans)
  
  Features:
    - Syllable blocks: 60%
    - Jamo decomposition: 40% (초성/중성/종성 separated)
    - Horizontal text only (Korean doesn't use vertical)
    - Mixed Hangul + Latin + numbers
  
  Text sources:
    - Korean Wikipedia paragraphs
    - News articles
    - Common phrases corpus

LATIN GENERATOR (EN + ID):
  Font families (5):
    - Sans Serif (Roboto)
    - Serif (Noto Serif)
    - Monospace (Noto Mono)
    - Display (Oswald)
    - Handwriting (Dancing Script)
  
  Text sources:
    - English: COCO captions, news
    - Indonesian: Wikipedia-ID, news headlines
    - 200K special: Indonesian-specific vocabulary (street names, food, places)

AUGMENTATIONS (on-the-fly during generation):
  - Rotation: ±15°
  - Perspective warp
  - Brightness: ±30%
  - Contrast: ±20%
  - Gaussian blur
  - Motion blur
  - JPEG compression artifacts
  - Random erasing (CutOut, 20% area)
  - Drop shadow rendering
  - Curved text (warp to cylinder/arc)
  - Background composition

RENDERING ENGINE:
  Library: Pillow + Cairo (for CJK text shaping)
  Size: 1920×1080 base canvas, crop RoI 300×80
  Format: JPEG quality 85

OUTPUT STATISTICS:
  ├── latin_en/       500K images  (5 fonts × 100K)
  ├── latin_id/       500K images  (5 fonts × 80K + ID-specific 100K)
  ├── japanese/       800K images  (5 fonts × 160K, 30% vertical)
  └── korean/         600K images  (4 fonts × 150K)
  TOTAL:             2.4M images  (~200 GB disk)

USAGE:
  python llm_data_pipeline/vision/synthtext_cjk.py \
    --config llm_data_pipeline/configs/vision_ocr.yaml \
    --output ../data/raw/vision/ocr/synthtiger_output/ \
    --workers 8 \
    --total 2400000
```

---

## 5. NEW MODULE: voice/ — Voice Data Pipeline

### 5a. voice/ssl_connector.py

```
PURPOSE: Download audio data for unsupervised SSL pretraining.

SOURCES:
  English (15K jam):
    LibriSpeech 960h + "light" 600h (openslr.org/12)
    GigaSpeech subset (10K jam, github.com/SpeechColab/GigaSpeech)
    Common Voice EN 17.0 (2K jam, commonvoice.mozilla.org)
    People's Speech (1K jam, mlcommons.org)
  
  Japanese (15K jam):
    ReazonSpeech (reazonspeech, HuggingFace)
    YouTube auto-captions JA — see youtube_scraper.py
    CSJ Corpus (NINJAL — need research application)
    CEJC (NINJAL)
  
  Korean (12K jam):
    YouTube auto-captions KO
    AI Hub Korea audio collections
  
  Indonesian (8K jam):
    YouTube auto-captions ID
    Podcast Indonesia
    RRI radio archives
  
  NOTE: SSL doesn't need transcripts. Audio only.
  Processing:
    Resample all to 16kHz mono
    Extract mel spectrogram (80 bins)
    Save as .npy arrays for fast CUDA loading

USAGE:
  python llm_data_pipeline/voice/ssl_connector.py \
    --config llm_data_pipeline/configs/voice_ssl.yaml \
    --output ../data/processed/voice/ssl/
```

### 5b. voice/asr_connector.py

```
PURPOSE: Download supervised ASR data (audio + transcripts).

OUTPUT FORMAT (per utterance):
  {
    "audio_path": "data/raw/voice/en/librispeech/...",
    "transcript": "the quick brown fox jumps over the lazy dog",
    "duration_sec": 3.5,
    "language": "en",
    "speaker_id": "spk_001",
    "source": "librispeech"
  }

SOURCES BY LANGUAGE — SEE LLM_IMPROVEMENT.md Section E.2

USAGE:
  python llm_data_pipeline/voice/asr_connector.py \
    --config llm_data_pipeline/configs/voice_asr.yaml \
    --output ../data/processed/voice/asr/
```

### 5c. voice/speaker_connector.py

```
PURPOSE: Download speaker identification datasets.

SOURCES:
  English (5,000 speakers):
    VoxCeleb1 + VoxCeleb2 (HuggingFace, ~500 GB)
  
  Japanese (2,000 speakers):
    JVS (100 speakers, sites.google.com/site/shinnosuketakai/research/jvs)
    YouTube auto-captions (extract speaker labels from video metadata)
    Self-collected JA speakers (crowdsource)
  
  Korean (1,500 speakers):
    AI Hub Korea speaker datasets
    KsponSpeech (speaker-labeled utterances)
    Self-collected KO speakers
  
  Indonesian (1,500 speakers):
    Common Voice ID (crowdsourced, 150+ speakers)
    YouTube ID channels (extract speaker labels)
    Self-collected: Toloka.ai campaign ($200, 1000 speakers × 50 utterances)

OUTPUT FORMAT:
  {
    "audio_segment_path": "data/raw/voice/ja/jvs/...",
    "speaker_id": "jvs_001",
    "duration_sec": 2.0,
    "language": "ja",
    "speaker_gender": "female"
  }

USAGE:
  python llm_data_pipeline/voice/speaker_connector.py \
    --config llm_data_pipeline/configs/voice_speaker.yaml \
    --output ../data/processed/voice/speaker/
```

### 5d. voice/youtube_scraper.py

```
PURPOSE: Scrape YouTube auto-captioned videos for ASR data.

LANGUAGES & QUERIES:
  Japanese (target 6,000 jam):
    Queries: "ニュース" "解説" "講義" "インタビュー" "ポッドキャスト"
            "ドキュメンタリー" "会議" "プレゼンテーション"
    Filter: duration 30s-300s, language=ja, caption confidence > 0.8
  
  Korean (target 4,000 jam):
    Queries: "뉴스" "강의" "인터뷰" "팟캐스트" "설명" "다큐멘터리"
    Filter: duration 30s-300s, language=ko, caption confidence > 0.8
  
  Indonesian (target 2,000 jam):
    Queries: "podcast indonesia" "berita" "kuliah" "ceramah" "diskusi"
            "wawancara" "review" "tutorial"
    Filter: duration 30s-300s, language=id, caption confidence > 0.75

TOOLS:
  yt-dlp (github.com/yt-dlp/yt-dlp)
  ffmpeg for audio extraction

WORKFLOW:
  1. Search YouTube for each query
  2. Filter videos with auto-captions
  3. Download audio only (--extract-audio --audio-format wav --audio-quality 0)
  4. Download auto-captions (--write-auto-subs --sub-lang {lang})
  5. Parse captions: segment by timestamp
  6. Align audio segments to caption timestamps
  7. Filter by caption confidence
  8. Resample to 16kHz mono
  9. Validate: remove segments with music-only or silence

ESTIMATED RUNTIME:
  12,000 jam video × 10× realtime download = ~5 hari (8 workers)

PREREQUISITES:
  pip install yt-dlp
  apt install ffmpeg

USAGE:
  python llm_data_pipeline/voice/youtube_scraper.py \
    --languages ja ko id \
    --output ../data/raw/voice/ \
    --workers 8 \
    --max-hours 12000
```

---

## 6. NEW MODULE: sft/ & dpo/ — SFT/DPO Data Builders

### 6a. sft/sft_builder.py

```
PURPOSE: Build 800K SFT examples from various sources.

CATEGORIES & SOURCES:
  1. Chat multi-turn (180K):
     - UltraChat (HuggingFace: stonedy/ultrachat_200k) — subset EN 90K
     - ID chat: generated via API (GPT-4 translation of EN chat)
     - JA chat: Carps-ja (Japanese chat dataset)
     - KO chat: Self-collected + translated

  2. Translation (120K):
     - OPUS parallel corpora (EN↔ID, EN↔JA, EN↔KO)
     - Format: "Translate to Japanese: {EN text}" → "{JA text}"

  3. Reasoning CoT (130K):
     - Flan/T0 datasets, subset for reasoning
     - Translated to ID/JA/KO via API

  4. Math (80K):
     - MetaMathQA, GSM8K, MATH
     - Step-by-step solutions formatted as CoT

  5. Summarization (50K):
     - CNN/DailyMail (EN), MLSUM (ID), multilingual XSum (JA/KO)

  6. Tool Calling (60K):
     - Gorilla (API-Bench), ToolBench
     - Format: <|tool_call|>{"name":"func","args":{}}<|tool_end|>

  7. Vision QA (66K):
     - LLaVA-665K subset
     - COCO captions + QA pairs

  8-11. Code-Switching, Safety, Persona, Instruction (24K+30K+30K+30K)

OUTPUT FORMAT:
  {
    "messages": [
      {"role": "system", "content": "Kamu adalah asisten..."},
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ],
    "category": "chat",
    "language": "id",
    "source": "ultrachat_translated",
    "difficulty": "medium"
  }

USAGE:
  python llm_data_pipeline/sft/sft_builder.py \
    --config llm_data_pipeline/configs/sft.yaml \
    --output ../data/processed/sft/
```

### 6b. dpo/dpo_builder.py

```
PURPOSE: Build 200K DPO preference pairs.

CATEGORIES:
  Chat quality (80K): helpful > unhelpful
  Safety (40K): refusal > compliance for unsafe prompts
  Math accuracy (30K): correct > wrong
  Summarization (20K): concise+accurate > verbose+inaccurate
  Translation (30K): human-accurate > machine translation

PAIR GENERATION METHODS:
  1. Human-annotated: UltraFeedback, HH-RLHF
  2. LLM-judged: generate 2 responses, GPT-4 ranks them
  3. Rule-based: length penalty for summarization, safety classifier
  4. Contrastive: SFT model output vs base model output

OUTPUT FORMAT:
  {
    "prompt": "Jelaskan cara membuat nasi goreng.",
    "chosen": "Berikut adalah langkah-langkah membuat nasi goreng:\n1. ...",
    "rejected": "nasi goreng enak",
    "category": "chat_quality",
    "language": "id"
  }

USAGE:
  python llm_data_pipeline/dpo/dpo_builder.py \
    --config llm_data_pipeline/configs/dpo.yaml \
    --output ../data/processed/dpo/
```

---

## 7. NEW: data_registry.py — Centralized Data Tracking

```
PURPOSE: Track all datasets across LLM, Vision, and Voice pipelines.

SCHEMA (SQLite):
  CREATE TABLE datasets (
    id INTEGER PRIMARY KEY,
    name TEXT,           -- "librispeech_960h"
    modality TEXT,       -- "llm" / "vision" / "voice"
    sub_modality TEXT,   -- "pretrain" / "siglip" / "asr" / "speaker"
    source_url TEXT,     -- download URL
    source_type TEXT,    -- "huggingface" / "youtube" / "manual" / "generated"
    language TEXT,       -- "en" / "id" / "ja" / "ko" / "mixed"
    raw_size_bytes INTEGER,
    raw_file_count INTEGER,
    download_status TEXT, -- "pending" / "downloading" / "complete" / "failed"
    processed_size_bytes INTEGER,
    token_count INTEGER,  -- for LLM text
    hour_count INTEGER,   -- for voice data
    image_count INTEGER,  -- for vision data
    checksum_sha256 TEXT,
    download_date TEXT,
    processing_date TEXT,
    notes TEXT
  );

USAGE:
  python llm_data_pipeline/data_registry.py --list
  python llm_data_pipeline/data_registry.py --verify
  python llm_data_pipeline/data_registry.py --status
```

---

## 8. NEW: multimodal_runner.py — Unified Orchestrator

```
PURPOSE: Run all pipelines (LLM + Vision + Voice + SFT + DPO).

MODES:
  --mode llm          → text pretraining pipeline only
  --mode vision       → siglip + detection + ocr pipelines
  --mode voice        → ssl + asr + speaker pipelines
  --mode sft          → SFT data builder
  --mode dpo          → DPO pair builder
  --mode all          → run everything sequentially
  --mode download     → download all data before any processing

DOWNLOAD PHASE (runs first):
  python multimodal_runner.py --mode download
    # Downloads ALL data to data/raw/
    # Uses data_registry.py to track progress
    # Resumable: skips already-downloaded datasets

PROCESSING PHASE (runs after download):
  python multimodal_runner.py --mode llm --config configs/pretrain.yaml
  python multimodal_runner.py --mode vision
  python multimodal_runner.py --mode voice
  python multimodal_runner.py --mode sft
  python multimodal_runner.py --mode dpo

FULL RUN:
  # Step 1: Download everything
  python multimodal_runner.py --mode download

  # Step 2: Process LLM text
  python multimodal_runner.py --mode llm

  # Step 3: Process Vision (can parallel with Voice)
  python multimodal_runner.py --mode vision &

  # Step 4: Process Voice (can parallel with Vision)
  python multimodal_runner.py --mode voice &

  # Step 5: After LLM is done, build SFT+DPO
  python multimodal_runner.py --mode sft
  python multimodal_runner.py --mode dpo
```

---

## 9. CONFIG FILES — ALL NEW

### configs/vision_siglip.yaml
```yaml
pipeline:
  name: "vision_siglip"
  data_root: "../data"
  output_root: "../data/processed/vision/siglip"
  image_size: [448, 448]
  augment: true

sources:
  laion:
    enabled: true
    target_pairs: 45000000
    aesthetic_threshold: 4.5
    safety_filter: true
    min_resolution: 256
    languages: ["en"]
  
  coco_captions:
    enabled: true
    target_pairs: 600000
  
  flickr30k:
    enabled: true
    target_pairs: 30000
  
  textcaps:
    enabled: true
    target_pairs: 140000
  
  multilingual_coco:
    enabled: true
    target_pairs: 4000000
    translate_to: ["id", "ja"]
```

### configs/vision_ocr.yaml
```yaml
pipeline:
  name: "vision_ocr"
  data_root: "../data"
  output_root: "../data/processed/vision/ocr"
  synth_output: "../data/raw/vision/ocr/synthtiger_output/"

synthetic:
  latin_en:
    enabled: true
    target_images: 500000
    fonts: ["Roboto", "Noto Serif", "Noto Mono", "Oswald", "Dancing Script"]
    text_source: "coco_captions"
  
  latin_id:
    enabled: true
    target_images: 500000
    fonts: ["Roboto", "Noto Serif", "Noto Mono", "Oswald", "Dancing Script"]
    text_source: "id_wikipedia"
    id_specific_pct: 0.20
  
  japanese:
    enabled: true
    target_images: 800000
    fonts: ["Noto Sans JP", "Noto Serif JP", "UD Digi Kyokasho", "Rounded M+", "Klee One"]
    text_source: "ja_wikipedia"
    vertical_pct: 0.30
    furigana_pct: 0.20
  
  korean:
    enabled: true
    target_images: 600000
    fonts: ["Noto Sans KR", "Noto Serif KR", "Hi Melody", "Black Han Sans"]
    text_source: "ko_wikipedia"
    jamo_decompose_pct: 0.40

  augmentations:
    rotation_range: [-15, 15]
    perspective_warp: true
    brightness_range: [-0.3, 0.3]
    contrast_range: [-0.2, 0.2]
    gaussian_blur: [0.5, 3.0]
    jpeg_quality: [60, 95]
    random_erasing_pct: 0.20
    curved_text: true
    drop_shadow: true

real_data:
  coco_text: {enabled: true}
  textocr: {enabled: true}
  icdar: {enabled: true, years: [2015, 2017, 2019, 2023]}
  uber_text: {enabled: true}
  mlt: {enabled: true, years: [2017, 2019], languages: ["en", "ja", "ko"]}
  ai_hub_korean: {enabled: true}
  id_specific: {enabled: true, target_images: 20000}
```

### configs/voice_asr.yaml
```yaml
pipeline:
  name: "voice_asr"
  data_root: "../data"
  output_root: "../data/processed/voice/asr"
  sample_rate: 16000
  channels: 1
  normalize: true

sources:
  english:
    librispeech_960h: {enabled: true}
    librispeech_light_600h: {enabled: true}
    gigaspeech_l: {enabled: true, target_hours: 10000}
    common_voice_en: {enabled: true, target_hours: 2000}
    tedlium_v3: {enabled: true}
    peoples_speech: {enabled: true, target_hours: 1000}
  
  japanese:
    reazonspeech: {enabled: true, target_hours: 20000}
    youtube_ja: {enabled: true, target_hours: 6000}
    csj: {enabled: true, application_required: true}
    cejc: {enabled: true, application_required: true}
    common_voice_ja: {enabled: true}
    jsut: {enabled: true}
    jvs: {enabled: true, with_transcript: true}
  
  korean:
    ai_hub_dialog: {enabled: true, target_hours: 2000}
    ai_hub_free_conv: {enabled: true, target_hours: 2000}
    ai_hub_educational: {enabled: true, target_hours: 1000}
    ai_hub_lecture: {enabled: true, target_hours: 1000}
    ai_hub_news: {enabled: true, target_hours: 1000}
    ksponspeech: {enabled: true}
    youtube_ko: {enabled: true, target_hours: 4000}
    common_voice_ko: {enabled: true}
    zeroth_korean: {enabled: true}
  
  indonesian:
    common_voice_id: {enabled: true}
    youtube_id: {enabled: true, target_hours: 2000}
    podcast_id: {enabled: true, target_hours: 1500}
    dpr_ri: {enabled: true, target_hours: 800}
    rri: {enabled: true, target_hours: 500}
    tedx_id: {enabled: true, target_hours: 200}
    crowdsource_id: {enabled: true, target_hours: 800, budget_usd: 200}
```

---

## 10. EXECUTION ORDER — DATA PIPELINE

```
PHASE │ AKTIVITAS                          │ WAKTU    │ OUTPUT
──────┼────────────────────────────────────┼──────────┼────────────────────────
 D1   │ DOWNLOAD PHASE (PARALEL)            │          │
      │ ├─ LLM text: HuggingFace sources   │ 2-3 hari │ data/raw/llm/
      │ ├─ Vision: LAION+COCO+OCR datasets │ 2-3 hari │ data/raw/vision/
      │ └─ Voice: ASR+SSL+Speaker          │ 3-4 hari │ data/raw/voice/
──────┼────────────────────────────────────┼──────────┼────────────────────────
 D2   │ SYNTHETIC GENERATION                │          │
      │ ├─ SynthText CJK mod (engineering) │ 7 hari   │ synthtext_cjk.py
      │ └─ Generate 2.4M synthetic images  │ 2-3 hari │ data/raw/vision/ocr/
──────┼────────────────────────────────────┼──────────┼────────────────────────
 D3   │ YouTube Scraping (PARALEL)          │          │
      │ ├─ JA: 6,000 jam                   │ 2-3 hari │ data/raw/voice/ja/
      │ ├─ KO: 4,000 jam                   │ 1-2 hari │ data/raw/voice/ko/
      │ └─ ID: 2,000 jam                   │ 1-2 hari │ data/raw/voice/id/
──────┼────────────────────────────────────┼──────────┼────────────────────────
 P1   │ LLM TEXT PIPELINE (14 stages)       │          │
      │ └─ pipeline_runner.py --mode llm   │ 4-5 hari │ data/binary/llm/
──────┼────────────────────────────────────┼──────────┼────────────────────────
 P2   │ VISION PIPELINE (PARALEL WITH P1)   │          │
      │ ├─ SigLIP: filter + pair          │ 1-2 hari │ data/processed/vision/
      │ ├─ Detection: format bboxes       │ 1 hari   │
      │ └─ OCR: format regions+text       │ 1 hari   │
──────┼────────────────────────────────────┼──────────┼────────────────────────
 P3   │ VOICE PIPELINE (PARALEL WITH P1)    │          │
      │ ├─ ASR: align audio→transcript    │ 2-3 hari │ data/processed/voice/
      │ ├─ Speaker: label speaker IDs     │ 1 hari   │
      │ ├─ SSL: extract mel spectrograms  │ 1-2 hari │
      │ └─ Resample all to 16kHz mono     │ 1 hari   │
──────┼────────────────────────────────────┼──────────┼────────────────────────
 P4   │ SFT + DPO BUILDERS                  │          │
      │ ├─ sft_builder.py (800K examples) │ 1-2 hari │ data/processed/sft/
      │ └─ dpo_builder.py (200K pairs)    │ 1 hari   │ data/processed/dpo/
──────┴────────────────────────────────────┴──────────┴────────────────────────

TOTAL DATA PREP: ~21-32 hari (bisa diparalelkan ke ~15-18 hari)
Semua biaya data prep: ~$305 (Toloka campaign + API translation + virtual KR number)
```

---

## 11. REQUIREMENTS.TXT EXTENSION

Tambahkan ke `requirements.txt` yang sudah ada:

```
# ─── NEW: Vision data processing ───
opencv-python>=4.8.0
Pillow>=10.0.0
pycairo>=1.24.0
albumentations>=1.3.0

# ─── NEW: Voice data processing ───
librosa>=0.10.0
soundfile>=0.12.0
audiomentations>=0.34.0
pydub>=0.25.0
yt-dlp>=2024.12.0

# ─── NEW: SFT/DPO data ───
openai>=1.0.0          # API translation
deep-translator>=1.11.0
```

---

## 12. QUICK START

```bash
# 1. Install dependencies
cd llm_data_pipeline
pip install -r requirements.txt

# 2. Download ALL data (runs in background, resumable)
python multimodal_runner.py --mode download
# Status check:
python data_registry.py --status

# 3. Generate synthetic OCR data (while downloads run)
python vision/synthtext_cjk.py --output ../data/raw/vision/ocr/synthtiger_output/

# 4. When downloads complete, process LLM text
python multimodal_runner.py --mode llm

# 5. In parallel: process Vision + Voice
python multimodal_runner.py --mode vision &
python multimodal_runner.py --mode voice &

# 6. Build SFT + DPO
python multimodal_runner.py --mode sft
python multimodal_runner.py --mode dpo

# 7. Verify all datasets ready
python data_registry.py --verify
```
