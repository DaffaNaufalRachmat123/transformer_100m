# Phase A: Pre-Training Data Pipeline — Implementation Plan

> **Scope:** Modules 1-6 (acquisition → cleaning)  
> **Status:** In Progress  
> **Source:** DATA_PIPELINE_PLAN.md (stages A.1–A.6)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│  MODULE 1: contamination/   ← CRITICAL — benchmark safety first        │
│  MODULE 2: quality_scoring/ ← Highest single-step quality gain (DCLM)   │
│  MODULE 3: acquisition/     ← Data ingestion + provenance              │
│  MODULE 4: extraction/      ← File validation + text extraction + norm  │
│  MODULE 5: language_id/     ← FastText + CLD3 dual classifier           │
│  MODULE 6: cleaning/        ← Web artifact removal + text repair        │
└──────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
HuggingFace / Web
       │
       ▼
[acquisition/sources.py]  ──→ raw JSONL + provenance UUIDs
       │
       ▼
[extraction/validator.py] ──→ validated files, corrupted quarantined
       │
       ▼
[extraction/text_extractor.py] ──→ Trafilatura > readability > jusText > BS4
       │                              + PDF, LaTeX, Markdown, Code parsers
       ▼
[extraction/normalizer.py]  ──→ Unicode NFC, ftfy repair, whitespace
       │
       ▼
[language_id/classifier.py] ──→ {lang, confidence, is_mixed, script}
       │
       ▼
[cleaning/web_artifact_removal.py] ──→ stripped HTML/CSS/JS/banners
       │
       ▼
[cleaning/text_repair.py]   ──→ ftfy v2, emoji norm, URL/email norm
       │
       ▼
[cleaning/boilerplate_detector.py] ──→ reject if main_content_ratio < 0.3
       │
       ▼
[quality_scoring/] (4 sub-modules)
  ├── heuristic.py         ──→ perplexity, readability, entropy, compression
  ├── model_based.py       ──→ fastText quality classifier (Wikipedia vs CC)
  ├── educational.py       ──→ edu score (FineWeb-Edu approach)
  └── composite.py         ──→ weighted combination score
       │
       ▼
[contamination/] (3 sub-modules) — runs as CHECK on final corpus
  ├── benchmark_loader.py  ──→ load 11 benchmark test sets
  ├── ngram_detector.py    ──→ 13-gram overlap (GPT-4 method)
  └── contamination_report.py ──→ per-benchmark statistics
       │
       ▼
  CLEAN CORPUS → Modules 7-12 (safety → dedup → filtering → mixing → tokenize → binary)
```

## Module Index

| # | Module | Stages | Files | Priority |
|---|--------|--------|-------|----------|
| 1 | `contamination/` | A.9 | 3 | **CRITICAL** |
| 2 | `quality_scoring/` | A.6 | 4 | **CRITICAL** |
| 3 | `acquisition/` | A.1 | 2 | Foundation |
| 4 | `extraction/` | A.2–A.3 | 3 | Foundation |
| 5 | `language_id/` | A.4 | 1 | Foundation |
| 6 | `cleaning/` | A.5 | 3 | Foundation |
| 7 | `safety/` | A.7 | 4 | Pending |
| 8 | `deduplication/` | A.8 | 4 | Pending |
| 9 | `filtering/` | A.10 | 3 | Pending |
| 10 | `mixing/` | A.11 | 3 | Pending |
| 11 | `tokenization/` | A.13 | 2 | Pending |
| 12 | `binary_dataset/` | A.14 | 2 | Pending |
| — | `provenance/` | F.2 | 1 | Cross-cut |
| — | `monitoring/` | F.3 | 1 | Cross-cut |

## File Manifest (Modules 1-6)

```
llm_data_pipeline/
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
├── __init__.py
├── configs/
│   └── pretrain.yaml
├── common/
│   └── __init__.py
├── contamination/
│   ├── __init__.py
│   ├── benchmark_loader.py        # ~200 lines
│   ├── ngram_detector.py          # ~180 lines
│   └── contamination_report.py    # ~120 lines
├── quality_scoring/
│   ├── __init__.py
│   ├── heuristic.py               # ~250 lines
│   ├── model_based.py             # ~220 lines
│   ├── educational.py             # ~180 lines
│   └── composite.py               # ~100 lines
├── acquisition/
│   ├── __init__.py
│   ├── sources.py                 # ~300 lines
│   └── provenance.py              # ~150 lines
├── extraction/
│   ├── __init__.py
│   ├── validator.py               # ~200 lines
│   ├── text_extractor.py          # ~250 lines
│   └── normalizer.py              # ~180 lines
├── language_id/
│   ├── __init__.py
│   └── classifier.py              # ~220 lines
└── cleaning/
    ├── __init__.py
    ├── web_artifact_removal.py    # ~200 lines
    ├── text_repair.py             # ~180 lines
    └── boilerplate_detector.py    # ~100 lines
```

## Key Design Decisions

1. **JSONL-based pipeline**: Every stage reads/writes JSONL (one JSON object per line). This allows resumability and parallel processing.
2. **Stage isolation**: Each module is independently runnable. A stage can be re-run without re-running prior stages.
3. **Provenance from day one**: Every document gets a UUID at ingestion. Every filter decision is logged.
4. **Smoke test first**: `smoke_test.py` validates the full pipeline on 1000 documents before full-scale run.
5. **Contamination runs as check, not removal**: The contamination module scans the final corpus and quarantines overlapping documents. The pipeline can be re-run with contaminated sources excluded.

## Usage

```bash
pip install -r llm_data_pipeline/requirements.txt
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O llm_data_pipeline/language_id/lid.176.bin

# Run individual modules:
python -m llm_data_pipeline.acquisition.sources --config configs/pretrain.yaml
python -m llm_data_pipeline.extraction.validator --input raw/ --output validated/
# ... etc

# Or run full pipeline:
python llm_data_pipeline/pipeline_runner.py --config configs/pretrain.yaml
```

## Estimated Token/Data Budget

| Stage | Processing Time | Output Size |
|-------|----------------|-------------|
| A.1 Acquisition | Download-bound | 20T+ raw tokens |
| A.2 Validation | ~1 CPU-hr per TB | ~5% rejection |
| A.3 Extraction | ~5 CPU-hr per TB | 15-30% size reduction |
| A.4 Language ID | ~2 CPU-hr per TB | Keep target langs only |
| A.5 Cleaning | ~3 CPU-hr per TB | 5-10% removal |
| A.6 Quality Scoring | ~10 CPU-hr per TB | 30-50% filtered out |
| A.9 Contamination | ~2 CPU-hr per TB | <0.1% removed |
