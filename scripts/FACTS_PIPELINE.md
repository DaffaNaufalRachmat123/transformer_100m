# Fact-Fixing SFT Pipeline (EN ⇄ JA facts)

Fixes the "language confusion" problem: the model was answering English
questions in Japanese and Japanese questions in English (because the dataset
was ~99.9% translation and almost no clean Q&A facts). This pipeline adds a
small, high-quality, **balanced EN/JA fact Q&A** SFT stage on top of the
existing translation model.

## Why facts were weak

`sft_translation_focused/train.jsonl` had **520,745 rows but only 152 fact
Q&A** (76 EN + 76 JA) = **0.03%**. The model learned translation shape almost
exclusively, and rarely saw clean fact answers → it "translates" the question
instead of answering it.

## Data

`scripts/build_facts_dataset.py` produces `sft_data/sft_facts_focused/train.jsonl`:

- **Curated gold** (`sft_facts_curated.jsonl`): 76 EN + 76 JA hand-verified facts.
- **Strict-filtered JA** (`sft_ja.jsonl`): only OBJECTIVE facts (capitals,
  units, chemistry, planets, rivers, mountains, population) with clean
  `〜です。` answers. Subjective/context/reading-comprehension rejected.
- **Strict-filtered EN** (`sft_en_facts.jsonl`): only clean `What/How/Which`
  questions with proper sentence answers. Task-style templates
  ("remove the words", "true or false", "fill in the blank", ...) rejected.
- **Balanced**: equal EN and JA so the model learns to answer in the language
  of the question.

Result: **~170 rows** (85 EN + 84 JA), validated: 108/108 EN in English,
all JA answers correct-language (bilingual facts like `CO2`/GBP are legit).

## Usage

```bash
# 1. Build the fact dataset
python3 scripts/build_facts_dataset.py

# 2. Two-stage SFT: facts first, then translation to preserve translation.
#    Optional env: FACTS_BASE_MODEL, FACTS_STEPS (2000), TRANSL_STEPS (15000)
bash scripts/run_facts_sft.sh
```

## Stage breakdown (3-stage, one folder per language)

| Stage | Dataset | Base | LR | Max steps | Checkpoint dir |
|---|---|---|---|---|---|
| 1 (JA facts) | `sft_facts_ja` | `ckpt_pretrain_final` | 1e-5 | 15000 | `checkpoints_facts_ja` |
| 2 (EN facts) | `sft_facts_en` | Stage 1 JA model | 1e-5 | 15000 | `checkpoints_facts_en` |
| 3 (translation) | `sft_translation_focused` | Stage 2 EN model | 3e-5 | 15000 | `checkpoints_transl` |

Rationale: the 892M model overfits JA (short, predictable answers) and underfits
EN. Training language-by-language in separate folders lets each stage focus on
one language, then translation is re-added last so it is not lost.

Each stage uses its OWN checkpoint dir so the pipeline's "stage already done"
skip (`--skip_done`, default on) never fires. A stage auto-advances when its
`ckpt_sft_final.safetensors` already exists (no state to resume); otherwise it
resumes from the last checkpoint. After Stage 3, the final model is copied to
`checkpoints/` and `translation_resources/`.

Datasets are built with mode flags:
```bash
python3 scripts/build_facts_dataset.py --mode ja_only       # -> sft_facts_ja
python3 scripts/build_facts_dataset.py --mode en_dominant   # -> sft_facts_en
python3 scripts/build_facts_dataset.py --mode all           # -> sft_facts_large
```

### Google Colab persistence

The stage dirs are **not** symlinked to Drive by default. Add symlinks in
Cell 4 (see `GOOGLE_COLLAB_FULL_TRAINING.md` Cell 4) for:
  `checkpoints_facts_ja` → `MyDrive/checkpoints_facts_ja_100m`
  `checkpoints_facts_en` → `MyDrive/checkpoints_facts_en_100m`
  `checkpoints_transl` → `MyDrive/checkpoints_transl_100m`
  `translation_resources` → `MyDrive/translation_resources_100m`

The script is symlink-aware: it clears the *contents* of a symlinked stage dir
without deleting the symlink itself, so Drive persistence is preserved.

## Evaluate

```bash
# Fact answering (language correctness + content accuracy)
python3 scripts/eval_facts.py \
    --model checkpoints_facts_en/ckpt_sft_final.safetensors --max_test 200

# Objective translation metric (novel sentences)
python3 scripts/eval_translation.py \
    --model checkpoints/ckpt_sft_final.safetensors \
    --eval sft_data/sft_clean/sft_translation_eval.jsonl \
    --max_test 200 --max_tokens 30
```

## Limits

- 892M model + 2B pretrain tokens (~12% of Chinchilla) caps generalization.
- Fact coverage is bounded by Wikidata (country/geography/elements); for more
  breadth, fetch additional categories (person, planets, languages) when
  Wikidata rate-limits allow.
- The 3-stage split trades total volume per stage for per-language focus, which
  helps the small model learn correct answer-language.
