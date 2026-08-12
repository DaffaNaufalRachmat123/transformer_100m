# EN ⇄ JA Parallel Corpus & SFT Data Pipeline

Drastically expands the EN<->JA parallel data used for SFT, fixing the
root cause of the poor translation quality: **data starvation → memorization**
instead of generalization.

## Why this was needed

The old SFT translation set had only **~4,500 unique source sentences**, far too
few for a model to learn translation *patterns*. The model was memorizing
training sentences verbatim and failing on novel word combinations.

## What the pipeline produces

| Artifact | File | Size |
|---|---|---|
| Clean parallel pairs (all sources, deduped) | `data/parallel_enja.tsv` | ~1.96M pairs |
| SFT train (both directions, templated) | `sft_data/sft_clean/sft_translation_large.jsonl` | 520K rows (260K pairs) |
| Held-out eval (novel sentences) | `sft_data/sft_clean/sft_translation_eval.jsonl` | 27K rows (13.7K pairs) |
| Final SFT train + facts | `sft_data/sft_translation_focused/train.jsonl` | 520K rows |

## Sources

- **Tatoeba** full export (~272K pairs) — clean, high-quality short sentences
- **OPUS OpenSubtitles** (subtitle dialogue) — conversational, short
- **OPUS KFTT** (Kyoto Free Translation Task) — high-quality news/encyclopedia
- **OPUS wikimedia** — Wikipedia article pairs

## Usage

```bash
# 1. Download + filter + dedup the parallel corpus
python3 scripts/fetch_parallel_enja.py \
    --sources tatoeba,opensubtitles,kftt,wikimedia \
    --max_len 80 --out data/parallel_enja.tsv

# 2. Build SFT rows (both directions) + held-out eval split
python3 scripts/build_sft_translation.py \
    --in data/parallel_enja.tsv \
    --out sft_data/sft_clean/sft_translation_large.jsonl \
    --eval_out sft_data/sft_clean/sft_translation_eval.jsonl \
    --max_pairs 300000 --eval_fraction 0.05

# 3. Build final balanced SFT dataset (90% translation + 10% facts)
python3 scripts/prepare_sft_translation_dominant.py

# 4. Deep quality filter (mojibake / untranslated / placeholders)
python3 scripts/clean_translation_quality.py
```

## Quality guarantees

- **0 untranslated rows** (every JA answer has Japanese; every EN answer has Latin)
- **0 duplicate rows**
- Balanced EN->JA / JA->EN
- Median EN length ~9 words (model-friendly)
- Held-out eval set is **99.1% novel** sentences (not in train) so you can
  measure real generalization, not memorization.

## Measuring improvement

Use the held-out eval set to score translation accuracy on novel sentences:

```bash
# Generate translations for each eval prompt, then compare against the
# reference answer (sft_data/sft_clean/sft_translation_eval.jsonl).
python3 scripts/eval_translation.py \
    --model translation_resources/ckpt_sft_final.safetensors \
    --eval sft_data/sft_clean/sft_translation_eval.jsonl \
    --max_test 200 --max_tokens 30 \
    --out results/translation_eval.json
```

### Baseline (current SFT model, BEFORE retraining with expanded data)

On 200 novel held-out prompts (100 per direction):

| Direction | BLEU | chrF |
|---|---|---|
| en-ja | 0.000 | 0.000 |
| ja-en | 0.000 | 0.003 |
| **combined** | **0.000** | **0.002** |

Only 64/200 prompts produced any output at all — the rest generated empty
text. This is the memorization signature: the model fails on sentences it has
not memorized verbatim.

**Retrain SFT with the expanded data, then re-run this command to measure the
gain. The combined BLEU/chrF should rise substantially if the model is now
generalizing instead of memorizing.**

### Score reference

- BLEU = 1.0 for identical, 0.0 for completely different (validated).
- chrF2-style character n-gram F-score (good for Japanese, where word
  tokenization is unreliable).