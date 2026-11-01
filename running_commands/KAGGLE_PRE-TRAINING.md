# KAGGLE — Pre-Training Only (EN / JA / Code / Math)

> **Updated for the v8 pipeline.** What changed and why it matters:
>
> | Change | Reason |
> |---|---|
> | `vocab_6` → **`vocab_7`**, `bpe_v6.bin` → **`bpe_v7.bin`** | `vocab_6` is no longer in the repo, and it strips newlines so code is untrainable. |
> | `bpe_v7.bin` regenerated from `vocab_7/` | The old binary still had `<\|phone_event\|>` / `<\|app_usage\|>` at ids 9–10 while `vocab.txt` had `<\|think\|>` / `<\|/think\|>`, so `--bpe` and `--vocab_dir` disagreed on two token ids. |
> | `data/corpus_final` → **`data/corpus_pretrain_v8`** | Five clean buckets (`english/ japanese/ code/ math/ mixed/`), documents separated by a `<\|endoftext\|>` line, porn/gambling/SEO/mojibake/duplicates removed. `mixed/` holds the generated fact-injection corpus and 10.8M EN↔JA sentence pairs. |
> | New **Cell 4 preflight** | Catches an empty bucket, a dangling symlink, a tokenizer mismatch or a lopsided token mix *before* the GPU clock starts. |
> | `--max_seq` is now honoured | The dataset previously forced `seq_len=256` regardless of the flag, so any other value trained on zero padding for the rest of every sequence. Other values are now safe. |
> | `--max_tokens` **2B → 10B** | Chinchilla for this architecture. See below. |

## Token budget (Chinchilla)

The architecture in `transformer/model_config.h` is 62.9M tied embedding plus
24 × 17.86M blocks = **491.6M parameters**. Chinchilla puts the compute-optimal
budget at 20 tokens/parameter:

```
491.6M x 20 = 9.8B tokens      <- --max_tokens 10000000000
```

Bucket shares, and the measured bytes/token that convert them to disk:

| bucket | share | tokens | bytes/token | corpus bytes |
|---|---|---|---|---|
| japanese | 36% | 3.6B | 3.18 | 11.4 GB |
| english | 32% | 3.2B | 3.53 | 11.3 GB |
| code | 20% | 2.0B | 3.64 | 7.3 GB |
| math | 12% | 1.2B | 3.70 | 4.4 GB |

**Japanese gets the largest share deliberately.** It packs 1.16 characters per
token against English's 4.15, so the same document costs more tokens; and the
model has to learn three scripts plus ~6,200 distinct kanji, which English does
not require. In a 45.8M-character sample the top 2,136 kanji cover 99.14% of
occurrences, but 2,531 kanji need ≥100 exposures each and only a large Japanese
token count delivers that.

The corpus does not have to hold 10B unique tokens. `build_pretrain_corpus.py`
builds ~20 GB ≈ 5.8B unique tokens, so the run makes ~1.7 passes. Under about
4 epochs repeated tokens are worth nearly as much as fresh ones
(Muennighoff et al., *Scaling Data-Constrained Language Models*); above that,
fetch more data instead of raising the epoch count.

---

## Cell 1: Copy Source Code

```python
%%bash
SRC=/kaggle/input/datasets/naufalrachmat/transformerarchitecture17
DST=/kaggle/working/transformer_100m_cuda
mkdir -p $DST
cp -r $SRC/bpe         $DST/
cp -r $SRC/cuda        $DST/
cp -r $SRC/scripts     $DST/
cp -r $SRC/tools       $DST/
cp -r $SRC/transformer $DST/
cp -r $SRC/vocab_8     $DST/
cp -r $SRC/infer_cpp   $DST/
cp $SRC/gdrive_token.json $DST/
cp $SRC/bpe_v8.bin     $DST/
cp $SRC/CMakeLists.txt $DST/
cp $SRC/config.ini     $DST/
cp $SRC/config_*.json  $DST/ 2>/dev/null || true

# Resume from the latest pretrain checkpoint
mkdir -p /kaggle/working/checkpoints
cp -r /kaggle/input/datasets/naufalrachmat/ckptlatest2/* /kaggle/working/checkpoints/ 2>/dev/null || true

echo "Copied to $DST"; ls $DST
echo "--- checkpoints ---"; ls -lh /kaggle/working/checkpoints/
```

## Cell 2: Build All Binaries

```python
%%bash
cd /kaggle/working/transformer_100m_cuda
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DUSE_DDP=ON 2>&1 | tail -15
cmake --build . -j$(nproc) 2>&1 | tail -30

echo ""; ls -lh train_pretrain_cuda train_sft_cuda

# Kernel correctness harness — never skip this.
echo ""
cmake --build . --target grad_check -j$(nproc) 2>&1 | tail -5
./grad_check
```

## Cell 3: Build Infer Binary

```python
import subprocess, shutil, os

infer_build = '/kaggle/working/transformer_100m_cuda/build_infer'
shutil.rmtree(infer_build, ignore_errors=True)
os.makedirs(infer_build, exist_ok=True)

subprocess.run(
    ['cmake', '/kaggle/working/transformer_100m_cuda/infer_cpp',
     '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=75'],
    cwd=infer_build, check=True)
subprocess.run(['make', f'-j{os.cpu_count()}'], cwd=infer_build, check=True)
print("infer binary built")
```

## Cell 4: Preflight — tokenizer + corpus (STOP if this fails)

```python
%%bash
cd /kaggle/working/transformer_100m_cuda
DATA=/kaggle/input/datasets/naufalrachmat/datatraining/data/corpus_pretrain_v8

python3 scripts/preflight_pretrain.py \
  --vocab_dir vocab_8 \
  --bpe bpe_v8.bin \
  --data_dir "$DATA" \
  --max_seq 512
```

Expect every line to read `[PASS]`, five non-empty buckets, and
`bpe binary merges == merges.txt merges`. A `[FAIL]` here is always a data or
artifact problem, never a transient one — fix it before spending GPU time.

If the attached dataset still has the old six-bucket layout (`indonesian/`,
`mixed/`), you can still point `--data_dir` at it: the trainer detects the layout
and prints which sampling weights it chose. Just confirm the printed line is the
one you intended.

## Cell 5: Run Pre-Training

```python
import subprocess, os

env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)
env['NCCL_P2P_DISABLE'] = '1'
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)

# Bucket sampling weights. Take these from the tail of
# logs/build_pretrain_corpus_v8.log — the builder prints the values that hit the
# intended token mix given what each bucket actually yielded.
# Japanese leads because it is the least compressible: 1.16 characters per token
# against 4.15 for English, so equal token counts buy very unequal amounts of
# text. `mixed` is fact injection plus the EN<->JA parallel corpus — it is the
# only bucket that trains facts and translation, so it must never be left at 0.
env['TW_W_JA']    = '0.300'
env['TW_W_EN']    = '0.260'
env['TW_W_CODE']  = '0.190'
env['TW_W_MATH']  = '0.140'
env['TW_W_MIXED'] = '0.110'

proc = subprocess.Popen(
    ['python3', 'scripts/train_pipeline.py',
     '--stages',            'pretrain',
     '--checkpoint_dir',    '/kaggle/working/checkpoints',
     '--data_dir',          '/kaggle/input/datasets/naufalrachmat/datatraining/data/corpus_pretrain_v8',
     '--vocab_dir',         'vocab_8',
     '--bpe',               'bpe_v8.bin',
     '--batch_size',        '16',
     '--max_seq',           '512',
     '--max_tokens',        '10000000000',
     '--lr',                '1.4e-4',
     '--grad_clip',         '5.0',
     '--loss_scale',        '4096',
     '--safety',            'light',
     '--save_every',        '200',
     '--log_every',         '5',
     '--upload_every',      '400',
     '--eval_every_tokens', '200000000',
     '--eval_infer_bin',    './build_infer/infer',
     '--ddp', '--num_gpus', '2',
     '--resume'],
    cwd='/kaggle/working/transformer_100m_cuda',
    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    encoding='utf-8', errors='replace', bufsize=1,
)
for line in proc.stdout:
    print(line, end='', flush=True)
proc.wait()
print(f"\n[EXIT CODE: {proc.returncode}]")
if proc.returncode == 2:
    print("[!] CUDA OOM — halve --batch_size and re-run; the checkpoint was not touched.")
```

### Batch size vs sequence length on 2×T4 (16 GB each)

`--max_seq` is now genuinely applied, so VRAM scales with `batch × seq`. Keep the
product near the value that already worked:

| `--max_seq` | `--batch_size` | tokens/step (2 GPUs) |
|---|---|---|
| 256 | 32 | 16,384 |
| 512 | 16 | 16,384 |
| 1024 | 8 | 16,384 |
| 2048 | 4 | 16,384 |

---

## Cell 5b: Stage B — annealing the last 1.5B tokens

Stop Stage A at ~8.5B tokens, then resume for the remaining 1.5B with a lower
peak LR and the `mixed/` bucket tripled.

The reason this is worth a separate stage: what a model learns late, while the
learning rate is collapsing, is what it retains most sharply. Facts and
translation pairs are exactly the kind of knowledge that benefits — they are
memorisation targets, not skills that need many epochs to generalise. Spending
the low-LR tail on them buys accuracy that the same data spread evenly across
the run does not.

```python
env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)
env['NCCL_P2P_DISABLE'] = '1'

# mixed 0.11 -> 0.33; the other four keep their relative proportions.
env['TW_W_JA']    = '0.23'
env['TW_W_EN']    = '0.20'
env['TW_W_CODE']  = '0.14'
env['TW_W_MATH']  = '0.10'
env['TW_W_MIXED'] = '0.33'
```

Then re-run Cell 5 with that `env` and **one** changed flag:

```python
     '--lr',  '1.4e-5',      # was 1.4e-4
```

`--max_tokens` stays `10000000000`. The LR schedule is token-based, so resuming
at 8.5B puts the run at 85% of the cosine curve; with a 10× lower peak the
effective LR drops from ~2.1e-5 to ~2.1e-6 and keeps decaying. Do **not** raise
`--lr` on resume — the trainer warns and re-warms when the new peak exceeds the
saved one by 1.5×, which is a signal you have restarted the schedule by accident.

Note `min_lr` is hardcoded to `0.1 × --lr`, so the run anneals toward 1.4e-6
rather than exactly zero. That is intentional; a true zero LR wastes the final
steps.

Longer windows are worth taking now that documents survive intact: the corpus is
written one document per record, so a 512- or 1024-token window usually stays
inside a single article or source file instead of straddling unrelated snippets.

## Cell 6: Sanity-check the first log lines

The first 30 seconds of output tell you whether the data path is right:

```
[TOK] Loaded: vocab_size=49152
[Dataset] EN/JA/Code/Math corpus detected — quad weights (EN 0.40 / JA 0.30 / Code 0.20 / Math 0.10)
[Dataset] Found 12 unique corpus files
  .../english/fineweb_edu_extra_00.txt  (1430 MB, doc-mode)
  ...
[prefetch] file=starcoder_00.txt (doc-mode) lines=... docs=... samples=...
[S1] step=10 seq=512 ... loss=10.8 ...
```

Stop the run if you see any of these:

- `line-mode` on the `code` or `math` bucket — the document separator is missing,
  so every source line is being trained as its own document.
- `WARNING: bucket 'math' exists but has 0 usable bytes` — that bucket is a no-op
  and the weights silently fall back to a different layout.
- `WARNING: N unreadable entries (dangling symlink ...)`.
- `[ERR] dataset seq_len=... != --max_seq ...`.
- Initial `loss` far from `ln(49152) ≈ 10.8` on a fresh (non-resumed) run.

---

## Rebuilding the corpus (only when the source data changes)

The `mixed/` bucket is generated, not scraped, so it has to exist before the
main build runs — `build_pretrain_corpus.py` only copies and filters it.

```bash
# 1. fact injection. Repetition is deliberate: 4,580 tier-1 facts at 1000
#    exposures and 44,973 expanded facts at 150, rendered through 32 template
#    categories in four document shapes (profile / contrast / QA / mixed) so the
#    model binds a subject to its attributes instead of memorising one sentence.
python3 scripts/build_fact_injection_corpus.py --tsv data/wikidata_facts_tier1.tsv --exposures 1000
python3 scripts/build_fact_injection_corpus.py --tsv data/wikidata_facts_expanded.tsv --exposures 150 --append

# 2. EN<->JA parallel. Fetch first (~2 GB of TSV), then render.
for d in par_tatoeba par_talks par_news par_opus100 par_subs par_wikimatrix par_ccmatrix; do
  python3 scripts/fetch_corpus.py --dataset $d --max_bytes 1500000000
done
python3 scripts/build_mixed_bucket.py      # -> 10.8M unique pairs, ~534M tokens

# 3. audit the main build: keep/drop rates per source, writes nothing
python3 scripts/build_pretrain_corpus.py --audit

# 4. full build (~2 h, ~20 GB out). Ends by printing the TW_W_* values to use.
python3 scripts/build_pretrain_corpus.py --out data/corpus_pretrain_v8
```

## Two known data gaps

**1. Math had no real source.** `data/corpus_1b/math/openwebmath.txt` is a
dangling symlink, and the English web corpora here contain almost no
mathematical notation — over 6,000 sampled fineweb_edu documents the
notation-density score is p99 = 1.5 and max = 7.5 hits per 1000 characters, so
there is nothing worth harvesting. Fill the bucket from HuggingFace:

```bash
pip install datasets
python3 scripts/fetch_corpus.py --dataset finemath    --max_bytes 3000000000
python3 scripts/fetch_corpus.py --dataset openwebmath --max_bytes 1500000000
```

**2. The code bucket was 94.6% Python.** Both starcoder extracts are
Python-only, so a general code model cannot come out of it. The Stack is gated
(needs `huggingface-cli login` plus accepting its terms); these stream without a
token:

```bash
python3 scripts/fetch_corpus.py --dataset gh_javascript  --max_bytes 900000000
python3 scripts/fetch_corpus.py --dataset csn_javascript --max_bytes 400000000
python3 scripts/fetch_corpus.py --dataset code_contests  --max_bytes 600000000  # C++/Java
python3 scripts/fetch_corpus.py --dataset rosetta        --max_bytes 200000000  # many languages
python3 scripts/fetch_corpus.py --dataset kstack_kotlin  --max_bytes 300000000
```

**3. Japanese is the tightest language bucket.** Everything local — `wiki_ja`
3.68 GB, `culturax_ja` 2.68 GB, `mc4_ja` 0.71 GB, an older `wiki_ja` 1.44 GB —
yields 7.15 GB after filtering, which is 2.25B tokens and 1.6 epochs at the
intended 36% share. No sibling project has any Japanese data. To add more:

```bash
python3 scripts/fetch_corpus.py --dataset fineweb2_ja --max_bytes 5000000000
```

After any fetch, rebuild — anything under `data/{en,ja,code,math}_raw/` is picked
up automatically, filtered by the same rules and deduplicated against the corpus
that already exists:

```bash
python3 scripts/build_pretrain_corpus.py --out data/corpus_pretrain_v8
python3 scripts/fetch_corpus.py --list    # everything available, per bucket
```

## Retrain the tokenizer before the long run

`vocab_7` is lopsided against Japanese and against code:

| slots | share |
|---|---|
| latin | 27,731 (56.4%) |
| partial-UTF8 fragments | 11,018 (22.4%) |
| japanese/CJK | 8,433 (17.2%) |

17.2% of the vocabulary for what will be 36% of the training tokens. Concretely,
**1,383 of the 2,136 most frequent kanji are not single tokens**, katakana
loanwords get no merges at all (コンピュータ = 6 tokens for 6 characters), and the
whole vocabulary contains exactly **one** token with a newline in it — so Python
indentation cannot be tokenized efficiently either.

The cause was the training mix (EN 2200 : JA 900 : code 500 MB) plus a corpus
reader that split on newlines and therefore never saw one. Both are fixed; a
4,000-token trial on the new layout learned 42 newline-bearing tokens including
`\n\n`, `\n    ` and `\n        `. Retrain on the v8 corpus:

```bash
cmake --build build --target train_bpe
./build/train_bpe --out_dir vocab_8 --vocab_size 49152 --min_freq 15 \
  --corpus data/corpus_pretrain_v8/japanese:2000 \
  --corpus data/corpus_pretrain_v8/english:1300 \
  --corpus data/corpus_pretrain_v8/code:1100 \
  --corpus data/corpus_pretrain_v8/math:600 \
  --binary bpe_v8.bin --checkpoint bpe_checkpoints/bpe_v8_ckpt.bin

python3 scripts/preflight_pretrain.py --vocab_dir vocab_8 --bpe bpe_v8.bin
```

Then switch `--vocab_dir vocab_8 --bpe bpe_v8.bin`. Changing the tokenizer
invalidates existing checkpoints, so do it before the long run, not during it.
