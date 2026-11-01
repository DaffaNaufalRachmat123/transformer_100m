# GOOGLE COLAB — Pre-Training Only (EN / JA / Code / Math)

> **Updated for the v8 pipeline.** Same changes as `KAGGLE_PRE-TRAINING.md`:
> `vocab_7` + `bpe_v7.bin` instead of `vocab_6` + `bpe_v6.bin`,
> `data/corpus_pretrain_v8` instead of `data/corpus_final`, a preflight cell, and
> a `--max_seq` flag that is actually applied by the dataset.

## Token budget (Chinchilla)

491.6M parameters × 20 tokens/parameter = **9.8B tokens** → `--max_tokens 10000000000`.

| bucket | share | tokens | bytes/token | corpus bytes |
|---|---|---|---|---|
| japanese | 36% | 3.6B | 3.18 | 11.4 GB |
| english | 32% | 3.2B | 3.53 | 11.3 GB |
| code | 20% | 2.0B | 3.64 | 7.3 GB |
| math | 12% | 1.2B | 3.70 | 4.4 GB |

Japanese takes the largest share on purpose: it packs 1.16 characters per token
against English's 4.15, and the model has to learn three scripts plus ~6,200
distinct kanji. The corpus holds ~5.8B unique tokens, so a 10B run makes ~1.7
passes — comfortably inside the ~4-epoch limit where repeated tokens still pay.

## Cell 1: Mount Drive and set paths

```python
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_ROOT = "/content/drive/MyDrive/transformer"   # adjust to your layout
WORK_DIR   = "/content/transformer_100m_cuda"
CKPT_DIR   = f"{DRIVE_ROOT}/checkpoints"            # survives a runtime restart
os.makedirs(CKPT_DIR, exist_ok=True)
print(os.listdir(DRIVE_ROOT))
```

## Cell 2: Extract source + link the corpus

```python
%%bash -s "$DRIVE_ROOT" "$WORK_DIR"
DRIVE_ROOT=$1; WORK_DIR=$2
rm -rf "$WORK_DIR"; mkdir -p "$WORK_DIR"

# Source tree (zip or folder on Drive)
if [ -f "$DRIVE_ROOT/transformer_100m_cuda.zip" ]; then
  unzip -q "$DRIVE_ROOT/transformer_100m_cuda.zip" -d /content/
else
  cp -r "$DRIVE_ROOT/transformer_100m_cuda/." "$WORK_DIR/"
fi

# The corpus stays on Drive; symlink it so nothing is copied into the VM disk.
mkdir -p "$WORK_DIR/data"
ln -sfn "$DRIVE_ROOT/data/corpus_pretrain_v8" "$WORK_DIR/data/corpus_pretrain_v8"

cd "$WORK_DIR"
ls -d vocab_8 bpe_v8.bin data/corpus_pretrain_v8
ls -lL data/corpus_pretrain_v8/*/ | head -30
```

> `ls -lL` follows symlinks on purpose. A dangling link prints
> `No such file or directory` here instead of silently removing a whole bucket
> from training — that is exactly how the math bucket disappeared previously.

## Cell 3: Build

```python
%%bash -s "$WORK_DIR"
cd "$1"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -10
cmake --build . -j$(nproc) 2>&1 | tail -20
ls -lh train_pretrain_cuda train_sft_cuda

cmake --build . --target grad_check -j$(nproc) 2>&1 | tail -5
./grad_check
```

```python
%%bash -s "$WORK_DIR"
cd "$1"
rm -rf build_infer && mkdir -p build_infer && cd build_infer
cmake ../infer_cpp -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
make -j$(nproc) 2>&1 | tail -5
ls -lh infer
```

## Cell 4: Preflight — tokenizer + corpus (STOP if this fails)

```python
%%bash -s "$WORK_DIR"
cd "$1"
python3 scripts/preflight_pretrain.py \
  --vocab_dir vocab_8 \
  --bpe bpe_v8.bin \
  --data_dir data/corpus_pretrain_v8 \
  --max_seq 512
```

## Cell 5: Run Pre-Training

Colab gives one GPU, so there is no `--ddp`. A T4 and an A100 need different
`--batch_size`; the table under Cell 6 has the values.

```python
import subprocess, os

WORK_DIR = "/content/transformer_100m_cuda"
CKPT_DIR = "/content/drive/MyDrive/transformer/checkpoints"

env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)
# Take these from the tail of logs/build_pretrain_corpus_v8.log.
# Japanese leads because it is the least compressible: 1.16 characters per token
# against 4.15 for English. `mixed` is fact injection plus the EN<->JA parallel
# corpus — the only bucket that trains facts and translation, never leave it 0.
env['TW_W_JA']    = '0.300'
env['TW_W_EN']    = '0.260'
env['TW_W_CODE']  = '0.190'
env['TW_W_MATH']  = '0.140'
env['TW_W_MIXED'] = '0.110'

proc = subprocess.Popen(
    ['python3', 'scripts/train_pipeline.py',
     '--stages',            'pretrain',
     '--checkpoint_dir',    CKPT_DIR,
     '--data_dir',          'data/corpus_pretrain_v8',
     '--vocab_dir',         'vocab_8',
     '--bpe',               'bpe_v8.bin',
     '--batch_size',        '12',
     '--max_seq',           '512',
     '--max_tokens',        '10000000000',
     '--lr',                '1.4e-4',
     '--grad_clip',         '5.0',
     '--loss_scale',        '4096',
     '--safety',            'light',
     '--save_every',        '150',
     '--log_every',         '5',
     '--upload_every',      '300',
     '--eval_every_tokens', '200000000',
     '--eval_infer_bin',    './build_infer/infer',
     '--resume'],
    cwd=WORK_DIR, env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
)
try:
    for raw in iter(proc.stdout.readline, b''):
        print(raw.decode('utf-8', errors='replace'), end='', flush=True)
except KeyboardInterrupt:
    proc.terminate()
proc.wait()
print(f"\n[EXIT CODE: {proc.returncode}]")
if proc.returncode == 2:
    print("[!] CUDA OOM — halve --batch_size and re-run; the checkpoint was not touched.")
```

Checkpoints are written straight to Drive, so a disconnect costs at most
`--save_every` steps. Re-running the same cell resumes: `--resume` restores the
weights, the optimizer state and the token counter, and the dataset seed is
derived from the token count so the second session reads a different slice of the
corpus rather than replaying the first one.

## Cell 5b: Stage B — annealing the last 1.5B tokens

Stop Stage A at ~8.5B tokens, then resume for the remaining 1.5B with a lower
peak LR and the `mixed/` bucket tripled.

What a model learns late, while the learning rate is collapsing, is what it
retains most sharply. Facts and translation pairs benefit most from that — they
are memorisation targets rather than skills that need many epochs to generalise,
so spending the low-LR tail on them buys accuracy that the same data spread
evenly across the run does not.

```python
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

`--max_tokens` stays `10000000000`. The schedule is token-based, so resuming at
8.5B puts the run at 85% of the cosine curve; with a 10× lower peak the effective
LR drops from ~2.1e-5 to ~2.1e-6 and keeps decaying. Do **not** raise `--lr` on
resume — the trainer warns and re-warms when the new peak exceeds the saved one
by 1.5×, which means you have restarted the schedule by accident.

`min_lr` is hardcoded to `0.1 × --lr`, so the run anneals toward 1.4e-6 rather
than exactly zero. That is intentional; a true zero LR wastes the final steps.

## Cell 6: Batch size per GPU

`--max_seq` is now genuinely applied, so VRAM scales with `batch × seq`.

| GPU | `--max_seq` 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|
| T4 (16 GB) | 24 | 12 | 6 | 3 |
| L4 (24 GB) | 40 | 20 | 10 | 5 |
| A100 (40 GB) | 72 | 36 | 18 | 9 |

Start one step below the value in the table on a fresh architecture; exit code 2
is a clean OOM abort that leaves the checkpoint intact.

## Cell 7: Sanity-check the first log lines

```
[TOK] Loaded: vocab_size=49152
[Dataset] EN/JA/Code/Math corpus detected — quad weights (EN 0.40 / JA 0.30 / Code 0.20 / Math 0.10)
  .../code/starcoder_00.txt  (1430 MB, doc-mode)
[S1] step=10 seq=512 ... loss=10.8 ...
```

Abort if you see `line-mode` on the `code` or `math` bucket, a
`bucket ... 0 usable bytes` warning, a dangling-symlink warning, or
`[ERR] dataset seq_len=... != --max_seq ...`.

---

## Rebuilding the corpus

```bash
python3 scripts/build_pretrain_corpus.py --audit                       # keep/drop rates
python3 scripts/build_pretrain_corpus.py --out data/corpus_pretrain_v8 # ~2 h, ~20 GB
```

The `mixed/` bucket is generated rather than scraped, so build it first or the
model gets no fact-injection and no translation training at all:

```bash
python3 scripts/build_fact_injection_corpus.py --tsv data/wikidata_facts_tier1.tsv --exposures 1000
python3 scripts/build_fact_injection_corpus.py --tsv data/wikidata_facts_expanded.tsv --exposures 150 --append
for d in par_tatoeba par_talks par_news par_opus100 par_subs par_wikimatrix par_ccmatrix; do
  python3 scripts/fetch_corpus.py --dataset $d --max_bytes 1500000000
done
python3 scripts/build_mixed_bucket.py
```

The build ends by printing the `TW_W_*` values that hit the intended token mix
given what each bucket actually yielded — use those, not the defaults.

**Three data gaps.** Math had no real source here (`data/corpus_1b/math/openwebmath.txt`
is a dangling symlink and the English corpora carry almost no notation), the code
bucket was 94.6% Python, and Japanese is the tightest language bucket at 1.6
epochs. Close them before the long run:

```bash
pip install datasets
python3 scripts/fetch_corpus.py --dataset finemath        --max_bytes 3000000000
python3 scripts/fetch_corpus.py --dataset openwebmath     --max_bytes 1500000000
python3 scripts/fetch_corpus.py --dataset gh_javascript   --max_bytes 900000000
python3 scripts/fetch_corpus.py --dataset code_contests   --max_bytes 600000000
python3 scripts/fetch_corpus.py --dataset rosetta         --max_bytes 200000000
python3 scripts/fetch_corpus.py --dataset fineweb2_ja     --max_bytes 5000000000
python3 scripts/build_pretrain_corpus.py --out data/corpus_pretrain_v8
```

`scripts/fetch_corpus.py --list` shows every registered source. The Stack is
gated, so run `huggingface-cli login` and accept its terms if you want the full
multi-language code set.

**Retrain the tokenizer first.** `vocab_7` gives Japanese 17.2% of its slots for
what will be 36% of the tokens: 1,383 of the 2,136 most frequent kanji are not
single tokens, katakana loanwords get no merges (コンピュータ = 6 tokens for 6
characters), and the vocabulary holds exactly one token containing a newline.

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

A tokenizer change invalidates existing checkpoints, so do this before the long
run rather than partway through it.
