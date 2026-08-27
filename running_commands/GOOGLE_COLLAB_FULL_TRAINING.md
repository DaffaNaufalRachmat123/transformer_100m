# GOOGLE COLAB — Full Training Pipeline (Pretrain → Facts SFT → Translation → DPO)

## Cell 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

## Cell 2: Configure Paths

```python
import os

DRIVE_ROOT = "/content/drive/MyDrive"

ZIP_PATH   = f"{DRIVE_ROOT}/transformer_100m_cuda.zip"
DATA_PATH  = f"{DRIVE_ROOT}/data"
SFT_PATH   = f"{DRIVE_ROOT}/sft_data"
DPO_PATH   = f"{DRIVE_ROOT}/sft_data/dpo"

WORK_DIR   = "/content/transformer_100m_cuda"
CKPT_DRIVE        = f"{DRIVE_ROOT}/checkpoints_100m"
CKPT_FACTS_DRIVE  = f"{DRIVE_ROOT}/checkpoints_facts_100m"
CKPT_TRANSL_DRIVE = f"{DRIVE_ROOT}/checkpoints_transl_100m"
TRANSL_RES_DRIVE  = f"{DRIVE_ROOT}/translation_resources_100m"

os.environ["WORK_DIR"]   = WORK_DIR
os.environ["ZIP_PATH"]   = ZIP_PATH
os.environ["DATA_PATH"]  = DATA_PATH
os.environ["SFT_PATH"]   = SFT_PATH
os.environ["CKPT_DRIVE"] = CKPT_DRIVE

errors = []
if not os.path.exists(ZIP_PATH):
    errors.append(f"  ✗ ZIP: {ZIP_PATH}")
if not os.path.isdir(DATA_PATH):
    errors.append(f"  ✗ data/: {DATA_PATH}")
if not os.path.isdir(SFT_PATH):
    errors.append(f"  ✗ sft_data/: {SFT_PATH}")

if errors:
    print("❌ PATH ERRORS:")
    for e in errors: print(e)
else:
    print("✓ All paths found")
    print(f"  ZIP       : {ZIP_PATH}")
    print(f"  data/     : {DATA_PATH}")
    print(f"  sft_data/ : {SFT_PATH}")
    print(f"  ckpt/     : {CKPT_DRIVE}")
```

## Cell 3: Extract Source Code

```python
import os, shutil

if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
    print(f"Cleaned old {WORK_DIR}")

os.makedirs(WORK_DIR, exist_ok=True)
!unzip -q "$ZIP_PATH" -d "$WORK_DIR"

assert os.path.exists(f"{WORK_DIR}/CMakeLists.txt"), \
    f"Extraction failed! Files: {os.listdir(WORK_DIR)[:10]}"
print(f"✓ Source extracted to {WORK_DIR}")
```

## Cell 4: Symlink Data & Checkpoints from Drive

```python
import os, shutil

def make_symlink(target, link):
    if os.path.islink(link):
        os.unlink(link)
    elif os.path.isdir(link):
        shutil.rmtree(link)
    elif os.path.exists(link):
        os.remove(link)
    os.symlink(target, link)
    print(f"✓ {os.path.basename(link)}/ -> {target}")

# Data and SFT
make_symlink(DATA_PATH, f"{WORK_DIR}/data")
make_symlink(SFT_PATH,  f"{WORK_DIR}/sft_data")

# Pretrain checkpoints (base model)
os.makedirs(CKPT_DRIVE, exist_ok=True)
make_symlink(CKPT_DRIVE, f"{WORK_DIR}/checkpoints")

# Facts SFT checkpoint (single combined EN+JA)
os.makedirs(CKPT_FACTS_DRIVE, exist_ok=True)
make_symlink(CKPT_FACTS_DRIVE, f"{WORK_DIR}/checkpoints_facts")

# Translation SFT checkpoint
os.makedirs(CKPT_TRANSL_DRIVE, exist_ok=True)
make_symlink(CKPT_TRANSL_DRIVE, f"{WORK_DIR}/checkpoints_transl")

# translation_resources (active model the infer binary loads)
os.makedirs(TRANSL_RES_DRIVE, exist_ok=True)
make_symlink(TRANSL_RES_DRIVE, f"{WORK_DIR}/translation_resources")
```

## Cell 5: Build All Binaries + Kernel Check + Infer

```python
%%bash
set -e
WORK_DIR="/content/transformer_100m_cuda"
cd "$WORK_DIR"

if ! command -v g++ &> /dev/null; then
    apt-get update -qq && apt-get install -y -qq build-essential cmake > /dev/null 2>&1
fi

export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME"

if   echo "$GPU_NAME" | grep -qi "T4";   then CUDA_ARCH=75
elif echo "$GPU_NAME" | grep -qi "V100"; then CUDA_ARCH=70
elif echo "$GPU_NAME" | grep -qi "A100"; then CUDA_ARCH=80
elif echo "$GPU_NAME" | grep -qi "L4";   then CUDA_ARCH=89
elif echo "$GPU_NAME" | grep -qi "H100"; then CUDA_ARCH=90
else CUDA_ARCH=75; echo "[WARN] Unknown GPU — defaulting to sm_75"; fi

echo "Building for: sm_${CUDA_ARCH}"

rm -rf build CMakeCache.txt
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH
make -j$(nproc)

echo ""
echo "Built binaries:"
ls -lh train_pretrain_cuda train_sft_cuda 2>/dev/null || true

echo ""
echo "Running grad_check..."
cmake --build . --target grad_check -j$(nproc)
./grad_check

echo ""
echo "Building infer binary..."
rm -rf "$WORK_DIR/build_infer"
mkdir -p "$WORK_DIR/build_infer" && cd "$WORK_DIR/build_infer"
cmake "$WORK_DIR/infer_cpp" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH
make -j$(nproc)

echo ""
echo "=== ALL BUILDS OK + KERNELS VERIFIED ==="
```

## Cell 6A: Continued Pretraining (2B → 4.5B)

> ✅ **Status: sudah berjalan** — checkpoint 3.85B-token sudah tersedia
> (`ckpt_pretrain_final.safetensors`, MD5 `cedf...`, Aug 25).
> Run this cell only if you want to CONTINUE pretraining further.
> Requires `checkpoints/ckpt_pretrain.safetensors` + `train_state.txt` + `ckpt_optim.safetensors`.
> Note: rename `ckpt_pretrain_final.safetensors` → `ckpt_pretrain.safetensors`
> and `ckpt_optim_final.safetensors` → `ckpt_optim.safetensors` first
> (the trainer looks for the non-`_final` names when resuming).

```python
import subprocess, os

WORK_DIR = "/content/transformer_100m_cuda"
env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)
env['NCCL_P2P_DISABLE'] = '1'

proc = subprocess.Popen(
    ['python3', 'scripts/train_pipeline.py',
     '--auto_continue',
     '--stages',            'pretrain',
     '--checkpoint_dir',    'checkpoints',
     '--data_dir',          'data/corpus_pretrain_2_5b',
     '--vocab_dir',         'vocab_6',
     '--bpe',               'bpe_v6.bin',
     # Pretrain — continued from 2B
     '--batch_size',        '35',
     '--max_seq',           '256',
     '--max_tokens',        '4500000000',     # 4.5B total (2B existing + 2.5B new)
     '--lr',                '7e-5',           # Lower LR for continued pretraining
     '--grad_clip',         '5.0',
     '--loss_scale',        '4096',
     '--safety',            'light',
     '--save_every',        '200',
     '--log_every',         '5',
     '--upload_every',      '400',
     '--eval_every_tokens', '200000000',
     '--eval_infer_bin',    './build_infer/infer',
     # Infra
     '--ddp', '--num_gpus', '2',
     '--resume'],
    cwd=WORK_DIR, env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
)
try:
    for raw in iter(proc.stdout.readline, b''):
        print(raw.decode('utf-8', errors='replace'), end='', flush=True)
except KeyboardInterrupt:
    proc.terminate()
    print("\nPretrain interrupted. Re-run to resume.")
proc.wait()
print(f"\n[EXIT CODE: {proc.returncode}]")
```

## Cell 6B: Facts SFT (from pretrain checkpoint)

> Run this after pretraining is done. Uses the V9 facts dataset (500K rows,
> EN 52% / JA 48%, 1,501 clean unique facts, strict category-specific prompts).
> Key V9 changes vs V5/V8: TSV data fixes (Nauru, euro, truncated values),
> superheavy placeholder elements dropped, JA binding share raised to fix the
> 10% subject-binding rate found by the honest evaluator.
> Single training run — no separate EN/JA stages.

```python
import subprocess, os

WORK_DIR = "/content/transformer_100m_cuda"
env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)

# Generate facts dataset if not present (v9_full profile, 500K rows)
proc_gen = subprocess.run(
    ['python3', 'scripts/build_facts_dataset.py',
     '--mode', 'oversample',
     '--target_rows', '500000',
     '--en_share', '0.52',
     '--eval_rows', '6000',
     '--profile', 'v9_full',
     '--out_dir', 'sft_data/sft_facts_v9_500k'],
    cwd=WORK_DIR, env=env,
    capture_output=True, text=True,
)
print(proc_gen.stdout)
if proc_gen.returncode != 0:
    print(proc_gen.stderr)

# Audit dataset (must PASS before training)
proc_audit = subprocess.run(
    ['python3', 'scripts/audit_facts_dataset.py',
     'sft_data/sft_facts_v9_500k/train.jsonl',
     'sft_data/sft_facts_v9_500k/eval.jsonl',
     '--max_category_share', '0.30'],
    cwd=WORK_DIR, env=env,
    capture_output=True, text=True,
)
print(proc_audit.stdout)

# Train facts SFT (20K steps from the 3.85B pretrain checkpoint)
proc = subprocess.Popen(
    ['python3', 'scripts/train_pipeline.py',
     '--auto_continue',
     '--stages',            'sft',
     '--checkpoint_dir',    'checkpoints_facts',
     '--vocab_dir',         'vocab_6',
     '--bpe',               'bpe_v6.bin',
     '--sft_dir',           'sft_data/sft_facts_v9_500k',
     '--sft_base_model',    'checkpoints/ckpt_pretrain_final.safetensors',
     '--sft_lr',            '1e-5',
     '--sft_epochs',        '2',
     '--sft_batch_size',    '20',
     '--sft_max_seq',       '256',
     '--sft_warmup',        '100',
     '--sft_max_steps',     '20000',
     '--sft_save_every',    '500',
     '--sft_upload_every',  '600',
     '--resume'],
    cwd=WORK_DIR, env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
)
try:
    for raw in iter(proc.stdout.readline, b''):
        print(raw.decode('utf-8', errors='replace'), end='', flush=True)
except KeyboardInterrupt:
    proc.terminate()
    print("\nFacts SFT interrupted. Re-run to resume.")
proc.wait()
print(f"\n[EXIT CODE: {proc.returncode}]")
```

## Cell 6C: Translation SFT (from facts checkpoint)

> Optional: train translation on top of the facts model.
> Skip if you only want facts QA.

```python
import subprocess, os

WORK_DIR = "/content/transformer_100m_cuda"
env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)

proc = subprocess.Popen(
    ['python3', 'scripts/train_pipeline.py',
     '--auto_continue',
     '--stages',            'sft',
     '--checkpoint_dir',    'checkpoints_transl',
     '--vocab_dir',         'vocab_6',
     '--bpe',               'bpe_v6.bin',
     '--sft_dir',           'sft_data/sft_translation_focused',
     '--sft_base_model',    'checkpoints_facts/ckpt_sft_final.safetensors',
     '--sft_lr',            '3e-5',
     '--sft_epochs',        '1',
     '--sft_batch_size',    '20',
     '--sft_max_seq',       '256',
     '--sft_warmup',        '50',
     '--sft_max_steps',     '15000',
     '--sft_save_every',    '200',
     '--sft_upload_every',  '600',
     '--resume'],
    cwd=WORK_DIR, env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
)
try:
    for raw in iter(proc.stdout.readline, b''):
        print(raw.decode('utf-8', errors='replace'), end='', flush=True)
except KeyboardInterrupt:
    proc.terminate()
    print("\nTranslation SFT interrupted. Re-run to resume.")
proc.wait()
print(f"\n[EXIT CODE: {proc.returncode}]")
```

## Cell 7: Evaluate Facts

> The evaluator now reports **subj-bind** (subject-binding: does the answer
> mention the asked subject?) and uses honest word-boundary value matching.
> Numbers will look LOWER than old V8 reports — that is expected: the old
> evaluator over-counted substring false positives and scored "right value,
> wrong country" as correct. Compare against the honest baseline below.

**Honest baseline (V8 checkpoint, 2B pretrain, buggy data):**

| Metric | EN | JA |
|---|---:|---:|
| lang-ok | 100% | 100% |
| attr-ok | 20% | 60% |
| value-hit | 20% | 40% |
| subj-bind | 60% | 10% |

```python
%%bash
cd /content/transformer_100m_cuda

echo "=== Facts Eval (20 samples) ==="
python3 scripts/eval_facts.py \
    --model checkpoints_facts/ckpt_sft_final.safetensors \
    --eval sft_data/sft_facts_v9_500k/eval.jsonl \
    --max_test 20 \
    --max_tokens 30

echo ""
echo "=== Facts Eval (200 samples, stable) ==="
python3 scripts/eval_facts.py \
    --model checkpoints_facts/ckpt_sft_final.safetensors \
    --eval sft_data/sft_facts_v9_500k/eval.jsonl \
    --max_test 200 \
    --max_tokens 30

echo ""
echo "=== Translation Eval ==="
python3 scripts/eval_translation.py \
    --model checkpoints_transl/ckpt_sft_final.safetensors \
    --max_test 20 \
    --max_tokens 30
```

## Cell 8: Copy Final Models

```python
%%bash
cd /content/transformer_100m_cuda

mkdir -p checkpoints translation_resources

# Facts model → active locations
cp checkpoints_facts/ckpt_sft_final.safetensors checkpoints/ckpt_sft_final.safetensors
cp checkpoints_facts/ckpt_sft_final.safetensors translation_resources/ckpt_sft_final.safetensors

# Also copy to facts subfolder for facts eval
mkdir -p translation_resources/facts
cp checkpoints_facts/ckpt_sft_final.safetensors translation_resources/facts/ckpt_sft_final.safetensors

echo "✓ Final models copied"
ls -lh checkpoints/ckpt_sft_final.safetensors translation_resources/ckpt_sft_final.safetensors
```

## Checkpoint Directory Structure

| Directory | Drive Path | Purpose |
|---|---|---|
| `checkpoints/` | `checkpoints_100m` | Pretrain base model |
| `checkpoints_facts/` | `checkpoints_facts_100m` | Facts SFT (combined EN+JA) |
| `checkpoints_transl/` | `checkpoints_transl_100m` | Translation SFT |
| `translation_resources/` | `translation_resources_100m` | Active model for inference |

## Facts Dataset Info (V9)

| Property | Value |
|---|---|
| Dataset | `sft_data/sft_facts_v9_500k` |
| Train rows | 500,000 (260K EN + 240K JA) |
| Eval rows | 4,800 (50/50, held-out by fact key) |
| EN share | 52% |
| Profile | `v9_full` (normal + binding + anti + contrast variants) |
| Unique facts | 1,501 (V8 had 1,224; +277 from euro/QID/Nauru fixes) |
| Data fixes | Naoero→Nauru, Q4916→euro (27 countries), France CFP dropped, truncated values dropped, 71 superheavy placeholder elements dropped |
| Overrides | `data/fact_overrides.json` (drop/subject/value/aliases — required by builder) |
| JA binding share | ~38% (raised to fix 10% subject-binding rate) |
| Categories | capital, currency, official_language, population, area, highest_point, continent_of, element_symbol, element_number |
| Excluded | head_of_state (volatile), mc4_ja (contaminated) |
| Audit | `scripts/audit_facts_dataset.py` (PASS required before training) |

## SFT config notes

- **Facts SFT** uses `sft_data/sft_facts_v9_500k` with strict
  category-specific prompts (no generic "Tell me about X"). Single training
  run (20K steps) from the 3.85B pretrain checkpoint, no separate EN/JA
  stages.
- **Translation SFT** uses `sft_data/sft_translation_focused` (520K rows).
  Run after facts SFT to add translation capability on top of facts.
- **DPO** is optional and currently skipped. The EN⇄JA DPO data is tiny and
  previously hurt the model.
- **Eval metrics are honest now** (word-boundary matching + subject-binding
  check). Do not compare new numbers against pre-V9 eval reports.

## Pretraining config notes

- **Current checkpoint: 3.85B tokens** (trained with `data/corpus_pretrain_2_5b`,
  clean EN/JA files only, `mc4_ja.txt` excluded — 5,092 エロ spam).
- **LR 7e-5** (half of initial 1.4e-4) for stable continued training.
- **max_tokens 4.5B** is cumulative — `--resume` restores `tokens_processed`
  from `train_state.txt`.
- Estimated time: ~2.5 days on T4 x2 (DDP).
```
