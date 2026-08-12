# GOOGLE COLAB — Full Training Pipeline (Pretrain → SFT → DPO)

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
CKPT_DRIVE = f"{DRIVE_ROOT}/checkpoints_100m"
CKPT_JA_DRIVE     = f"{DRIVE_ROOT}/checkpoints_facts_ja_100m"
CKPT_EN_DRIVE     = f"{DRIVE_ROOT}/checkpoints_facts_en_100m"
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
if not os.path.isdir(DPO_PATH):
    errors.append(f"  ✗ sft_data/dpo/: {DPO_PATH}")

if errors:
    print("❌ PATH ERRORS:")
    for e in errors: print(e)
else:
    print("✓ All paths found")
    print(f"  ZIP       : {ZIP_PATH}")
    print(f"  data/     : {DATA_PATH}")
    print(f"  sft_data/ : {SFT_PATH}")
    print(f"  dpo/      : {DPO_PATH}")
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

## Cell 4: Symlink Data from Drive

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

make_symlink(DATA_PATH, f"{WORK_DIR}/data")
make_symlink(SFT_PATH,  f"{WORK_DIR}/sft_data")

os.makedirs(CKPT_DRIVE, exist_ok=True)
make_symlink(CKPT_DRIVE, f"{WORK_DIR}/checkpoints")

# Also persist the stage-specific dirs used by run_facts_sft.sh to Drive,
# so checkpoints_facts_ja / checkpoints_facts_en / checkpoints_transl survive.
CKPT_JA_DRIVE     = f"{DRIVE_ROOT}/checkpoints_facts_ja_100m"
CKPT_EN_DRIVE     = f"{DRIVE_ROOT}/checkpoints_facts_en_100m"
CKPT_TRANSL_DRIVE = f"{DRIVE_ROOT}/checkpoints_transl_100m"
os.makedirs(CKPT_JA_DRIVE,     exist_ok=True)
os.makedirs(CKPT_EN_DRIVE,     exist_ok=True)
os.makedirs(CKPT_TRANSL_DRIVE, exist_ok=True)
make_symlink(CKPT_JA_DRIVE,     f"{WORK_DIR}/checkpoints_facts_ja")
make_symlink(CKPT_EN_DRIVE,     f"{WORK_DIR}/checkpoints_facts_en")
make_symlink(CKPT_TRANSL_DRIVE, f"{WORK_DIR}/checkpoints_transl")

# translation_resources (active model the infer binary loads) → Drive too.
TRANSL_RES_DRIVE = f"{DRIVE_ROOT}/translation_resources_100m"
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

## Cell 6: Run Full Pipeline (Pretrain → SFT → DPO)

```python
import subprocess, os

WORK_DIR = "/content/transformer_100m_cuda"
env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)

proc = subprocess.Popen(
    ['python3', 'scripts/train_pipeline.py',
     '--auto_continue',
     '--stages',            'pretrain,sft,dpo',
     '--checkpoint_dir',    'checkpoints',
     '--data_dir',          'data/corpus_final',
     '--vocab_dir',         'vocab_6',
     '--bpe',               'bpe_v6.bin',
     # Pretrain
     '--batch_size',        '24',
     '--max_seq',           '256',
     '--max_tokens',        '2000000000',
     '--lr',                '1.4e-4',
     '--grad_clip',         '5.0',
     '--loss_scale',        '4096',
     '--safety',            'light',
     '--save_every',        '150',
     '--log_every',         '5',
     '--upload_every',      '300',
     '--eval_every_tokens', '200000000',
     '--eval_infer_bin',    './build_infer/infer',
     # SFT
     '--sft_dir',           'sft_data/sft_translation_focused',
     '--sft_lr',            '3e-5',
     '--sft_epochs',        '1',
     '--sft_batch_size',    '8',
     '--sft_max_seq',       '512',
     '--sft_warmup',        '100',
     '--sft_save_every',    '200',
     '--sft_upload_every',  '200',
     # DPO
     '--dpo_dir',           'sft_data/dpo',
     '--dpo_lr',            '1e-5',
     '--dpo_epochs',        '3',
     '--dpo_beta',          '0.1',
     '--dpo_batch_size',    '4',
     '--dpo_max_seq',       '512',
     '--dpo_save_every',    '200',
     '--dpo_upload_every',  '200',
     # Infra
     '--resume'],
    cwd=WORK_DIR, env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
)
try:
    for raw in iter(proc.stdout.readline, b''):
        print(raw.decode('utf-8', errors='replace'), end='', flush=True)
except KeyboardInterrupt:
    proc.terminate()
    print("\nPipeline interrupted. Re-run to resume.")
proc.wait()
print(f"\n[EXIT CODE: {proc.returncode}]")
```

## SFT config notes (updated for the expanded parallel corpus)

The SFT stage was changed to match the new data pipeline
(`scripts/fetch_parallel_enja.py` → `build_sft_translation.py` →
`prepare_sft_translation_dominant.py`):

- **`--sft_dir sft_data/sft_translation_focused`** (was `sft_data/sft_clean`).
  `train_sft_cuda` reads **every `.jsonl` file** in the directory. The old
  `sft_data/sft_clean` contained the **held-out eval set
  (`sft_translation_eval.jsonl`)** — training on it would have contaminated the
  eval and produced a fake "improvement". It also contained old noisy data
  (sft_en, sft_id, sft_codeswitched, sft_multiturn). The new dir contains only
  `train.jsonl` (520K translation rows, 90% translation + 10% facts).
- **`--sft_epochs 1`** (was 3). The dataset is **~53× larger** (9.8K → 520K
  rows). 3 epochs would be ~390K steps — far too long. 1 epoch over 520K rows
  is already far more exposure than the old 3 epochs over 9.8K rows.
- **`--sft_batch_size 8`** (was 4) to keep step count manageable (~65K steps
  for 1 epoch). If training is too slow on a T4, raise `--sft_batch_size`
  further; the step budget scales as `rows*epochs/batch_size`.

> ⚠️ **If you only want to retrain SFT** (recommended after a data change),
> run only the SFT stage and skip DPO:
> `--stages 'sft'` and remove the `--dpo_*` args, or set `--dpo_max_steps 0`.
> The EN⇄JA DPO data is tiny and previously hurt the model (see
> `scripts/run_enja_pipeline.sh`).
```
