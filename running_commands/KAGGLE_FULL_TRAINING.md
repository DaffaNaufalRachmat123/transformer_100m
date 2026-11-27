# KAGGLE — Full Ceiling Training Pipeline (Mid-Train → SFT v4 → DPO Ceiling → Verifiable RL)

> **Update (Ultra-Max Ceiling Edition — 491.58M @ 10B `corpus_pretrain_v8` on 2× Tesla T4 14.6 GB):**
> 1. **Mid-Training Annealing (opsional namun direkomendasikan pada sisa ~50% Pretrain):**
>    Jalankan `bash scripts/apply_midtrain_addons.sh /kaggle/input/datasets/naufalrachmat/corpus-pretrain-updated-1/corpus_pretrain_v8 --kaggle-overlay /kaggle/working/corpus_v8_midtrain` sebelum `--resume` pretrain agar fase penurunan *cosine LR* menyerap 8 shard mid-train (`411 MB`: `GSM8K-JA`, `OpenMathInstruct-JA`, `MetaMathQA-EN`, `Phi-1 Python Textbook`, `OSS Code`, `OPUS KFTT`, `SciQ`, `JCommonsenseQA`, `Dolly-15K` EN+JA) tanpa menyalin `26.8 GB` korpus read-only.
> 2. **SFT v4 Ultra-Max (`sft_data/sft_mixed_enja`):** `631,727` train / `6,000` eval (`25.48M` *masked tokens*: 33% Translasi, 24% Fakta+QA, 23% Math EN+JA Option B1, 20% Coding `max_seq=1024`; `99.02% <= 512` token). Pada Tesla T4 (`14.6 GB`), gunakan `--sft_batch_size 4`.
> 3. **DPO Ceiling (`sft_data/dpo_ceiling_enja`):** `22,999` *matched-length pairs* (`p50=1.00`, 4 domain penuh). Seluruh *bug* historis DPO (`BOS` menimpa *embedding* `SYSTEM/USER/ASSISTANT`, skalar *batch* `masked_log_probs`, dan *prefix-truncation*) telah diperbaiki. Pada Tesla T4 (`14.6 GB`), gunakan `--dpo_batch_size 2` (`--dpo_lr 5e-6`, `1` epoch).
> 4. **Stage 5 Verifiable RL (`rl_data/ceiling_enja`):** `12,498` task eksekusi kode & matematika (`GSM8K` EN+JA, `MBPP`, `Phi-1` unit-tests, `code_stdin`). Jalankan `bash scripts/run_rl.sh --ddp` (otomatis mendeteksi VRAM T4 -> `batch_size=1, n_samples=4, max_seq=768, rl_max_new=384`).

## Cell 1: Copy Source Code

```python
%%bash
SRC=/kaggle/input/datasets/naufalrachmat/transformerarchitecture17
DST=/kaggle/working/transformer_500m_cuda
mkdir -p $DST
cp -r $SRC/bpe         $DST/
cp -r $SRC/cuda        $DST/
cp -r $SRC/scripts     $DST/
cp -r $SRC/tools       $DST/
cp -r $SRC/transformer $DST/
cp -r $SRC/vocab_9     $DST/
cp -r $SRC/infer_cpp   $DST/
cp -r $SRC/rl_data     $DST/ 2>/dev/null || true
cp $SRC/gdrive_token.json $DST/ 2>/dev/null || true
cp $SRC/bpe_v9.bin     $DST/
cp $SRC/config_*.json  $DST/
cp $SRC/CMakeLists.txt $DST/
cp $SRC/config.ini     $DST/

# Copy SFT data (translation-focused + sources)
cp -r /kaggle/input/datasets/naufalrachmat/sftdata/sft_data $DST/

# Copy checkpoints
mkdir -p /kaggle/working/checkpoints
cp -r /kaggle/input/datasets/naufalrachmat/ckptlatest2/* /kaggle/working/checkpoints/
cp -r /kaggle/input/datasets/naufalrachmat/sftckptlatest/* /kaggle/working/checkpoints/

echo "Copied to $DST"
ls $DST
echo "---"
echo "Checkpoints:"
ls -lh /kaggle/working/checkpoints/
echo "---"
echo "SFT data:"
ls -lh $DST/sft_data/
```

## Cell 2: Build All Binaries

```python
%%bash
cd /kaggle/working/transformer_500m_cuda
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DUSE_DDP=ON 2>&1 | tail -15
cmake --build . -j$(nproc) 2>&1 | tail -30

echo ""
echo "Built binaries:"
ls -lh train_pretrain_cuda train_sft_cuda

# grad_check — kernel correctness (WAJIB sebelum training)
echo ""
echo "Running grad_check..."
cmake --build . --target grad_check -j$(nproc) 2>&1 | tail -5
./grad_check
```

## Cell 3: Build Infer Binary

```python
import subprocess, shutil, os

infer_build = '/kaggle/working/transformer_500m_cuda/build_infer'
if os.path.exists(infer_build):
    shutil.rmtree(infer_build)
os.makedirs(infer_build, exist_ok=True)

subprocess.run(
    ['cmake', '/kaggle/working/transformer_500m_cuda/infer_cpp',
     '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=75'],
    cwd=infer_build, check=True
)
subprocess.run(['make', f'-j{os.cpu_count()}'], cwd=infer_build, check=True)
print("✓ Infer binary built")
```

## Cell 4: Run Full Pipeline (Pretrain → SFT)

> Setelah pretrain selesai, SFT dijalankan dengan data terjemahan bersih
> (`sft_data/sft_translation_focused`). DPO di-skip (lihat catatan atas).
>
> ⚠️ **PENTING:** Karena pipeline `--auto_continue` memakai `--skip_done`
> (default True), SFT akan **DILEWATI** kalau `checkpoints/ckpt_sft_final.safetensors`
> sudah ada (mis. ikut ter-copy dari `sftckptlatest` di Cell 1).
> Untuk **memaksakan re-run SFT** dengan data terjemahan baru, hapus dulu:
> ```bash
> rm -f /kaggle/working/checkpoints/ckpt_sft_final.safetensors \
>       /kaggle/working/checkpoints/train_state_sft.txt
> ```

```python
import subprocess, os

env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)
env['NCCL_P2P_DISABLE'] = '1'
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)

proc = subprocess.Popen(
    ['python3', 'scripts/train_pipeline.py',
     '--auto_continue',
     '--stages',            'pretrain,sft,dpo',
     '--checkpoint_dir',    '/kaggle/working/checkpoints',
     '--data_dir',          '/kaggle/input/datasets/naufalrachmat/corpus-pretrain-updated-1/corpus_pretrain_v8',
     '--vocab_dir',         'vocab_9',
     '--bpe',               'bpe_v9.bin',
     '--arch_config',       'config_500m.json',
     # Pretrain (Tesla T4 14.6 GB x2: batch=7, seq=512 -> RoPE table tetap 2048)
     '--batch_size',        '7',
     '--seq',               '512',
     '--max_tokens',        '10000000000',
     '--lr',                '1.0e-4',
     '--grad_clip',         '0.5',
     '--loss_scale',        '4096',
     '--safety',            'light',
     '--save_every',        '200',
     '--log_every',         '5',
     '--upload_every',      '400',
     '--eval_every_tokens', '200000000',
     '--eval_infer_bin',    './build_infer/infer',
     # SFT v4 Ultra-Max (631.7K rows, max_seq=1024, dynamic batch S; T4 safe batch=4)
     '--sft_dir',           'sft_data/sft_mixed_enja',
     '--sft_lr',            '2e-5',
     '--sft_epochs',        '2',
     '--sft_batch_size',    '4',
     '--sft_max_seq',       '1024',
     '--sft_warmup',        '200',
     '--sft_save_every',    '300',
     '--sft_upload_every',  '600',
     # DPO Ceiling (23K pairs, per-sample kernel + matched-length; T4 safe batch=2)
     '--dpo_dir',           'sft_data/dpo_ceiling_enja',
     '--dpo_lr',            '5e-6',
     '--dpo_epochs',        '1',
     '--dpo_batch_size',    '2',
     '--dpo_max_seq',       '1024',
     '--dpo_beta',          '0.1',
     '--dpo_max_margin',    '4.0',
     # Infra
     '--ddp', '--num_gpus', '2',
     '--resume'],
    cwd='/kaggle/working/transformer_500m_cuda',
    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    encoding='utf-8', errors='replace', bufsize=1,
)
for line in proc.stdout:
    print(line, end='', flush=True)
proc.wait()
print(f"\n[EXIT CODE: {proc.returncode}]")
if proc.returncode == 2:
    print("[!] Training stopped due to CUDA OOM.")
```

## Cell 5: Stage 5 — Verifiable RL / GRPO-lite (Coding + Math EN/JA)

```bash
cd /kaggle/working/transformer_500m_cuda
bash scripts/run_rl.sh --ddp
```
