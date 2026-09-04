# KAGGLE — Full Training Pipeline (Pretrain → SFT)

> **Update (2026-08-09):** DPO dihapus dari pipeline.
> Data DPO EN↔JA yang tersedia sangat kecil & berkualitas rendah sehingga
> *menurunkan* kualitas model (margin meledak ke -131). Untuk model 892M
> yang fokus terjemahan EN↔JA, **SFT dengan data paralel bersih sudah cukup**.
> Lihat `scripts/run_enja_pipeline.sh` untuk versi ringkas.

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
cp -r $SRC/vocab_6     $DST/
cp -r $SRC/infer_cpp   $DST/
cp $SRC/gdrive_token.json $DST/
cp $SRC/bpe_v6.bin     $DST/
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
cd /kaggle/working/transformer_100m_cuda
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

infer_build = '/kaggle/working/transformer_100m_cuda/build_infer'
if os.path.exists(infer_build):
    shutil.rmtree(infer_build)
os.makedirs(infer_build, exist_ok=True)

subprocess.run(
    ['cmake', '/kaggle/working/transformer_100m_cuda/infer_cpp',
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
     '--stages',            'pretrain,sft',
     '--checkpoint_dir',    '/kaggle/working/checkpoints',
     '--data_dir',          '/kaggle/input/datasets/naufalrachmat/datatraining/data/corpus_final',
     '--vocab_dir',         'vocab_6',
     '--bpe',               'bpe_v6.bin',
     # Pretrain
     '--batch_size',        '35',
     '--max_seq',           '256',
     '--max_tokens',        '2000000000',
     '--lr',                '1.4e-4',
     '--grad_clip',         '5.0',
     '--loss_scale',        '4096',
     '--safety',            'light',
     '--save_every',        '200',
     '--log_every',         '5',
     '--upload_every',      '400',
     '--eval_every_tokens', '200000000',
     '--eval_infer_bin',    './build_infer/infer',
     # SFT (translation-focused, dari checkpoint pretrain)
     '--sft_dir',           'sft_data/sft_translation_focused',
     '--sft_lr',            '3e-5',
     '--sft_epochs',        '1',
     '--sft_batch_size',    '8',
     '--sft_max_seq',       '512',
     '--sft_warmup',        '100',
     '--sft_save_every',    '300',
     '--sft_upload_every',  '600',
     # Infra
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
    print("[!] Training stopped due to CUDA OOM.")
```
