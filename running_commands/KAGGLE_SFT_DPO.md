# KAGGLE — SFT → DPO (Auto-Continue, skip pretrain)

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

# Copy SFT/DPO data
cp -r /kaggle/input/datasets/naufalrachmat/sftdata/sft_data $DST/

# Copy checkpoints (pretrain final + SFT resume)
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

## Cell 2: Build

```python
%%bash
cd /kaggle/working/transformer_100m_cuda
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75 -DUSE_DDP=ON 2>&1 | tail -15
cmake --build . -j$(nproc) 2>&1 | tail -30
ls -lh train_sft_cuda
cmake --build . --target grad_check -j$(nproc) 2>&1 | tail -5
./grad_check
```

## Cell 3: Run SFT → DPO (Auto-Continue)

> Requires `ckpt_pretrain_final.safetensors` in checkpoint_dir.

```python
import subprocess, os

env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)
env['NCCL_P2P_DISABLE'] = '1'

proc = subprocess.Popen(
    ['python3', 'scripts/train_pipeline.py',
     '--auto_continue',
     '--stages',            'sft,dpo',
     '--checkpoint_dir',    '/kaggle/working/checkpoints',
     '--vocab_dir',         'vocab_8',
     '--bpe',               'bpe_v8.bin',
     '--log_every',         '5',
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
```
