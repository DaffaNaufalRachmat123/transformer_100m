# KAGGLE — DPO Only (Single Stage)

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

# Copy SFT/DPO data
cp -r /kaggle/input/datasets/naufalrachmat/sftdata/sft_data $DST/

# Copy checkpoints (pretrain final + SFT final for DPO)
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

## Cell 3: Run DPO Only

> Requires `ckpt_sft_final.safetensors` (or latest sft step checkpoint) in checkpoint_dir.
> Uses `ckpt_pretrain_final.safetensors` as frozen reference model.

```python
import subprocess, os

env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)
env['NCCL_P2P_DISABLE'] = '1'

proc = subprocess.Popen(
    ['python3', 'scripts/train_pipeline.py',
     '--stages',            'dpo',
     '--checkpoint_dir',    '/kaggle/working/checkpoints',
     '--vocab_dir',         'vocab_6',
     '--bpe',               'bpe_v6.bin',
     '--log_every',         '5',
     '--dpo_dir',           'sft_data/dpo',
     '--dpo_lr',            '1e-5',
     '--dpo_epochs',        '3',
     '--dpo_beta',          '0.1',
     '--dpo_batch_size',    '4',
     '--dpo_max_seq',       '512',
     '--dpo_warmup',        '50',
     '--dpo_save_every',    '200',
     '--dpo_upload_every',  '200',
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
