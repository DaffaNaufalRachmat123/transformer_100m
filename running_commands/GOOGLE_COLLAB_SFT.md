# GOOGLE COLAB — SFT Only (Single Stage)

## Cell 1–5: Same as GOOGLE_COLLAB_FULL_TRAINING.md (Mount, Paths, Extract, Symlink, Build)

See `GOOGLE_COLLAB_FULL_TRAINING.md` cells 1–5.

## Cell 6: Run SFT Only

> Requires `ckpt_pretrain_final.safetensors` in checkpoint_dir (Google Drive).

```python
import subprocess, os

WORK_DIR = "/content/transformer_100m_cuda"
env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)

proc = subprocess.Popen(
    ['python3', 'scripts/train_pipeline.py',
     '--stages',            'sft',
     '--checkpoint_dir',    'checkpoints',
     '--vocab_dir',         'vocab_6',
     '--bpe',               'bpe_v6.bin',
     '--log_every',         '5',
     '--sft_dir',           'sft_data/sft_translation_focused',
     '--sft_lr',            '3e-5',
     '--sft_epochs',        '1',
     '--sft_batch_size',    '8',
     '--sft_max_seq',       '512',
     '--sft_warmup',        '100',
     '--sft_save_every',    '200',
     '--sft_upload_every',  '200',
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
```
