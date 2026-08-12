# GOOGLE COLAB — DPO Only (Single Stage)

## Cell 1–5: Same as GOOGLE_COLLAB_FULL_TRAINING.md (Mount, Paths, Extract, Symlink, Build)

See `GOOGLE_COLLAB_FULL_TRAINING.md` cells 1–5.

## Cell 6: Run DPO Only

> Requires `ckpt_sft_final.safetensors` in checkpoint_dir.
> Uses `ckpt_pretrain_final.safetensors` as frozen reference model.

```python
import subprocess, os

WORK_DIR = "/content/transformer_100m_cuda"
env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)

proc = subprocess.Popen(
    ['python3', 'scripts/train_pipeline.py',
     '--stages',            'dpo',
     '--checkpoint_dir',    'checkpoints',
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
