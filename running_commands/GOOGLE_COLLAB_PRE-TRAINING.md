# GOOGLE COLAB — Pre-Training Only (Single Stage)

## Cell 1–5: Same as GOOGLE_COLLAB_FULL_TRAINING.md (Mount, Paths, Extract, Symlink, Build)

See `GOOGLE_COLLAB_FULL_TRAINING.md` cells 1–5.

## Cell 6: Run Pre-Training Only

```python
import subprocess, os

WORK_DIR = "/content/transformer_100m_cuda"
env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)

proc = subprocess.Popen(
    ['python3', 'scripts/train_pipeline.py',
     '--stages',            'pretrain',
     '--checkpoint_dir',    'checkpoints',
     '--data_dir',          'data/corpus_final',
     '--vocab_dir',         'vocab_6',
     '--bpe',               'bpe_v6.bin',
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
