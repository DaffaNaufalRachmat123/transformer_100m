# GOOGLE COLAB — SFT Only (Facts, Single Stage)

## Cell 1–5: Same as GOOGLE_COLLAB_FULL_TRAINING.md (Mount, Paths, Extract, Symlink, Build)

See `GOOGLE_COLLAB_FULL_TRAINING.md` cells 1–5.

> Cell 4 symlinks `checkpoints_facts` → `checkpoints_facts_100m` on Drive —
> make sure that cell ran so the facts checkpoint dir exists.

## Cell 6: Generate + Audit Facts Dataset (V9)

> Generates `sft_data/sft_facts_v9_500k` (500K rows, 1,501 clean unique facts)
> if not already present, then audits it. Audit must PASS before training.
> Requires `data/wikidata_facts_all.tsv` + `data/fact_overrides.json` in the
> extracted source (both ship inside the zip).

```python
import subprocess, os

WORK_DIR = "/content/transformer_100m_cuda"
env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)

# Skip generation if the dataset already exists on Drive (symlinked sft_data)
if not os.path.exists(f"{WORK_DIR}/sft_data/sft_facts_v9_500k/train.jsonl"):
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
else:
    print("✓ sft_facts_v9_500k already present, skipping generation")

# Audit (must PASS)
proc_audit = subprocess.run(
    ['python3', 'scripts/audit_facts_dataset.py',
     'sft_data/sft_facts_v9_500k/train.jsonl',
     'sft_data/sft_facts_v9_500k/eval.jsonl',
     '--max_category_share', '0.30'],
    cwd=WORK_DIR, env=env,
    capture_output=True, text=True,
)
print(proc_audit.stdout)
if proc_audit.returncode != 0:
    print("❌ AUDIT FAILED — do not train on this dataset")
```

## Cell 7: Run Facts SFT Only

> Requires `checkpoints/ckpt_pretrain_final.safetensors` (the 3.85B-token
> pretrain checkpoint) on Drive. 20K steps, batch 20, LR 1e-5.
> Checkpoint output goes to `checkpoints_facts/` (symlinked to Drive).

```python
import subprocess, os

WORK_DIR = "/content/transformer_100m_cuda"
env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)

proc = subprocess.Popen(
    ['python3', 'scripts/train_pipeline.py',
     '--auto_continue',
     '--stages',            'sft',
     '--checkpoint_dir',    'checkpoints_facts',
     '--vocab_dir',         'vocab_8',
     '--bpe',               'bpe_v8.bin',
     '--log_every',         '5',
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
    print("\nFacts SFT interrupted. Re-run to resume (auto_continue).")
proc.wait()
print(f"\n[EXIT CODE: {proc.returncode}]")
```

## Cell 8: Evaluate Facts (honest metrics)

> Metrics now include **subj-bind** (subject-binding) and use word-boundary
> value matching. Numbers are LOWER than old pre-V9 reports by design —
> the old evaluator over-counted. Compare against the honest baseline below.

**Honest baseline (V8 checkpoint, 2B pretrain):**

| Metric | EN | JA |
|---|---:|---:|
| lang-ok | 100% | 100% |
| attr-ok | 20% | 60% |
| value-hit | 20% | 40% |
| subj-bind | 60% | 10% |

```python
%%bash
cd /content/transformer_100m_cuda

echo "=== Facts Eval (20 samples, quick look) ==="
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
```

## Cell 9: Copy Final Model to Active Locations

```python
%%bash
cd /content/transformer_100m_cuda

mkdir -p checkpoints translation_resources translation_resources/facts

cp checkpoints_facts/ckpt_sft_final.safetensors checkpoints/ckpt_sft_final.safetensors
cp checkpoints_facts/ckpt_sft_final.safetensors translation_resources/ckpt_sft_final.safetensors
cp checkpoints_facts/ckpt_sft_final.safetensors translation_resources/facts/ckpt_sft_final.safetensors

echo "✓ Final models copied"
ls -lh checkpoints/ckpt_sft_final.safetensors translation_resources/facts/ckpt_sft_final.safetensors
```

## Config Summary

| Property | Value |
|---|---|
| Base model | `ckpt_pretrain_final.safetensors` (3.85B tokens) |
| Dataset | `sft_data/sft_facts_v9_500k` (500K rows, 1,501 facts) |
| Steps | 20,000 |
| Batch / Seq | 20 / 256 |
| LR | 1e-5, warmup 100 |
| EN / JA share | 52% / 48% |
| Checkpoint dir | `checkpoints_facts` → Drive `checkpoints_facts_100m` |
| Est. time | ~3h on T4 x2 DDP, ~6h on single T4 |

## Translation SFT (optional, after facts)

To add translation on top of the facts model, see `GOOGLE_COLLAB_FULL_TRAINING.md` Cell 6C.
