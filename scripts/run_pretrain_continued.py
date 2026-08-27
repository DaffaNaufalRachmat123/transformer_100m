#!/usr/bin/env python3
"""run_pretrain_continued.py — Continue pretraining from 2B to 4.5B tokens.

This script resumes from the existing pretrain checkpoint and trains
2.5B more tokens with a lower learning rate (7e-5).

Usage on Kaggle:
    python3 scripts/run_pretrain_continued.py

Or copy the command into a Colab/Kaggle cell.
"""

import subprocess
import os
import sys

WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working/transformer_100m_cuda")
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input/datasets/naufalrachmat/datatraining/data/corpus_final")
CKPT_DIR = os.environ.get("CKPT_DIR", "/kaggle/working/checkpoints")

env = os.environ.copy()
env.pop("CUDA_VISIBLE_DEVICES", None)
env["NCCL_P2P_DISABLE"] = "1"
os.makedirs(CKPT_DIR, exist_ok=True)

cmd = [
    "python3", "scripts/train_pipeline.py",
    "--auto_continue",
    "--stages",            "pretrain",
    "--checkpoint_dir",    CKPT_DIR,
    "--data_dir",          DATA_DIR,
    "--vocab_dir",         "vocab_6",
    "--bpe",               "bpe_v6.bin",
    "--batch_size",        "35",
    "--max_seq",           "256",
    "--max_tokens",        "4500000000",
    "--lr",                "7e-5",
    "--grad_clip",         "5.0",
    "--loss_scale",        "4096",
    "--safety",            "light",
    "--save_every",        "200",
    "--log_every",         "5",
    "--upload_every",      "400",
    "--eval_every_tokens", "200000000",
    "--eval_infer_bin",    "./build_infer/infer",
    "--ddp", "--num_gpus", "2",
    "--resume",
]

print("=" * 60)
print("Continued Pretraining: 2B -> 4.5B tokens (+2.5B)")
print("=" * 60)
print(f"  Work dir:   {WORK_DIR}")
print(f"  Data dir:   {DATA_DIR}")
print(f"  Checkpoint: {CKPT_DIR}")
print(f"  Max tokens: 4.5B (cumulative)")
print(f"  LR:         7e-5")
print(f"  Batch:      35 x 256 x 2 GPU")
print(f"  Est. time:  ~10 hours")
print("=" * 60)
print()

proc = subprocess.Popen(
    cmd,
    cwd=WORK_DIR,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    encoding="utf-8",
    errors="replace",
    bufsize=1,
)

for line in proc.stdout:
    print(line, end="", flush=True)

proc.wait()
print(f"\n[EXIT CODE: {proc.returncode}]")
if proc.returncode == 2:
    print("[!] Training stopped due to CUDA OOM.")
    print("    Try reducing batch_size to 30 or max_seq to 192.")
