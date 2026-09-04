#!/usr/bin/env python3
"""Validate an architecture JSON against the C++ trainer's expectations.

Mirrors ModelConfig::param_count() in transformer/model_config.h exactly, and
adds the T4 capacity numbers that decide whether a config is trainable at all:
optimizer/master-weight VRAM, per-sample activation cost, and the resulting
batch ceiling. Throughput is extrapolated from the measured 204M baseline
(10.5K tok/s on Kaggle T4 x2 @ seq=256, see loss_log.txt).

  python3 scripts/check_arch.py config_492m.json
  python3 scripts/check_arch.py config_*.json --seq 256 --vram 14.6
"""
import argparse
import json
import sys
from pathlib import Path

# Measured on Kaggle T4 x2 DDP, FP16 + grad checkpointing, seq=256, batch=35/GPU.
BASE_PARAMS = 204_304_128
BASE_TOK_S = 10_450
BASE_VRAM_ACT_GB = 9.7   # 13.0 GB observed minus 3.3 GB static
BASE_BATCH = 35
BASE_D_MODEL = 768
BASE_LAYERS = 18


def param_count(c):
    """Byte-for-byte port of ModelConfig::param_count()."""
    v, d, L = c["vocab_size"], c["d_model"], c["n_layers"]
    q_dim = c["n_q_heads"] * c["d_head"]
    kv_dim = c["n_kv_heads"] * c["d_head"]
    ffn = c["ffn_dim"]
    embed = v * d
    per_layer = (
        d * q_dim + d * kv_dim + d * kv_dim + q_dim * d   # wq wk wv wo
        + d * ffn + ffn * d + d * ffn                     # w1 w2 w3
        + d * 2                                           # rms_attn rms_ffn
    )
    return embed + per_layer * L + d


def validate(c):
    """Same checks as arch::load() in transformer/arch_runtime.h."""
    errs = []
    if c["n_q_heads"] * c["d_head"] != c["d_model"]:
        errs.append(
            f"n_q_heads({c['n_q_heads']}) * d_head({c['d_head']}) "
            f"!= d_model({c['d_model']})"
        )
    if c["n_kv_heads"] == 0 or c["n_q_heads"] % c["n_kv_heads"] != 0:
        errs.append(
            f"n_q_heads({c['n_q_heads']}) not divisible by "
            f"n_kv_heads({c['n_kv_heads']})"
        )
    if c["d_head"] % 2:
        errs.append(f"d_head({c['d_head']}) must be even (RoPE rotates pairs)")
    return errs


def report(path, seq, vram_gb):
    c = json.loads(Path(path).read_text())
    errs = validate(c)
    n = param_count(c)

    print(f"\n=== {path} ===")
    print(f"  D={c['d_model']} L={c['n_layers']} QH={c['n_q_heads']} "
          f"KVH={c['n_kv_heads']} DH={c['d_head']} FFN={c['ffn_dim']} "
          f"V={c['vocab_size']}")
    print(f"  params          : {n/1e6:.1f}M")

    # AdamW keeps FP32 master + m + v alongside FP16 weights and grads.
    static_gb = n * (2 + 4 + 4 + 4 + 2) / 1024**3
    print(f"  static VRAM     : {static_gb:.2f} GB  (fp16 w+g, fp32 master+m+v)")

    # Activations scale with width x depth; logits scale with vocab.
    act_scale = (c["d_model"] / BASE_D_MODEL) * (c["n_layers"] / BASE_LAYERS)
    logits_gb = seq * c["vocab_size"] * 6 / 1024**3          # fp16 + fp32 softmax
    act_per_sample = (BASE_VRAM_ACT_GB / BASE_BATCH) * act_scale * (seq / 256)
    per_sample = act_per_sample + logits_gb
    budget = vram_gb - static_gb
    batch = int(budget / per_sample) if per_sample > 0 else 0

    print(f"  per-sample act  : {per_sample*1024:.0f} MB @ seq={seq}")
    if batch < 1:
        print(f"  batch/GPU       : DOES NOT FIT in {vram_gb} GB")
    else:
        print(f"  batch/GPU       : ~{batch} (in {budget:.2f} GB free)")

    tok_s = BASE_TOK_S * BASE_PARAMS / n
    print(f"  est throughput  : {tok_s/1000:.1f}K tok/s (T4 x2, seq=256)")
    for budget_tok in (3e9, 6e9, 10e9):
        hours = budget_tok / tok_s / 3600
        print(f"    {budget_tok/1e9:>4.0f}B tokens : {hours:5.0f} GPU-hours "
              f"= {hours/30:4.1f} weeks of Kaggle quota")

    if errs:
        for e in errs:
            print(f"  INVALID: {e}")
        return False
    print("  OK")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--vram", type=float, default=14.6,
                    help="usable VRAM per GPU in GB (T4 reports 14.6)")
    a = ap.parse_args()
    ok = all(report(p, a.seq, a.vram) for p in a.configs)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
