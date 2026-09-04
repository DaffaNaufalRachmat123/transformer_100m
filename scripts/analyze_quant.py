#!/usr/bin/env python3
"""analyze_quant.py — Quantization sensitivity probe for the 204M checkpoints.

For each tensor class (embed / wq / wk / wv / wo / w1 / w2 / w3) it simulates
block-wise Q8_0 and Q4_0 round-trip and reports relative RMS error, so the
mixed-precision recipe is chosen from data rather than guesswork.

Also reports the L2 distance between two checkpoints, which tells us whether a
weight-space merge ("model soup") of the translation and facts SFT runs is
even plausible.

Usage:
  python3 scripts/analyze_quant.py \
      --a translation_resources/ckpt_sft_final.safetensors \
      --b translation_resources/facts/ckpt_sft_final.safetensors
"""

import argparse
import json
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_st(path):
    """Minimal safetensors reader -> {name: (numpy_array_fp32, shape)}."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            s, e = meta["data_offsets"]
            f.seek(base + s)
            buf = f.read(e - s)
            dt = {"F16": np.float16, "F32": np.float32, "BF16": np.uint16}[meta["dtype"]]
            arr = np.frombuffer(buf, dtype=dt)
            if meta["dtype"] == "BF16":
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
            out[name] = arr.astype(np.float32).reshape(meta["shape"])
    return out


def quant_rt(w, bits, block=32, asym=False):
    """Block-wise round-trip quantize/dequantize. Returns dequantized array."""
    flat = w.reshape(-1).astype(np.float32)
    pad = (-flat.size) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.float32)])
    b = flat.reshape(-1, block)
    if asym:  # Q4_1 style: scale + min
        mn = b.min(1, keepdims=True)
        mx = b.max(1, keepdims=True)
        levels = (1 << bits) - 1
        scale = np.where(mx - mn == 0, 1.0, (mx - mn) / levels)
        q = np.clip(np.rint((b - mn) / scale), 0, levels)
        deq = q * scale + mn
    else:     # Q*_0 style: symmetric absmax
        amax = np.abs(b).max(1, keepdims=True)
        half = (1 << (bits - 1)) - 1
        scale = np.where(amax == 0, 1.0, amax / half)
        q = np.clip(np.rint(b / scale), -half - 1, half)
        deq = q * scale
    return deq.reshape(-1)[: w.size].reshape(w.shape)


def rel_rms(w, dq):
    d = w - dq
    den = np.sqrt(np.mean(w.astype(np.float64) ** 2)) + 1e-12
    return float(np.sqrt(np.mean(d.astype(np.float64) ** 2)) / den)


def tensor_class(name):
    if name.startswith("embed"):
        return "embed"
    m = re.match(r"layer\.\d+\.(\w+)\.weight", name)
    return m.group(1) if m else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="translation_resources/ckpt_sft_final.safetensors")
    ap.add_argument("--b", default="translation_resources/facts/ckpt_sft_final.safetensors")
    ap.add_argument("--layers", type=int, default=3, help="layers sampled per class")
    args = ap.parse_args()

    A = load_st(ROOT / args.a)
    print(f"loaded A: {args.a}  ({len(A)} tensors)\n")

    bits_per_w = {"Q8_0": 8 + 16 / 32, "Q4_0": 4 + 16 / 32, "Q4_1": 4 + 32 / 32}

    # ── per-class quantization error ────────────────────────────────────
    groups = defaultdict(list)
    for name in A:
        groups[tensor_class(name)].append(name)

    print(f"{'class':<10}{'shape':>14}{'params':>13}{'absmax':>9}"
          f"{'kurt':>8}{'Q8_0':>9}{'Q4_0':>9}{'Q4_1':>9}")
    print("-" * 84)
    order = ["embed", "wq", "wk", "wv", "wo", "w1", "w2", "w3", "rms_attn", "rms_ffn"]
    totals = {}
    for cls in order:
        names = sorted(groups.get(cls, []))[: args.layers] if cls != "embed" else groups.get(cls, [])
        if not names:
            continue
        e8 = e4 = e41 = 0.0
        kurt = amax = 0.0
        for nm in names:
            w = A[nm]
            e8 += rel_rms(w, quant_rt(w, 8))
            e4 += rel_rms(w, quant_rt(w, 4))
            e41 += rel_rms(w, quant_rt(w, 4, asym=True))
            x = w.reshape(-1).astype(np.float64)
            sd = x.std() + 1e-12
            kurt += float(np.mean(((x - x.mean()) / sd) ** 4))
            amax += float(np.abs(w).max())
        k = len(names)
        total_params = sum(A[n].size for n in groups[cls])
        totals[cls] = total_params
        print(f"{cls:<10}{str(list(A[names[0]].shape)):>14}{total_params:>13,}"
              f"{amax/k:>9.3f}{kurt/k:>8.1f}{e8/k:>9.4f}{e4/k:>9.4f}{e41/k:>9.4f}")

    # ── size table ──────────────────────────────────────────────────────
    n_all = sum(a.size for a in A.values())
    n_embed = totals.get("embed", 0)
    n_rest = n_all - n_embed
    print(f"\ntotal params: {n_all:,}   embed: {n_embed:,} ({100*n_embed/n_all:.1f}%)")
    print(f"\n{'scheme':<26}{'MB':>10}")
    print("-" * 36)
    print(f"{'FP16 (current)':<26}{n_all*2/2**20:>10.1f}")
    for nm, bpw in bits_per_w.items():
        print(f"{'all ' + nm:<26}{n_all*bpw/8/2**20:>10.1f}")
    mix = (n_embed * bits_per_w["Q8_0"] + n_rest * bits_per_w["Q4_0"]) / 8 / 2**20
    print(f"{'embed Q8_0 + rest Q4_0':<26}{mix:>10.1f}   <- recommended")
    mix2 = (n_embed * bits_per_w["Q8_0"] + n_rest * bits_per_w["Q4_1"]) / 8 / 2**20
    print(f"{'embed Q8_0 + rest Q4_1':<26}{mix2:>10.1f}")

    # ── checkpoint distance (merge feasibility) ─────────────────────────
    pb = ROOT / args.b
    if pb.exists():
        B = load_st(pb)
        print(f"\ncheckpoint distance  A={Path(args.a).parent.name}/{Path(args.a).name}"
              f"  B={Path(args.b).parent.name}/{Path(args.b).name}")
        print(f"{'class':<10}{'rel L2 |A-B|/|A|':>20}{'cos sim':>10}")
        print("-" * 40)
        for cls in order:
            names = sorted(groups.get(cls, []))[:6]
            if not names or names[0] not in B:
                continue
            rl = cs = 0.0
            for nm in names:
                a, b = A[nm].reshape(-1), B[nm].reshape(-1)
                rl += float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
                cs += float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            print(f"{cls:<10}{rl/len(names):>20.4f}{cs/len(names):>10.4f}")


if __name__ == "__main__":
    main()
