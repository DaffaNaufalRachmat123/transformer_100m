#!/usr/bin/env python3
"""convert_to_gguf.py — Convert a trilingual transformer checkpoint to GGUF.

Designed so swapping checkpoints is a one-flag change:

    python3 scripts/convert_to_gguf.py --model path/to/any_ckpt.safetensors

The architecture (d_model, layers, heads, ffn) is DETECTED from tensor shapes,
so a future checkpoint that does facts + translation — or a differently sized
one — converts with no code edits. Only values that cannot be inferred from
weights (rope_theta, d_head, context length, eps, special token ids) come from
the config JSON, and each has a sane default.

Run `--inspect` to see what would be produced without writing anything.

Layout notes (verified against transformer/transformer.cpp):
  * Linear weights are stored [D_in, D_out] because the runtime computes
    y = x @ W.  GGUF/ggml needs [D_out, D_in], so linears are TRANSPOSED.
  * embed.weight is already [vocab, d_model] == token_embd layout, no transpose.
  * RoPE is interleaved (head[2j], head[2j+1]) == ggml rope type NORM, which is
    what LLM_ARCH_LLAMA uses. So the HuggingFace q/k permutation must NOT be
    applied here.
  * weight_tying=True -> no `output.weight`; llama.cpp reuses token_embd.
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("missing dependency: pip install gguf")

ROOT = Path(__file__).resolve().parent.parent

# Source tensor suffix -> (GGUF name template, needs_transpose)
# Everything the converter knows about lives in this one table.
LINEAR_MAP = {
    "wq": ("blk.{i}.attn_q.weight",      True),
    "wk": ("blk.{i}.attn_k.weight",      True),
    "wv": ("blk.{i}.attn_v.weight",      True),
    "wo": ("blk.{i}.attn_output.weight", True),
    # SwiGLU operand order is SWAPPED vs llama. transformer.cpp computes
    #   ff = w1(x) * silu(w3(x))
    # while llama.cpp computes
    #   ff = silu(ffn_gate(x)) * ffn_up(x)
    # SiLU is applied to w3 here, so w3 -> ffn_gate and w1 -> ffn_up.
    "w1": ("blk.{i}.ffn_up.weight",      True),
    "w2": ("blk.{i}.ffn_down.weight",    True),
    "w3": ("blk.{i}.ffn_gate.weight",    True),
    "rms_attn": ("blk.{i}.attn_norm.weight", False),
    "rms_ffn":  ("blk.{i}.ffn_norm.weight",  False),
}
GLOBAL_MAP = {
    "embed.weight":     ("token_embd.weight",  False),
    "final_rms.weight": ("output_norm.weight", False),
}

# Defaults for anything not recoverable from the weights themselves.
DEFAULTS = {
    "d_head": 64,
    "rope_theta": 500000.0,
    "max_seq_len": 2048,
    "rms_eps": 1e-6,          # transformer.cpp: const float eps = 1e-6f
    "bos_token_id": 0,        # <|endoftext|>
    "eos_token_id": 4,        # <|end|>
    "pad_token_id": 15,       # <|pad|>
    "n_special": 16,          # ids 0..15 are <|...|> control tokens
}


# ── safetensors ─────────────────────────────────────────────────────────────

def read_safetensors(path):
    """Minimal safetensors reader -> {name: np.ndarray(float32)}."""
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len
        tensors = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            f.seek(data_start + start)
            raw = f.read(end - start)
            dtype = {"F16": np.float16, "F32": np.float32,
                     "BF16": np.uint16}[meta["dtype"]]
            arr = np.frombuffer(raw, dtype=dtype)
            if meta["dtype"] == "BF16":
                arr = (arr.astype(np.uint32) << 16).view(np.float32)
            tensors[name] = arr.astype(np.float32).reshape(meta["shape"])
    return tensors


# ── architecture detection ──────────────────────────────────────────────────

def detect_arch(tensors, cfg):
    """Infer model geometry from tensor shapes; cfg only fills the rest."""
    if "embed.weight" not in tensors:
        sys.exit("checkpoint has no 'embed.weight' - unsupported layout")

    vocab_size, d_model = tensors["embed.weight"].shape
    layer_ids = {int(m.group(1)) for n in tensors
                 if (m := re.match(r"layer\.(\d+)\.", n))}
    if not layer_ids:
        sys.exit("no 'layer.N.*' tensors found - unsupported layout")
    n_layers = max(layer_ids) + 1

    d_head = int(cfg.get("d_head", DEFAULTS["d_head"]))
    q_dim = tensors["layer.0.wq.weight"].shape[1]
    kv_dim = tensors["layer.0.wk.weight"].shape[1]
    ffn_dim = tensors["layer.0.w1.weight"].shape[1]

    for label, dim in (("q_dim", q_dim), ("kv_dim", kv_dim)):
        if dim % d_head:
            sys.exit(f"{label}={dim} not divisible by d_head={d_head}; "
                     f"pass the right --config")

    return {
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "n_layers": int(n_layers),
        "n_q_heads": q_dim // d_head,
        "n_kv_heads": kv_dim // d_head,
        "d_head": d_head,
        "ffn_dim": int(ffn_dim),
        "n_params": sum(t.size for t in tensors.values()),
        "context_length": int(cfg.get("max_seq_len", DEFAULTS["max_seq_len"])),
        "rope_theta": float(cfg.get("rope_theta", DEFAULTS["rope_theta"])),
        "rms_eps": float(cfg.get("rms_eps", DEFAULTS["rms_eps"])),
        "bos": int(cfg.get("bos_token_id", DEFAULTS["bos_token_id"])),
        "eos": int(cfg.get("eos_token_id", DEFAULTS["eos_token_id"])),
        "pad": int(cfg.get("pad_token_id", DEFAULTS["pad_token_id"])),
        "weight_tying": bool(cfg.get("weight_tying", True)),
    }


# ── tokenizer ───────────────────────────────────────────────────────────────

def read_vocab(vocab_dir, n_special):
    """Load vocab.txt + merges.txt, mirroring BPEEncoder::load_text exactly.

    The C++ loader drops every line whose first char is '#' -- not just the
    '#version:' header but also 42 real merges such as '# f' and '# #'. The
    trained model therefore only ever saw 65222 merges, so we must drop them
    here too or the GGUF vocab would not match the weights.
    """
    vocab_dir = Path(vocab_dir)
    raw = (vocab_dir / "vocab.txt").read_text(encoding="utf-8")
    tokens = raw.split("\n")
    if tokens and tokens[-1] == "":
        tokens.pop()

    merges = []
    for line in (vocab_dir / "merges.txt").read_text(encoding="utf-8").split("\n"):
        if not line or line[0] == "#":
            continue
        if " " in line:
            merges.append(line)

    types = [gguf.TokenType.CONTROL if i < n_special else gguf.TokenType.NORMAL
             for i in range(len(tokens))]
    return tokens, merges, types


# ── conversion ──────────────────────────────────────────────────────────────

def build_tensor_list(tensors, arch):
    """Yield (gguf_name, ndarray) in GGUF layout. Raises on anything unmapped."""
    out = []
    consumed = set()

    for src, (dst, transpose) in GLOBAL_MAP.items():
        if src not in tensors:
            if src == "final_rms.weight":
                sys.exit("missing final_rms.weight")
            continue
        arr = tensors[src]
        out.append((dst, arr.T.copy() if transpose else arr))
        consumed.add(src)

    for i in range(arch["n_layers"]):
        for suffix, (tmpl, transpose) in LINEAR_MAP.items():
            src = f"layer.{i}.{suffix}.weight"
            if src not in tensors:
                sys.exit(f"missing tensor {src}")
            arr = tensors[src]
            out.append((tmpl.format(i=i), arr.T.copy() if transpose else arr))
            consumed.add(src)

    leftover = set(tensors) - consumed
    if leftover:
        sys.exit(f"unmapped tensors (update LINEAR_MAP/GLOBAL_MAP): "
                 f"{sorted(leftover)[:8]}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Convert a trilingual transformer checkpoint to GGUF.")
    ap.add_argument("--model", default="translation_resources/facts/ckpt_sft_final.safetensors",
                    help="source .safetensors checkpoint (swap this to change models)")
    ap.add_argument("--config", default="config_204m.json",
                    help="JSON with rope_theta/d_head/max_seq_len/special ids")
    ap.add_argument("--vocab", default="vocab_6", help="dir with vocab.txt + merges.txt")
    ap.add_argument("--outfile", default="", help="output .gguf (default: auto-named)")
    ap.add_argument("--outtype", default="f16", choices=["f16", "f32"])
    ap.add_argument("--name", default="", help="model name recorded in GGUF")
    ap.add_argument("--inspect", action="store_true",
                    help="print detected architecture and exit without writing")
    args = ap.parse_args()

    model_path = (ROOT / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    if not model_path.exists():
        sys.exit(f"model not found: {model_path}")

    cfg_path = (ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    if not cfg:
        print(f"[warn] no config at {cfg_path}; using defaults {DEFAULTS}")

    print(f"reading {model_path.name} …")
    tensors = read_safetensors(model_path)
    arch = detect_arch(tensors, cfg)

    print("\ndetected architecture")
    print("─" * 46)
    for k in ("vocab_size", "d_model", "n_layers", "n_q_heads", "n_kv_heads",
              "d_head", "ffn_dim", "context_length", "rope_theta", "rms_eps",
              "weight_tying"):
        print(f"  {k:<16} {arch[k]}")
    print(f"  {'n_params':<16} {arch['n_params']:,}")
    print(f"  {'n_tensors':<16} {len(tensors)}")
    print("─" * 46)

    vocab_dir = (ROOT / args.vocab) if not Path(args.vocab).is_absolute() else Path(args.vocab)
    tokens, merges, types = read_vocab(vocab_dir, DEFAULTS["n_special"])
    print(f"tokenizer: {len(tokens)} tokens, {len(merges)} merges")
    if len(tokens) != arch["vocab_size"]:
        sys.exit(f"vocab.txt has {len(tokens)} tokens but embed expects "
                 f"{arch['vocab_size']} - wrong --vocab dir?")

    if args.inspect:
        print("\n--inspect: nothing written.")
        return

    stem = args.name or f"{model_path.parent.name}-{arch['n_params'] // 10**6}m"
    out = Path(args.outfile) if args.outfile else ROOT / "gguf" / f"{stem}-{args.outtype}.gguf"
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(str(out), "llama")
    writer.add_name(stem)
    writer.add_context_length(arch["context_length"])
    writer.add_embedding_length(arch["d_model"])
    writer.add_block_count(arch["n_layers"])
    writer.add_feed_forward_length(arch["ffn_dim"])
    writer.add_head_count(arch["n_q_heads"])
    writer.add_head_count_kv(arch["n_kv_heads"])
    writer.add_rope_dimension_count(arch["d_head"])
    writer.add_rope_freq_base(arch["rope_theta"])
    writer.add_layer_norm_rms_eps(arch["rms_eps"])
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16 if args.outtype == "f16"
                         else gguf.LlamaFileType.ALL_F32)

    # NOTE: the runtime tokenizer is NOT GPT-2 compatible (whitespace
    # pre-tokenizer + longest-match seeding). "gpt2" is recorded so llama.cpp
    # loads and detokenizes correctly; prompts must be ENCODED with
    # scripts/bpe_reference.py or the Kotlin/Swift port.
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(arch["bos"])
    writer.add_eos_token_id(arch["eos"])
    writer.add_pad_token_id(arch["pad"])
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)

    np_dtype = np.float16 if args.outtype == "f16" else np.float32
    total = 0
    for name, arr in build_tensor_list(tensors, arch):
        data = arr.astype(np.float32 if arr.ndim == 1 else np_dtype)
        writer.add_tensor(name, data)
        total += data.size

    print(f"\nwriting {out} …")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    size_mb = out.stat().st_size / 2 ** 20
    print(f"\ndone: {out}  ({size_mb:.1f} MB, {total:,} values)")
    print("\nnext:")
    print(f"  llama-cli -m {out} -p '<|system|>...' -n 32 --temp 0")


if __name__ == "__main__":
    main()
