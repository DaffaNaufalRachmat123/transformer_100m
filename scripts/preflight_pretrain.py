#!/usr/bin/env python3
"""preflight_pretrain.py — verify tokenizer + corpus before burning GPU hours.

Every check here exists because the corresponding failure was silent: a corpus
bucket whose only file was a dangling symlink, a merges.txt whose '#' rules were
dropped by the text loader, a --max_seq the dataset ignored. Run it before every
pretraining job.

    python3 scripts/preflight_pretrain.py \
        --vocab_dir vocab_7 --bpe bpe_v7.bin --data_dir data/corpus_pretrain_v8

Exit code 0 = safe to train, 1 = at least one check failed.
"""

import argparse
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

DOC_SEP = "<|endoftext|>"
BUCKETS = ("english", "japanese", "code", "math", "mixed")
# Measured with vocab_7 over real corpus samples; keep in sync with
# scripts/build_pretrain_corpus.py.
BYTES_PER_TOKEN = {"english": 3.84, "japanese": 3.69, "code": 3.06, "math": 3.49,
                   "mixed": 3.78}
TARGET_SHARE = {"english": 0.27, "japanese": 0.30, "code": 0.19, "math": 0.15,
                "mixed": 0.09}
SPECIALS = ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
            "<|code|>", "<|/code|>", "<|translate|>", "<|summary|>",
            "<|think|>", "<|/think|>", "<|lang_id|>", "<|lang_en|>",
            "<|lang_ja|>", "<|lang_ko|>", "<|pad|>"]

PROBES = {
    "EN": "The mitochondria is the powerhouse of the cell.",
    "JA": "日本語の自然言語処理において、形態素解析は重要な前処理ステップです。",
    "CODE": "#!/usr/bin/env python3\n# fib\ndef fib(n):\n    return n if n < 2 else fib(n-1)+fib(n-2)\n",
    "MATH": "Let $f(x)=\\int_0^x e^{-t^2}dt$. Then $f'(x)=e^{-x^2}$.",
}

ok = True


def check(cond, msg, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}" + (f" — {detail}" if detail else ""))
    if not cond:
        ok = False
    return cond


def warn(msg, detail=""):
    print(f"  [WARN] {msg}" + (f" — {detail}" if detail else ""))


def load_merges_from_text(vocab_dir: Path, tok2id):
    """Mirrors BPEEncoder::load_text after the '#'-comment fix."""
    merges = {}
    dropped_if_buggy = 0
    for line in (vocab_dir / "merges.txt").read_text(encoding="utf-8").split("\n"):
        if not line or line.startswith("#version:"):
            continue
        sp = line.find(" ")
        if sp < 0:
            continue
        left, right = line[:sp], line[sp + 1:]
        a, b, m = tok2id.get(left), tok2id.get(right), tok2id.get(left + right)
        if None in (a, b, m):
            continue
        merges[(a, b)] = m
        if line[0] == "#":
            dropped_if_buggy += 1
    return merges, dropped_if_buggy


def load_merges_from_bin(path: Path):
    with path.open("rb") as f:
        if f.read(4)[:3] != b"BP2":
            raise ValueError("bad magic")
        (vs,) = struct.unpack("<i", f.read(4))
        toks = []
        for _ in range(vs):
            (n,) = struct.unpack("<i", f.read(4))
            toks.append(f.read(n).decode("utf-8", "replace"))
        (nm,) = struct.unpack("<i", f.read(4))
        tok2id = {t: i for i, t in enumerate(toks)}
        merges = {}
        for _ in range(nm):
            a, b = struct.unpack("<ii", f.read(8))
            m = tok2id.get(toks[a] + toks[b])
            if m is not None:
                merges[(a, b)] = m
    return toks, merges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_dir", default="vocab_7")
    ap.add_argument("--bpe", default="bpe_v7.bin")
    ap.add_argument("--data_dir", default="data/corpus_pretrain_v8")
    ap.add_argument("--max_seq", type=int, default=0,
                    help="if set, also report the token budget per step")
    ap.add_argument("--model_vocab", type=int, default=49152)
    ap.add_argument("--params", type=float, default=491.6e6,
                    help="parameter count for the Chinchilla budget [492M]")
    ap.add_argument("--tokens_per_param", type=float, default=20.0)
    args = ap.parse_args()

    vocab_dir = ROOT / args.vocab_dir
    print(f"\n=== tokenizer: {args.vocab_dir} ===")
    if not check((vocab_dir / "vocab.txt").exists(), "vocab.txt present"):
        return 1
    toks = (vocab_dir / "vocab.txt").read_text(encoding="utf-8").split("\n")
    if toks and toks[-1] == "":
        toks.pop()
    tok2id = {t: i for i, t in enumerate(toks)}

    check(len(toks) == len(set(toks)), "no duplicate token strings",
          f"{len(toks) - len(set(toks))} duplicates")
    check(len(toks) == args.model_vocab, "vocab size matches model",
          f"{len(toks)} vs {args.model_vocab}")
    check(toks[:16] == SPECIALS, "special tokens occupy ids 0..15",
          str(toks[:5]))

    text_merges, hash_merges = load_merges_from_text(vocab_dir, tok2id)
    print(f"  ... {len(text_merges)} merges from merges.txt "
          f"({hash_merges} of them start with '#')")

    bpe_path = ROOT / args.bpe
    if bpe_path.exists():
        bin_toks, bin_merges = load_merges_from_bin(bpe_path)
        check(bin_toks == toks, "bpe binary vocab == vocab.txt")
        check(set(bin_merges) == set(text_merges),
              "bpe binary merges == merges.txt merges",
              f"binary={len(bin_merges)} text={len(text_merges)}; "
              f"missing from text: {len(set(bin_merges) - set(text_merges))}")
    else:
        warn(f"{args.bpe} not found — the trainer will use --vocab_dir")

    # Byte coverage: ids 16..271 must be the 256 byte tokens or arbitrary UTF-8
    # cannot round-trip.
    check(len(set(toks[16:272])) == 256, "256 byte-level base tokens present")

    print("\n=== tokenizer probes (bytes per token) ===")
    try:
        from bpe_reference import BPEReference
        enc = BPEReference(vocab_dir)
        for name, s in PROBES.items():
            ids = enc.encode(s)
            rt = enc.decode(ids, skip_special=False) == s
            nb = len(s.encode())
            print(f"  {name:5s} {nb:5d} B -> {len(ids):4d} tok "
                  f"({nb/max(1,len(ids)):.2f} B/tok) roundtrip={'ok' if rt else 'BROKEN'}")
            check(rt, f"{name} round-trips through encode/decode")
    except Exception as e:  # noqa: BLE001 - preflight must report, not crash
        warn(f"reference tokenizer unavailable: {e}")

    print(f"\n=== corpus: {args.data_dir} ===")
    data_dir = ROOT / args.data_dir
    if not check(data_dir.is_dir(), "corpus directory exists", str(data_dir)):
        return 1

    bucket_bytes = {}
    for b in BUCKETS:
        d = data_dir / b
        if not d.is_dir():
            check(False, f"bucket '{b}' exists")
            bucket_bytes[b] = 0
            continue
        files = sorted(d.glob("*.txt"))
        dangling = [p for p in files if not p.exists()]
        size = sum(p.stat().st_size for p in files if p.exists())
        bucket_bytes[b] = size
        check(size > 0, f"bucket '{b}' non-empty",
              f"{len(files)} files, {size/1e9:.2f} GB")
        if dangling:
            check(False, f"bucket '{b}' has dangling symlinks", str(dangling))
        # Document separator: without it the loader falls back to
        # one-document-per-line and multi-line code is destroyed.
        if files and files[0].exists():
            head = files[0].open("r", encoding="utf-8", errors="replace").read(1 << 20)
            has_sep = any(ln.strip() == DOC_SEP for ln in head.split("\n"))
            if b in ("code", "math"):
                check(has_sep, f"bucket '{b}' uses the '{DOC_SEP}' separator")
            elif not has_sep:
                warn(f"bucket '{b}' has no document separator — "
                     f"it will be read one document per line")

    # ---- balance vs Chinchilla -------------------------------------------
    print(f"\n=== balance ({args.params/1e6:.0f}M params, "
          f"{args.tokens_per_param} tokens/param) ===")
    budget = args.params * args.tokens_per_param
    tokens = {b: bucket_bytes[b] / BYTES_PER_TOKEN[b] for b in BUCKETS}
    total = sum(tokens.values())
    print(f"  {'bucket':9} {'GB':>7} {'tokens':>10} {'share':>7} {'target':>7} "
          f"{'epochs':>7}")
    for b in BUCKETS:
        share = tokens[b] / max(1.0, total)
        want = TARGET_SHARE[b]
        ep = (budget * want) / tokens[b] if tokens[b] > 0 else float("inf")
        ep_s = "  empty" if tokens[b] <= 0 else f"{ep:7.2f}"
        print(f"  {b:9} {bucket_bytes[b]/1e9:7.2f} {tokens[b]/1e9:9.2f}B "
              f"{100*share:6.1f}% {100*want:6.1f}% {ep_s}")
    print(f"  {'TOTAL':9} {sum(bucket_bytes.values())/1e9:7.2f} "
          f"{total/1e9:9.2f}B  -> Chinchilla budget {budget/1e9:.1f}B tokens "
          f"({budget/max(1.0,total):.2f} epochs)")

    check(total > 0, "corpus has tokens")
    for b in BUCKETS:
        ep = (budget * TARGET_SHARE[b]) / max(1.0, tokens[b])
        if tokens[b] <= 0:
            continue
        if ep > 4.0:
            warn(f"bucket '{b}' would repeat {ep:.1f}x to hit its {100*TARGET_SHARE[b]:.0f}% "
                 f"share; above ~4 epochs that stops paying off")
    if args.max_seq:
        print(f"  ~{total/args.max_seq:,.0f} sequences of {args.max_seq} tokens")


    print("\n" + ("PREFLIGHT OK" if ok else "PREFLIGHT FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
