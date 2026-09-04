#!/usr/bin/env python3
"""ask.py — Quick local test driver for the quantized GGUF models on macOS.

Wraps the three steps (build chat prompt -> encode with our BPE -> run
llama.cpp) into one command, and picks the right checkpoint per task because no
single checkpoint does both facts and translation.

    python3 scripts/ask.py fact   "What is the capital of Japan?"
    python3 scripts/ask.py fact-ja "フランスの首都はどこですか？"
    python3 scripts/ask.py to-ja  "Good evening, everyone."
    python3 scripts/ask.py to-en  "私は毎朝コーヒーを飲みます。"

    python3 scripts/ask.py fact --repl          # interactive
    python3 scripts/ask.py to-en --file in.txt  # one input per line
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
EOT = "<<<EOT>>>"

SYS_FACT_EN = "You are a helpful, accurate multilingual assistant."
SYS_FACT_JA = "あなたは多言語AIアシスタントです。親切で正直で正確に回答します。"
SYS_TR_JA = "You are a helpful translation engine. Output only the translation."
SYS_TR_EN = "You are a multilingual translation assistant."

FACTS_MODEL = "gguf/trilingual-204m-facts-Q4_K_M.gguf"
TRANS_MODEL = "gguf/trilingual-204m-translate-Q4_K_M.gguf"

# mode -> (system prompt, user-text template, default model)
MODES = {
    "fact":    (SYS_FACT_EN, "{t}",                          FACTS_MODEL),
    "fact-ja": (SYS_FACT_JA, "{t}",                          FACTS_MODEL),
    "to-ja":   (SYS_TR_JA,   "日本語に翻訳: {t}",             TRANS_MODEL),
    "to-en":   (SYS_TR_EN,   "Translate to English: {t}",    TRANS_MODEL),
    "to-id":   (SYS_TR_EN,   "Terjemahkan ke bahasa Indonesia: {t}", TRANS_MODEL),
}


def main():
    ap = argparse.ArgumentParser(
        description="Local test driver for the quantized trilingual model.")
    ap.add_argument("mode", choices=sorted(MODES))
    ap.add_argument("text", nargs="*", help="input text (omit with --repl/--file)")
    ap.add_argument("--model", default="", help="override the .gguf model")
    ap.add_argument("--vocab", default="vocab_6")
    ap.add_argument("--runner", default="build_infer/llama_runner")
    ap.add_argument("--n-predict", type=int, default=48)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpu", action="store_true", help="use Metal (default: CPU)")
    ap.add_argument("--file", default="", help="read one input per line")
    ap.add_argument("--repl", action="store_true", help="interactive loop")
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    system, tmpl, default_model = MODES[args.mode]
    model = ROOT / (args.model or default_model)
    runner = ROOT / args.runner
    if not model.exists():
        sys.exit(f"model not found: {model}\n"
                 f"convert one first: ./.venv/bin/python scripts/convert_to_gguf.py")
    if not runner.exists():
        sys.exit(f"runner not found: {runner}  (see mobile/README.md to build it)")

    from bpe_reference import BPEReference
    enc = BPEReference(ROOT / args.vocab)

    cmd = [str(runner), "--model", str(model),
           "--n-predict", str(args.n_predict), "--ctx", str(args.ctx),
           "--threads", str(args.threads),
           "-ngl", "99" if args.gpu else "0", "--eot-marker"]

    def build(text):
        return f"<|system|>{system}<|user|>{tmpl.format(t=text)}<|assistant|>"

    def run(texts):
        prompts = [build(t) for t in texts]
        if args.show_prompt:
            for p in prompts:
                print(f"  prompt: {p}", file=sys.stderr)
        ids = "\n".join(" ".join(str(i) for i in enc.encode(p)) for p in prompts)
        proc = subprocess.run(cmd, input=(ids + "\n").encode("utf-8"),
                              capture_output=True)
        outs = proc.stdout.decode("utf-8", "replace").split(EOT)
        return [o.strip() for o in outs][:len(prompts)]

    if args.repl:
        print(f"[{args.mode}] {model.name} — Ctrl-D to quit")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                print()
                break
            if line:
                print(run([line])[0])
        return

    if args.file:
        texts = [l.strip() for l in open(ROOT / args.file, encoding="utf-8") if l.strip()]
    elif args.text:
        texts = [" ".join(args.text)]
    else:
        texts = [l.strip() for l in sys.stdin if l.strip()]

    for t, out in zip(texts, run(texts)):
        if len(texts) > 1:
            print(f"IN : {t}")
            print(f"OUT: {out}\n")
        else:
            print(out)


if __name__ == "__main__":
    main()
