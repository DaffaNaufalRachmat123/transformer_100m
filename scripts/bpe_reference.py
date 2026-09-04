#!/usr/bin/env python3
"""bpe_reference.py — Exact Python port of the C++ BPEEncoder.

This tokenizer is NOT GPT-2/HuggingFace compatible, so llama.cpp cannot encode
prompts for this model. The differences (all verified against
bpe/bpe_tokenizer.cpp) are:

  1. Pre-tokenizer splits on runs of bytes <= 0x20 and COLLAPSES them. Only the
     first word of a segment lacks the 'Ġ' space prefix.
  2. Word seeding is greedy LONGEST-MATCH over up to 8 *bytes*, not per
     character.
  3. Merges are applied as repeated left-to-right passes over any adjacent pair
     present in the merge table -- NOT lowest-rank-first as in classic BPE.
  4. '<|...|>' specials are split out before any of the above.
  5. merges.txt lines starting with '#' are dropped by the C++ loader, which
     silently removes 42 real merges. We reproduce that.

Use as a library (encode/decode) or a CLI:

    python3 scripts/bpe_reference.py --text '<|user|>Hello<|assistant|>'
    python3 scripts/bpe_reference.py --selftest
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def bytes_to_unicode():
    """GPT-2 byte->glyph table; matches BYTE_ENCODER[] in bpe_tokenizer.cpp."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}

SPACE_GLYPH = BYTE_ENCODER[0x20]          # 'Ġ'
MAX_MATCH_BYTES = 8                        # C++: len = min(size-pos, 8)
MAX_WORD_BYTES = 512                       # C++: word length guard


class BPEReference:
    def __init__(self, vocab_dir=ROOT / "vocab_6"):
        vocab_dir = Path(vocab_dir)
        raw = (vocab_dir / "vocab.txt").read_text(encoding="utf-8")
        self.tokens = raw.split("\n")
        if self.tokens and self.tokens[-1] == "":
            self.tokens.pop()

        self.tok2id = {t: i for i, t in enumerate(self.tokens)}
        # byte view: the C++ longest-match indexes std::string by BYTE
        self.bytes2id = {t.encode("utf-8"): i for i, t in enumerate(self.tokens)}

        self.merges = {}
        for line in (vocab_dir / "merges.txt").read_text(encoding="utf-8").split("\n"):
            if not line or line[0] == "#":       # mirrors the C++ loader
                continue
            sp = line.find(" ")
            if sp < 0:
                continue
            left, right = line[:sp], line[sp + 1:]
            a, b = self.tok2id.get(left), self.tok2id.get(right)
            m = self.tok2id.get(left + right)
            if a is None or b is None or m is None:
                continue
            self.merges[(a, b)] = m

        self.bos = self.tok2id.get("<|endoftext|>", 0)
        self.eos = self.tok2id.get("<|end|>", 4)
        self.pad = self.tok2id.get("<|pad|>", 15)
        self.n_special = 16

    # ── stage 1: whitespace pre-tokenizer ────────────────────────────────
    def pretokenize(self, segment: bytes):
        out = []
        p, n, first = 0, len(segment), True
        while p < n:
            while p < n and segment[p] <= 0x20:
                p += 1
            if p >= n:
                break
            start = p
            while p < n and segment[p] > 0x20:
                p += 1
            word = "" if first else SPACE_GLYPH
            word += "".join(BYTE_ENCODER[b] for b in segment[start:p])
            first = False
            out.append(word)
        return out

    # ── stage 2: longest-match seed + left-to-right merge passes ─────────
    def encode_word(self, word: str):
        wb = word.encode("utf-8")[:MAX_WORD_BYTES]
        seq, pos, n = [], 0, len(wb)
        while pos < n:
            hit = False
            for ln in range(min(n - pos, MAX_MATCH_BYTES), 0, -1):
                tid = self.bytes2id.get(wb[pos:pos + ln])
                if tid is not None:
                    seq.append(tid)
                    pos += ln
                    hit = True
                    break
            if not hit:
                pos += 1                      # C++ skips the unmatched byte
        changed = True
        while changed and len(seq) > 1:
            changed = False
            merged, i = [], 0
            while i < len(seq):
                if i + 1 < len(seq):
                    m = self.merges.get((seq[i], seq[i + 1]))
                    if m is not None:
                        merged.append(m)
                        i += 2
                        changed = True
                        continue
                merged.append(seq[i])
                i += 1
            seq = merged
        return seq

    # ── stage 3: split specials, then encode each segment ────────────────
    def encode(self, text: str, add_bos=False, add_eos=False):
        result = []
        if add_bos:
            result.append(self.bos)
        bb = text.encode("utf-8")
        p, end = 0, len(bb)
        LT, PIPE, GT = ord("<"), ord("|"), ord(">")
        while p < end:
            if end - p >= 3 and bb[p] == LT and bb[p + 1] == PIPE:
                scan_end = end if end - p < 128 else p + 128
                sp_end = None
                q = p + 2
                while q + 1 < scan_end:
                    if bb[q] == PIPE and bb[q + 1] == GT:
                        sp_end = q + 2
                        break
                    q += 1
                if sp_end is not None:
                    tid = self.tok2id.get(bb[p:sp_end].decode("utf-8", "replace"))
                    if tid is not None:
                        result.append(tid)
                        p = sp_end
                        continue
            seg_start = p
            while p < end:
                if end - p >= 2 and bb[p] == LT and bb[p + 1] == PIPE:
                    break
                p += 1
            if p > seg_start:
                for word in self.pretokenize(bb[seg_start:p]):
                    result.extend(self.encode_word(word))
            else:
                p += 2                        # unmatched '<|', avoid infinite loop
        if add_eos:
            result.append(self.eos)
        return result

    def decode(self, ids, skip_special=True):
        glyphs = []
        for i in ids:
            if i < 0 or i >= len(self.tokens):
                continue
            if skip_special and i < self.n_special:
                continue
            glyphs.append(self.tokens[i])
        raw = bytes(BYTE_DECODER[c] for c in "".join(glyphs) if c in BYTE_DECODER)
        return raw.decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=str(ROOT / "vocab_6"))
    ap.add_argument("--text", default="")
    ap.add_argument("--stdin", action="store_true",
                    help="read one text per line, print space-separated ids")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    enc = BPEReference(args.vocab)

    if args.selftest:
        # byte table must match the hand-written C++ one at known indices
        assert BYTE_ENCODER[0x20] == "\u0120", "space glyph mismatch"
        assert BYTE_ENCODER[0x00] == "\u0100" and BYTE_ENCODER[0x7f] == "\u0121"
        assert BYTE_ENCODER[0xa0] == "\u0142" and BYTE_ENCODER[0xad] == "\u0143"
        print(f"vocab={len(enc.tokens)} merges={len(enc.merges)}")
        assert len(enc.tokens) == 65536
        assert len(enc.merges) == 65222, f"merges={len(enc.merges)}"
        cases = [
            "Hello world",
            "What is the capital of Japan?",
            "日本の首都は東京です。",
            "Terjemahkan ke bahasa Indonesia: Good morning.",
            "<|system|>You are a helpful, accurate multilingual assistant."
            "<|user|>What is the capital of France?<|assistant|>",
        ]
        ok = True
        for c in cases:
            ids = enc.encode(c)
            back = enc.decode(ids, skip_special=False)
            # whitespace is intentionally lossy (runs collapse to one space)
            expect = " ".join(c.split())
            got = " ".join(back.split())
            flag = "ok " if got == expect else "DIFF"
            if got != expect:
                ok = False
            print(f"  [{flag}] {len(ids):3d} ids | {c[:52]}")
            if got != expect:
                print(f"         expect: {expect[:70]}")
                print(f"         got   : {got[:70]}")
        print("selftest:", "PASS" if ok else "FAIL (see DIFF above)")
        return

    if args.stdin:
        for line in sys.stdin:
            line = line.rstrip("\n")
            if not line:
                continue
            print(" ".join(str(i) for i in enc.encode(line)))
        return

    text = args.text or "<|user|>Hello<|assistant|>"
    ids = enc.encode(text)
    print(json.dumps({"text": text, "n": len(ids), "ids": ids,
                      "decoded": enc.decode(ids)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
