#!/usr/bin/env python3
"""bpe_reference.py — Exact Python port of the C++ BPEEncoder.

This tokenizer is NOT GPT-2/HuggingFace compatible, so llama.cpp cannot encode
prompts for this model. The differences (all verified against
bpe/bpe_tokenizer.cpp) are:

  1. Pre-tokenizer splits on runs of bytes <= 0x20 and KEEPS the whitespace
     verbatim; only ONE space fuses to the following word (GPT-2 style).
  2. Word seeding is greedy LONGEST-MATCH over up to 8 *bytes*, not per
     character.
  3. Merges are applied as repeated left-to-right passes over any adjacent pair
     present in the merge table -- NOT lowest-rank-first as in classic BPE.
  4. '<|...|>' specials are split out before any of the above.
  5. Only the leading '#version:' banner in merges.txt is a comment. Real merge
     rules whose left token is '#' ('#!', '#!/usr/bin/env', '##'...) are kept.

Use as a library (encode/decode) or a CLI:

    python3 scripts/bpe_reference.py --text '<|user|>Hello<|assistant|>'
    python3 scripts/bpe_reference.py --selftest
"""

import argparse
import json
import struct
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
MAX_WORD_BYTES = 512                       # C++: kMaxWordBytes (chunk size)


class BPEReference:
    def __init__(self, vocab_dir=ROOT / "vocab_9"):
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
            if not line or line.startswith("#version:"):   # mirrors the C++ loader
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
        """Split on whitespace, keeping the whitespace bytes verbatim.

        Collapsing runs to a single space (the previous behaviour) flattens
        multi-line Python onto one line, so generated code cannot parse.
        Only ONE space fuses to the following word, GPT-2 style; the rest of
        the run is its own token. Fusing the whole run instead makes
        "\\n    def" and "\\n        def" distinct entries, multiplying every
        word by its indentation depth.
        """
        out = []
        p, n = 0, len(segment)
        while p < n:
            ws_start = p
            while p < n and segment[p] <= 0x20:
                p += 1
            ws_end = p
            word_start = p
            while p < n and segment[p] > 0x20:
                p += 1
            word_end = p
            if ws_start == ws_end and word_start == word_end:
                break

            ws_tail = ws_end
            if word_start != word_end and ws_end > ws_start \
                    and segment[ws_end - 1] == 0x20:
                ws_tail = ws_end - 1

            if ws_tail > ws_start:
                out.append("".join(BYTE_ENCODER[b]
                                   for b in segment[ws_start:ws_tail]))

            word = "".join(BYTE_ENCODER[b] for b in segment[ws_tail:ws_end])
            word += "".join(BYTE_ENCODER[b] for b in segment[word_start:word_end])
            if word:
                out.append(word)
        return out

    # ── stage 2: longest-match seed + left-to-right merge passes ─────────
    def _encode_chunk(self, wb: bytes):
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

    def encode_word(self, word: str):
        wb = word.encode("utf-8")
        if len(wb) <= MAX_WORD_BYTES:
            return self._encode_chunk(wb)
        # Split, never truncate: Japanese has no spaces, so a whole sentence is
        # one pre-token and truncating dropped 43% of wiki_ja characters.
        out, pos = [], 0
        while pos < len(wb):
            take = min(MAX_WORD_BYTES, len(wb) - pos)
            if pos + take < len(wb):
                while take > 1 and (wb[pos + take] & 0xC0) == 0x80:
                    take -= 1
            out.extend(self._encode_chunk(wb[pos:pos + take]))
            pos += take
        return out

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
            # A '<|' that is not a known special is ordinary text: step over it
            # so the scan can advance, but keep it in the segment.
            if end - p >= 2 and bb[p] == LT and bb[p + 1] == PIPE:
                p += 2
            while p < end:
                if end - p >= 2 and bb[p] == LT and bb[p + 1] == PIPE:
                    break
                p += 1
            for word in self.pretokenize(bb[seg_start:p]):
                result.extend(self.encode_word(word))
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

    def export_binary(self, path):
        """Write the BP2 file that BPEEncoder::load_binary reads.

        Needed because the two artifacts drift: bpe_v7.bin still carried the
        pre-rename specials <|phone_event|>/<|app_usage|> at ids 9/10 while
        vocab_7/vocab.txt had <|think|>/<|/think|>, so `--bpe` and
        `--vocab_dir` disagreed on two token ids.
        """
        pairs = sorted(self.merges.items(), key=lambda kv: kv[1])
        with open(path, "wb") as f:
            f.write(b"BP2\0")
            f.write(struct.pack("<i", len(self.tokens)))
            for t in self.tokens:
                b = t.encode("utf-8")
                f.write(struct.pack("<i", len(b)))
                f.write(b)
            f.write(struct.pack("<i", len(pairs)))
            for (a, b), _ in pairs:
                f.write(struct.pack("<ii", a, b))
        return len(self.tokens), len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=str(ROOT / "vocab_9"))
    ap.add_argument("--text", default="")
    ap.add_argument("--stdin", action="store_true",
                    help="read one text per line, print space-separated ids")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--export-bin", default="",
                    help="write a BP2 binary tokenizer from --vocab and exit")
    args = ap.parse_args()

    enc = BPEReference(args.vocab)

    if args.export_bin:
        nv, nm = enc.export_binary(args.export_bin)
        print(f"wrote {args.export_bin}: vocab={nv} merges={nm}")
        return

    if args.selftest:
        # byte table must match the hand-written C++ one at known indices
        assert BYTE_ENCODER[0x20] == "\u0120", "space glyph mismatch"
        assert BYTE_ENCODER[0x00] == "\u0100" and BYTE_ENCODER[0x7f] == "\u0121"
        assert BYTE_ENCODER[0xa0] == "\u0142" and BYTE_ENCODER[0xad] == "\u0143"
        print(f"vocab={len(enc.tokens)} merges={len(enc.merges)}")
        cases = [
            "Hello world",
            "What is the capital of Japan?",
            "日本の首都は東京です。",
            "<|system|>You are a helpful, accurate multilingual assistant."
            "<|user|>What is the capital of France?<|assistant|>",
            # Indentation and newlines must survive or generated Python cannot
            # parse, which silently pins every RL code reward at zero.
            "def solve(n):\n    total = 0\n    for i in range(n):\n"
            "        total += i * i\n    return total\nprint(solve(10))",
        ]
        ok = True
        for c in cases:
            ids = enc.encode(c)
            back = enc.decode(ids, skip_special=False)
            same = back == c
            if not same:
                ok = False
            print(f"  [{'ok ' if same else 'DIFF'}] {len(ids):3d} ids | "
                  f"{c[:52]!r}")
            if not same:
                print(f"         expect: {c[:70]!r}")
                print(f"         got   : {back[:70]!r}")
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
