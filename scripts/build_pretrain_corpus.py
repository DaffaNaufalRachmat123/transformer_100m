#!/usr/bin/env python3
"""build_pretrain_corpus.py — build the EN / JA / CODE / MATH pretraining corpus.

Reads the raw corpora scattered under data/, reconstructs real documents from
them, filters them, deduplicates them, and writes a single clean corpus tree:

    <out>/english/*.txt
    <out>/japanese/*.txt
    <out>/code/*.txt
    <out>/math/*.txt

OUTPUT FORMAT
-------------
Documents are separated by a line containing exactly ``<|endoftext|>``.
PretrainDataset detects that separator and tokenizes each document as ONE unit
with its newlines intact. The previous corpora were one *line* per record, so
every source line of code became its own BOS/EOS-wrapped document — roughly 20%
of all code tokens were separators and multi-line structure was unlearnable.

WHY THE SOURCES ARE READ DIFFERENTLY
------------------------------------
Measured line statistics differ by an order of magnitude between sources:

    fineweb_edu_extra.txt   avg 3704 B/line   -> one document per line
    culturax_ja_new.txt     avg 5474 B/line   -> one document per line
    wiki_en / fineweb / openwebtext / wiki_ja  avg ~200 B/line
                                              -> one *paragraph* per line;
                                                 consecutive lines are the same
                                                 article, so they are grouped
    starcoder_clean.txt     avg 32 B/line     -> raw source lines, split on the
                                                 <reponame> marker

Usage
-----
    # what would be kept/dropped, no writes (fast, samples each source)
    python3 scripts/build_pretrain_corpus.py --audit

    # real build with per-bucket byte budgets
    python3 scripts/build_pretrain_corpus.py --out data/corpus_pretrain_v8
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIBLINGS = ROOT.parent          # .../TransformerSample — other projects live here
DOC_SEP = "<|endoftext|>"

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
# bucket, label, path, reader ("auto" picks doc-per-line vs paragraph-group by
# measuring mean line length). Paths starting with "../" resolve against the
# sibling-project root.
#
# Source selection is deliberate, not "everything we can find":
#
#   english/fineweb_edu_extra -> the 650m copy (6.03 GB, 236 B/line) instead of
#     the local one (3.29 GB, 3663 B/line). Same corpus, but the local copy had
#     its paragraph newlines flattened into one line per document and holds
#     ~45% fewer bytes.
#
#   code/starcoder_extra -> the 650m raw copy (3.21 GB) instead of the local
#     starcoder_extra_clean.txt (2.68 GB), whose "cleaning" replaced every
#     newline with U+23CE RETURN SYMBOL (59,948 of 60,000 sampled lines).
#
#   japanese/culturax -> the RAW capture, not culturax_ja_new.txt. The "new"
#     file is the raw one minus ~2% and was filtered by a weaker ruleset; this
#     script's filters are stricter, so start from raw and filter once.
#
# Order matters: the deduplicator is first-wins, so the highest-quality source
# in each bucket is listed first.
SOURCES = [
    # ---- English ---------------------------------------------------------
    ("english", "fineweb_edu_extra", "../transformer_650m_mobile/scripts/data/corpus_extra/english/fineweb_edu_extra.txt", "auto"),
    ("english", "wiki_en",           "data/corpus_pretrain_2_5b/english/wiki_en.txt",     "auto"),
    ("english", "fineweb_edu",       "data/corpus_pretrain_2_5b/english/fineweb_edu.txt", "auto"),
    ("english", "openwebtext",       "data/corpus_pretrain_2_5b/english/openwebtext.txt", "auto"),
    ("english", "fineweb",           "data/corpus_pretrain_2_5b/english/fineweb.txt",     "auto"),
    # ---- Japanese --------------------------------------------------------
    ("japanese", "wiki_ja",       "data/corpus_pretrain_2_5b/japanese/wiki_ja.txt", "auto"),
    ("japanese", "culturax_ja",   "data/culturax_raw/culturax_ja.txt",              "auto"),
    ("japanese", "mc4_ja",        "data/corpus_final/japanese/mc4_ja.txt",          "auto"),
    ("japanese", "wiki_ja_older", "data/corpus_final/japanese/wiki_ja.txt",         "auto"),
    # ---- Code ------------------------------------------------------------
    ("code", "starcoder_extra", "../transformer_650m_mobile/scripts/data/corpus_extra/code/starcoder_extra.txt", "code"),
    ("code", "starcoder",       "data/corpus_1b/code/starcoder_clean.txt", "code"),
    # ---- Mixed: knowledge injection + EN<->JA parallel --------------------
    # Both are generated (scripts/build_fact_injection_corpus.py and
    # scripts/build_mixed_bucket.py) and already document-separated. They are
    # the only sources that directly move facts and translation, so they get
    # their own bucket rather than being diluted into english/ and japanese/.
    ("mixed", "facts_en",      "data/corpus_inject_facts/facts_en.txt", "auto"),
    ("mixed", "facts_ja",      "data/corpus_inject_facts/facts_ja.txt", "auto"),
    ("mixed", "parallel_enja", "data/corpus_inject_parallel/parallel_enja.txt", "auto"),
]


# Math is not a downloaded corpus here: no math source survived in the tree
# (data/corpus_1b/math/openwebmath.txt is a dangling symlink to another
# project), and the English web corpora present contain almost no mathematical
# notation — the notation-only density score peaks at 7.5 per 1000 chars over
# 6000 fineweb_edu documents, so there is nothing meaningful to harvest. The
# bucket is therefore assembled from:
#   1. the local worked-solution datasets, which are real math prose;
#   2. whatever math-dense documents the English sources do contain;
#   3. anything scripts/fetch_corpus.py has downloaded into data/math_raw/
#      (run that to get a corpus of real size).
MATH_JSONL = [
    ("gsm8k_train", "data/gsm8k/train.jsonl", ("question", "answer")),
    ("rl_math",     "rl_data/math.jsonl",     None),
    ("rl_math_v2",  "rl_data/math_v2.jsonl",  None),
    ("rl_math_pot", "rl_data/math_pot.jsonl", None),
]

# SFT sets are chat-formatted; the markup is stripped and Indonesian rows are
# dropped so pretraining never sees a chat template or a fourth language.
MATH_SFT = [
    ("sft_math",       "sft_data/sft_math/train.jsonl"),
    ("sft_math_v2",    "sft_data/sft_math_v2/train.jsonl"),
    ("sft_math_pot",   "sft_data/sft_math_pot/train.jsonl"),
    ("sft_think_math", "sft_data/sft_think_math/train.jsonl"),
]

# Plain-text corpora downloaded by scripts/fetch_corpus.py. Anything dropped in
# these directories is filtered and deduplicated like every other source, so
# closing a data gap is "fetch, then rebuild" with no code change.
RAW_DIRS = {
    "english":  "data/en_raw",
    "japanese": "data/ja_raw",
    "code":     "data/code_raw",
    "math":     "data/math_raw",
}

# ---------------------------------------------------------------------------
# Budgets — derived from Chinchilla for the 492M architecture in model_config.h
# ---------------------------------------------------------------------------
# Parameters: 62.9M tied embedding + 24 x 17.86M blocks = 491.6M.
# Chinchilla-optimal is 20 tokens/parameter = 9.8B tokens; model_config.h
# targets ~23 tokens/param = 11.3B.
#
# Bytes per token, measured with vocab_7 on real corpus samples:
#     EN 3.53   JA 3.18   CODE 3.64   MATH 3.70
#
# Token mix. Japanese gets the largest share on purpose:
#   - JA packs 1.16 characters per token against EN's 4.15, so the same
#     document costs ~1.15x more tokens in Japanese even after accounting for
#     kanji being denser than letters;
#   - the model has to learn three scripts and ~6,200 distinct kanji, none of
#     which English requires. In a 45.8M-character sample the top 2,136 kanji
#     already cover 99.14% of occurrences, but 2,531 kanji need >=100
#     exposures each, and that only happens at a large JA token count.
#
#     JA 36%  EN 32%  CODE 20%  MATH 12%   of 10B tokens
#
# The byte budgets below spend ~20 GB, which is what the disk allows, and buy
# ~5.8B unique tokens — i.e. ~1.7 epochs to reach the 10B Chinchilla target.
# Under 4 epochs, repeated tokens are worth nearly as much as fresh ones
# (Muennighoff et al., "Scaling Data-Constrained Language Models").
# Measured on data/corpus_pretrain_v8 with vocab_8. vocab_7's rates (JA 3.18,
# EN 3.53, code 3.64, math 3.70) no longer apply: the script-aware pretokenizer
# gave Japanese 22% better compression and shifted every other bucket too.
BYTES_PER_TOKEN = {"english": 3.84, "japanese": 3.69, "code": 3.06, "math": 3.49,
                   "mixed": 3.78}
TARGET_TOKEN_SHARE = {"english": 0.27, "japanese": 0.30, "code": 0.19,
                      "math": 0.15, "mixed": 0.09}
CHINCHILLA_TOKENS = 10_000_000_000

DEFAULT_BUDGET = {
    "english":  6_500_000_000,
    "japanese": 7_000_000_000,
    "code":     4_500_000_000,
    # 14% of 10B tokens is 1.4B, which at 3.49 bytes/token needs ~4.9 GB for a
    # single epoch. The old 3.5 GB ceiling predated openwebmath and left math at
    # 0.80B tokens / 1.75 epochs while raw math sat unused.
    "math":     5_000_000_000,
    "mixed":    4_000_000_000,
}

# Ceiling on how much of a bucket one source may fill. Per-bucket because the
# buckets have opposite problems: the first build let fineweb_edu_extra take the
# entire English budget, and both code sources are Python-only extracts (the
# resulting bucket measured 94.6% Python), while Japanese has only four sources
# and needs volume out of each of them.
SOURCE_SHARE = {
    "english":  0.30,   # 5 sources
    "japanese": 0.50,   # 4 sources, volume-limited
    "code":     0.25,   # 7 sources, diversity-limited
    "math":     0.60,   # finemath + openwebmath, so neither can own the bucket
    "mixed":    0.50,   # facts_en, facts_ja, parallel_enja
}
SHARD_BYTES = 1_500_000_000  # split each bucket into shards of this size


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
REPLACEMENT_RE = re.compile("\ufffd")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
REPEAT_RE = re.compile(r"(.)\1{15,}")
B64_RE = re.compile(r"[A-Za-z0-9+/]{256,}={0,2}")
WS_RE = re.compile(r"\s+")

LATIN_RE = re.compile(r"[A-Za-z]")
ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)
KANA_RE = re.compile(r"[\u3040-\u30ff]")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")
HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
JA_SENT_END_RE = re.compile(r"[。！？]")
# Anchored at the start of a LINE, not anywhere in the text. Counting dates
# anywhere dropped 26% of Japanese Wikipedia — articles on history and
# biography are dense with "ｙｙｙｙ年ｍ月" and are exactly the prose we want.
JA_LINE_DATE_RE = re.compile(
    r"^\s*(?:\d{4}\s*年|\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})"
)

# Newline substitutes. starcoder_extra_clean.txt stores every newline as
# U+23CE RETURN SYMBOL (verified: 24,022 of the first 20 MB of lines contain
# it, and vocab_7 grew 213 merges around it against exactly 1 for a real
# newline). Code trained on that can never emit a usable newline, so restore
# the real character.
PSEUDO_NEWLINES = {
    "\u23ce": "\n",   # ⏎ RETURN SYMBOL
    "\u21b5": "\n",   # ↵ DOWNWARDS ARROW WITH CORNER LEFTWARDS
    "\u2424": "\n",   # ␤ SYMBOL FOR NEWLINE
    "\u240a": "\n",   # ␊ SYMBOL FOR LINE FEED
    "\u2028": "\n",
    "\u2029": "\n",
}
PSEUDO_NEWLINE_RE = re.compile("[" + "".join(PSEUDO_NEWLINES) + "]")


def repair_newlines(text: str) -> str:
    if PSEUDO_NEWLINE_RE.search(text):
        return PSEUDO_NEWLINE_RE.sub("\n", text)
    return text


# Adult / gambling use a strong+weak split. A single "casino" or "人妻" is normal
# in an encyclopedia article; hard-dropping on one hit removed 8% of wiki_ja and
# 2% of wiki_en in the first audit pass, which is censorship of legitimate text,
# not cleaning.
ADULT_STRONG = re.compile(
    r"\b(?:porn|pornhub|xvideos|xhamster|redtube|youporn|xnxx|hentai|onlyfans|"
    r"camgirls?|webcam ?girls?|blowjobs?|handjobs?|cumshots?|creampie|gangbang|"
    r"bukkake|deepthroat|anal ?sex|sex ?tape|sex ?cams?|adult ?video|"
    r"adult ?dating|escort ?service|call ?girls?|nude ?pics?|naked ?pics?|"
    r"nsfw|jailbait|rule ?34|erotic ?stor\w+|xxx ?(?:video|movie|porn))\b",
    re.I,
)
ADULT_WEAK = re.compile(
    r"\b(?:sexual|sex|erotic|nude|naked|milf|fetish|bdsm|orgasm|masturbat\w+|"
    r"stripper|brothel|prostitut\w+|incest|swingers|threesome)\b",
    re.I,
)
ADULT_STRONG_JA = re.compile(
    "エロ動画|エロ画像|無修正|アダルト動画|AV女優|裏ビデオ|出会い系|デリヘル|"
    "ソープランド|風俗店|風俗嬢|ハメ撮り|ライブチャット|エロゲ|援助交際|"
    "ハプニングバー|JKビジネス|性感エステ|素人動画|中出し"
)
ADULT_WEAK_JA = re.compile(
    "エロ|アダルト|セックス|オナニー|痴漢|巨乳|爆乳|人妻|熟女|童貞|"
    "寝取られ|援交|ヌード"
)
GAMBLE_STRONG = re.compile(
    r"\b(?:online ?casinos?|casino ?bonus|free ?spins|no ?deposit ?bonus|"
    r"betting ?site|sportsbook|bookmaker|bet365|judi ?online|slot ?gacor|"
    r"situs ?slot|togel ?(?:online|hk|sgp)|bandar ?(?:qq|togel|judi)|"
    r"daftar ?slot|rtp ?slot|capsa ?susun)\b",
    re.I,
)
GAMBLE_WEAK = re.compile(
    r"\b(?:casino|poker|blackjack|roulette|baccarat|slot ?machines?|betting|"
    r"gambling|jackpot|wager|bookie)\b",
    re.I,
)
GAMBLE_STRONG_JA = re.compile(
    "オンラインカジノ|ネットカジノ|入金不要ボーナス|競馬予想|競艇予想|"
    "競輪予想|ブックメーカー|パチスロ実機|ギャンブル必勝"
)
GAMBLE_WEAK_JA = re.compile("パチンコ|パチスロ|カジノ|ギャンブル|必勝法|換金率")

# Commercial / SEO / navigation boilerplate. Counted as signals, never a single
# hard drop, because legitimate prose mentions these too.
SPAM_EN = re.compile(
    r"(?:buy ?now|click ?here|order ?now|shop ?now|add ?to ?cart|shopping ?cart|"
    r"limited ?(?:time )?offer|discount ?code|coupon ?code|promo ?code|"
    r"free ?shipping|best ?price|lowest ?price|out ?of ?stock|"
    r"subscribe ?now|sign ?up ?(?:now|free|today)|newsletter|"
    r"all ?rights ?reserved|cookie ?policy|privacy ?policy|terms ?of ?(?:service|use)|"
    r"enable ?javascript|accept ?cookies|comments ?are ?closed|"
    r"make ?money ?online|work ?from ?home|payday ?loan|viagra|cialis|"
    r"seo ?service|backlinks?|guest ?post)",
    re.I,
)
SPAM_JA = re.compile(
    "送料無料|最安値|激安|購入はこちら|今すぐ購入|カートに入れる|即日発送|"
    "無料登録|会員登録|お問い合わせ|全国送料|ポイント還元|楽天市場|"
    "コメントは受け付けていません|利用規約|プライバシーポリシー|無断転載"
)
# License headers duplicated across millions of code files.
CODE_LICENSE_RE = re.compile(
    r"(?:Licensed under the Apache License|GNU GENERAL PUBLIC LICENSE|"
    r"Permission is hereby granted, free of charge|WITHOUT WARRANTIES OR CONDITIONS)",
    re.I,
)
CODE_SIGNAL_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:def |class |function |import |from |#include|package |"
    r"public |private |protected |const |let |var |func |fn |impl |module |"
    r"namespace |if |for |while |return |switch |struct |enum |type |interface )"
)
CODE_GENERATED_RE = re.compile(
    r"(?:DO NOT EDIT|Code generated by|auto-?generated|@generated|"
    r"This file was automatically generated)",
    re.I,
)

# Math routing keys on NOTATION and vocabulary only. An earlier digit-density
# score ranked agricultural census tables and bible verse references as the most
# "mathematical" documents in fineweb_edu, so digits are excluded entirely.
MATH_SYMBOL_RE = re.compile(
    r"\\(?:frac|sum|int|prod|lim|sqrt|alpha|beta|gamma|theta|lambda|sigma|"
    r"partial|infty|cdot|times|leq|geq|neq|approx|equiv|mathbb|mathcal|"
    r"begin\{(?:equation|align|matrix|cases|pmatrix|bmatrix)|end\{)"
    r"|\$\$?[^$\n]{3,}\$\$?"
    r"|[\u2211\u220f\u222b\u221a\u221e\u2264\u2265\u2260\u2248\u2261\u00b1\u2202\u2207\u2208\u2209\u2282\u2286\u222a\u2229\u2192\u2190\u2194\u2200\u2203]"
    r"|\b[a-zA-Z]\s*[=<>]\s*[-+]?\d"
    r"|\b\d+\s*[+\-*/^]\s*\d+\s*="
)
MATH_WORD_RE = re.compile(
    r"\b(?:theorem|lemma|corollary|proof|proposition|axiom|conjecture|"
    r"derivative|integral|matrix|matrices|eigenvalue|eigenvector|polynomial|"
    r"logarithm|factorial|modulo|congruence|variance|deviation|"
    r"algebra|algebraic|calculus|geometry|topology|equation|inequality|"
    r"hypotenuse|quadratic|binomial|permutation|factorise|factorize|"
    r"numerator|denominator|coefficient|exponent|asymptote|vector ?space)\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Bloom filter — bounded-memory duplicate detection
# ---------------------------------------------------------------------------
class Bloom:
    """Fixed-size Bloom filter. ~0.04% false positives at 40M documents with
    the default 128 MiB / k=4, which is far below the duplicate rates we are
    trying to remove."""

    def __init__(self, mib: int = 128, k: int = 4):
        self.nbits = mib * 1024 * 1024 * 8
        self.bits = bytearray(self.nbits // 8)
        self.k = k
        self.n = 0

    def add_if_new(self, data: bytes) -> bool:
        h = hashlib.blake2b(data, digest_size=16).digest()
        h1 = int.from_bytes(h[:8], "little")
        h2 = int.from_bytes(h[8:], "little") | 1
        seen = True
        for i in range(self.k):
            idx = (h1 + i * h2) % self.nbits
            byte, bit = idx >> 3, idx & 7
            if not (self.bits[byte] >> bit) & 1:
                seen = False
                self.bits[byte] |= 1 << bit
        if not seen:
            self.n += 1
        return not seen


def dedup_key(text: str) -> bytes:
    """Normalized signature: kills near-duplicates that differ only in
    whitespace, case, punctuation or a trailing boilerplate tail.

    Only the head is normalized — two documents sharing a 1.5 KB normalized
    prefix are duplicates for our purposes, and normalizing whole multi-KB
    documents is pure cost."""
    t = unicodedata.normalize("NFKC", text[:1536]).lower()
    t = NONWORD_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip()[:1024].encode("utf-8", "ignore")


NONWORD_RE = re.compile(r"[^\w\u3040-\u30ff\u4e00-\u9fff]+")
LONGTOKEN_RE = re.compile(r"\S{200,}")

# ---------------------------------------------------------------------------
# Cheap literal anchors. The big alternation regexes cost ~250 us per document;
# a `substring in str` test costs ~1 us and clears >99% of documents, so every
# expensive pattern is gated behind its anchors. This also lets the expensive
# scan run over the WHOLE document instead of a head slice, which is strictly
# better recall for the content we actually want gone.
# ---------------------------------------------------------------------------
ADULT_ANCHORS = (
    "porn", "sex", "nude", "naked", "xxx", "escort", "nsfw", "erotic", "hentai",
    "onlyfans", "camgirl", "webcam", "milf", "fetish", "bdsm", "blowjob",
    "cumshot", "gangbang", "prostitut", "stripper", "brothel", "incest",
    "swinger", "threesome", "masturbat", "orgasm", "jailbait", "rule 34",
    "bukkake", "creampie", "deepthroat", "handjob", "adult",
)
ADULT_ANCHORS_JA = (
    "エロ", "無修正", "アダルト", "AV女優", "裏ビデオ", "出会い", "デリヘル", "ソープ",
    "風俗", "セックス", "ハメ撮り", "オナニー", "痴漢", "巨乳", "爆乳", "人妻", "熟女",
    "童貞", "ライブチャット", "援交", "援助交際", "寝取られ", "中出し", "ヌード",
)
GAMBLE_ANCHORS = (
    "casino", "poker", "blackjack", "roulette", "baccarat", "slot", "betting",
    "gambling", "jackpot", "wager", "bookie", "bookmaker", "sportsbook",
    "judi", "togel", "bandar", "bet365",
)
GAMBLE_ANCHORS_JA = (
    "パチンコ", "パチスロ", "カジノ", "ギャンブル", "必勝", "換金率", "予想",
    "ブックメーカー", "ボーナス",
)
SPAM_ANCHORS = (
    "buy now", "click here", "order now", "shop now", "cart", "offer", "coupon",
    "promo", "shipping", "price", "stock", "subscribe", "sign up", "newsletter",
    "rights reserved", "cookie", "privacy policy", "terms of", "javascript",
    "comments are closed", "money online", "work from home", "payday", "viagra",
    "cialis", "seo ", "backlink", "guest post",
)
SPAM_ANCHORS_JA = (
    "送料", "最安値", "激安", "購入", "カート", "発送", "登録", "問い合わせ",
    "ポイント還元", "楽天", "コメントは受け付け", "利用規約", "プライバシー", "無断転載",
)
# Blog/aggregator shells: a page whose entire body is "posts tagged X" plus a
# service footer. They pass the sentence-ending test because the boilerplate
# itself is a sentence, so they need their own rule.
JA_AGGREGATOR_RE = re.compile(
    "タグまとめ|のタグがついている|新着記事と人気記事|をまとめました|"
    "エキサイトブログ|にほんブログ村|ブログランキング|ランキング参加中|"
    "アメブロ|ライブドアブログ|FC2ブログ|Seesaaブログ|ココログ|"
    "ございます。このページ|検索結果|関連する記事|この記事をシェア|"
    "もくじ|カテゴリー一覧|月別アーカイブ"
)
MATH_ANCHORS = (
    "$", "\\frac", "\\sum", "\\int", "\\sqrt", "\\begin{", "\\lim", "\\prod",
    "theorem", "lemma", "corollary", "proof", "derivative", "integral",
    "matrix", "eigen", "polynomial", "logarithm", "factorial", "modulo",
    "algebra", "calculus", "geometry", "topology", "equation", "inequality",
    "quadratic", "binomial", "permutation", "numerator", "denominator",
    "coefficient", "exponent", "asymptote", "∑", "∏", "∫", "√", "∞", "≤", "≥",
    "≠", "≈", "≡", "∂", "∇", "∈",
)


def _any_anchor(lc: str, anchors) -> bool:
    for a in anchors:
        if a in lc:
            return True
    return False


# Repetition filters, after Rae et al. (Gopher / MassiveText). Keyword lists
# only catch spam somebody thought of; a page that is one phrase stamped out N
# times is junk regardless of the words in it, and that is what most surviving
# SEO shells look like.
def word_repetition(words: list[str]) -> float:
    """Share of word bigrams taken by the single most common bigram."""
    if len(words) < 50:
        return 0.0
    big = collections.Counter(zip(words, words[1:]))
    return big.most_common(1)[0][1] / (len(words) - 1)


def char_ngram_repetition(text: str, n: int = 8, stride: int = 3,
                          cap: int = 3000) -> float:
    """Share of sampled character n-grams that are repeats.

    Strided and capped so this stays ~0.1 ms per document; Japanese has no word
    boundaries, so the word-bigram test above does not apply to it.
    """
    t = text[:cap]
    if len(t) < 4 * n:
        return 0.0
    seen = set()
    total = dup = 0
    for i in range(0, len(t) - n + 1, stride):
        g = t[i:i + n]
        total += 1
        if g in seen:
            dup += 1
        else:
            seen.add(g)
    return dup / max(1, total)


# ---------------------------------------------------------------------------
# Per-bucket document filters. Each returns None to keep, or a reason to drop.
#
# Character-class ratios are measured on a bounded head slice: a 40 KB document
# does not need a full scan to decide whether it is English.
# ---------------------------------------------------------------------------
SLICE = 4000
RATIO_SLICE = 1200


def _shared_reject(doc: str, n: int, head: str) -> str | None:
    if "\ufffd" in head:
        return "mojibake"
    if CTRL_RE.search(head):
        return "control_bytes"
    if REPEAT_RE.search(head):
        return "repeat_run"
    if LONGTOKEN_RE.search(doc) and B64_RE.search(doc):
        return "binary_blob"
    if "http" in head or "www." in head:
        urls = URL_RE.findall(head)
        if len(urls) > 12 or (urls and sum(map(len, urls)) / len(head) > 0.25):
            return "url_spam"
    return None


def _adult_gamble(doc: str, lc: str, ja: bool) -> str | None:
    if _any_anchor(lc, ADULT_ANCHORS) or (ja and _any_anchor(doc, ADULT_ANCHORS_JA)):
        if ADULT_STRONG.search(doc) or (ja and ADULT_STRONG_JA.search(doc)):
            return "adult"
        weak = len(set(m.lower() for m in ADULT_WEAK.findall(doc)))
        if ja:
            weak += len(set(ADULT_WEAK_JA.findall(doc)))
        if weak >= 3:
            return "adult"
    if _any_anchor(lc, GAMBLE_ANCHORS) or (ja and _any_anchor(doc, GAMBLE_ANCHORS_JA)):
        if GAMBLE_STRONG.search(doc) or (ja and GAMBLE_STRONG_JA.search(doc)):
            return "gambling"
        weak = len(set(m.lower() for m in GAMBLE_WEAK.findall(doc)))
        if ja:
            weak += len(set(GAMBLE_WEAK_JA.findall(doc)))
        if weak >= 3:
            return "gambling"
    return None


def _spam_hits(doc: str, lc: str) -> int:
    if not _any_anchor(lc, SPAM_ANCHORS):
        return 0
    return len(set(m.lower() for m in SPAM_EN.findall(doc)))


def reject_en(doc: str) -> str | None:
    n = len(doc)
    if n < 200:
        return "too_short"
    words = doc.split()
    if len(words) < 40:
        return "too_few_words"
    if len(set(words)) < 0.25 * len(words):
        return "low_lexical_diversity"
    head = doc[:SLICE]
    r = _shared_reject(doc, n, head)
    if r:
        return r
    lc = doc.lower()
    r = _adult_gamble(doc, lc, ja=False)
    if r:
        return r
    if _spam_hits(doc, lc) >= (3 if n < 2000 else 5):
        return "spam_seo"
    if word_repetition(words) > 0.06:
        return "repetitive"
    ratio_head = doc[:RATIO_SLICE]
    alpha = len(ALPHA_RE.findall(ratio_head))
    if alpha < 0.5 * len(ratio_head):
        return "low_alpha"
    if len(LATIN_RE.findall(ratio_head)) < 0.8 * max(1, alpha):
        return "not_english"
    return None


def reject_ja(doc: str) -> str | None:
    n = len(doc)
    if n < 120:
        return "too_short"
    head = doc[:SLICE]
    r = _shared_reject(doc, n, head)
    if r:
        return r
    lc = doc.lower()
    r = _adult_gamble(doc, lc, ja=True)
    if r:
        return r
    ja_spam = len(set(SPAM_JA.findall(doc))) if _any_anchor(doc, SPAM_ANCHORS_JA) else 0
    if ja_spam >= 2 or _spam_hits(doc, lc) >= 4:
        return "spam_seo"
    if JA_AGGREGATOR_RE.search(head):
        return "blog_aggregator"
    if char_ngram_repetition(doc) > 0.28:
        return "repetitive"

    ratio_head = doc[:RATIO_SLICE]
    m = len(ratio_head)
    kana = len(KANA_RE.findall(ratio_head))
    if kana < 0.08 * m or kana < 20:
        # No kana means Chinese, a link farm, or a bare title list — that was
        # 9.4% of wiki_ja lines in the one-line-per-record layout. The absolute
        # floor also drops short captions that are pure kanji compounds.
        return "not_japanese"
    if kana == 0 and len(HAN_RE.findall(ratio_head)) > 20:
        return "chinese"
    if len(HANGUL_RE.findall(ratio_head)) > 10:
        return "korean"
    if len(LATIN_RE.findall(ratio_head)) > 0.5 * m:
        return "mostly_latin"

    # Japanese prose ends sentences with 。 (or ！？). A document with almost no
    # sentence terminators is a navigation menu, a product grid or a tag cloud —
    # the dominant failure mode of JA web crawls, and the one that shows up as
    # long runs of noun phrases the model cannot learn anything from.
    sentences = len(JA_SENT_END_RE.findall(doc))
    if sentences < 2 or n / sentences > 400:
        return "no_sentences"

    nonempty = [x.strip() for x in doc.split("\n") if x.strip()]
    if len(nonempty) >= 8:
        # An index page is short fragments that never finish a sentence.
        frag = sum(1 for x in nonempty if len(x) < 30 and "。" not in x)
        if frag > 0.7 * len(nonempty):
            return "listing_page"
        dated = sum(1 for x in nonempty
                    if "。" not in x and JA_LINE_DATE_RE.match(x))
        if dated > 0.4 * len(nonempty):
            return "date_index"
    # Digit/symbol soup: price tables, spec sheets, ISBN lists, schedule grids.
    digits = sum(c.isdigit() for c in ratio_head)
    if digits > 0.25 * m:
        return "numeric_table"
    # A single phrase repeated over and over (template spam).
    toks = doc.split("。")
    if len(toks) >= 6 and len(set(t.strip() for t in toks)) < 0.5 * len(toks):
        return "repeated_sentences"
    return None


def reject_code(doc: str) -> str | None:
    n = len(doc)
    if n < 120:
        return "too_short"
    head = doc[:SLICE]
    if "\ufffd" in head:
        return "mojibake"
    if CODE_GENERATED_RE.search(head):
        return "generated"
    lines = doc.split("\n")
    if len(lines) < 5:
        return "too_few_lines"
    if max(map(len, lines)) > 2000:
        return "minified"
    if LONGTOKEN_RE.search(doc) and B64_RE.search(doc):
        return "binary_blob"
    if not CODE_SIGNAL_RE.search(doc):
        return "not_code"
    lc = doc.lower()
    if _any_anchor(lc, ADULT_ANCHORS) and ADULT_STRONG.search(doc):
        return "adult"
    # A file that is almost entirely a licence header carries no signal and is
    # duplicated across millions of repositories.
    if n < 1500 and CODE_LICENSE_RE.search(head):
        return "license_only"
    return None


def reject_math(doc: str) -> str | None:
    n = len(doc)
    if n < 60:
        return "too_short"
    head = doc[:SLICE]
    r = _shared_reject(doc, n, head)
    if r:
        return r
    return _adult_gamble(doc, doc.lower(), ja=False)


def reject_mixed(doc: str) -> str | None:
    """Generated bilingual/fact text: it is curated, so only sanity checks.

    The language rules must not apply — every document here is deliberately half
    English and half Japanese, which reject_en calls "not_english" and reject_ja
    calls "mostly_latin". Deduplication is also skipped by the caller, because
    repetition is the entire point of knowledge injection.
    """
    n = len(doc)
    if n < 40:
        return "too_short"
    head = doc[:SLICE]
    if "\ufffd" in head:
        return "mojibake"
    if CTRL_RE.search(head):
        return "control_bytes"
    return _adult_gamble(doc, doc.lower(), ja=True)


def math_score(head: str) -> tuple[float, int]:
    """(density per 1000 chars, absolute hit count) of mathematical notation."""
    lc = head.lower()
    if not _any_anchor(lc, MATH_ANCHORS):
        return 0.0, 0
    hits = len(MATH_SYMBOL_RE.findall(head)) * 2 + len(MATH_WORD_RE.findall(head))
    return 1000.0 * hits / max(1, len(head)), hits


# Both a density and an absolute floor: density alone promotes a two-sentence
# fragment that happens to say "equation", and an absolute count alone promotes
# any long document. Measured on 6000 fineweb_edu documents the notation-only
# density is p99=1.5 / max=7.5, so 6.0 sits far out in the tail.
MATH_ROUTE_THRESHOLD = 6.0
MATH_MIN_HITS = 8


# ---------------------------------------------------------------------------
# Document readers
# ---------------------------------------------------------------------------
def mean_line_bytes(path: Path, probe: int = 4_000_000) -> float:
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(size // 3)
        f.readline()
        n = tot = 0
        while tot < probe:
            line = f.readline()
            if not line:
                break
            tot += len(line)
            n += 1
    return tot / max(1, n)


def iter_docs_per_line(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if line.strip():
                yield line


def iter_docs_paragraph(path: Path, target: int = 4096):
    """Group consecutive lines into pseudo-documents.

    Verified on wiki_en: consecutive lines are consecutive paragraphs of the
    same article, so grouping restores document-level coherence that the
    one-line-per-record layout had destroyed. A blank line ends a group.
    """
    buf: list[str] = []
    size = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                if buf:
                    yield "\n".join(buf)
                    buf, size = [], 0
                continue
            buf.append(line)
            size += len(line) + 1
            if size >= target:
                yield "\n".join(buf)
                buf, size = [], 0
    if buf:
        yield "\n".join(buf)


REPO_RE = re.compile(r"^<reponame>\s*(\S+)\s*$")


def iter_docs_code_lines(path: Path, target: int = 6144):
    """Reassemble source files from raw source lines.

    starcoder_clean.txt is one source line per record with <reponame> markers;
    blank lines are content (they separate functions) and must survive.
    """
    buf: list[str] = []
    size = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if REPO_RE.match(line):
                if size:
                    text = "\n".join(buf).strip("\n")
                    buf, size = [], 0
                    if text:
                        yield text
                else:
                    buf, size = [], 0
                continue
            buf.append(line)
            size += len(line) + 1
            # Break on a blank line so a function is never cut in half.
            if size >= target and not line.strip():
                text = "\n".join(buf).strip("\n")
                buf, size = [], 0
                if text:
                    yield text
    if size:
        text = "\n".join(buf).strip("\n")
        if text:
            yield text


def iter_docs_math_jsonl(path: Path, fields):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys = fields or ("prompt", "question", "problem", "instruction",
                              "input", "answer", "solution", "reference",
                              "response", "output", "rationale", "cot")
            parts = []
            for k in keys:
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
            text = "\n".join(parts).strip()
            if text:
                yield repair_newlines(text)


CHAT_TAG_RE = re.compile(r"<\|(?:system|user|assistant|end|endoftext|think|/think|"
                         r"code|/code|translate|summary|pad|lang_\w+)\|>")


def iter_docs_math_sft(path: Path):
    """Local math SFT rows as plain prose.

    The chat markup is removed: pretraining must not learn the conversation
    template, and the Indonesian rows are dropped because this corpus is
    EN/JA/code/math only.
    """
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("lang") == "id":
                continue
            text = obj.get("text")
            if not isinstance(text, str):
                continue
            text = CHAT_TAG_RE.sub("\n", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if text:
                yield repair_newlines(text)


def file_has_separator(path: Path, probe: int = 1 << 20) -> bool:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        read = 0
        for line in f:
            read += len(line)
            if line.rstrip("\n").rstrip("\r") == DOC_SEP:
                return True
            if read >= probe:
                break
    return False


def iter_docs_separated(path: Path):
    """Documents delimited by a line holding exactly DOC_SEP, newlines intact.

    This is what scripts/fetch_corpus.py writes, and the only layout that keeps
    source code trainable — the builder rejects single-line code."""
    buf: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if line == DOC_SEP:
                if buf:
                    text = "\n".join(buf).strip("\n")
                    buf = []
                    if text:
                        yield text
                continue
            buf.append(line)
    if buf:
        text = "\n".join(buf).strip("\n")
        if text:
            yield text


def make_reader(path: Path, mode: str):
    if file_has_separator(path):
        return iter_docs_separated(path), "doc-separated"
    avg = mean_line_bytes(path)
    if mode == "code":
        if avg < 200:
            # raw source lines -> reassemble whole files
            return iter_docs_code_lines(path), f"code-reassemble(avg={avg:.0f}B)"
        # already chunked by clean_code_corpus.py: one chunk per line, with
        # newlines stored as U+23CE. repair_newlines() undoes that.
        return (repair_newlines(d) for d in iter_docs_per_line(path)), \
               f"code-chunk(avg={avg:.0f}B)"
    if mode == "auto":
        if avg >= 1200:
            return (repair_newlines(d) for d in iter_docs_per_line(path)), \
                   f"doc-per-line(avg={avg:.0f}B)"
        return iter_docs_paragraph(path), f"paragraph-group(avg={avg:.0f}B)"
    return iter_docs_per_line(path), mode


# ---------------------------------------------------------------------------
# Sharded writer
# ---------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, out_dir: Path, label: str, shard_bytes: int):
        self.dir = out_dir
        self.label = label
        self.shard_bytes = shard_bytes
        self.idx = 0
        self.written = 0
        self.shard_written = 0
        self.docs = 0
        self.f = None

    def _open(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / f"{self.label}_{self.idx:02d}.txt"
        self.f = p.open("wb")

    def write(self, blob: bytes):
        if self.f is None or self.shard_written >= self.shard_bytes:
            if self.f:
                self.f.close()
                self.idx += 1
            self.shard_written = 0
            self._open()
        self.f.write(blob)
        self.written += len(blob)
        self.shard_written += len(blob)
        self.docs += 1

    def close(self):
        if self.f:
            self.f.close()
            self.f = None


def resolve(rel: str) -> Path:
    """'../x' resolves against the sibling-project root, everything else against
    the repo root."""
    return (SIBLINGS / rel[3:]) if rel.startswith("../") else (ROOT / rel)


# ---------------------------------------------------------------------------
def build(args):
    out_root = ROOT / args.out
    budget = dict(DEFAULT_BUDGET)
    for k in budget:
        v = getattr(args, f"budget_{k}", None)
        if v:
            budget[k] = v

    bloom = Bloom(mib=args.bloom_mib)
    stats: dict[str, dict] = {}
    writers: dict[str, ShardWriter] = {}
    bucket_bytes = {k: 0 for k in budget}

    def writer_for(bucket: str, label: str) -> ShardWriter:
        key = f"{bucket}/{label}"
        if key not in writers:
            writers[key] = ShardWriter(out_root / bucket, label, args.shard_bytes)
        return writers[key]

    rejectors = {
        "english": reject_en,
        "japanese": reject_ja,
        "code": reject_code,
        "math": reject_math,
        "mixed": reject_mixed,
    }

    def process(bucket, label, docs, reader_desc, byte_cap):
        st = {
            "source": label,
            "bucket": bucket,
            "reader": reader_desc,
            "read_docs": 0,
            "read_bytes": 0,
            "kept_docs": 0,
            "kept_bytes": 0,
            "routed_math": 0,
            "drop": {},
        }
        reject = rejectors[bucket]
        t0 = time.time()
        print(f"  {label:20s} {reader_desc:28s} reading...", flush=True)
        next_tick = t0 + args.progress_secs
        # No single source may own more than this much of its bucket, so the
        # mix stays diverse. Without it fineweb_edu_extra alone filled 6.5 GB
        # of the 6.5 GB English budget and wiki_en/openwebtext/fineweb were
        # skipped entirely.
        share = args.max_source_share or SOURCE_SHARE[bucket]
        source_cap = budget[bucket] * share
        sep = ("\n" + DOC_SEP + "\n").encode("utf-8")
        for doc in docs:
            st["read_docs"] += 1
            st["read_bytes"] += len(doc)

            target_bucket = bucket
            if bucket == "english" and bucket_bytes["math"] < budget["math"]:
                dens, hits = math_score(doc[:SLICE])
                if dens >= MATH_ROUTE_THRESHOLD and hits >= MATH_MIN_HITS:
                    target_bucket = "math"

            reason = rejectors[target_bucket](doc)
            if reason:
                st["drop"][reason] = st["drop"].get(reason, 0) + 1
                continue
            # Knowledge injection is repetition by design — deduplicating it
            # would delete the whole bucket after the first exposure.
            if target_bucket != "mixed" and not bloom.add_if_new(dedup_key(doc)):
                st["drop"]["duplicate"] = st["drop"].get("duplicate", 0) + 1
                continue

            if bucket_bytes[target_bucket] >= budget[target_bucket]:
                if target_bucket == bucket:
                    break
                st["drop"]["math_budget_full"] = st["drop"].get("math_budget_full", 0) + 1
                continue

            blob = doc.encode("utf-8") + sep
            if not args.audit:
                writer_for(target_bucket, label).write(blob)
            bucket_bytes[target_bucket] += len(blob)
            if target_bucket == bucket:
                st["kept_docs"] += 1
                st["kept_bytes"] += len(blob)
            else:
                st["routed_math"] += 1

            if st["read_bytes"] >= byte_cap:
                break
            if st["kept_bytes"] >= source_cap:
                print(f"      [{label}] source cap "
                      f"{source_cap/1e9:.2f} GB reached", flush=True)
                break
            if args.audit and st["read_docs"] >= args.audit_docs:
                break

            now = time.time()
            if now >= next_tick:
                next_tick = now + args.progress_secs
                el = now - t0
                print(f"      [{label}] read {st['read_bytes']/1e9:5.2f} GB -> "
                      f"{bucket} {bucket_bytes[bucket]/1e9:5.2f}/"
                      f"{budget[bucket]/1e9:.1f} GB "
                      f"({100*bucket_bytes[bucket]/budget[bucket]:4.1f}%)  "
                      f"kept {100.0*st['kept_docs']/max(1,st['read_docs']):4.1f}%  "
                      f"{st['read_bytes']/1e6/max(el,1):5.1f} MB/s", flush=True)
        st["seconds"] = round(time.time() - t0, 1)
        stats[f"{bucket}/{label}"] = st
        pct = 100.0 * st["kept_docs"] / max(1, st["read_docs"])
        print(
            f"  {label:20s} {reader_desc:28s} read={st['read_docs']:>9,d} "
            f"kept={st['kept_docs']:>9,d} ({pct:5.1f}%) "
            f"math={st['routed_math']:>7,d}  {st['seconds']:>6.1f}s",
            flush=True,
        )
        top = sorted(st["drop"].items(), key=lambda kv: -kv[1])[:6]
        if top:
            print("      drops: " + ", ".join(f"{k}={v:,}" for k, v in top), flush=True)

    print(f"[build] out={out_root}  audit={args.audit}")
    print(f"[build] budgets: " + ", ".join(f"{k}={v/1e9:.1f}GB" for k, v in budget.items()))

    for bucket, label, rel, mode in SOURCES:
        path = resolve(rel)
        if not path.exists():
            print(f"  !! MISSING {rel}", flush=True)
            continue
        if bucket_bytes[bucket] >= budget[bucket]:
            print(f"  -- {label}: {bucket} budget already met, skipping", flush=True)
            continue
        docs, desc = make_reader(path, mode)
        cap = args.audit_bytes if args.audit else args.max_read_per_source
        process(bucket, label, docs, desc, cap)

    for label, rel, fields in MATH_JSONL:
        path = ROOT / rel
        if not path.exists():
            print(f"  !! MISSING {rel}", flush=True)
            continue
        cap = args.audit_bytes if args.audit else args.max_read_per_source
        process("math", label, iter_docs_math_jsonl(path, fields), "jsonl", cap)

    for label, rel in MATH_SFT:
        path = ROOT / rel
        if not path.exists():
            print(f"  !! MISSING {rel}", flush=True)
            continue
        cap = args.audit_bytes if args.audit else args.max_read_per_source
        process("math", label, iter_docs_math_sft(path), "sft-jsonl", cap)

    raw_math = ROOT / RAW_DIRS["math"]
    if raw_math.is_dir():
        for p in sorted(raw_math.glob("*.txt")):
            cap = args.audit_bytes if args.audit else args.max_read_per_source
            docs, desc = make_reader(p, "auto")
            process("math", p.stem, docs, desc, cap)

    # Fetched corpora for the other buckets, read last so curated sources win
    # the first-wins deduplication.
    for bucket in ("english", "japanese", "code"):
        d = ROOT / RAW_DIRS[bucket]
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.txt")):
            if bucket_bytes[bucket] >= budget[bucket]:
                print(f"  -- {p.stem}: {bucket} budget already met, skipping",
                      flush=True)
                continue
            cap = args.audit_bytes if args.audit else args.max_read_per_source
            docs, desc = make_reader(p, "code" if bucket == "code" else "auto")
            process(bucket, p.stem, docs, desc, cap)
    for w in writers.values():
        w.close()

    # ---- balance report --------------------------------------------------
    # The point of this block is that a bucket which came up short must not
    # silently shrink the model's exposure to that language. Byte budgets are
    # a proxy; what the trainer samples is tokens, so convert and print the
    # TW_W_* values that restore the intended token mix over whatever was
    # actually built.
    tokens = {k: bucket_bytes[k] / BYTES_PER_TOKEN[k] for k in bucket_bytes}
    unique_total = sum(tokens.values())
    print("\n[build] bucket totals")
    print(f"  {'bucket':9} {'GB':>7} {'budget':>8} {'unique tok':>12} {'share':>7} "
          f"{'target':>7} {'epochs@10B':>11}")
    for k in ("english", "japanese", "code", "math", "mixed"):
        share = tokens[k] / max(1.0, unique_total)
        want = TARGET_TOKEN_SHARE[k]
        ep = (CHINCHILLA_TOKENS * want) / max(1.0, tokens[k])
        print(f"  {k:9} {bucket_bytes[k]/1e9:7.2f} {budget[k]/1e9:8.1f} "
              f"{tokens[k]/1e9:11.2f}B {100*share:6.1f}% {100*want:6.1f}% {ep:11.2f}")
    print(f"  {'TOTAL':9} {sum(bucket_bytes.values())/1e9:7.2f} "
          f"{sum(budget.values())/1e9:8.1f} {unique_total/1e9:11.2f}B")
    print(f"  Chinchilla target for 492M @ 20 tok/param: "
          f"{CHINCHILLA_TOKENS/1e9:.1f}B tokens "
          f"({CHINCHILLA_TOKENS/max(1.0, unique_total):.2f} epochs over this corpus)")

    # Sampling weights are relative, so normalising the target share by the
    # available tokens makes every bucket finish its share at the same time.
    weights = {k: TARGET_TOKEN_SHARE[k] for k in tokens}
    live = {k: v for k, v in weights.items() if tokens[k] > 0}
    s = sum(live.values()) or 1.0
    weights = {k: (live.get(k, 0.0) / s) for k in tokens}
    print("\n[build] sampling weights for the trainer "
          "(export before train_pretrain_cuda):")
    env_names = {"english": "TW_W_EN", "japanese": "TW_W_JA",
                 "code": "TW_W_CODE", "math": "TW_W_MATH",
                 "mixed": "TW_W_MIXED"}
    for k in ("english", "japanese", "code", "math", "mixed"):
        flag = "" if tokens[k] > 0 else "   # bucket is empty"
        print(f"  export {env_names[k]}={weights[k]:.3f}{flag}")
    for k in ("english", "japanese", "code", "math", "mixed"):
        ep = (CHINCHILLA_TOKENS * TARGET_TOKEN_SHARE[k]) / max(1.0, tokens[k])
        if tokens[k] > 0 and ep > 4.0 and k != "mixed":
            print(f"  !! '{k}' would be repeated {ep:.1f}x to hit its share. "
                  f"Above ~4 epochs repeated tokens stop paying for themselves "
                  f"— fetch more {k} data or lower {env_names[k]}.")

    report = {
        "out": str(out_root),
        "audit": args.audit,
        "doc_separator": DOC_SEP,
        "budget": budget,
        "bucket_bytes": bucket_bytes,
        "bucket_tokens_est": tokens,
        "bytes_per_token": BYTES_PER_TOKEN,
        "target_token_share": TARGET_TOKEN_SHARE,
        "chinchilla_tokens": CHINCHILLA_TOKENS,
        "sampling_weights": weights,
        "unique_docs_seen": bloom.n,
        "sources": stats,
    }
    rp = ROOT / "logs" / ("pretrain_corpus_audit.json" if args.audit
                          else "pretrain_corpus_build.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[build] report -> {rp}")

    if not args.audit:
        (out_root / "stats.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for k in ("english", "japanese", "code", "math", "mixed"):
            if bucket_bytes[k] == 0:
                print(f"  !! WARNING: bucket '{k}' is EMPTY — the trainer will "
                      f"silently fall back to a different sampling mode")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/corpus_pretrain_v8")
    ap.add_argument("--audit", action="store_true",
                    help="sample each source, report keep/drop rates, write nothing")
    ap.add_argument("--audit-docs", type=int, default=40_000)
    ap.add_argument("--audit-bytes", type=int, default=60_000_000)
    ap.add_argument("--max-read-per-source", type=int, default=40_000_000_000)
    ap.add_argument("--shard-bytes", type=int, default=SHARD_BYTES)
    ap.add_argument("--bloom-mib", type=int, default=128)
    ap.add_argument("--progress-secs", type=float, default=30.0,
                    help="seconds between progress lines while reading a source")
    ap.add_argument("--max-source-share", type=float, default=0.0,
                    help="override the per-bucket cap on one source's share")
    for k, v in DEFAULT_BUDGET.items():
        ap.add_argument(f"--budget-{k}", type=int, default=v)
    args = ap.parse_args()
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
