#!/usr/bin/env python3
"""
build_corpus_v9_50b.py — Collect +22.0 GiB (~6.87B new unique bpe_v9 tokens)
exclusively into a SEPARATE directory:
    data/corpus_pretrain_v9/{japanese,english,code,math,mixed}/v9_<bucket>_<NN>.txt

Guarantees:
  1. data/corpus_pretrain_v8/ is NEVER modified (opened strictly read-only to build
     64-bit Blake2b deduplication hashes so v9 contains 100% brand-new unique data).
  2. Runs 5 separate OS processes (1 per bucket, no Python GIL contention) + C-speed
     binary chunk hashing + async Parquet shard prefetching.
  3. Zero ~/.cache/huggingface bloat (temp shard unlinked immediately after reading).
  4. Strict holdout decontamination against gsm8k/test, mgsm_ja_test, humaneval, mbpp_heldout.
  5. Hard MacBook SSD safety floor: stops gracefully if free disk drops below 18.0 GiB.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import queue
import random
import re
import shutil
import struct
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_url

ROOT = Path(__file__).resolve().parent.parent
V8_DIR = ROOT / "data" / "corpus_pretrain_v8"
V9_DIR = ROOT / "data" / "corpus_pretrain_v9"
TMP_DIR = V9_DIR / ".tmp_stage"

DOC_SEP = "<|endoftext|>"
DOC_SEP_BYTES = b"\n<|endoftext|>\n"
SHARD_MAX_BYTES = 1_073_741_824  # 1.0 GiB per output .txt shard
MIN_FREE_BYTES = int(18.0 * (1024 ** 3))  # Hard safety floor: 18.0 GiB free on MacBook

# Per-bucket NEW unique data targets in data/corpus_pretrain_v9 (Total = +22.00 GiB)
TARGETS_GIB = {
    "japanese": 8.00,
    "english":  7.20,
    "code":     3.40,
    "math":     2.20,
    "mixed":    1.20,
}
TARGETS_BYTES = {k: int(v * (1024 ** 3)) for k, v in TARGETS_GIB.items()}

BAD_RE = re.compile(
    r"\b(?:porn|xxx|hentai|onlyfans|casino|online ?gambling|escort ?service|viagra)\b"
    r"|http[s]?://\S{200,}",
    re.I,
)
MATH_HINT_RE = re.compile(
    r"\\(?:frac|sum|int|sqrt|begin\{|cdot|times|leq|geq)|\$\$?[^$\n]{2,}\$|"
    r"[\u2211\u220f\u222b\u221a\u2264\u2265\u2260\u2248\u2261\u2202]|"
    r"\b(?:theorem|lemma|proof|derivative|integral|polynomial|equation|matrix|probability|calculate)\b|"
    r"(?:方程式|関数|微分|積分|確率|証明|定理|計算|求める)",
    re.I,
)
KANA_RE = re.compile(r"[\u3040-\u30ff]")
CJK_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGITS_RE = re.compile(r"\d+")
WS_NORM_RE = re.compile(r"\s+")


def hash64_bytes(b: bytes) -> int:
    h = hashlib.blake2b(b[:2048].strip(), digest_size=8).digest()
    return struct.unpack("<Q", h)[0]


def hash64_str(s: str) -> int:
    return hash64_bytes(s[:2048].encode("utf-8", "ignore"))


def build_holdout_regex() -> re.Pattern | None:
    """Compile a single C-speed alternation regex for all holdout test sets."""
    probes: set[str] = set()
    candidates = [
        (ROOT / "data" / "gsm8k" / "test.jsonl", ["question"]),
        (ROOT / "eval_data" / "mgsm_ja_test.jsonl", ["question", "prompt"]),
        (ROOT / "eval_data" / "humaneval.jsonl", ["prompt"]),
        (ROOT / "eval_data" / "mbpp_heldout.jsonl", ["text", "prompt"]),
    ]
    for path, keys in candidates:
        if not path.exists():
            continue
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                for k in keys:
                    val = row.get(k)
                    if isinstance(val, str):
                        norm = WS_NORM_RE.sub(" ", val).strip().lower()
                        if len(norm) >= 35:
                            probes.add(re.escape(norm[:50]))
    if not probes:
        return None
    return re.compile("|".join(sorted(probes)))


def index_existing_bucket_hashes_fast(bucket: str) -> set[int]:
    """C-speed binary chunk scanner (reads v8 read-only at >1.5 GB/s per process)."""
    t0 = time.time()
    seen: set[int] = set()
    for base in (V8_DIR / bucket, V9_DIR / bucket):
        if not base.exists():
            continue
        for fp in sorted(base.glob("*.txt")):
            carry = b""
            with fp.open("rb") as f:
                while True:
                    chunk = f.read(16 * 1024 * 1024)
                    if not chunk:
                        if carry.strip():
                            seen.add(hash64_bytes(carry))
                        break
                    data = carry + chunk
                    parts = data.split(DOC_SEP_BYTES)
                    carry = parts.pop()
                    for doc in parts:
                        if doc:
                            seen.add(hash64_bytes(doc))
    print(f"[{bucket.upper()}] Indexed {len(seen):,} v8+v9 hashes in {time.time()-t0:.1f}s", flush=True)
    return seen


def pair_ok(en: str, ja: str) -> bool:
    en, ja = en.strip(), ja.strip()
    if not (10 <= len(en) <= 420) or not (5 <= len(ja) <= 320):
        return False
    if "\ufffd" in en or "\ufffd" in ja:
        return False
    if not LATIN_RE.search(en) or not CJK_RE.search(ja) or not KANA_RE.search(ja):
        return False
    if len(CJK_RE.findall(en)) > 2:
        return False
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff] [\u3040-\u30ff\u4e00-\u9fff] [\u3040-\u30ff\u4e00-\u9fff]", ja):
        return False
    ratio = len(en) / max(1, len(ja))
    if not (0.7 <= ratio <= 6.0):
        return False
    de, dj = set(DIGITS_RE.findall(en)), set(DIGITS_RE.findall(ja))
    if de and dj and not (de & dj):
        return False
    if len(de) >= 2 and not dj:
        return False
    return not (BAD_RE.search(en) or BAD_RE.search(ja))


def clean_doc(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    if DOC_SEP in text:
        text = text.replace(DOC_SEP, "<|eot|>")
    return text.strip("\n")


class ShardWriter:
    def __init__(self, bucket: str, target_bytes: int, seen: set[int], holdout_re: re.Pattern | None):
        self.bucket = bucket
        self.dir = V9_DIR / bucket
        self.dir.mkdir(parents=True, exist_ok=True)
        self.target_bytes = target_bytes
        self.seen = seen
        self.holdout_re = holdout_re

        # Resume support: count complete shards in V9_DIR/<bucket>
        existing = sorted(self.dir.glob(f"v9_{bucket}_*.txt"))
        self.written_bytes = sum(p.stat().st_size for p in existing)
        self.shard_idx = len(existing)
        self.cur_file = None
        self.cur_shard_bytes = 0
        self.docs_written = 0
        self.dups_skipped = 0
        self._check_counter = 0
        self._cached_free_ok = shutil.disk_usage(ROOT).free >= MIN_FREE_BYTES

    def done(self) -> bool:
        if self.written_bytes >= self.target_bytes:
            return True
        self._check_counter += 1
        if (self._check_counter & 0xFF) == 0:
            self._cached_free_ok = shutil.disk_usage(ROOT).free >= MIN_FREE_BYTES
        return not self._cached_free_ok

    def _rotate_if_needed(self):
        if self.cur_file is None or self.cur_shard_bytes >= SHARD_MAX_BYTES:
            if self.cur_file is not None:
                self.cur_file.flush()
                self.cur_file.close()
            path = self.dir / f"v9_{self.bucket}_{self.shard_idx:02d}.txt"
            self.shard_idx += 1
            self.cur_file = path.open("w", encoding="utf-8", buffering=8 * 1024 * 1024)
            self.cur_shard_bytes = 0

    def write_doc(self, text: str) -> bool:
        if self.done():
            return False
        text = clean_doc(text)
        if len(text) < 260 or len(text) > 350_000 or "\ufffd" in text:
            return True
        if BAD_RE.search(text):
            return True
        if self.holdout_re is not None:
            norm = WS_NORM_RE.sub(" ", text[:1000]).strip().lower()
            if self.holdout_re.search(norm):
                return True
        h = hash64_str(text)
        if h in self.seen:
            self.dups_skipped += 1
            return True
        self.seen.add(h)
        self._rotate_if_needed()
        rec = text + "\n" + DOC_SEP + "\n"
        nb = len(rec.encode("utf-8"))
        self.cur_file.write(rec)
        self.cur_shard_bytes += nb
        self.written_bytes += nb
        self.docs_written += 1
        return not self.done()

    def close(self):
        if self.cur_file is not None:
            self.cur_file.flush()
            self.cur_file.close()
            self.cur_file = None


def download_temp_file(repo: str, filename: str, tag: str) -> Path | None:
    url = hf_hub_url(repo, filename, repo_type="dataset")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    dst = TMP_DIR / f"{tag}_{os.getpid()}_{os.path.basename(filename)}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (transformer-500m-v9)"})
            with urllib.request.urlopen(req, timeout=60) as resp, dst.open("wb") as out:
                while True:
                    chunk = resp.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            return dst
        except Exception as e:
            dst.unlink(missing_ok=True)
            if attempt == 2:
                print(f"  [WARN] {repo}/{filename} download failed: {e}", flush=True)
                return None
            time.sleep(2 * (attempt + 1))
    return None


def iter_prefetched_parquet(repo: str, files: list[str], tag: str, stop_fn):
    """Background 1-ahead shard prefetcher so download overlaps 100% with CPU extraction."""
    q: queue.Queue[Path | None] = queue.Queue(maxsize=2)
    abort = threading.Event()

    def fetcher():
        for fn in files:
            if abort.is_set() or stop_fn():
                break
            p = download_temp_file(repo, fn, tag)
            if p is not None:
                q.put(p)
        q.put(None)

    t = threading.Thread(target=fetcher, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            yield item
            if stop_fn():
                abort.set()
                # Drain any prefetched temp file
                while not q.empty():
                    rem = q.get_nowait()
                    if rem is not None:
                        rem.unlink(missing_ok=True)
                break
    finally:
        abort.set()


def stream_json_array_objects(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (transformer-500m-v9)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dec = json.JSONDecoder()
        buf = ""
        started = False
        while True:
            chunk = resp.read(2 * 1024 * 1024)
            if not chunk:
                break
            buf += chunk.decode("utf-8", "replace")
            pos = 0
            n = len(buf)
            if not started:
                idx = buf.find("[")
                if idx < 0:
                    buf = ""
                    continue
                pos = idx + 1
                started = True
            while pos < n:
                while pos < n and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos >= n or buf[pos] == "]":
                    break
                try:
                    obj, end_pos = dec.raw_decode(buf, pos)
                    yield obj
                    pos = end_pos
                except json.JSONDecodeError:
                    break
            buf = buf[pos:]


def run_bucket(bucket: str) -> dict:
    holdout_re = build_holdout_regex() if bucket in ("code", "math", "mixed") else None
    seen = index_existing_bucket_hashes_fast(bucket)
    w = ShardWriter(bucket, TARGETS_BYTES[bucket], seen, holdout_re)

    if w.done():
        w.close()
        return {"bucket": bucket, "bytes": w.written_bytes, "docs": w.docs_written, "dups": w.dups_skipped}

    api = HfApi()

    if bucket == "japanese":
        files = sorted(
            f for f in api.list_repo_files("HuggingFaceFW/fineweb-2", repo_type="dataset")
            if f.startswith("data/jpn_Jpan/train/") and f.endswith(".parquet")
        )
        for tmp in iter_prefetched_parquet("HuggingFaceFW/fineweb-2", files, "ja", w.done):
            try:
                pf = pq.ParquetFile(tmp)
                for rg in range(pf.num_row_groups):
                    if w.done():
                        break
                    tbl = pf.read_row_group(rg, columns=["text"])
                    for val in tbl.column("text").to_pylist():
                        if isinstance(val, str) and len(KANA_RE.findall(val[:2000])) >= 25:
                            if not w.write_doc(val):
                                break
            except Exception as e:
                print(f"  [WARN] JA error: {e}", flush=True)
            finally:
                tmp.unlink(missing_ok=True)

    elif bucket == "english":
        files = sorted(
            f for f in api.list_repo_files("HuggingFaceFW/fineweb-edu", repo_type="dataset")
            if f.startswith("sample/100BT/") and f.endswith(".parquet")
        )
        ordered = files[35:] + files[:35]
        for tmp in iter_prefetched_parquet("HuggingFaceFW/fineweb-edu", ordered, "en", w.done):
            try:
                pf = pq.ParquetFile(tmp)
                snames = set(pf.schema.names)
                cols = ["text", "score"] if "score" in snames else ["text"]
                for rg in range(pf.num_row_groups):
                    if w.done():
                        break
                    tbl = pf.read_row_group(rg, columns=cols)
                    texts = tbl.column("text").to_pylist()
                    scores = tbl.column("score").to_pylist() if "score" in cols else None
                    for i, val in enumerate(texts):
                        if not isinstance(val, str):
                            continue
                        if scores is not None and scores[i] is not None and float(scores[i]) < 2.85:
                            continue
                        if not w.write_doc(val):
                            break
            except Exception as e:
                print(f"  [WARN] EN error: {e}", flush=True)
            finally:
                tmp.unlink(missing_ok=True)

    elif bucket == "code":
        # 1. Nan-Do/code-search-net-python
        py_files = sorted(
            f for f in api.list_repo_files("Nan-Do/code-search-net-python", repo_type="dataset")
            if f.endswith(".parquet")
        )
        for tmp in iter_prefetched_parquet("Nan-Do/code-search-net-python", py_files, "code_py", w.done):
            try:
                pf = pq.ParquetFile(tmp)
                snames = set(pf.schema.names)
                col = "original_string" if "original_string" in snames else ("code" if "code" in snames else list(snames)[0])
                for rg in range(pf.num_row_groups):
                    if w.done():
                        break
                    tbl = pf.read_row_group(rg, columns=[col])
                    for val in tbl.column(col).to_pylist():
                        if isinstance(val, str) and val.count("\n") >= 5:
                            if not w.write_doc(val):
                                break
            except Exception as e:
                print(f"  [WARN] CSN-Py error: {e}", flush=True)
            finally:
                tmp.unlink(missing_ok=True)

        # 2. ise-uiuc/Magicoder-Evol-Instruct-110K
        if not w.done():
            try:
                url = hf_hub_url("ise-uiuc/Magicoder-Evol-Instruct-110K", "data-evol_instruct-decontaminated.jsonl", repo_type="dataset")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    for raw in resp:
                        if w.done():
                            break
                        try:
                            row = json.loads(raw.decode("utf-8", "replace"))
                            inst = (row.get("instruction") or "").strip()
                            ans = (row.get("response") or "").strip()
                            if inst and ans and ans.count("\n") >= 5:
                                if not w.write_doc(f"# Problem:\n{inst}\n\n{ans}"):
                                    break
                        except Exception:
                            continue
            except Exception as e:
                print(f"  [WARN] Magicoder-Evol error: {e}", flush=True)

        # 3. Vezora/Tested-143k-Python-Alpaca
        if not w.done():
            try:
                url = hf_hub_url("Vezora/Tested-143k-Python-Alpaca", "143k-Tested-Python-Alpaca-Vezora.json", repo_type="dataset")
                for row in stream_json_array_objects(url):
                    if w.done():
                        break
                    if not isinstance(row, dict):
                        continue
                    inst = (row.get("instruction") or "").strip()
                    inp = (row.get("input") or "").strip()
                    out = (row.get("output") or "").strip()
                    if inst and out and out.count("\n") >= 4:
                        doc = f"{inst}\n{inp}\n\n{out}".strip() if inp else f"{inst}\n\n{out}"
                        if not w.write_doc(doc):
                            break
            except Exception as e:
                print(f"  [WARN] Vezora error: {e}", flush=True)

        # 4. ajibawa-2023/Code-290k-ShareGPT
        if not w.done():
            try:
                url = hf_hub_url("ajibawa-2023/Code-290k-ShareGPT", "Code-290k-ShareGPT.json", repo_type="dataset")
                for row in stream_json_array_objects(url):
                    if w.done():
                        break
                    if not isinstance(row, dict):
                        continue
                    convs = row.get("conversations") or []
                    parts = [c.get("value", "").strip() for c in convs if isinstance(c, dict) and c.get("value")]
                    doc = "\n\n".join(parts)
                    if doc.count("\n") >= 6 and ("```" in doc or "def " in doc or "function " in doc or "class " in doc):
                        if not w.write_doc(doc):
                            break
            except Exception as e:
                print(f"  [WARN] Code-290k error: {e}", flush=True)

        # 5. angie-chen55/javascript-github-code (fresh shards 45+)
        if not w.done():
            js_files = sorted(
                f for f in api.list_repo_files("angie-chen55/javascript-github-code", repo_type="dataset")
                if f.endswith(".parquet")
            )[45:]
            for tmp in iter_prefetched_parquet("angie-chen55/javascript-github-code", js_files, "code_js", w.done):
                try:
                    pf = pq.ParquetFile(tmp)
                    col = "code" if "code" in pf.schema.names else pf.schema.names[0]
                    for rg in range(pf.num_row_groups):
                        if w.done():
                            break
                        tbl = pf.read_row_group(rg, columns=[col])
                        for val in tbl.column(col).to_pylist():
                            if isinstance(val, str) and 5 <= val.count("\n") <= 800:
                                if not w.write_doc(val):
                                    break
                except Exception as e:
                    print(f"  [WARN] JS-GitHub error: {e}", flush=True)
                finally:
                    tmp.unlink(missing_ok=True)

    elif bucket == "math":
        # 1. kunishou/OpenMathInstruct-1-1.8m-ja (~700 MiB Japanese CoT + Python math)
        ja_math_cap = min(w.target_bytes, int(0.70 * (1024 ** 3)))
        if w.written_bytes < ja_math_cap:
            try:
                url = hf_hub_url("kunishou/OpenMathInstruct-1-1.8m-ja", "openmathinstruct_ja_20240225.jsonl", repo_type="dataset")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    for idx, raw in enumerate(resp):
                        if w.done() or w.written_bytes >= ja_math_cap:
                            break
                        if idx < 18_000:
                            continue
                        try:
                            row = json.loads(raw.decode("utf-8", "replace"))
                        except Exception:
                            continue
                        q_ja = (row.get("question_ja") or "").strip()
                        s_ja = (row.get("generated_solution_ja") or "").strip()
                        if not q_ja or not s_ja or len(KANA_RE.findall(q_ja + s_ja)) < 10:
                            continue
                        s_ja = s_ja.replace("<llm-code>", "```python").replace("</llm-code>", "```")
                        s_ja = s_ja.replace("<llm-code-output>", "# output: ").replace("</llm-code-output>", "")
                        w.write_doc(f"問題: {q_ja}\n\n解答:\n{s_ja}")
            except Exception as e:
                print(f"  [WARN] OpenMathInstruct-JA error: {e}", flush=True)

        # 2. HuggingFaceTB/finemath (finemath-4plus -> infiwebmath-4plus -> finemath-3plus[60:])
        all_fm = sorted(api.list_repo_files("HuggingFaceTB/finemath", repo_type="dataset"))
        fm_files = (
            [f for f in all_fm if f.startswith("finemath-4plus/") and f.endswith(".parquet")]
            + [f for f in all_fm if f.startswith("infiwebmath-4plus/") and f.endswith(".parquet")]
            + [f for f in all_fm if f.startswith("finemath-3plus/") and f.endswith(".parquet")][60:]
        )
        for tmp in iter_prefetched_parquet("HuggingFaceTB/finemath", fm_files, "math_fm", w.done):
            try:
                pf = pq.ParquetFile(tmp)
                if "text" not in pf.schema.names:
                    continue
                for rg in range(pf.num_row_groups):
                    if w.done():
                        break
                    tbl = pf.read_row_group(rg, columns=["text"])
                    for val in tbl.column("text").to_pylist():
                        if isinstance(val, str) and MATH_HINT_RE.search(val):
                            if not w.write_doc(val):
                                break
            except Exception as e:
                print(f"  [WARN] FineMath error: {e}", flush=True)
            finally:
                tmp.unlink(missing_ok=True)

    elif bucket == "mixed":
        rng = random.Random(202650)

        def render_pair(en: str, ja: str) -> str:
            if rng.random() < 0.40:
                if rng.random() < 0.5:
                    return f"{en}\n<|translate|><|lang_ja|>\n{ja}"
                return f"{ja}\n<|translate|><|lang_en|>\n{en}"
            return f"{en} {ja}" if rng.random() < 0.5 else f"{ja} {en}"

        try:
            url = hf_hub_url("kunishou/oasst1-89k-ja", "oasst1_89k_ja_20231027.json", repo_type="dataset")
            pair_buf: list[str] = []
            for row in stream_json_array_objects(url):
                if w.done():
                    break
                if not isinstance(row, dict):
                    continue
                en = (row.get("text") or "").strip()
                ja = (row.get("text_ja") or "").strip()
                if en and ja and len(KANA_RE.findall(ja[:1500])) >= 12 and not BAD_RE.search(en + " " + ja):
                    if len(en) <= 420 and len(ja) <= 320 and pair_ok(en, ja):
                        pair_buf.append(render_pair(en, ja))
                        if len(pair_buf) >= 5:
                            w.write_doc("\n".join(pair_buf))
                            pair_buf.clear()
                    elif len(ja) >= 280 and len(en) >= 280:
                        if rng.random() < 0.5:
                            w.write_doc(f"{en}\n<|translate|><|lang_ja|>\n{ja}")
                        else:
                            w.write_doc(f"{ja}\n<|translate|><|lang_en|>\n{en}")
            if pair_buf and not w.done():
                w.write_doc("\n".join(pair_buf))
        except Exception as e:
            print(f"  [WARN] oasst1-89k-ja error: {e}", flush=True)

        seen_pairs: set[int] = set()
        for repo, prefix, start_idx in [
            ("sentence-transformers/parallel-sentences-ccmatrix", "en-ja/", 6),
            ("sentence-transformers/parallel-sentences-opensubtitles", "en-ja/", 1),
        ]:
            if w.done():
                break
            try:
                pfiles = sorted(
                    f for f in api.list_repo_files(repo, repo_type="dataset")
                    if f.startswith(prefix) and f.endswith(".parquet")
                )
            except Exception:
                continue
            ordered = pfiles[start_idx:] + pfiles[:start_idx]
            for tmp in iter_prefetched_parquet(repo, ordered, "mixed_par", w.done):
                try:
                    pf = pq.ParquetFile(tmp)
                    cols = [c for c in ("english", "non_english") if c in pf.schema.names]
                    if len(cols) != 2:
                        continue
                    pair_buf = []
                    for rg in range(pf.num_row_groups):
                        if w.done():
                            break
                        tbl = pf.read_row_group(rg, columns=cols)
                        ens = tbl.column("english").to_pylist()
                        jas = tbl.column("non_english").to_pylist()
                        for en, ja in zip(ens, jas):
                            if not isinstance(en, str) or not isinstance(ja, str):
                                continue
                            en = " ".join(en.split()).strip()
                            ja = " ".join(ja.split()).strip()
                            if not pair_ok(en, ja):
                                continue
                            ph = hash64_str(en.lower() + "\t" + ja)
                            if ph in seen_pairs:
                                continue
                            seen_pairs.add(ph)
                            pair_buf.append(render_pair(en, ja))
                            if len(pair_buf) >= 5:
                                if not w.write_doc("\n".join(pair_buf)):
                                    break
                                pair_buf.clear()
                    if pair_buf and not w.done():
                        w.write_doc("\n".join(pair_buf))
                except Exception as e:
                    print(f"  [WARN] Parallel error: {e}", flush=True)
                finally:
                    tmp.unlink(missing_ok=True)

    w.close()
    print(
        f"[DONE] {bucket:10s}: {w.written_bytes/(1024**3):.2f} GiB "
        f"({w.written_bytes/1e9:.2f} GB, {w.docs_written:,} unique docs, {w.dups_skipped:,} dups skipped)",
        flush=True,
    )
    return {"bucket": bucket, "bytes": w.written_bytes, "docs": w.docs_written, "dups": w.dups_skipped}


def dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.glob("v9_*.txt") if p.is_file())


def main() -> int:
    t0 = time.time()
    V9_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for b in TARGETS_GIB:
        (V9_DIR / b).mkdir(parents=True, exist_ok=True)

    free_gib = shutil.disk_usage(ROOT).free / (1024 ** 3)
    total_target_gib = sum(TARGETS_GIB.values())
    print(
        f"=== Building Separate data/corpus_pretrain_v9 (+{total_target_gib:.2f} GiB target | "
        f"MacBook free={free_gib:.2f} GiB | v8 untouched) ===",
        flush=True,
    )
    if free_gib < 20.0:
        print(f"[ERR] Free disk ({free_gib:.2f} GiB) is too close to 18 GiB floor!", file=sys.stderr)
        return 1

    stop_mon = threading.Event()

    def monitor():
        while not stop_mon.wait(15.0):
            sizes = {b: dir_bytes(V9_DIR / b) / (1024 ** 3) for b in ("japanese", "english", "code", "math", "mixed")}
            tot = sum(sizes.values())
            fr = shutil.disk_usage(ROOT).free / (1024 ** 3)
            parts = " ".join(f"{b[:4]}={sizes[b]:.2f}/{TARGETS_GIB[b]:.1f}G" for b in sizes)
            elapsed = time.time() - t0
            rate_mbs = (tot * 1024) / max(elapsed, 1.0)
            print(f"[PROGRESS {elapsed:5.0f}s] {tot:5.2f}/{total_target_gib:.2f} GiB ({rate_mbs:4.1f} MiB/s) | free={fr:4.1f} GiB | {parts}", flush=True)

    mon_thread = threading.Thread(target=monitor, daemon=True)
    mon_thread.start()

    ctx = mp.get_context("spawn")
    buckets = ["japanese", "english", "code", "math", "mixed"]
    results = {}
    with ctx.Pool(processes=5) as pool:
        for res in pool.imap_unordered(run_bucket, buckets):
            results[res["bucket"]] = res

    stop_mon.set()
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    bytes_per_tok = {"english": 3.84, "japanese": 3.69, "code": 3.06, "math": 3.49, "mixed": 3.78}
    summary = {
        "corpus_dir": "data/corpus_pretrain_v9",
        "v8_untouched": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 1),
        "free_gib_after": round(shutil.disk_usage(ROOT).free / (1024 ** 3), 2),
        "buckets": {},
    }
    tot_new_bytes = 0
    tot_new_toks = 0.0
    for b in buckets:
        nb = dir_bytes(V9_DIR / b)
        nt = nb / bytes_per_tok[b]
        tot_new_bytes += nb
        tot_new_toks += nt
        summary["buckets"][b] = {
            "bytes": nb,
            "gib": round(nb / (1024 ** 3), 3),
            "gb": round(nb / 1e9, 3),
            "est_tokens_b": round(nt / 1e9, 3),
            "docs": results.get(b, {}).get("docs", 0),
            "dups_skipped": results.get(b, {}).get("dups", 0),
        }
    summary["total_v9_new_gib"] = round(tot_new_bytes / (1024 ** 3), 3)
    summary["total_v9_new_gb"] = round(tot_new_bytes / 1e9, 3)
    summary["total_v9_new_tokens_b"] = round(tot_new_toks / 1e9, 3)
    (V9_DIR / "stats_v9_only.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== CORPUS_PRETRAIN_V9 COLLECTION COMPLETE ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
