#!/usr/bin/env python3
"""Build a single interleaved multi-task SFT set (EN/JA) from the existing corpora.

The staged pipeline (run_facts_sft.sh: JA facts -> EN facts -> translation)
trains each skill on top of the previous one, and each stage overwrites the
last: the facts checkpoint scores 0.003 chrF on translation while the
translation checkpoint answers 0/10 EN facts. Interleaving every task into one
shuffled set removes the sequential overwrite entirely.

Sampling is quota-driven, not head-of-file: each family gets a token budget,
which is split evenly across its language buckets so EN and JA stay balanced
even though the source files are trilingual and unevenly ordered.

  python3 scripts/build_mixed_sft.py --out sft_data/sft_mixed_enja --total 400000
  python3 scripts/build_mixed_sft.py --out /tmp/probe --total 20000 \
      --ratio facts=0.5,translation=0.5
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Family -> source dataset dirs under sft_data/. Listed newest/best first;
# a family draws from every listed dir, so add a dir to widen coverage.
SOURCES = {
    "facts": ["sft_facts_expanded", "sft_facts_v11"],
    "translation": ["sft_translation_tri", "sft_translation_bank"],
    "coding": ["sft_coding_v3", "sft_code_explain", "sft_code_bugfix",
               "sft_coding_mbpp"],
    "math": ["sft_math_pot", "sft_math_v2"],
}

DEFAULT_RATIO = {"facts": 0.35, "translation": 0.35, "coding": 0.20, "math": 0.10}


def lang_bucket(rec, keep):
    """Return the balancing bucket for a record, or None if it must be dropped.

    Monolingual rows bucket by `lang`; translation rows bucket by direction and
    are dropped unless BOTH sides are languages we keep.
    """
    lang = str(rec.get("lang", "")).lower()
    if "-" in lang:
        src, _, tgt = lang.partition("-")
        if src not in keep or tgt not in keep:
            return None
        return lang
    return lang if lang in keep else None


def scan(path, keep, cap_per_bucket, rng, seen):
    """Reservoir-sample up to cap_per_bucket unique rows per language bucket.

    Dedup happens here rather than after quota allocation: the coding and facts
    sources reuse a small task pool across many template variants, so ~70% of a
    naive sample collapses and the family would silently miss its quota.
    """
    buckets = defaultdict(list)
    seen_count = defaultdict(int)
    if not path.exists():
        return buckets
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text")
            if not text:
                continue
            b = lang_bucket(rec, keep)
            if b is None:
                continue
            h = hashlib.sha1(text.encode("utf-8")).digest()
            if h in seen:
                continue
            seen.add(h)
            seen_count[b] += 1
            pool = buckets[b]
            if len(pool) < cap_per_bucket:
                pool.append(rec)
            else:
                # Reservoir: keeps a uniform sample without holding the file in RAM.
                j = rng.randrange(seen_count[b])
                if j < cap_per_bucket:
                    pool[j] = rec
    return buckets


def collect(family, dirs, quota, keep, root, rng, seen):
    """Gather `quota` unique rows for one family, split evenly across buckets."""
    merged = defaultdict(list)
    for name in dirs:
        for bucket, rows in scan(root / name / "train.jsonl", keep,
                                 quota, rng, seen).items():
            merged[bucket].extend(rows)
    if not merged:
        print(f"  [WARN] {family}: no usable rows in {dirs}", file=sys.stderr)
        return []

    per_bucket = max(1, quota // len(merged))
    out, leftover = [], []
    for bucket, rows in sorted(merged.items()):
        rng.shuffle(rows)
        out.extend(rows[:per_bucket])
        leftover.extend(rows[per_bucket:])
    # Backfill from whatever remains when a bucket could not meet its share.
    rng.shuffle(leftover)
    out.extend(leftover[: max(0, quota - len(out))])
    for r in out:
        r["family"] = family
    return out[:quota]


def parse_ratio(s):
    if not s:
        return dict(DEFAULT_RATIO)
    out = {}
    for part in s.split(","):
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in SOURCES:
            raise SystemExit(f"unknown family '{k}' (known: {sorted(SOURCES)})")
        out[k] = float(v)
    total = sum(out.values())
    if total <= 0:
        raise SystemExit("ratios must sum to > 0")
    return {k: v / total for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sft_root", default="sft_data")
    ap.add_argument("--total", type=int, default=400_000)
    ap.add_argument("--ratio", default="",
                    help="e.g. facts=0.35,translation=0.35,coding=0.2,math=0.1")
    ap.add_argument("--langs", default="en,ja")
    ap.add_argument("--eval_frac", type=float, default=0.02)
    ap.add_argument("--strict_ratio", action="store_true",
                    help="shrink total so every family hits its exact share "
                         "instead of letting well-stocked families dominate")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    keep = {x.strip().lower() for x in a.langs.split(",") if x.strip()}
    ratio = parse_ratio(a.ratio)
    root = Path(a.sft_root)

    print(f"[MIX] langs={sorted(keep)} total={a.total} seed={a.seed}")
    seen = set()
    per_family = {}
    for family, frac in sorted(ratio.items()):
        quota = int(a.total * frac)
        got = collect(family, SOURCES[family], quota, keep, root, rng, seen)
        per_family[family] = got
        short = "  SHORT" if len(got) < quota else ""
        print(f"  {family:12s} quota={quota:7d} got={len(got):7d}{short}  "
              f"from {SOURCES[family]}")

    # A family that ran short silently rewrites the mix, so surface the ceiling
    # the scarcest family imposes on an exactly-proportioned dataset.
    caps = [len(per_family[f]) / ratio[f] for f in ratio if ratio[f] > 0]
    max_total = int(min(caps)) if caps else 0
    if max_total < a.total:
        binding = min(ratio, key=lambda f: len(per_family[f]) / ratio[f])
        print(f"[MIX] ceiling: {max_total} rows keep the requested ratio "
              f"(bound by '{binding}' with {len(per_family[binding])} unique rows)")

    if a.strict_ratio:
        for family in per_family:
            keep_n = int(max_total * ratio[family])
            rng.shuffle(per_family[family])
            per_family[family] = per_family[family][:keep_n]
        print(f"[MIX] strict_ratio: capped total to {max_total}")

    uniq = [r for f in sorted(per_family) for r in per_family[f]]

    rng.shuffle(uniq)
    n_eval = min(int(len(uniq) * a.eval_frac), 20_000)
    ev, tr = uniq[:n_eval], uniq[n_eval:]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", tr), ("eval", ev)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fam = Counter(r["family"] for r in tr)
    lang = Counter(str(r.get("lang", "?")) for r in tr)
    stats = {
        "train": len(tr), "eval": len(ev),
        "family": dict(fam.most_common()),
        "lang": dict(lang.most_common()),
        "ratio": ratio, "langs": sorted(keep), "seed": a.seed,
        "strict_ratio": a.strict_ratio,
        "max_total_at_ratio": max_total,
        "unique_available": {f: len(v) for f, v in per_family.items()},
        "sources": {k: SOURCES[k] for k in ratio},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"[MIX] train={len(tr)} eval={len(ev)} -> {out}")
    print(f"  family: {dict(fam.most_common())}")
    print(f"  lang  : {dict(lang.most_common(10))}")


if __name__ == "__main__":
    main()
