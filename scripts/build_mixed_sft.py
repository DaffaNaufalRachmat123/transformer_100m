#!/usr/bin/env python3
r"""Build a single interleaved multi-task SFT set (EN/JA) balanced by MASKED LOSS TOKENS.

Ceiling Edition v3 (aligned with 491.58M @ 10B corpus_pretrain_v8, vocab_9):
  * Token-budget allocation across families so long code/math answers cannot starve
    short translation/factual answers under 1/tok_count CE normalization.
  * Priority-ordered source queues within each family so gold Tier-1 Wikidata facts,
    GSM8K EN+JA Option B1 traces, SciQ/OASST2 multi-turn, and Phi-1 textbook Python
    exercises are 100% retained before filling remaining quota from broader pools.
  * Exact BPE v9 length guard (`<= 1020` tokens) on long candidates so 0% of rows
    exceed `max_seq=1024` or lose `<|end|>`.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_sft import audit_row, split_turn  # noqa: E402
from bpe_reference import BPEReference  # noqa: E402

# Ordered from highest-priority gold sources to broader volume pools within each family
SOURCES = {
    "translation": [
        "sft_translation_curated",
    ],
    "facts": [
        "sft_science_multiturn",
        "sft_facts_expanded",
    ],
    "math": [
        "sft_gsm8k_pot",
        "sft_gsm8k_ja_pot",
        "sft_metamath_ja",
        "sft_metamath_en",
        "sft_think_math",
    ],
    "coding": [
        "sft_phi_code_exercises",
        "sft_code_instruct_aligned",
        "sft_think_code",
    ],
}

DEFAULT_RATIO = {
    "translation": 0.33,
    "facts": 0.19,
    "math": 0.25,
    "coding": 0.23,
}

BYTES_PER_TOK = {
    "translation": 3.75,
    "facts": 3.75,
    "math": 3.40,
    "coding": 3.10,
}

ASST_SPANS_RE = re.compile(r"<\|assistant\|>(.*?)<\|end\|>", re.DOTALL)


def est_masked_tokens(text, fallback_ans, family):
    spans = ASST_SPANS_RE.findall(text)
    if spans:
        b = sum(len(s.encode("utf-8")) for s in spans)
        n_turns = len(spans)
    else:
        b = len((fallback_ans or "").encode("utf-8"))
        n_turns = 1
    return max(2, int(round(b / BYTES_PER_TOK.get(family, 3.5))) + 2 * n_turns)


def lang_bucket(rec, keep):
    lang = str(rec.get("lang", "")).lower()
    if "-" in lang:
        src, _, tgt = lang.partition("-")
        if src not in keep or tgt not in keep:
            return None
        return lang
    return lang if lang in keep else None


def row_priority_tier(rec, src_name):
    """Tier 0 = real/curated gold sets + Option B1 multi-digit calculator anchor; Tier 1 = real/LLM volume."""
    src_field = str(rec.get("source", ""))
    if src_name in ("sft_gsm8k_pot", "sft_gsm8k_ja_pot", "sft_science_multiturn", "sft_phi_code_exercises"):
        return 0
    if src_name == "sft_think_code" and "mbpp" in src_field:
        return 0
    # Retain ~30% deterministic hash slice (~10K EN+JA rows) of Option B1 `# = val` arithmetic drills in Tier 0!
    if src_name == "sft_think_math":
        h0 = hashlib.md5(rec.get("text", "").encode("utf-8")).digest()[0]
        if h0 < 77:  # 77 / 256 ~= 30% (~5,000 JA + ~5,000 EN Option B1 `# = val` traces!)
            return 0
    if "tier1" in src_field:
        return 0
    if src_name == "sft_translation_curated" and rec.get("prompt_style") == "freeform_q_guard":
        return 0
    if src_name in ("sft_metamath_ja", "sft_metamath_en", "sft_code_instruct_aligned", "sft_facts_expanded", "sft_translation_curated"):
        return 1
    return 2


def scan_dir(path, src_name, family, keep, rng, seen, reject, seen_qa, tok_ref, max_tok=1020):
    buckets = defaultdict(list)
    if not path.exists():
        return buckets
    with path.open(encoding="utf-8") as fh:
        lines = [l for l in fh if l.strip()]
    rng.shuffle(lines)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = rec.get("text", "")
        if not text or len(text) > 4200:
            reject["over_char_cap"] += 1
            continue
        if "\r" in text:
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            rec["text"] = text
        b = lang_bucket(rec, keep)
        if b is None:
            continue
        # Fast exact BPE v9 check whenever UTF-8 byte length or digit density could approach max_seq=1024
        utf8_len = len(text.encode("utf-8"))
        digit_cnt = sum(ch.isdigit() for ch in text)
        if utf8_len > 2100 or (utf8_len + digit_cnt * 2) > 2200:
            ids = tok_ref.encode(text, add_bos=True, add_eos=False)
            if len(ids) > max_tok:
                reject["over_1020_bpe_tok"] += 1
                continue

        h = hashlib.sha1(text.encode("utf-8")).digest()
        if h in seen:
            continue
        problems = audit_row(rec, seen, seen_qa)
        if problems:
            for p in problems:
                reject[p] += 1
            continue
        _, ans = split_turn(text)
        rec["family"] = family
        rec["_mtok"] = est_masked_tokens(text, ans, family)
        rec["_prio"] = row_priority_tier(rec, src_name)
        buckets[b].append(rec)

    # Within a single source, keep tier-0 rows ahead while preserving random order inside each tier
    for b in buckets:
        buckets[b].sort(key=lambda r: r["_prio"])
    return buckets


def collect_pool(family, dirs, keep, root, rng, seen, reject, seen_qa, tok_ref):
    merged = defaultdict(list)
    for name in dirs:
        p = root / name / "train.jsonl"
        part = scan_dir(p, name, family, keep, rng, seen, reject, seen_qa, tok_ref)
        src_rows = sum(len(v) for v in part.values())
        src_tok = sum(r["_mtok"] for v in part.values() for r in v)
        print(f"    + {name:28s}: {src_rows:7,d} rows (~{src_tok:9,d} masked tok)")
        for b, rows in part.items():
            merged[b].extend(rows)
    # Shuffle across sources and then stable-sort by _prio so Tier-0 gold sets stay 100% intact
    # while Tier-1 pools (e.g. Phi-1 Python + multi-lang CodeInstruct, MetaMath + ThinkMath) interleave evenly!
    for b in merged:
        rng.shuffle(merged[b])
        merged[b].sort(key=lambda r: r["_prio"])
    return merged


def take_token_quota(buckets, target_tok, rng):
    """Draw rows across language buckets in priority order until `target_tok` masked tokens are met."""
    if not buckets or target_tok <= 0:
        return [], 0
    per_b = target_tok / len(buckets)
    selected = []
    leftover = []
    got_tok = 0
    for b in sorted(buckets):
        btok = 0
        rows = buckets[b]
        i = 0
        while i < len(rows) and btok < per_b:
            r = rows[i]
            selected.append(r)
            btok += r["_mtok"]
            got_tok += r["_mtok"]
            i += 1
        leftover.extend(rows[i:])
    if got_tok < target_tok and leftover:
        # Preserve priority tiers in leftover spill-over too, shuffling within same priority
        rng.shuffle(leftover)
        leftover.sort(key=lambda r: r.get("_prio", 1))
        for r in leftover:
            if got_tok >= target_tok:
                break
            selected.append(r)
            got_tok += r["_mtok"]
    return selected, got_tok


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
    ap.add_argument("--out", default="sft_data/sft_mixed_enja")
    ap.add_argument("--sft_root", default="sft_data")
    ap.add_argument("--vocab_dir", default="vocab_9")
    ap.add_argument("--target_masked_tokens", type=int, default=14_500_000,
                    help="total masked assistant tokens requested across all families")
    ap.add_argument("--total", type=int, default=0,
                    help="legacy alias (ignored in favor of token budget)")
    ap.add_argument("--ratio", default="",
                    help="masked-token ratio, e.g. translation=0.41,facts=0.21,math=0.21,coding=0.17")
    ap.add_argument("--langs", default="en,ja")
    ap.add_argument("--eval_frac", type=float, default=0.012)
    ap.add_argument("--strict_ratio", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    keep = {x.strip().lower() for x in a.langs.split(",") if x.strip()}
    ratio = parse_ratio(a.ratio)
    root = Path(a.sft_root)
    tok_ref = BPEReference(a.vocab_dir)

    print(f"[MIX-CEILING-v3] langs={sorted(keep)} target_masked_tokens={a.target_masked_tokens:,} ratio={ratio}")
    seen, seen_qa, reject = set(), {}, Counter()
    pools = {}
    avail_tok = {}

    for family in sorted(ratio):
        print(f"  scanning family '{family}':")
        buckets = collect_pool(family, SOURCES[family], keep, root, rng, seen, reject, seen_qa, tok_ref)
        pools[family] = buckets
        avail = sum(r["_mtok"] for rows in buckets.values() for r in rows)
        avail_rows = sum(len(rows) for rows in buckets.values())
        avail_tok[family] = avail
        print(f"  => pool {family:12s}: {avail_rows:7,d} rows (~{avail:9,d} masked tok)")

    if reject:
        print(f"[MIX-CEILING-v3] rejected during scan: {dict(reject.most_common())}")

    caps = [avail_tok[f] / ratio[f] for f in ratio if ratio[f] > 0]
    ceiling_tok = int(min(caps)) if caps else 0
    eff_budget = min(a.target_masked_tokens, ceiling_tok) if a.strict_ratio else a.target_masked_tokens
    if eff_budget < a.target_masked_tokens:
        binding = min(ratio, key=lambda f: avail_tok[f] / ratio[f])
        print(f"[MIX-CEILING-v3] strict_ratio capped masked-token budget to {eff_budget:,} (bound by '{binding}')")

    all_rows = []
    actual_tok = {}
    for family, frac in sorted(ratio.items()):
        q_tok = int(eff_budget * frac)
        picked, got_t = take_token_quota(pools[family], q_tok, rng)
        actual_tok[family] = got_t
        all_rows.extend(picked)
        print(f"  selected {family:12s}: {len(picked):7,d} rows (~{got_t:9,d} masked tok, {100*got_t/max(1,eff_budget):.1f}%)")

    for r in all_rows:
        r.pop("_mtok", None)
        r.pop("_prio", None)

    rng.shuffle(all_rows)
    n_eval = min(int(len(all_rows) * a.eval_frac), 6000)

    # Prefer non-pretrain-leaked math rows in eval, and keep gold Tier-1 facts in train
    pt_math_path = Path("data/corpus_pretrain_v8/math/sft_think_math_00.txt")
    pt_math_blob = pt_math_path.read_text(encoding="utf-8", errors="ignore") if pt_math_path.exists() else ""
    ev, tr = [], []
    for r in all_rows:
        if len(ev) < n_eval:
            if "tier1" in str(r.get("source", "")):
                tr.append(r)
                continue
            if r.get("family") == "math" and pt_math_blob:
                q, _ = split_turn(r["text"])
                if q and q[:60] in pt_math_blob:
                    tr.append(r)
                    continue
            ev.append(r)
        else:
            tr.append(r)
    rng.shuffle(tr)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", tr), ("eval", ev)):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    fam_rows = Counter(r["family"] for r in tr)
    lang_rows = Counter(str(r.get("lang", "?")) for r in tr)
    tot_act_tok = sum(actual_tok.values())
    stats = {
        "train": len(tr),
        "eval": len(ev),
        "family_rows": dict(fam_rows.most_common()),
        "family_masked_tokens_est": actual_tok,
        "family_masked_token_share_est": {k: round(v / max(1, tot_act_tok), 4) for k, v in actual_tok.items()},
        "lang_rows": dict(lang_rows.most_common()),
        "target_token_ratio": ratio,
        "langs": sorted(keep),
        "seed": a.seed,
        "strict_ratio": a.strict_ratio,
        "sources": {k: SOURCES[k] for k in ratio},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[MIX-CEILING-v3] DONE -> train={len(tr):,} eval={len(ev):,} written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
