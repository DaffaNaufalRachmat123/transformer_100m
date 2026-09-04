"""
DPO preference data inspector — analyze sft_data/dpo/.

Sections:
 1. File inventory — pair count, sizes, broken lines
 2. Language & task distribution
 3. Length analysis — chosen vs rejected length ratio
 4. Gap analysis — delta length, preference margin
 5. Quality — flag trivially similar pairs
 6. Response quality — readability/perplexity of chosen vs rejected
"""

import json
import os
import re
import time
from collections import defaultdict, Counter
from typing import Dict, List

from ..common import text_sha256


class DPOInspector:
    def __init__(self, dpo_dirs: List[str], sample_pairs: int = 2000):
        self.dpo_dirs = dpo_dirs
        self.sample_size = sample_pairs
        self.results = {
            "files": {},
            "language_distribution": {},
            "task_distribution": {},
            "length_analysis": {},
            "gap_analysis": {},
            "quality_flags": {},
            "response_quality": {},
            "cross_dedup": {},
            "alerts": [],
        }

    def run(self) -> Dict:
        print("\n" + "=" * 70)
        print(" DPO DATA INSPECTION")
        print("=" * 70)
        t0 = time.time()

        all_pairs = self._load_all_pairs()

        self._inspect_files(all_pairs)
        self._check_distributions(all_pairs)
        self._check_lengths(all_pairs)
        self._check_gaps(all_pairs)
        self._check_quality(all_pairs)
        self._check_cross_dedup(all_pairs)

        elapsed = time.time() - t0
        self.results["_totals"] = {
            "total_files": len(self.results["files"]),
            "total_pairs": sum(v["pairs"] for v in self.results["files"].values()),
        }
        self.results["metadata"] = {"elapsed_s": round(elapsed, 1), "inspected_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        print(f"\n[dpo] Inspection complete in {elapsed:.1f}s")
        return self.results

    def _load_all_pairs(self) -> Dict[str, List[Dict]]:
        all_pairs = {}
        for dpo_dir in self.dpo_dirs:
            if not os.path.isdir(dpo_dir):
                continue
            for fname in sorted(os.listdir(dpo_dir)):
                if not fname.endswith(".jsonl"):
                    continue
                fpath = os.path.join(dpo_dir, fname)
                key = os.path.relpath(fpath)
                pairs = []
                broken = 0
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                pairs.append(json.loads(line))
                            except json.JSONDecodeError:
                                broken += 1
                except Exception:
                    continue
                all_pairs[key] = pairs
        return all_pairs

    def _inspect_files(self, all_pairs: Dict[str, List[Dict]]):
        print("\n[1/6] File inventory...")
        for key, pairs in all_pairs.items():
            langs = Counter(p.get("lang", "unknown") for p in pairs)
            tasks = Counter(p.get("task", "unknown") for p in pairs)
            self.results["files"][key] = {
                "path": key,
                "pairs": len(pairs),
                "file_size_mb": round(os.path.getsize(key) / (1024*1024), 1) if os.path.isfile(key) else 0,
                "languages": dict(langs.most_common(5)),
                "tasks": dict(tasks.most_common(10)),
            }
        total = sum(f["pairs"] for f in self.results["files"].values())
        print(f"  Found {len(all_pairs)} files, {total:,} pairs")

    def _check_distributions(self, all_pairs: Dict[str, List[Dict]]):
        print("\n[2/6] Language & task distribution...")
        global_langs = Counter()
        global_tasks = Counter()
        for pairs in all_pairs.values():
            for p in pairs:
                global_langs[p.get("lang", "unknown")] += 1
                global_tasks[p.get("task", "unknown")] += 1

        total = sum(global_langs.values()) or 1
        self.results["language_distribution"] = {k: {"count": v, "pct": round(100*v/total, 1)} for k, v in global_langs.most_common()}
        self.results["task_distribution"] = {k: {"count": v, "pct": round(100*v/total, 1)} for k, v in global_tasks.most_common()}

    def _check_lengths(self, all_pairs: Dict[str, List[Dict]]):
        print("\n[3/6] Length analysis (chosen vs rejected)...")
        chosen_lens = []
        rejected_lens = []
        chosen_longer = 0
        rejected_longer = 0
        same_length = 0

        for pairs in all_pairs.values():
            for p in pairs:
                cl = len(p.get("chosen", ""))
                rl = len(p.get("rejected", ""))
                chosen_lens.append(cl)
                rejected_lens.append(rl)
                if cl > rl:
                    chosen_longer += 1
                elif rl > cl:
                    rejected_longer += 1
                else:
                    same_length += 1

        total = len(chosen_lens) or 1
        self.results["length_analysis"] = {
            "total_pairs": total,
            "avg_chosen_len": round(sum(chosen_lens)/total, 1),
            "avg_rejected_len": round(sum(rejected_lens)/total, 1),
            "chosen_longer": chosen_longer,
            "chosen_longer_pct": round(100*chosen_longer/total, 1),
            "rejected_longer": rejected_longer,
            "same_length": same_length,
        }

        if rejected_longer > total * 0.3:
            self.results["alerts"].append(f"Length bias: {rejected_longer}/{total} pairs have longer rejected ({100*rejected_longer//total}%)")
        print(f"  Chosen longer: {chosen_longer} ({100*chosen_longer//total}%), rejected longer: {rejected_longer} ({100*rejected_longer//total}%)")

    def _check_gaps(self, all_pairs: Dict[str, List[Dict]]):
        print("\n[4/6] Gap analysis...")
        gaps = []
        trivial = 0
        extreme_gap = 0

        for pairs in all_pairs.values():
            for p in pairs:
                cl = p.get("chosen", "").strip()
                rl = p.get("rejected", "").strip()
                delta = abs(len(cl) - len(rl))
                gaps.append(delta)

                if cl == rl:
                    trivial += 1
                if delta < 5 and cl != rl:
                    trivial += 1
                if delta > 5000:
                    extreme_gap += 1

        total = len(gaps) or 1
        gaps_sorted = sorted(gaps)
        self.results["gap_analysis"] = {
            "total_pairs": total,
            "avg_gap_chars": round(sum(gaps)/total, 1),
            "p50_gap": gaps_sorted[len(gaps_sorted)//2] if gaps_sorted else 0,
            "p90_gap": gaps_sorted[int(len(gaps_sorted)*0.9)] if gaps_sorted else 0,
            "trivial_pairs": trivial,
            "extreme_gap_pairs": extreme_gap,
        }
        print(f"  Trivial pairs: {trivial}, extreme gaps: {extreme_gap}")

    def _check_quality(self, all_pairs: Dict[str, List[Dict]]):
        print("\n[5/6] Pair quality flags...")
        missing_fields = 0
        empty_chosen = 0
        empty_rejected = 0
        missing_task = 0

        for pairs in all_pairs.values():
            for p in pairs:
                if "chosen" not in p or "rejected" not in p or "prompt" not in p:
                    missing_fields += 1
                if not p.get("chosen", "").strip():
                    empty_chosen += 1
                if not p.get("rejected", "").strip():
                    empty_rejected += 1
                if not p.get("task"):
                    missing_task += 1

        total = sum(v["pairs"] for v in self.results["files"].values()) or 1
        self.results["quality_flags"] = {
            "missing_fields": missing_fields,
            "empty_chosen": empty_chosen,
            "empty_rejected": empty_rejected,
            "missing_task": missing_task,
        }
        flags = sum([missing_fields, empty_chosen, empty_rejected])
        print(f"  Quality flags: {flags} issues in {total:,} pairs")

    def _check_cross_dedup(self, all_pairs: Dict[str, List[Dict]]):
        print("\n[6/6] Cross-file dedup (prompt-level)...")
        seen = {}
        dup_count = 0

        for key, pairs in all_pairs.items():
            for p in pairs:
                prompt = p.get("prompt", "")
                h = text_sha256(prompt)
                if h in seen:
                    dup_count += 1
                else:
                    seen[h] = key

        self.results["cross_dedup"] = {
            "unique_prompts": len(seen),
            "duplicate_prompts": dup_count,
        }
        print(f"  Duplicate prompts: {dup_count}")
