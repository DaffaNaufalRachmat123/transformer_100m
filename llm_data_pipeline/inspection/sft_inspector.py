"""
SFT data inspector — analyze sft_data/sft/ and sft_data/sft_clean/.

Sections:
 1. File inventory — conversation count, tokens, avg turns, sizes
 2. Task distribution — per-file and global task/category distribution
 3. Language distribution — per-file and global language distribution
 4. Chat template validation — ChatML format correctness
 5. Turn statistics — avg turns, prompt length, response length
 6. Response quality — readability, perplexity of assistant responses
 7. Response style — boilerplate phrases, code fence correctness
 8. Instruction diversity — near-duplicate detection via similarity
 9. Cross-file dedup — exact conversation overlap
10. Tokenization stats — tokens per conversation
"""

import json
import os
import re
import sys
import time
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

from ..common import text_sha256


CHATML_PATTERN = re.compile(r"<\|system\|>(.*?)<\|end\|>\s*<\|user\|>(.*?)<\|end\|>\s*<\|assistant\|>(.*?)<\|end\|>", re.DOTALL)

BOILERPLATE_RESPONSE_PATTERNS = [
    r"(?i)as an AI language model",
    r"(?i)I am an AI",
    r"(?i)I am a large language model",
    r"(?i)I don't have personal",
    r"(?i)I cannot provide",
    r"(?i)I apologize, but I (?:cannot|am unable)",
    r"(?i)as a responsible AI",
    r"(?i)I don't have the ability",
    r"(?i)I'm just an AI",
]


class SFTInspector:
    def __init__(self, sft_dirs: List[str], sample_conversations: int = 5000):
        self.sft_dirs = sft_dirs
        self.sample_size = sample_conversations
        self.results = {
            "files": {},
            "task_distribution": {},
            "language_distribution": {},
            "template_validation": {},
            "turn_statistics": {},
            "response_quality": {},
            "response_style": {},
            "cross_dedup": {},
            "tokenization": {},
            "alerts": [],
        }

    def run(self) -> Dict:
        print("\n" + "=" * 70)
        print(" SFT DATA INSPECTION")
        print("=" * 70)
        t0 = time.time()

        all_convs = self._load_all_conversations()

        self._inspect_files(all_convs)
        self._check_task_distribution(all_convs)
        self._check_language_distribution(all_convs)
        self._validate_chat_template(all_convs)
        self._check_turn_statistics(all_convs)
        self._check_response_quality(all_convs)
        self._check_response_style(all_convs)
        self._check_cross_dedup(all_convs)
        self._check_tokenization(all_convs)

        elapsed = time.time() - t0
        self.results["_totals"] = {
            "total_files": len(self.results["files"]),
            "total_conversations": sum(v["conversations"] for v in self.results["files"].values()),
        }
        self.results["metadata"] = {"elapsed_s": round(elapsed, 1), "inspected_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        print(f"\n[sft] Inspection complete in {elapsed:.1f}s")
        return self.results

    def _load_all_conversations(self) -> Dict[str, List[Dict]]:
        all_convs = {}
        for sft_dir in self.sft_dirs:
            if not os.path.isdir(sft_dir):
                continue
            for fname in sorted(os.listdir(sft_dir)):
                if not fname.endswith(".jsonl"):
                    continue
                fpath = os.path.join(sft_dir, fname)
                key = os.path.relpath(fpath)
                convs = []
                broken = 0
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                convs.append(json.loads(line))
                            except json.JSONDecodeError:
                                broken += 1
                except Exception:
                    continue
                all_convs[key] = convs
        return all_convs

    # ─── 1. File Inventory ────────────────────────────────────────────────

    def _inspect_files(self, all_convs: Dict[str, List[Dict]]):
        print("\n[1/8] File inventory...")
        for key, convs in all_convs.items():
            total_turns = 0
            langs = Counter()
            tasks = Counter()
            for c in convs:
                text = c.get("text", "")
                turns = len(re.findall(r"<\|user\|>", text))
                total_turns += max(1, turns)
                langs[c.get("lang", "unknown")] += 1
                tasks[c.get("task", "unknown")] += 1

            self.results["files"][key] = {
                "path": key,
                "conversations": len(convs),
                "total_turns": total_turns,
                "avg_turns": round(total_turns / max(1, len(convs)), 1),
                "file_size_mb": round(os.path.getsize(key) / (1024*1024), 1) if os.path.isfile(key) else 0,
                "languages": dict(langs.most_common(5)),
                "tasks": dict(tasks.most_common(10)),
            }

        total = sum(f["conversations"] for f in self.results["files"].values())
        print(f"  Found {len(all_convs)} files, {total:,} conversations")

    # ─── 2. Task Distribution ─────────────────────────────────────────────

    def _check_task_distribution(self, all_convs: Dict[str, List[Dict]]):
        print("\n[2/8] Task distribution...")
        global_tasks = Counter()
        lang_tasks = defaultdict(Counter)

        for convs in all_convs.values():
            for c in convs:
                task = c.get("task", "unknown")
                lang = c.get("lang", "unknown")
                global_tasks[task] += 1
                lang_tasks[lang][task] += 1

        total = sum(global_tasks.values()) or 1
        self.results["task_distribution"] = {
            "global": {k: {"count": v, "pct": round(100*v/total, 1)} for k, v in global_tasks.most_common()},
            "by_language": {lang: {t: {"count": c, "pct": round(100*c/max(1, sum(tasks.values())), 1)} for t, c in tasks.most_common(10)}
                           for lang, tasks in sorted(lang_tasks.items())},
        }
        print(f"  Tasks: {len(global_tasks)} unique, top={global_tasks.most_common(1)[0]}")

    # ─── 3. Language Distribution ─────────────────────────────────────────

    def _check_language_distribution(self, all_convs: Dict[str, List[Dict]]):
        print("\n[3/8] Language distribution...")
        global_langs = Counter()
        for convs in all_convs.values():
            for c in convs:
                global_langs[c.get("lang", "unknown")] += 1

        total = sum(global_langs.values()) or 1
        self.results["language_distribution"] = {
            k: {"count": v, "pct": round(100*v/total, 1)} for k, v in global_langs.most_common()
        }
        print(f"  Languages: {dict(global_langs.most_common())}")

    # ─── 4. Chat Template Validation ──────────────────────────────────────

    def _validate_chat_template(self, all_convs: Dict[str, List[Dict]]):
        print("\n[4/8] Chat template validation...")
        results = {}
        total_valid = 0
        total_invalid = 0
        reasons = Counter()

        for key, convs in all_convs.items():
            valid = 0
            invalid = 0
            for c in convs:
                text = c.get("text", "")
                if not text.strip():
                    invalid += 1
                    reasons["empty_text"] += 1
                    continue

                has_system = "<|system|>" in text
                has_user = "<|user|>" in text
                has_assistant = "<|assistant|>" in text
                has_end = "<|end|>" in text

                if not has_user:
                    invalid += 1
                    reasons["missing_user"] += 1
                elif not has_assistant:
                    invalid += 1
                    reasons["missing_assistant"] += 1
                elif not has_end:
                    invalid += 1
                    reasons["missing_end"] += 1
                else:
                    turns = re.findall(r"<\|user\|>(.*?)<\|end\|>", text, re.DOTALL)
                    responses = re.findall(r"<\|assistant\|>(.*?)<\|end\|>", text, re.DOTALL)
                    if not turns:
                        invalid += 1
                        reasons["no_user_turns"] += 1
                    elif not responses:
                        invalid += 1
                        reasons["no_assistant_turns"] += 1
                    else:
                        valid += 1

            results[key] = {"valid": valid, "invalid": invalid, "total": valid + invalid}
            total_valid += valid
            total_invalid += invalid

        self.results["template_validation"] = {
            "total_valid": total_valid,
            "total_invalid": total_invalid,
            "validity_pct": round(100 * total_valid / max(1, total_valid + total_invalid), 1),
            "reasons": dict(reasons.most_common()),
            "per_file": results,
        }
        print(f"  Valid: {total_valid:,}, Invalid: {total_invalid:,} ({100*total_valid//max(1,total_valid+total_invalid)}%)")

        if total_invalid > total_valid * 0.1:
            self.results["alerts"].append(f"High ChatML invalid rate: {total_invalid}/{total_valid+total_invalid}")

    # ─── 5. Turn Statistics ───────────────────────────────────────────────

    def _check_turn_statistics(self, all_convs: Dict[str, List[Dict]]):
        print("\n[5/8] Turn statistics...")
        turn_dist = Counter()
        prompt_lens = []
        response_lens = []
        conv_lens = []

        for convs in all_convs.values():
            for c in convs:
                text = c.get("text", "")
                user_turns = re.findall(r"<\|user\|>(.*?)<\|end\|>", text, re.DOTALL)
                asst_turns = re.findall(r"<\|assistant\|>(.*?)<\|end\|>", text, re.DOTALL)
                n_turns = len(user_turns)
                turn_dist[n_turns] += 1

                for u in user_turns:
                    prompt_lens.append(len(u.strip()))
                for a in asst_turns:
                    response_lens.append(len(a.strip()))
                conv_lens.append(len(text))

        def _pct(arr, p):
            s = sorted(arr)
            return s[int(len(s) * p / 100)] if s else 0

        self.results["turn_statistics"] = {
            "turn_distribution": dict(sorted(turn_dist.items())),
            "prompt_chars": {"avg": round(sum(prompt_lens)/max(1,len(prompt_lens)), 1), "p50": _pct(prompt_lens, 50), "p90": _pct(prompt_lens, 90)},
            "response_chars": {"avg": round(sum(response_lens)/max(1,len(response_lens)), 1), "p50": _pct(response_lens, 50), "p90": _pct(response_lens, 90)},
            "conversation_chars": {"avg": round(sum(conv_lens)/max(1,len(conv_lens)), 1), "p50": _pct(conv_lens, 50)},
        }
        print(f"  Avg turns: {sum(k*v for k,v in turn_dist.items())//max(1,sum(turn_dist.values()))}, "
              f"avg prompt: {self.results['turn_statistics']['prompt_chars']['avg']} chars")

    # ─── 6. Response Quality ──────────────────────────────────────────────

    def _check_response_quality(self, all_convs: Dict[str, List[Dict]]):
        print("\n[6/8] Response quality...")
        try:
            from ..quality_scoring.heuristic import HeuristicQualityScorer
            scorer = HeuristicQualityScorer()
        except Exception:
            self.results["response_quality"] = {"status": "scorer_unavailable"}
            return

        all_readability = []
        all_perplexity = []
        by_lang = defaultdict(lambda: {"readability": [], "perplexity": []})

        count = 0
        for convs in all_convs.values():
            for c in convs:
                if count >= self.sample_size:
                    break
                text = c.get("text", "")
                lang = c.get("lang", "en")
                asst_turns = re.findall(r"<\|assistant\|>(.*?)<\|end\|>", text, re.DOTALL)
                for a in asst_turns:
                    a = a.strip()
                    if len(a) < 20:
                        continue
                    s = scorer.score(a, lang)
                    all_readability.append(s["readability"])
                    all_perplexity.append(s["perplexity"])
                    by_lang[lang]["readability"].append(s["readability"])
                    by_lang[lang]["perplexity"].append(s["perplexity"])
                count += 1

        def _avg(arr):
            return round(sum(arr)/max(1, len(arr)), 3)

        self.results["response_quality"] = {
            "samples": len(all_readability),
            "avg_readability": _avg(all_readability),
            "avg_perplexity": _avg(all_perplexity),
            "by_language": {lang: {"avg_readability": _avg(v["readability"]), "avg_perplexity": _avg(v["perplexity"]), "samples": len(v["readability"])} for lang, v in by_lang.items()},
        }
        print(f"  Avg readability: {_avg(all_readability):.3f}, avg perplexity: {_avg(all_perplexity):.1f}")

    # ─── 7. Response Style ────────────────────────────────────────────────

    def _check_response_style(self, all_convs: Dict[str, List[Dict]]):
        print("\n[7/8] Response style...")
        boilerplate_count = 0
        missing_code_fence = 0
        total_responses = 0

        for convs in all_convs.values():
            for c in convs:
                text = c.get("text", "")
                asst_turns = re.findall(r"<\|assistant\|>(.*?)<\|end\|>", text, re.DOTALL)
                for a in asst_turns:
                    total_responses += 1
                    a_stripped = a.strip()
                    for bp in BOILERPLATE_RESPONSE_PATTERNS:
                        if re.search(bp, a_stripped):
                            boilerplate_count += 1
                            break
                    if "```" in text and "```" in a_stripped:
                        fences = re.findall(r"```(\w*)", a_stripped)
                        if any(not lang for lang in fences):
                            missing_code_fence += 1

        self.results["response_style"] = {
            "total_responses_sampled": total_responses,
            "boilerplate_responses": boilerplate_count,
            "boilerplate_pct": round(100 * boilerplate_count / max(1, total_responses), 1),
            "missing_code_language_tag": missing_code_fence,
        }
        print(f"  Boilerplate: {boilerplate_count}/{total_responses} ({100*boilerplate_count//max(1,total_responses)}%)")

    # ─── 8. Cross-File Dedup ──────────────────────────────────────────────

    def _check_cross_dedup(self, all_convs: Dict[str, List[Dict]]):
        print("\n[8/8] Cross-file dedup (conversation-level)...")
        seen = {}
        dup_count = 0
        dup_examples = []

        for key, convs in all_convs.items():
            for c in convs:
                text = c.get("text", "")
                h = text_sha256(text)
                if h in seen:
                    dup_count += 1
                    if len(dup_examples) < 20:
                        dup_examples.append({"file_a": seen[h], "file_b": key, "hash": h[:16]})
                else:
                    seen[h] = key

        self.results["cross_dedup"] = {
            "total_unique_hashes": len(seen),
            "duplicates_found": dup_count,
            "samples": dup_examples,
        }
        print(f"  Exact duplicates across files: {dup_count}")

    # ─── 9. Tokenization ─────────────────────────────────────────────────

    def _check_tokenization(self, all_convs: Dict[str, List[Dict]]):
        print("\n[Tokenization] Counting token estimates...")
        vocab_path = "vocab_6/vocab.txt"
        merges_path = "vocab_6/merges.txt"
        if not os.path.exists(vocab_path):
            self.results["tokenization"] = {"status": "skipped", "reason": "vocab_6 not found"}
            return

        try:
            from ..tokenization.encoder import TokenizationEncoder
            encoder = TokenizationEncoder(vocab_path=vocab_path, merges_path=merges_path)
            encoder.load_vocab()
        except Exception as e:
            self.results["tokenization"] = {"status": "error", "reason": str(e)}
            return

        total_chars = 0
        total_tokens = 0
        count = 0
        by_lang = defaultdict(lambda: {"chars": 0, "tokens": 0, "count": 0})

        for convs in all_convs.values():
            for c in convs:
                if count >= self.sample_size:
                    break
                text = c.get("text", "")
                lang = c.get("lang", "en")
                ids = encoder.encode(text, add_bos=True, add_eos=True, max_len=4096)
                total_chars += len(text)
                total_tokens += len(ids)
                by_lang[lang]["chars"] += len(text)
                by_lang[lang]["tokens"] += len(ids)
                by_lang[lang]["count"] += 1
                count += 1

        self.results["tokenization"] = {
            "samples": count,
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "chars_per_token": round(total_chars / max(1, total_tokens), 1),
            "by_language": {lang: {"chars_per_token": round(v["chars"]/max(1,v["tokens"]), 1), "samples": v["count"]} for lang, v in by_lang.items()},
        }
        # Estimate total SFT tokens
        total_convs = sum(len(c) for c in all_convs.values())
        self.results["tokenization"]["estimated_total_sft_tokens"] = total_convs * (total_tokens // max(1, count))
        print(f"  Chars/token: {self.results['tokenization']['chars_per_token']}")
