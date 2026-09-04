"""
Pretraining corpus inspector — analyze data/corpus_final/ and data/corpus_extra/.

Sections:
 1. File inventory — sizes, line counts, symlinks, estimated tokens
 2. Language verification — FastText check per language directory
 3. Quality score distribution — perplexity, readability, entropy per file
 4. Repetition check — line dup ratio, n-gram repetition
 5. Composition check — alpha/numeric/symbol ratios
 6. Boilerplate ratio — boilerplate line detection
 7. Tokenization stats — encode samples with vocab_6
 8. Contamination scan — 13-gram overlap with benchmarks
 9. Cross-file dedup — SHA-256 overlap check
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..common import text_sha256


class CorpusInspector:
    def __init__(self, corpus_dirs: List[str], sample_size: int = 5000, seed: int = 42):
        self.corpus_dirs = corpus_dirs
        self.sample_size = sample_size
        self.seed = seed
        self._sample_cache: Dict[str, List[str]] = {}
        self.results = {
            "files": {},
            "language_summary": {},
            "quality_distribution": {},
            "repetition": {},
            "composition": {},
            "boilerplate": {},
            "tokenization": {},
            "contamination": {},
            "cross_dedup": {},
            "alerts": [],
        }

    def run(self) -> Dict:
        print("\n" + "=" * 70)
        print(" CORPUS INSPECTION")
        print("=" * 70)
        t0 = time.time()

        steps = [
            ("File inventory", self._inspect_files),
            ("Language verification", self._check_language),
            ("Quality scoring", self._check_quality),
            ("Repetition check", self._check_repetition),
            ("Composition check", self._check_composition),
            ("Boilerplate check", self._check_boilerplate),
            ("Tokenization stats", self._check_tokenization),
            ("Cross-file dedup", self._check_cross_dedup),
            ("Contamination scan", self._check_contamination),
        ]
        for step_name, step_fn in steps:
            t_step = time.time()
            step_fn()
            print(f"  [{step_name}] {time.time()-t_step:.1f}s")

        elapsed = time.time() - t0
        self.results["metadata"] = {"elapsed_s": round(elapsed, 1), "inspected_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        print(f"\n[corpus] Inspection complete in {elapsed:.1f}s")
        return self.results

    # ─── 1. File Inventory ────────────────────────────────────────────────

    def _inspect_files(self):
        print("\n[1/8] File inventory...")
        for corpus_dir in self.corpus_dirs:
            if not os.path.isdir(corpus_dir):
                continue
            for lang_dir in sorted(os.listdir(corpus_dir)):
                lang_path = os.path.join(corpus_dir, lang_dir)
                if not os.path.isdir(lang_path) or lang_dir.startswith("."):
                    continue
                for fname in sorted(os.listdir(lang_path)):
                    fpath = os.path.join(lang_path, fname)
                    is_symlink = os.path.islink(fpath)
                    size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
                    rpath = os.path.realpath(fpath) if is_symlink else fpath
                    if os.path.isfile(rpath):
                        real_size = os.path.getsize(rpath)
                        lines = self._count_lines(rpath)
                    else:
                        real_size = size
                        lines = 0

                    key = os.path.relpath(fpath)
                    self.results["files"][key] = {
                        "path": key,
                        "language": lang_dir,
                        "source_dir": corpus_dir,
                        "is_symlink": is_symlink,
                        "symlink_target": os.readlink(fpath) if is_symlink else None,
                        "size_bytes": real_size,
                        "size_mb": round(real_size / (1024 * 1024), 1),
                        "line_count": lines,
                        "estimated_tokens": lines * 100,
                        "extension": os.path.splitext(fname)[1],
                    }

        total_size = sum(f["size_bytes"] for f in self.results["files"].values())
        total_lines = sum(f["line_count"] for f in self.results["files"].values())
        lang_counts = defaultdict(lambda: {"files": 0, "size_mb": 0, "lines": 0})
        for f in self.results["files"].values():
            l = f["language"]
            lang_counts[l]["files"] += 1
            lang_counts[l]["size_mb"] += f["size_mb"]
            lang_counts[l]["lines"] += f["line_count"]

        self.results["language_summary"] = {
            lang: dict(stats) for lang, stats in sorted(lang_counts.items())
        }
        self.results["_totals"] = {
            "total_files": len(self.results["files"]),
            "total_size_mb": round(total_size / (1024 * 1024), 1),
            "total_lines": total_lines,
            "estimated_tokens": total_lines * 100,
        }
        print(f"  Found {len(self.results['files'])} files, {total_size/(1024**3):.1f} GB, {total_lines:,} lines")

    # ─── 2. Language Verification ─────────────────────────────────────────

    def _check_language(self):
        print("\n[2/8] Language verification...")
        try:
            from ..language_id.classifier import LanguageClassifier
            classifier = LanguageClassifier(paragraph_level=False, primary_threshold=0.5)
        except Exception as e:
            self.results["alerts"].append(f"Language classifier unavailable: {e}")
            return

        LANG_DIR_TO_CODE = {
            "english": "en", "indonesian": "id", "japanese": "ja", "korean": "ko",
            "code": "en", "math": "en", "mixed": "en",
        }
        lang_results = {}
        for key, finfo in self.results["files"].items():
            expected_lang = finfo["language"]
            expected_code = LANG_DIR_TO_CODE.get(expected_lang, expected_lang)
            fpath = self._resolve_path(key)
            if not os.path.isfile(fpath):
                continue
            lines = self._get_cached_sample(fpath, 30)
            if not lines:
                continue

            detected = defaultdict(int)
            for line in lines:
                if len(line.strip()) < 30:
                    continue
                res = classifier.classify(line)
                detected[res["language"]] += 1

            total = sum(detected.values()) or 1
            top_lang = max(detected, key=detected.get) if detected else "unknown"
            top_pct = 100 * detected[top_lang] / total

            lang_results[key] = {
                "expected_language": expected_lang,
                "expected_code": expected_code,
                "detected_language": top_lang,
                "match": top_lang == expected_code,
                "confidence": round(top_pct, 1),
                "distribution": {k: round(100 * v / total, 1) for k, v in sorted(detected.items(), key=lambda x: -x[1])[:5]},
            }

            if top_lang != expected_code and top_pct > 50:
                alert = f"MISMATCH: {key} → expected={expected_lang}({expected_code}), detected={top_lang} ({top_pct:.0f}%)"
                self.results["alerts"].append(alert)
                print(f"  [WARN] {alert}")

        self.results["language_verification"] = lang_results
        mismatches = sum(1 for v in lang_results.values() if v["match"] is False)
        print(f"  Language mismatches: {mismatches}/{len(lang_results)}")

    # ─── 3. Quality Score Distribution ─────────────────────────────────────

    def _check_quality(self):
        print("\n[3/8] Quality scoring (sampled)...")
        try:
            from ..quality_scoring.heuristic import HeuristicQualityScorer
            scorer = HeuristicQualityScorer()
        except Exception as e:
            self.results["alerts"].append(f"Quality scorer unavailable: {e}")
            return

        q_results = {}
        for key, finfo in self.results["files"].items():
            fpath = self._resolve_path(key)
            if not os.path.isfile(fpath) or finfo["size_mb"] < 0.5:
                continue
            lines = self._get_cached_sample(fpath, min(self.sample_size, 50))
            if not lines:
                continue

            all_scores = []
            for line in lines:
                if len(line.strip()) < 100:
                    continue
                s = scorer.score(line, finfo["language"])
                all_scores.append(s)
                if len(all_scores) >= 200:
                    break

            if not all_scores:
                continue

            keys_avg = ["perplexity", "readability", "entropy", "compression_ratio", "info_density"]
            avgs = {k: round(sum(s[k] for s in all_scores) / len(all_scores), 2) for k in keys_avg}
            perplexities = sorted([s["perplexity"] for s in all_scores])
            entropies = sorted([s["entropy"] for s in all_scores])

            q_results[key] = {
                "samples": len(all_scores),
                "averages": avgs,
                "perplexity_p50": perplexities[len(perplexities) // 2],
                "perplexity_p90": perplexities[int(len(perplexities) * 0.9)],
                "entropy_p50": entropies[len(entropies) // 2],
            }

            if avgs["perplexity"] > 5000:
                self.results["alerts"].append(f"High avg perplexity ({avgs['perplexity']:.0f}) in {key}")

        self.results["quality_distribution"] = q_results

        # Language-level averages
        lang_quality = defaultdict(lambda: defaultdict(list))
        for key, q in q_results.items():
            lang = self.results["files"].get(key, {}).get("language", "unknown")
            for k, v in q.get("averages", {}).items():
                lang_quality[lang][k].append(v)

        self.results["quality_by_language"] = {}
        for lang, metrics in lang_quality.items():
            self.results["quality_by_language"][lang] = {
                k: round(sum(v) / len(v), 2) for k, v in metrics.items()
            }
        print(f"  Scored {len(q_results)} files")

    # ─── 4. Repetition Check ───────────────────────────────────────────────

    def _check_repetition(self):
        print("\n[4/8] Repetition check...")
        from ..filtering.repetition_filter import RepetitionFilter
        rfilter = RepetitionFilter()
        rep_results = {}

        for key, finfo in self.results["files"].items():
            fpath = self._resolve_path(key)
            if not os.path.isfile(fpath) or finfo["size_mb"] < 0.5:
                continue
            lines = self._get_cached_sample(fpath, min(self.sample_size, 50))
            if not lines:
                continue

            failed = 0
            reasons = defaultdict(int)
            for line in lines:
                ok, info = rfilter.check(line)
                if not ok:
                    failed += 1
                    for r in info.get("reasons", []):
                        reasons[r] += 1

            rep_results[key] = {
                "samples": len(lines),
                "repetition_failures": failed,
                "failure_rate": round(100 * failed / max(1, len(lines)), 1),
                "reasons": dict(reasons),
            }

            if failed > max(5, len(lines) * 0.5):
                self.results["alerts"].append(f"High repetition ({100*failed//max(1,len(lines))}%) in {key}")

        self.results["repetition"] = rep_results
        print(f"  Checked {len(rep_results)} files")

    # ─── 5. Composition Check ──────────────────────────────────────────────

    def _check_composition(self):
        print("\n[5/8] Composition check...")
        from ..filtering.composition_filter import CompositionFilter
        cfilter = CompositionFilter(alpha_min_ratio=0.3, require_stopwords=False)
        comp_results = {}

        for key, finfo in self.results["files"].items():
            fpath = self._resolve_path(key)
            if not os.path.isfile(fpath) or finfo["size_mb"] < 0.5:
                continue
            lines = self._get_cached_sample(fpath, min(self.sample_size, 50))
            if not lines:
                continue

            failed = 0
            for line in lines:
                ok, _ = cfilter.check(line, finfo["language"])
                if not ok:
                    failed += 1

            comp_results[key] = {
                "samples": len(lines),
                "composition_failures": failed,
                "failure_rate": round(100 * failed / max(1, len(lines)), 1),
            }

        self.results["composition"] = comp_results
        print(f"  Checked {len(comp_results)} files")

    # ─── 6. Boilerplate Ratio ─────────────────────────────────────────────

    def _check_boilerplate(self):
        print("\n[6/8] Boilerplate check...")
        from ..cleaning.boilerplate_detector import BoilerplateDetector
        bd = BoilerplateDetector(main_content_ratio_threshold=0.3)
        bp_results = {}

        for key, finfo in self.results["files"].items():
            fpath = self._resolve_path(key)
            if not os.path.isfile(fpath) or finfo["size_mb"] < 0.5:
                continue
            lines = self._get_cached_sample(fpath, min(self.sample_size, 50))
            text = "\n".join(lines[:500])
            if not text.strip():
                continue
            ratio, acceptable = bd.detect(text)
            bp_results[key] = {
                "content_ratio": round(ratio, 3),
                "acceptable": acceptable,
            }

        self.results["boilerplate"] = bp_results
        low_bp = sum(1 for v in bp_results.values() if not v["acceptable"])
        print(f"  Low content ratio files: {low_bp}/{len(bp_results)}")

    # ─── 7. Tokenization Stats ─────────────────────────────────────────────

    def _check_tokenization(self):
        print("\n[7/8] Tokenization stats...")
        vocab_path = "vocab_6/vocab.txt"
        merges_path = "vocab_6/merges.txt"
        if not os.path.exists(vocab_path):
            self.results["alerts"].append("vocab_6/ not found — skipping tokenization stats")
            return

        try:
            from ..tokenization.encoder import TokenizationEncoder
            encoder = TokenizationEncoder(vocab_path=vocab_path, merges_path=merges_path, max_seq_len=2048)
            encoder.load_vocab()
        except Exception as e:
            self.results["alerts"].append(f"Tokenizer unavailable: {e}")
            return

        tok_results = {}
        tok_sample = min(self.sample_size, 5)
        for key, finfo in self.results["files"].items():
            fpath = self._resolve_path(key)
            if not os.path.isfile(fpath) or finfo["size_mb"] < 0.5:
                continue
            lines = self._get_cached_sample(fpath, tok_sample)
            if not lines:
                continue

            total_chars = 0
            total_tokens = 0
            count = 0
            for line in lines:
                if len(line.strip()) < 50:
                    continue
                ids = encoder.encode(line, add_bos=False, add_eos=False, max_len=2048)
                total_chars += len(line)
                total_tokens += len(ids)
                count += 1
                if count >= 30:
                    break

            ratio = round(total_chars / max(1, total_tokens), 1)
            tok_results[key] = {
                "samples": count,
                "total_chars": total_chars,
                "total_tokens": total_tokens,
                "chars_per_token": ratio,
            }

        self.results["tokenization"] = tok_results

        # Per-language averages
        lang_tok = defaultdict(list)
        for key, t in tok_results.items():
            lang = self.results["files"].get(key, {}).get("language", "unknown")
            lang_tok[lang].append(t["chars_per_token"])

        self.results["tokenization_by_language"] = {
            lang: round(sum(vals) / len(vals), 1) for lang, vals in lang_tok.items()
        }
        print(f"  Tokenized {len(tok_results)} files")

    # ─── 8. Cross-File Dedup ──────────────────────────────────────────────

    def _check_cross_dedup(self):
        print("\n[8/8] Cross-file dedup check...")
        seen_hashes: Dict[str, str] = {}
        dup_results = []

        for key, finfo in self.results["files"].items():
            fpath = self._resolve_path(key)
            if not os.path.isfile(fpath) or finfo["size_mb"] < 0.5:
                continue
            lines = self._get_cached_sample(fpath, min(self.sample_size, 50))
            duplicates = 0
            for line in lines:
                h = text_sha256(line.strip())
                if h in seen_hashes:
                    dup_results.append({
                        "file_a": seen_hashes[h],
                        "file_b": key,
                        "hash": h[:16],
                    })
                    duplicates += 1
                else:
                    seen_hashes[h] = key

        self.results["cross_dedup"] = {
            "duplicates_found": len(dup_results),
            "samples": list(dup_results[:20]),
        }
        print(f"  Cross-file duplicates found: {len(dup_results)}")

    # ─── 9. Contamination ─────────────────────────────────────────────────

    def _check_contamination(self):
        print("\n[Contamination] Checking...")
        from ..contamination.benchmark_loader import BenchmarkLoader
        from ..contamination.ngram_detector import NgramContaminationDetector

        loader = BenchmarkLoader(cache_dir="data/benchmarks")
        benchmarks = loader.load_cached()
        if not benchmarks:
            print("  No cached benchmarks — skipping (run benchmark_loader first)")
            self.results["contamination"] = {"status": "skipped", "reason": "no_cached_benchmarks"}
            return

        all_bench_texts = loader.extract_all_texts(benchmarks)
        detector = NgramContaminationDetector(ngram_size=13)
        print(f"  Building index from {len(all_bench_texts):,} benchmark items...")
        detector.build_index(all_bench_texts)

        total_contaminated = 0
        file_contamination = {}
        for key, finfo in self.results["files"].items():
            fpath = self._resolve_path(key)
            if not os.path.isfile(fpath) or finfo["size_mb"] < 1:
                continue
            lines = self._get_cached_sample(fpath, min(self.sample_size, 500))
            contaminated = 0
            for line in lines:
                is_cont, _ = detector.check_document(key, line.strip())
                if is_cont:
                    contaminated += 1

            if contaminated > 0:
                file_contamination[key] = {
                    "samples": len(lines),
                    "contaminated": contaminated,
                    "rate_pct": round(100 * contaminated / max(1, len(lines)), 3),
                }
                total_contaminated += contaminated

        self.results["contamination"] = {
            "status": "completed",
            "ngram_size": 13,
            "benchmark_items": len(all_bench_texts),
            "total_contaminated_samples": total_contaminated,
            "per_file": file_contamination,
        }

        if total_contaminated > 0:
            self.results["alerts"].append(f"CONTAMINATION: {total_contaminated} samples overlap with benchmarks!")
        else:
            print("  ZERO contamination detected ✓")

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _resolve_path(self, key: str) -> str:
        fpath = os.path.join(os.getcwd(), key)
        if os.path.islink(fpath):
            return os.path.realpath(fpath)
        return fpath if os.path.isfile(fpath) else ""

    def _get_cached_sample(self, path: str, n: int) -> List[str]:
        """Return cached sample lines, or sample and cache them."""
        cache_key = f"{path}:n={min(n, self.sample_size)}"
        if cache_key not in self._sample_cache:
            self._sample_cache[cache_key] = self._sample_lines(path, min(n, self.sample_size))
        cached = self._sample_cache[cache_key]
        return cached[:n]

    def _sample_lines(self, path: str, n: int) -> List[str]:
        """Reservoir sampling — O(N) time, O(n) memory. Never loads full file."""
        if not os.path.isfile(path):
            return []
        try:
            rng = random.Random(self.seed)
            reservoir = []
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    line = line.rstrip("\n\r")
                    if i < n:
                        reservoir.append(line)
                    else:
                        j = rng.randint(0, i)
                        if j < n:
                            reservoir[j] = line
            return reservoir
        except Exception:
            return []

    def _count_lines(self, path: str) -> int:
        """Fast line count using buffered byte scanning."""
        try:
            size = os.path.getsize(path)
            if size > 500_000_000:
                return int(size / 200)
            with open(path, "rb") as f:
                buf = f.read(65536)
                avg_line = max(1, len(buf) // max(1, buf.count(b"\n")))
                total = size // avg_line if avg_line > 0 else 0
            return total
        except Exception:
            return 0
