"""
Benchmark Loader — Download and cache test/validation sets for 12 benchmarks.

Benchmarks loaded (per DATA_PIPELINE_PLAN A.9):
  MMLU (57 subjects, 14,042 questions)
  ARC-Easy + ARC-Challenge (7,787 questions)
  HellaSwag (10,042 examples)
  WinoGrande (1,267 examples)
  GSM8K (1,319 test problems)
  HumanEval (164 programming problems)
  MBPP (500 programming problems)
  TruthfulQA (817 questions)
  BBH (6,511 examples across 27 tasks)
  MATH (5,000 competition problems)
  IFEval (541 instructions)
"""

import json
import os
import re
from typing import Dict, List, Optional, Set

from ..common import ensure_dir, write_jsonl, load_jsonl


BENCHMARK_NAMES = [
    "mmlu",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "gsm8k",
    "humaneval",
    "mbpp",
    "truthfulqa",
    "bbh",
    "math",
    "ifeval",
]

HF_BENCHMARK_MAP = {
    "mmlu":          ("cais/mmlu", "test", "all"),
    "arc_easy":      ("ai2_arc", "ARC-Easy", "test"),
    "arc_challenge": ("ai2_arc", "ARC-Challenge", "test"),
    "hellaswag":     ("Rowan/hellaswag", None, "test"),
    "winogrande":    ("winogrande", "winogrande_debiased", "test"),
    "gsm8k":         ("gsm8k", "main", "test"),
    "humaneval":     ("openai_humaneval", None, "test"),
    "mbpp":          ("mbpp", "full", "test"),
    "truthfulqa":    ("truthful_qa", "multiple_choice", "validation"),
    "bbh":           ("lukaemon/bbh", None, None),
    "math":          ("hendrycks/competition_math", None, "test"),
    "ifeval":        ("google/IFEval", None, None),
}


class BenchmarkLoader:
    """Download, cache, and extract text from all benchmark test sets."""

    def __init__(self, cache_dir: str = "data/benchmarks", benchmarks: Optional[List[str]] = None):
        self.cache_dir = cache_dir
        self.benchmarks = benchmarks or BENCHMARK_NAMES
        ensure_dir(cache_dir)

    def load_all(self, force_download: bool = False) -> Dict[str, List[str]]:
        """Download all benchmarks and return dict of name -> list of text strings."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install 'datasets': pip install datasets")

        results: Dict[str, List[str]] = {}
        for bname in self.benchmarks:
            texts = self._load_benchmark(bname, load_dataset, force_download)
            if texts:
                results[bname] = texts
                self._cache_texts(bname, texts)
        return results

    def load_cached(self) -> Dict[str, List[str]]:
        """Load all benchmarks from cache (must have been downloaded first)."""
        results: Dict[str, List[str]] = {}
        for bname in self.benchmarks:
            cached = self._load_from_cache(bname)
            if cached:
                results[bname] = cached
        return results

    def extract_all_texts(self, benchmarks: Dict[str, List[str]]) -> List[str]:
        """Flatten all benchmark texts into a single list for n-gram extraction."""
        all_texts = []
        for texts in benchmarks.values():
            all_texts.extend(texts)
        return all_texts

    def _load_benchmark(self, name: str, load_fn, force: bool) -> List[str]:
        """Load a single benchmark from HuggingFace datasets."""
        cached = None if force else self._load_from_cache(name)
        if cached:
            return cached

        info = HF_BENCHMARK_MAP.get(name)
        if info is None:
            return []

        dataset_name, config_name, split = info
        try:
            if name == "mmlu":
                return self._load_mmlu(load_fn)
            elif name == "bbh":
                return self._load_bbh(load_fn)
            elif name == "ifeval":
                return self._load_ifeval(load_fn)
            elif name == "math":
                return self._load_math(load_fn)
            elif name == "humaneval":
                return self._load_humaneval(load_fn)
            elif name == "mbpp":
                return self._load_mbpp(load_fn)
            else:
                return self._load_generic(load_fn, dataset_name, config_name, split, name)
        except Exception as e:
            print(f"[benchmark] Failed to load {name}: {e}")
            return []

    def _load_generic(self, load_fn, dataset_name, config, split, name) -> List[str]:
        """Generic loader for simple benchmark datasets."""
        kwargs = {"path": dataset_name, "split": split}
        if config:
            kwargs["name"] = config
        ds = load_fn(**kwargs)
        texts = []
        for row in ds:
            text = self._extract_text_from_row(row, name)
            if text:
                texts.append(text)
        return texts

    def _load_mmlu(self, load_fn) -> List[str]:
        """MMLU: 57 subjects, load all dev/test splits."""
        texts = []
        subjects = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology",
            "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "human_sexuality",
            "international_law", "jurisprudence", "logical_fallacies",
            "machine_learning", "management", "marketing", "medical_genetics",
            "miscellaneous", "moral_disputes", "moral_scenarios",
            "nutrition", "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology",
            "us_foreign_policy", "virology", "world_religions",
        ]
        for subject in subjects:
            try:
                ds = load_fn("cais/mmlu", subject, split="test")
                for row in ds:
                    text = self._extract_text_from_row(row, "mmlu")
                    if text:
                        texts.append(text)
            except Exception:
                continue
        return texts

    def _load_bbh(self, load_fn) -> List[str]:
        """BBH: Big Bench Hard — 27 tasks."""
        texts = []
        tasks = [
            "boolean_expressions", "causal_judgement", "date_understanding",
            "disambiguation_qa", "dyck_languages", "formal_fallacies",
            "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
            "logical_deduction_seven_objects", "logical_deduction_three_objects",
            "movie_recommendation", "multistep_arithmetic_two",
            "navigate", "object_counting", "penguins_in_a_table",
            "reasoning_about_colored_objects", "ruin_names",
            "salient_translation_error_detection", "snarks",
            "sports_understanding", "temporal_sequences",
            "tracking_shuffled_objects_five_objects",
            "tracking_shuffled_objects_seven_objects",
            "tracking_shuffled_objects_three_objects",
            "web_of_lies", "word_sorting",
        ]
        for task in tasks:
            try:
                ds = load_fn("lukaemon/bbh", task, split="test")
                for row in ds:
                    text = self._extract_text_from_row(row, "bbh")
                    if text:
                        texts.append(text)
            except Exception:
                continue
        return texts

    def _load_ifeval(self, load_fn) -> List[str]:
        """IFEval: Instruction Following Evaluation."""
        texts = []
        try:
            ds = load_fn("google/IFEval", split="train")
            for row in ds:
                prompt = row.get("prompt", "")
                if prompt and prompt.strip():
                    texts.append(prompt.strip())
        except Exception:
            pass
        return texts

    def _load_math(self, load_fn) -> List[str]:
        """MATH: Competition math problems."""
        texts = []
        try:
            ds = load_fn("hendrycks/competition_math", split="test")
            for row in ds:
                problem = row.get("problem", "")
                solution = row.get("solution", "")
                combined = f"Problem: {problem}\nSolution: {solution}"
                texts.append(combined)
        except Exception:
            pass
        return texts

    def _load_humaneval(self, load_fn) -> List[str]:
        """HumanEval: Programming problems."""
        texts = []
        try:
            ds = load_fn("openai_humaneval", split="test")
            for row in ds:
                prompt = row.get("prompt", "")
                canonical = row.get("canonical_solution", "")
                entry = row.get("entry_point", "")
                combined = f"{prompt}\n{canonical}"
                if entry:
                    combined += f"\nFunction: {entry}"
                texts.append(combined)
        except Exception:
            pass
        return texts

    def _load_mbpp(self, load_fn) -> List[str]:
        """MBPP: Mostly Basic Python Problems."""
        texts = []
        try:
            ds = load_fn("mbpp", split="test")
            for row in ds:
                text = row.get("text", "")
                code = row.get("code", "")
                combined = f"{text}\n{code}"
                texts.append(combined)
        except Exception:
            pass
        return texts

    def _extract_text_from_row(self, row: Dict, benchmark_name: str) -> str:
        """Extract relevant text fields from a benchmark row."""
        if benchmark_name == "gsm8k":
            return f"Q: {row.get('question', '')}\nA: {row.get('answer', '')}"
        if benchmark_name == "truthfulqa":
            return f"Q: {row.get('question', '')}\nA: {'; '.join(row.get('mc1_targets', {}).get('choices', []))}"

        # Generic extraction: try common keys
        for key in ["question", "prompt", "input", "text", "ctx", "context", "sentence"]:
            val = row.get(key, "")
            if isinstance(val, str) and val.strip():
                return val.strip()

        # Arc-specific
        if "question" in row and "choices" in row:
            q = row["question"]
            choices = row.get("choices", {})
            if isinstance(choices, dict):
                choice_texts = choices.get("text", [])
            else:
                choice_texts = choices if isinstance(choices, list) else []
            return f"{q}\n" + "\n".join(choice_texts)

        # HellaSwag
        if "ctx" in row and "endings" in row:
            ctx = row["ctx"]
            endings = row.get("endings", [])
            return f"{ctx} " + " ".join(endings)

        # Fallback
        return json.dumps(row, ensure_ascii=False)

    def _cache_texts(self, name: str, texts: List[str]):
        """Write benchmark texts to JSONL cache."""
        path = os.path.join(self.cache_dir, f"{name}.jsonl")
        records = [{"id": i, "text": t} for i, t in enumerate(texts)]
        write_jsonl(path, records)

    def _load_from_cache(self, name: str) -> Optional[List[str]]:
        """Load benchmark texts from JSONL cache."""
        path = os.path.join(self.cache_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            return None
        return [r["text"] for r in load_jsonl(path)]

    def print_stats(self, benchmarks: Dict[str, List[str]]):
        """Print benchmark statistics."""
        print("\n" + "=" * 70)
        print(" BENCHMARK CONTAMINATION REFERENCE SET ")
        print("=" * 70)
        total = 0
        for name in sorted(benchmarks.keys()):
            count = len(benchmarks[name])
            total += count
            total_chars = sum(len(t) for t in benchmarks[name])
            print(f"  {name:<20s}  {count:>6d} items  ({total_chars/1024:>8.1f} KB)")
        print("-" * 70)
        print(f"  {'TOTAL':<20s}  {total:>6d} items")
        print("=" * 70)
