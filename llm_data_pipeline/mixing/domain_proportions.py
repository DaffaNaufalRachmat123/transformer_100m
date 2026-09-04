"""
Domain Proportion Mixer — per DATA_PIPELINE_PLAN A.11.

Assigns per-document sampling weights based on:
  - Domain proportions (web 55%, code 12%, math 5%, science 8%, books 5%, wiki 5%, ...)
  - Language proportions (EN 70%, ID 10%, JA 10%, KO 10%)
  - Freshness weighting (recent data weighted 1.5x)

Output: per-document {domain, language, sampling_weight} for the dataloader.
"""

import math
from datetime import datetime
from typing import Dict, List, Tuple


DEFAULT_DOMAIN_WEIGHTS = {
    "web_text": 0.55,
    "code": 0.12,
    "math": 0.05,
    "science": 0.08,
    "books": 0.05,
    "wikipedia": 0.05,
    "news": 0.03,
    "legal_medical": 0.02,
    "educational": 0.05,
}

DEFAULT_LANGUAGE_WEIGHTS = {
    "en": 0.70,
    "id": 0.10,
    "ja": 0.10,
    "ko": 0.10,
}

SOURCE_TO_DOMAIN = {
    "fineweb_edu": "web_text",
    "dclm_baseline": "web_text",
    "cc_news": "news",
    "wikipedia": "wikipedia",
    "wikimedia/wikipedia": "wikipedia",
    "arxiv": "science",
    "scientific_papers": "science",
    "pubmed_oa": "legal_medical",
    "pubmed_qa": "legal_medical",
    "the_stack_v2": "code",
    "the_stack_v2/python": "code",
    "the_stack_v2/javascript": "code",
    "the_stack_v2/java": "code",
    "mathpile": "math",
    "openwebmath": "math",
    "cc100": "web_text",
    "culturax": "web_text",
    "mc4": "web_text",
    "gutenberg": "books",
    "open_library": "books",
}


class DomainProportionMixer:
    """Compute per-document sampling weights."""

    def __init__(
        self,
        domain_weights: Dict[str, float] = None,
        language_weights: Dict[str, float] = None,
        freshness_weight: float = 1.5,
        freshness_window_days: int = 365,
        default_domain: str = "web_text",
        default_language: str = "en",
    ):
        self.domain_weights = domain_weights or DEFAULT_DOMAIN_WEIGHTS.copy()
        self.language_weights = language_weights or DEFAULT_LANGUAGE_WEIGHTS.copy()
        self.freshness_weight = freshness_weight
        self.freshness_window = freshness_window_days
        self.default_domain = default_domain
        self.default_language = default_language

        self._normalize_weights()

    def get_domain(self, source: str) -> str:
        """Map source name to domain category."""
        source_lower = source.lower()
        for key, domain in SOURCE_TO_DOMAIN.items():
            if key in source_lower:
                return domain
        if "code" in source_lower or source_lower.endswith("/python") or source_lower.endswith("/java"):
            return "code"
        if "math" in source_lower:
            return "math"
        if "science" in source_lower or "paper" in source_lower:
            return "science"
        if "book" in source_lower:
            return "books"
        if "wiki" in source_lower:
            return "wikipedia"
        if "news" in source_lower:
            return "news"
        if "medical" in source_lower or "legal" in source_lower or "pubmed" in source_lower:
            return "legal_medical"
        if "edu" in source_lower:
            return "educational"
        return self.default_domain

    def compute_weight(self, doc: Dict) -> float:
        """Compute sampling weight for a single document."""
        source = doc.get("source", "")
        language = doc.get("language", self.default_language)
        domain = self.get_domain(source)

        dw = self.domain_weights.get(domain, 0.01)
        lw = self.language_weights.get(language, 0.01)

        weight = dw * lw

        # Freshness boost
        crawl_date = doc.get("crawl_date", "")
        if crawl_date and self.freshness_weight > 1.0:
            try:
                if isinstance(crawl_date, str):
                    dt = datetime.fromisoformat(crawl_date.replace("Z", "+00:00"))
                elif isinstance(crawl_date, datetime):
                    dt = crawl_date
                else:
                    dt = None

                if dt:
                    days_ago = (datetime.utcnow() - dt.replace(tzinfo=None)).days
                    if 0 <= days_ago <= self.freshness_window:
                        freshness_factor = 1.0 + (self.freshness_weight - 1.0) * (
                            1.0 - days_ago / self.freshness_window
                        )
                        weight *= freshness_factor
            except (ValueError, TypeError):
                pass

        return max(0.001, weight)

    def assign_weights(self, documents: List[Dict]) -> List[Dict]:
        """Assign domain, language, and sampling_weight to all documents."""
        results = []
        for doc in documents:
            source = doc.get("source", "")
            language = doc.get("language", self.default_language)
            domain = self.get_domain(source)
            weight = self.compute_weight(doc)

            doc_out = dict(doc)
            doc_out["domain"] = domain
            doc_out["sampling_weight"] = weight
            doc_out["language"] = language
            results.append(doc_out)

        return results

    def print_distribution(self, documents: List[Dict]):
        """Print domain and language distribution."""
        from collections import defaultdict
        domain_counts = defaultdict(int)
        lang_counts = defaultdict(int)
        domain_tokens = defaultdict(int)

        for doc in documents:
            domain = doc.get("domain", self.default_domain)
            lang = doc.get("language", self.default_language)
            text = doc.get("text", "")
            domain_counts[domain] += 1
            lang_counts[lang] += 1
            domain_tokens[domain] += len(text.split())

        total = len(documents)
        print(f"\n{'='*60}\n DOMAIN DISTRIBUTION\n{'='*60}")
        print(f"  Total documents     : {total:,}")
        print("-" * 60)
        for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / max(1, total)
            est_tokens = domain_tokens.get(domain, 0)
            print(f"  {domain:<20s}: {count:>10,} ({pct:>5.1f}%)  ~{est_tokens//1000000:,}M tokens")
        print("-" * 60)
        print("  Language distribution:")
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / max(1, total)
            print(f"    {lang:<10s}: {count:>10,} ({pct:>5.1f}%)")
        print("=" * 60)

    def _normalize_weights(self):
        """Ensure weights sum to 1.0."""
        dw_sum = sum(self.domain_weights.values())
        if dw_sum > 0:
            self.domain_weights = {k: v / dw_sum for k, v in self.domain_weights.items()}
        lw_sum = sum(self.language_weights.values())
        if lw_sum > 0:
            self.language_weights = {k: v / lw_sum for k, v in self.language_weights.items()}
