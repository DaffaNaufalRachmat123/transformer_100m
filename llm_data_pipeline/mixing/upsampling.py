"""
Domain Upsampling — per DATA_PIPELINE_PLAN A.11.

High-quality domains upsampled relative to volume:
  - Code: 3x (crucial for reasoning, small raw volume)
  - Math: 5x (small volume, high impact on reasoning)
  - Wikipedia: 3x (factual anchor, ground truth)
  - Educational: 2x (FineWeb-Edu findings)

Cap: no domain repeated > 5 epochs.
"""

import math
from typing import Dict, List


DEFAULT_UPSAMPLE_FACTORS = {
    "code": 3.0,
    "math": 5.0,
    "wikipedia": 3.0,
    "educational": 2.0,
}


class DomainUpsampler:
    """Upsample high-quality domains with controlled repetition."""

    def __init__(
        self,
        upsample_factors: Dict[str, float] = None,
        max_epochs: int = 5,
        min_weight: float = 0.001,
    ):
        self.upsample_factors = upsample_factors or DEFAULT_UPSAMPLE_FACTORS.copy()
        self.max_epochs = max_epochs
        self.min_weight = min_weight

    def apply(self, documents: List[Dict]) -> List[Dict]:
        """Apply upsampling to documents by adjusting sampling weights.

        Upsampling works by multiplying the existing sampling_weight by the
        upsample factor. The cap of max_epochs prevents any domain from being
        repeated excessively.
        """
        domain_counts: Dict[str, int] = {}
        domain_total_weight: Dict[str, float] = {}
        total_weight = 0.0

        for doc in documents:
            domain = doc.get("domain", "web_text")
            w = doc.get("sampling_weight", 1.0)
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            domain_total_weight[domain] = domain_total_weight.get(domain, 0) + w
            total_weight += w

        # Compute base epoch count for each domain
        domain_epochs = {}
        for domain, count in domain_counts.items():
            raw_epochs = total_weight / max(domain_total_weight.get(domain, 1.0), 0.001)
            domain_epochs[domain] = raw_epochs

        # Apply upsampling with epoch cap
        results = []
        for doc in documents:
            domain = doc.get("domain", "web_text")
            factor = self.upsample_factors.get(domain, 1.0)

            if domain in domain_epochs:
                current_epochs = domain_epochs[domain]
                effective_factor = min(factor, self.max_epochs / max(current_epochs, 0.001))
                effective_factor = max(1.0, effective_factor)
            else:
                effective_factor = 1.0

            doc_out = dict(doc)
            original_weight = doc.get("sampling_weight", 1.0)
            doc_out["sampling_weight"] = max(self.min_weight, original_weight * effective_factor)
            doc_out["upsample_factor"] = effective_factor
            results.append(doc_out)

        return results

    def print_upsampling_plan(self, documents: List[Dict]):
        """Print the upsampling plan for review."""
        from collections import defaultdict
        domain_weights = defaultdict(float)
        domain_counts = defaultdict(int)

        for doc in documents:
            domain = doc.get("domain", "web_text")
            domain_weights[domain] += doc.get("sampling_weight", 1.0)
            domain_counts[domain] += 1

        total_w = sum(domain_weights.values())

        print(f"\n{'='*60}\n DOMAIN UPSAMPLING PLAN\n{'='*60}")
        print(f"  {'Domain':<20s} {'Docs':>10s} {'Weight%':>10s} {'Factor':>10s}")
        print("-" * 56)
        for domain in sorted(domain_weights.keys(), key=lambda d: -domain_weights[d]):
            count = domain_counts[domain]
            w_pct = 100 * domain_weights[domain] / max(1, total_w)
            factor = self.upsample_factors.get(domain, 1.0)
            print(f"  {domain:<20s} {count:>10,} {w_pct:>9.1f}% {factor:>9.1f}x")
        print("=" * 60)
