"""
Pipeline Stage Metrics — per DATA_PIPELINE_PLAN F.3.

Collects per-stage metrics:
  - Throughput: documents/second, tokens/second
  - Rejection rate per filter (alert if > 50%)
  - Score distributions (histogram per quality signal)
  - Pipeline latency (end-to-end processing time)
  - Data drift detection (KL divergence between snapshot distributions)
"""

import json
import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class StageMetrics:
    """Collect and report metrics for a single pipeline stage."""

    def __init__(self, stage_name: str):
        self.stage_name = stage_name
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.documents_in = 0
        self.documents_out = 0
        self.documents_rejected = 0
        self.bytes_processed = 0
        self.scores: List[float] = []
        self.rejection_reasons: Dict[str, int] = defaultdict(int)
        self.sub_stages: Dict[str, "StageMetrics"] = {}

    def add_sub_stage(self, name: str) -> "StageMetrics":
        sm = StageMetrics(name)
        self.sub_stages[name] = sm
        return sm

    def finish(self):
        self.end_time = time.time()

    @property
    def elapsed_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def docs_per_second(self) -> float:
        return self.documents_in / max(0.001, self.elapsed_seconds)

    @property
    def rejection_rate(self) -> float:
        return self.documents_rejected / max(1, self.documents_in)

    @property
    def bytes_per_second(self) -> float:
        return self.bytes_processed / max(0.001, self.elapsed_seconds)

    @property
    def score_mean(self) -> float:
        return sum(self.scores) / max(1, len(self.scores))

    @property
    def score_median(self) -> float:
        if not self.scores:
            return 0.0
        s = sorted(self.scores)
        n = len(s)
        if n % 2 == 0:
            return (s[n // 2 - 1] + s[n // 2]) / 2
        return s[n // 2]

    def to_dict(self) -> Dict:
        return {
            "stage": self.stage_name,
            "elapsed_s": round(self.elapsed_seconds, 2),
            "documents_in": self.documents_in,
            "documents_out": self.documents_out,
            "documents_rejected": self.documents_rejected,
            "rejection_rate": round(self.rejection_rate, 4),
            "docs_per_second": round(self.docs_per_second, 1),
            "bytes_processed": self.bytes_processed,
            "bytes_per_second": round(self.bytes_per_second, 1),
            "score_mean": round(self.score_mean, 4) if self.scores else None,
            "score_median": round(self.score_median, 4) if self.scores else None,
            "scores_count": len(self.scores),
            "rejection_reasons": dict(self.rejection_reasons),
            "sub_stages": {k: v.to_dict() for k, v in self.sub_stages.items()},
        }


class PipelineMonitor:
    """Monitor and report metrics across all pipeline stages."""

    def __init__(self, log_dir: str = "logs", track_drift: bool = True):
        self.log_dir = log_dir
        self.track_drift = track_drift
        os.makedirs(log_dir, exist_ok=True)

        self.start_time = time.time()
        self.stages: Dict[str, StageMetrics] = {}
        self.alerts: List[str] = []
        self._prev_score_distributions: Dict[str, List[float]] = {}

    def get_stage(self, name: str) -> StageMetrics:
        if name not in self.stages:
            self.stages[name] = StageMetrics(name)
        return self.stages[name]

    def record_stage(self, name: str, docs_in: int, docs_out: int,
                     scores: Optional[List[float]] = None,
                     rejection_reasons: Optional[Dict[str, int]] = None,
                     bytes_processed: int = 0):
        stage = self.get_stage(name)
        stage.documents_in += docs_in
        stage.documents_out += docs_out
        stage.documents_rejected += docs_in - docs_out
        stage.bytes_processed += bytes_processed
        if scores:
            stage.scores.extend(scores)
        if rejection_reasons:
            for reason, count in rejection_reasons.items():
                stage.rejection_reasons[reason] += count
        stage.finish()

        if stage.rejection_rate > 0.5:
            alert = f"[ALERT] {name}: rejection rate {stage.rejection_rate:.1%} exceeds 50% threshold"
            self.alerts.append(alert)
            print(alert)

        if self.track_drift and scores:
            self._check_drift(name, scores)

        return stage

    def print_summary(self):
        """Print pipeline monitoring summary."""
        total_elapsed = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(" PIPELINE MONITORING SUMMARY")
        print("=" * 60)
        print(f"  Total elapsed        : {total_elapsed:.1f}s")
        print(f"  Stages completed     : {len(self.stages)}")
        print("-" * 60)

        for name, stage in self.stages.items():
            s = stage.to_dict()
            print(f"  [{name}]")
            print(f"    Docs in/out/rej    : {s['documents_in']:,} / {s['documents_out']:,} / {s['documents_rejected']:,}")
            print(f"    Rejection rate     : {s['rejection_rate']:.2%}")
            print(f"    Throughput         : {s['docs_per_second']:.0f} docs/s")
            if s["scores_count"] > 0:
                print(f"    Score mean/median  : {s['score_mean']:.3f} / {s['score_median']:.3f}")

        if self.alerts:
            print("-" * 60)
            print("  ALERTS:")
            for alert in self.alerts:
                print(f"    {alert}")

        print("=" * 60)

    def save_report(self, path: Optional[str] = None):
        """Save monitoring report to JSON."""
        path = path or os.path.join(self.log_dir, "pipeline_report.json")
        report = {
            "pipeline_version": "pretrain_v1",
            "total_elapsed_s": time.time() - self.start_time,
            "stages": {name: stage.to_dict() for name, stage in self.stages.items()},
            "alerts": self.alerts,
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    def _check_drift(self, stage_name: str, scores: List[float]):
        """Check for distribution drift vs previous runs."""
        if stage_name in self._prev_score_distributions:
            prev = self._prev_score_distributions[stage_name]
            kl = self._kl_divergence_approx(prev, scores)
            if kl > 0.1:
                alert = f"[DRIFT] {stage_name}: KL divergence {kl:.3f} exceeds threshold 0.1"
                self.alerts.append(alert)
                print(alert)

        self._prev_score_distributions[stage_name] = scores[:10000]

    def _kl_divergence_approx(self, dist1: List[float], dist2: List[float],
                               bins: int = 50) -> float:
        """Compute approximate KL divergence between two score distributions."""
        if not dist1 or not dist2:
            return 0.0

        min_val = min(min(dist1), min(dist2))
        max_val = max(max(dist1), max(dist2))
        if max_val == min_val:
            return 0.0

        bin_width = (max_val - min_val) / bins
        eps = 1e-10

        hist1 = [eps] * bins
        hist2 = [eps] * bins

        for v in dist1:
            idx = min(bins - 1, int((v - min_val) / bin_width))
            hist1[idx] += 1
        for v in dist2:
            idx = min(bins - 1, int((v - min_val) / bin_width))
            hist2[idx] += 1

        n1 = sum(hist1)
        n2 = sum(hist2)

        kl = 0.0
        for i in range(bins):
            p = hist1[i] / n1
            q = hist2[i] / n2
            if p > 0 and q > 0:
                kl += p * math.log(p / q)

        return kl
