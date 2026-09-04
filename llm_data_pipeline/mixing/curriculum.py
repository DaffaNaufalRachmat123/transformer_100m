"""
Curriculum Scheduler — per DATA_PIPELINE_PLAN A.11 + A.15.

Curriculum phases:
  Phase 1 (0-35% tokens): seq_len=256, easier text (lower perplexity)
  Phase 2 (35-75% tokens): seq_len=512, full distribution
  Phase 3 (75-100% tokens): seq_len=1024, higher quality emphasis

Each phase has different:
  - Sequence length
  - Language sampling weights (JA/KO weighted higher early for character coverage)
  - Quality emphasis (stricter filtering in later phases)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple


class CurriculumPhase(Enum):
    EASY = "phase_1_easy"
    FULL = "phase_2_full"
    QUALITY = "phase_3_quality"


PHASE_DEFINITIONS = {
    CurriculumPhase.EASY: {
        "name": "Phase 1 — Easy (Curriculum Start)",
        "seq_len": 256,
        "token_pct": 0.35,
        "description": "Lower perplexity, JA/KO weighted higher for character coverage",
        "language_weights": {
            "en": 0.22, "id": 0.20, "ja": 0.26, "ko": 0.22, "code": 0.10,
        },
        "min_quality_score": 0.2,
        "max_perplexity": 5000,
    },
    CurriculumPhase.FULL: {
        "name": "Phase 2 — Full Distribution",
        "seq_len": 512,
        "token_pct": 0.40,
        "description": "Full distribution, all quality levels included",
        "language_weights": {
            "en": 0.26, "id": 0.23, "ja": 0.24, "ko": 0.19, "code": 0.08,
        },
        "min_quality_score": 0.15,
        "max_perplexity": 10000,
    },
    CurriculumPhase.QUALITY: {
        "name": "Phase 3 — Quality Emphasis",
        "seq_len": 1024,
        "token_pct": 0.25,
        "description": "Higher quality emphasis, longer sequences",
        "language_weights": {
            "en": 0.29, "id": 0.26, "ja": 0.23, "ko": 0.17, "code": 0.05,
        },
        "min_quality_score": 0.3,
        "max_perplexity": 8000,
    },
}


class CurriculumScheduler:
    """Manage curriculum phases for pretraining."""

    def __init__(
        self,
        total_target_tokens: int = 10_000_000_000,
        phases: Optional[List[CurriculumPhase]] = None,
        current_phase: CurriculumPhase = CurriculumPhase.EASY,
        tokens_consumed: int = 0,
    ):
        self.total_target_tokens = total_target_tokens
        self.phases = phases or list(CurriculumPhase)
        self.current_phase = current_phase
        self.tokens_consumed = tokens_consumed

        self._phase_boundaries: Dict[CurriculumPhase, Tuple[int, int]] = {}
        self._compute_boundaries()

    def get_current_phase(self) -> CurriculumPhase:
        """Return current curriculum phase."""
        return self.current_phase

    def get_phase_config(self, phase: Optional[CurriculumPhase] = None) -> Dict:
        """Return configuration for a phase."""
        p = phase or self.current_phase
        return PHASE_DEFINITIONS.get(p, PHASE_DEFINITIONS[CurriculumPhase.FULL])

    def get_seq_len(self, phase: Optional[CurriculumPhase] = None) -> int:
        """Return sequence length for a phase."""
        return self.get_phase_config(phase)["seq_len"]

    def get_language_weights(self, phase: Optional[CurriculumPhase] = None) -> Dict[str, float]:
        """Return language sampling weights for a phase."""
        return self.get_phase_config(phase)["language_weights"]

    def advance(self) -> Optional[CurriculumPhase]:
        """Advance to the next curriculum phase if token budget exceeded."""
        current_idx = self.phases.index(self.current_phase)
        if current_idx + 1 >= len(self.phases):
            return None

        phase = self.current_phase
        if phase in self._phase_boundaries:
            _, end = self._phase_boundaries[phase]
            if self.tokens_consumed >= end:
                self.current_phase = self.phases[current_idx + 1]
                return self.current_phase

        return None

    def should_advance(self) -> bool:
        """Check if we should advance to the next phase."""
        phase = self.current_phase
        if phase in self._phase_boundaries:
            _, end = self._phase_boundaries[phase]
            return self.tokens_consumed >= end
        return False

    def get_tokens_remaining_in_phase(self) -> int:
        """Get tokens remaining in current phase."""
        phase = self.current_phase
        if phase in self._phase_boundaries:
            _, end = self._phase_boundaries[phase]
            return max(0, end - self.tokens_consumed)
        return self.total_target_tokens

    def get_phase_progress_pct(self) -> float:
        """Get progress through current phase (0-100)."""
        phase = self.current_phase
        if phase in self._phase_boundaries:
            start, end = self._phase_boundaries[phase]
            if end > start:
                return 100.0 * min(1.0, (self.tokens_consumed - start) / (end - start))
        return 0.0

    def add_tokens(self, tokens: int):
        """Record tokens consumed."""
        self.tokens_consumed += tokens

    def filter_by_phase(self, documents: List[Dict]) -> List[Dict]:
        """Filter documents based on current phase quality requirements."""
        config = self.get_phase_config()
        min_quality = config.get("min_quality_score", 0)
        max_ppl = config.get("max_perplexity", 10000)

        filtered = []
        for doc in documents:
            q_score = doc.get("quality_final", 0.5)
            ppl = doc.get("perplexity", 5000)

            if q_score >= min_quality and ppl <= max_ppl:
                filtered.append(doc)

        return filtered

    def _compute_boundaries(self):
        """Compute token boundaries for each phase."""
        start = 0
        for phase in self.phases:
            config = PHASE_DEFINITIONS.get(phase, {})
            phase_tokens = int(self.total_target_tokens * config.get("token_pct", 1.0 / len(self.phases)))
            self._phase_boundaries[phase] = (start, start + phase_tokens)
            start += phase_tokens

        # Adjust last phase to capture any rounding
        if self.phases:
            last = self.phases[-1]
            self._phase_boundaries[last] = (
                self._phase_boundaries[last][0],
                self.total_target_tokens,
            )

    def get_phase_plan(self) -> Dict:
        """Return full phase plan with token budgets."""
        plan = {}
        for phase in self.phases:
            config = self.get_phase_config(phase)
            start, end = self._phase_boundaries.get(phase, (0, 0))
            plan[phase.value] = {
                "name": config["name"],
                "seq_len": config["seq_len"],
                "tokens": end - start,
                "pct": config["token_pct"],
                "language_weights": config["language_weights"],
                "description": config["description"],
            }
        return plan

    def print_plan(self):
        """Print the curriculum plan."""
        plan = self.get_phase_plan()
        print(f"\n{'='*60}\n CURRICULUM PLAN ({self.total_target_tokens//1_000_000_000}B tokens)\n{'='*60}")
        print(f"  {'Phase':<25s} {'Seq':>5s} {'Tokens':>12s} {'Pct':>6s}")
        print("-" * 52)
        for key, info in plan.items():
            tokens_str = f"{info['tokens']//1_000_000}M"
            print(f"  {key:<25s} {info['seq_len']:>4d}  {tokens_str:>11s} {info['pct']*100:>5.0f}%")
        print("=" * 60)
