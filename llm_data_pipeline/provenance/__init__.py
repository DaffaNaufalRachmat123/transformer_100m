"""
Lineage Tracker — per DATA_PIPELINE_PLAN F.2.

Tracks every document transformation through the pipeline:
  - Every filter decision logged per document
  - Store: append-only log (SQLite via acquisition/provenance.py)
  - Lineage query: "show me all transformations for doc X"
  - Aggregate stats: "what % rejected at stage Y?"
"""

from ..acquisition.provenance import ProvenanceTracker

__all__ = ["ProvenanceTracker"]
