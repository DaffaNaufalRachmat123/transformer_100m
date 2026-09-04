"""Binary dataset generation per DATA_PIPELINE_PLAN A.14.

Sub-modules:
  shard_writer  — Write memory-mapped binary token files
  index_builder — Build global index mapping idx → (shard_id, offset)
"""

from .shard_writer import ShardWriter
from .index_builder import IndexBuilder

__all__ = ["ShardWriter", "IndexBuilder"]
