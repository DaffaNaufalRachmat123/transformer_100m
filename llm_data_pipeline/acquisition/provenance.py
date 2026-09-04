"""
Provenance Tracking — per DATA_PIPELINE_PLAN A.1 + F.2.

Assigns a UUID to every document at ingestion and records:
  - source_url, crawl_date, snapshot_id, license
  - Every filter decision: {doc_id, stage, filter_name, passed/rejected, score, timestamp, pipeline_version}

Storage: SQLite database with append-only log.
"""

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from ..common import ensure_dir


DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_url TEXT,
    crawl_date TEXT,
    language TEXT,
    license TEXT,
    ingested_at TEXT NOT NULL,
    pipeline_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS filter_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    filter_name TEXT NOT NULL,
    passed INTEGER NOT NULL,
    score REAL,
    details TEXT,
    timestamp TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE INDEX IF NOT EXISTS idx_filter_doc ON filter_decisions(doc_id);
CREATE INDEX IF NOT EXISTS idx_filter_stage ON filter_decisions(stage);
CREATE INDEX IF NOT EXISTS idx_filter_passed ON filter_decisions(passed);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    pipeline_version TEXT NOT NULL,
    config_hash TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT DEFAULT 'running'
);
"""


class ProvenanceTracker:
    """Track document lineage through all pipeline stages."""

    def __init__(self, db_path: str = "data/provenance.db", pipeline_version: str = "pretrain_v1"):
        self.db_path = db_path
        self.pipeline_version = pipeline_version
        self._conn: Optional[sqlite3.Connection] = None
        ensure_dir(os.path.dirname(db_path))

    def open(self):
        """Open database connection and ensure schema exists."""
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(DB_SCHEMA)
        self._conn.commit()

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def register_document(self, doc_id: str, source: str, language: str = "en",
                          source_url: str = "", crawl_date: str = "",
                          license_: str = "unknown") -> bool:
        """Register a new document at ingestion time."""
        if not self._conn:
            return False
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?)",
                (doc_id, source, source_url, crawl_date, language, license_, now, self.pipeline_version),
            )
            self._conn.commit()
            return True
        except sqlite3.Error:
            return False

    def register_batch(self, docs: list):
        """Register many documents in a single transaction."""
        if not self._conn:
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for doc in docs:
            rows.append((
                doc.get("doc_id", ""),
                doc.get("source", ""),
                doc.get("source_url", ""),
                doc.get("crawl_date", ""),
                doc.get("language", "en"),
                doc.get("license", "unknown"),
                now,
                self.pipeline_version,
            ))
        try:
            self._conn.executemany(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()
        except sqlite3.Error as e:
            print(f"[provenance] Batch insert error: {e}")

    def log_filter(self, doc_id: str, stage: str, filter_name: str,
                   passed: bool, score: Optional[float] = None,
                   details: Optional[str] = None):
        """Record a filter decision for a document."""
        if not self._conn:
            return
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._conn.execute(
                "INSERT INTO filter_decisions VALUES (NULL,?,?,?,?,?,?,?,?)",
                (doc_id, stage, filter_name, int(passed), score, details, now, self.pipeline_version),
            )
        except sqlite3.Error:
            pass

    def log_filter_batch(self, decisions: list):
        """Record multiple filter decisions at once."""
        if not self._conn or not decisions:
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (d["doc_id"], d["stage"], d["filter_name"],
             int(d["passed"]), d.get("score"), d.get("details"),
             now, self.pipeline_version)
            for d in decisions
        ]
        try:
            self._conn.executemany(
                "INSERT INTO filter_decisions VALUES (NULL,?,?,?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()
        except sqlite3.Error as e:
            print(f"[provenance] Batch log error: {e}")

    def get_lineage(self, doc_id: str) -> list:
        """Query all filter decisions for a document."""
        if not self._conn:
            return []
        cursor = self._conn.execute(
            "SELECT stage, filter_name, passed, score, details, timestamp "
            "FROM filter_decisions WHERE doc_id = ? ORDER BY timestamp",
            (doc_id,),
        )
        return cursor.fetchall()

    def get_stage_stats(self, stage: str) -> Dict:
        """Get pass/fail statistics for a pipeline stage."""
        if not self._conn:
            return {}
        cursor = self._conn.execute(
            "SELECT COUNT(*), SUM(passed) FROM filter_decisions WHERE stage = ?",
            (stage,),
        )
        total, passed = cursor.fetchone()
        return {
            "stage": stage,
            "total_documents": total or 0,
            "passed": passed or 0,
            "rejected": (total or 0) - (passed or 0),
            "pass_rate": (passed / total) if total else 0,
        }

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document metadata from the provenance database."""
        if not self._conn:
            return None
        cursor = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "doc_id": row[0], "source": row[1], "source_url": row[2],
                "crawl_date": row[3], "language": row[4], "license": row[5],
                "ingested_at": row[6], "pipeline_version": row[7],
            }
        return None

    def start_run(self, config_hash: str = "") -> str:
        """Record a new pipeline run."""
        import uuid
        run_id = str(uuid.uuid4())
        if self._conn:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "INSERT INTO pipeline_runs VALUES (?,?,?,?,NULL,'running')",
                (run_id, self.pipeline_version, config_hash, now),
            )
            self._conn.commit()
        return run_id

    def finish_run(self, run_id: str, status: str = "completed"):
        """Mark a pipeline run as completed."""
        if self._conn:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "UPDATE pipeline_runs SET finished_at = ?, status = ? WHERE run_id = ?",
                (now, status, run_id),
            )
            self._conn.commit()

    def commit(self):
        """Force commit pending transactions."""
        if self._conn:
            self._conn.commit()
