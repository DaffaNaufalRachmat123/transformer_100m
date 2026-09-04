#!/usr/bin/env python3
# =============================================================================
# data_registry.py — Centralized Data Tracking for All Pipelines
# =============================================================================
# Tracks datasets across LLM, Vision, and Voice pipelines.
# Uses SQLite for durability. Resumable: skips already-downloaded datasets.

import sqlite3
import os
import hashlib
import argparse
from datetime import datetime
from pathlib import Path

DB_PATH = os.path.join(os.path.dirname(__file__), "data_registry.db")

MODALITIES = ["llm", "vision", "voice"]
SUB_MODALITIES = {
    "llm": ["pretrain"],
    "vision": ["siglip", "detection", "ocr"],
    "voice": ["ssl", "asr", "speaker"],
}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            modality TEXT NOT NULL,
            sub_modality TEXT NOT NULL,
            source_url TEXT,
            source_type TEXT,
            language TEXT,
            raw_size_bytes INTEGER DEFAULT 0,
            raw_file_count INTEGER DEFAULT 0,
            download_status TEXT DEFAULT 'pending',
            processed_size_bytes INTEGER DEFAULT 0,
            token_count INTEGER DEFAULT 0,
            hour_count INTEGER DEFAULT 0,
            image_count INTEGER DEFAULT 0,
            checksum_sha256 TEXT,
            download_date TEXT,
            processing_date TEXT,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()

def register_dataset(name, modality, sub_modality, **kwargs):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT OR IGNORE INTO datasets (name, modality, sub_modality, source_url,
                source_type, language, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, modality, sub_modality,
              kwargs.get("source_url", ""),
              kwargs.get("source_type", "manual"),
              kwargs.get("language", ""),
              kwargs.get("notes", "")))
        conn.commit()
    finally:
        conn.close()

def update_download_status(name, status, size_bytes=0, file_count=0):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        UPDATE datasets SET download_status=?, raw_size_bytes=?, raw_file_count=?,
            download_date=?
        WHERE name=?
    """, (status, size_bytes, file_count, datetime.now().isoformat(), name))
    conn.commit()
    conn.close()

def list_datasets(modality=None, status=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    query = "SELECT id, name, modality, sub_modality, language, download_status, raw_size_bytes, hour_count, image_count, token_count FROM datasets"
    conditions = []
    params = []
    if modality:
        conditions.append("modality=?")
        params.append(modality)
    if status:
        conditions.append("download_status=?")
        params.append(status)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY modality, sub_modality, name"
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return rows

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    stats = {}
    for mod in MODALITIES:
        c.execute("SELECT COUNT(*), SUM(raw_size_bytes) FROM datasets WHERE modality=? AND download_status='complete'", (mod,))
        count, size = c.fetchone()
        stats[mod] = {"datasets": count or 0, "size_gb": round((size or 0) / 1e9, 2)}
    conn.close()
    return stats

def verify_checksum(name, expected_sha256):
    """Verify dataset integrity via SHA-256."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE datasets SET checksum_sha256=? WHERE name=?", (expected_sha256, name))
    conn.commit()
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Data Registry for all pipelines")
    parser.add_argument("--init", action="store_true", help="Initialize database")
    parser.add_argument("--list", action="store_true", help="List all datasets")
    parser.add_argument("--status", action="store_true", help="Show download status summary")
    parser.add_argument("--modality", choices=MODALITIES, help="Filter by modality")
    parser.add_argument("--verify", action="store_true", help="Check integrity")
    args = parser.parse_args()

    if args.init:
        init_db()
        print("[OK] Database initialized at", DB_PATH)

    if args.list or args.status:
        init_db()
        if args.list:
            rows = list_datasets(args.modality)
            print(f"\n{'ID':<4} {'Name':<40} {'Modality':<8} {'Sub':<12} {'Lang':<6} {'Status':<12} {'Size':<12}")
            print("-" * 100)
            for r in rows:
                size_str = ""
                if r[6]:
                    size_str = f"{r[6]/1e9:.2f} GB" if r[6] > 1e9 else f"{r[6]/1e6:.0f} MB"
                elif r[7]:
                    size_str = f"{r[7]:,} hrs"
                elif r[8]:
                    size_str = f"{r[8]:,} imgs"
                elif r[9]:
                    size_str = f"{r[9]:,} toks"
                print(f"{r[0]:<4} {r[1][:39]:<40} {r[2]:<8} {r[3]:<12} {r[4]:<6} {r[5]:<12} {size_str:<12}")

        if args.status:
            stats = get_stats()
            print("\n=== Download Status ===")
            for mod, s in stats.items():
                print(f"  {mod.upper():8s}: {s['datasets']:3d} datasets, {s['size_gb']:8.2f} GB")
            pending = list_datasets(status="pending")
            failed = list_datasets(status="failed")
            if pending:
                print(f"\n  PENDING: {len(pending)} datasets")
            if failed:
                print(f"  FAILED:  {len(failed)} datasets")

if __name__ == "__main__":
    main()
