"""Common utilities shared across pipeline modules."""

import hashlib
import json
import os
import uuid
from typing import Any, Dict, Optional


def load_jsonl(path: str):
    """Generator: yield parsed JSON objects from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, records, mode: str = "w"):
    """Write an iterable of dicts to a JSONL file."""
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def generate_doc_id() -> str:
    """Generate a UUID-based document identifier."""
    return str(uuid.uuid4())


def file_blake3(path: str) -> str:
    """Compute BLAKE3 hash of a file (fast, cryptographic)."""
    try:
        import blake3
    except ImportError:
        return file_sha256(path)
    h = blake3.blake3()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def text_sha256(text: str) -> str:
    """Compute SHA-256 of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
