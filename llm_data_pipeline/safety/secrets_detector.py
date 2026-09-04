"""
Secrets Detector — per DATA_PIPELINE_PLAN A.7.

Detects secrets/keys/tokens in training data:
  - AWS access keys (AKIA...), AWS secret keys
  - GCP service account keys, GCP API keys
  - GitHub personal access tokens
  - SSH private keys (-----BEGIN.*PRIVATE KEY-----)
  - PGP private keys
  - Generic API keys (Bearer tokens, OAuth secrets)
  - Slack tokens, Stripe keys, JWT tokens
  - Database connection strings

Action: REMOVE entire document if secrets detected (secrets = compromised data).
"""

import re
from typing import Dict, List, Tuple

SECRET_PATTERNS = [
    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("AWS_SECRET_KEY", re.compile(r"\b[0-9a-zA-Z/+]{40}\b")),
    ("GCP_API_KEY", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("GCP_SERVICE_ACCOUNT", re.compile(r"\"type\":\s*\"service_account\"")),
    ("GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,255}\b")),
    ("GITHUB_PAT_CLASSIC", re.compile(r"\bghp_[0-9a-zA-Z]{36}\b")),
    ("SSH_PRIVATE_KEY", re.compile(r"-----BEGIN\s+(?:RSA|DSA|EC|OPENSSH|PGP)\s+PRIVATE\s+KEY-----")),
    ("PGP_PRIVATE_KEY", re.compile(r"-----BEGIN\s+PGP\s+PRIVATE\s+KEY\s+BLOCK-----")),
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[0-9a-zA-Z\-]{10,}\b")),
    ("STRIPE_SECRET", re.compile(r"\bsk_(?:live|test)_[0-9a-zA-Z]{24,}\b")),
    ("STRIPE_PUBLISHABLE", re.compile(r"\bpk_(?:live|test)_[0-9a-zA-Z]{24,}\b")),
    ("JWT_TOKEN", re.compile(r"\beyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b")),
    ("HEROKU_API_KEY", re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")),
    ("DISCORD_TOKEN", re.compile(r"\b[NO][A-Za-z0-9]{23}\.[A-Za-z0-9\-_]{6}\.[A-Za-z0-9\-_]{27}\b")),
    ("NPM_TOKEN", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    ("CONNECTION_STRING", re.compile(r"\b(?:mongodb|postgresql|mysql|redis|jdbc)://[^@\s]+:[^@\s]+@[\w.]+", re.IGNORECASE)),
    ("GENERIC_API_KEY", re.compile(r"\b(?:api[_-]?key|apikey|api[_-]?secret|access[_-]?token|auth[_-]?token)\s*[:=]\s*[\"']?[A-Za-z0-9+/=_-]{20,}[\"']?", re.IGNORECASE)),
    ("PRIVATE_KEY_PEM", re.compile(r"(?:private|secret)\s*key\s*[:=]\s*[\"']?-----BEGIN")),
    ("AUTH_BEARER", re.compile(r"Authorization\s*:\s*Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)),
    ("HARDCODED_PASSWORD", re.compile(r"\b(?:password|passwd|pwd)\s*[:=]\s*[\"'][^\"']{4,}[\"']", re.IGNORECASE)),
    ("SIGNING_KEY", re.compile(r"-----BEGIN\s+(?:EC|RSA)\s+(?:PRIVATE|SIGNING)\s+KEY-----")),
]

COMPROMISED_DOC_INDICATORS = [
    "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "GCP_SERVICE_ACCOUNT",
    "SSH_PRIVATE_KEY", "PGP_PRIVATE_KEY", "HARDCODED_PASSWORD",
    "SIGNING_KEY",
]


class SecretsDetector:
    """Detect secrets/keys/tokens and remove compromised documents."""

    def __init__(self):
        self.stats = {
            "total_documents": 0,
            "clean": 0,
            "compromised": 0,
            "secret_types": {},
        }

    def detect(self, text: str) -> Tuple[bool, List[Dict]]:
        has_secrets = False
        detections = []

        for secret_type, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                detections.append({
                    "type": secret_type,
                    "match": match.group()[:40] + "..." if len(match.group()) > 43 else match.group(),
                    "start": match.start(),
                })
                has_secrets = True
                self.stats["secret_types"][secret_type] = self.stats["secret_types"].get(secret_type, 0) + 1

        return has_secrets, detections

    def is_compromised(self, detections: List[Dict]) -> bool:
        """Check if detected secrets mean the document should be entirely removed."""
        for d in detections:
            if d["type"] in COMPROMISED_DOC_INDICATORS:
                return True
        return len(detections) >= 3

    def filter_documents(self, documents: List[Dict], text_field: str = "text") -> Tuple[List[Dict], List[Dict]]:
        clean = []
        compromised = []

        for doc in documents:
            text = doc.get(text_field, "")
            self.stats["total_documents"] += 1

            has_secrets, detections = self.detect(text)

            if has_secrets and self.is_compromised(detections):
                doc_out = dict(doc)
                doc_out["secrets_detected"] = detections
                compromised.append(doc_out)
                self.stats["compromised"] += 1
            else:
                clean.append(doc)
                self.stats["clean"] += 1

        return clean, compromised

    def get_stats(self) -> Dict:
        s = dict(self.stats)
        s["secret_types"] = dict(s["secret_types"])
        return s

    def print_summary(self):
        s = self.get_stats()
        print(f"\n{'='*60}\n SECRETS DETECTION SUMMARY\n{'='*60}")
        print(f"  Documents scanned : {s['total_documents']:,}")
        print(f"  Clean             : {s['clean']:,}")
        print(f"  Compromised       : {s['compromised']:,}")
        if s["secret_types"]:
            print("-" * 60)
            print("  Secret types found:")
            for stype, count in sorted(s["secret_types"].items(), key=lambda x: -x[1]):
                print(f"    {stype:<30s}: {count:>6,}")
        print("=" * 60)
