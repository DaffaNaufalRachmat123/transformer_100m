"""
PII Detection & Redaction — per DATA_PIPELINE_PLAN A.7.

Detects and redacts personally identifiable information:
  - Regex: SSN, credit card numbers, phone numbers, email addresses, IP
  - NER-based: names, addresses, organizations (spaCy + Presidio)

Action: REDACT with [PII_TYPE] placeholder — don't remove the document.
Content surrounding PII may still be valuable for training.
"""

import re
from typing import Dict, List, Tuple

PII_PATTERNS = [
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("EMAIL", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("PHONE_US", re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("IPV4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("IPV6", re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b")),
    ("PASSPORT_UK", re.compile(r"\b\d{9}\b")),
    ("PASSPORT_US", re.compile(r"\b[A-Z]\d{8}\b")),
    ("PASSPORT_ID", re.compile(r"\b[A-Z]\d{7}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b")),
    ("BITCOIN_ADDR", re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b")),
    ("ETH_ADDR", re.compile(r"\b0x[a-fA-F0-9]{40}\b")),
    ("DOB", re.compile(r"\b(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])\b")),
    ("LAT_LON", re.compile(r"\b[-+]?\d{1,2}\.\d{4,}\s*,\s*[-+]?\d{1,3}\.\d{4,}\b")),
    ("VIN", re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")),
    ("MEDICAL_RECORD", re.compile(r"\bMRN[:\s]*\d{5,10}\b", re.IGNORECASE)),
    ("DRIVER_LICENSE_US", re.compile(r"\b[A-Z]\d{7,8}\b")),
]

PLACEHOLDER_MAP = {name: f"[{name}]" for name, _ in PII_PATTERNS}

SPACY_MODELS = {
    "en": "en_core_web_sm",
    "id": "xx_ent_wiki_sm",  # multilingual fallback
    "ja": "xx_ent_wiki_sm",
    "ko": "xx_ent_wiki_sm",
}


class PIIDetector:
    """Detect and redact PII from text."""

    def __init__(
        self,
        use_regex: bool = True,
        use_spacy: bool = True,
        use_presidio: bool = True,
        redaction_strategy: str = "placeholder",
    ):
        self.use_regex = use_regex
        self.use_spacy = use_spacy
        self.use_presidio = use_presidio
        self.redaction_strategy = redaction_strategy

        self.stats = {
            "total_documents": 0,
            "pii_detections": {},
            "total_redactions": 0,
            "documents_with_pii": 0,
        }

        self._spacy_loaded = False
        self._presidio_loaded = False

    def detect(self, text: str, language: str = "en") -> Tuple[List[Dict], bool]:
        self.stats["total_documents"] += 1
        detections = []

        if self.use_regex:
            detections.extend(self._regex_detect(text))

        if self.use_presidio and self._ensure_presidio():
            detections.extend(self._presidio_detect(text, language))
        elif self.use_spacy and self._ensure_spacy():
            detections.extend(self._spacy_detect(text, language))

        has_pii = len(detections) > 0
        if has_pii:
            self.stats["documents_with_pii"] += 1
            for d in detections:
                key = d["type"]
                self.stats["pii_detections"][key] = self.stats["pii_detections"].get(key, 0) + 1

        return detections, has_pii

    def redact(self, text: str, language: str = "en") -> Tuple[str, List[Dict]]:
        detections, has_pii = self.detect(text, language)
        if not has_pii:
            return text, []

        result = text
        for d in sorted(detections, key=lambda x: -len(x["match"])):
            ptype = d["type"]
            placeholder = PLACEHOLDER_MAP.get(ptype, f"[{ptype}]")
            if d["match"] in result:
                result = result.replace(d["match"], placeholder, 1)

        self.stats["total_redactions"] += len(detections)
        return result, detections

    def redact_documents(self, documents: List[Dict], text_field: str = "text",
                         language_field: str = "language") -> List[Dict]:
        results = []
        for doc in documents:
            text = doc.get(text_field, "")
            lang = doc.get(language_field, "en")
            redacted, detections = self.redact(text, lang)
            doc_out = dict(doc)
            doc_out[text_field] = redacted
            doc_out["pii_detections"] = detections
            doc_out["pii_count"] = len(detections)
            results.append(doc_out)
        return results

    def _regex_detect(self, text: str) -> List[Dict]:
        detections = []
        for name, pattern in PII_PATTERNS:
            for match in pattern.finditer(text):
                matched_text = match.group()
                if name == "IPV4":
                    try:
                        parts = [int(p) for p in matched_text.split(".")]
                        if any(p > 255 for p in parts):
                            continue
                    except Exception:
                        continue
                if name == "CREDIT_CARD":
                    digits = re.sub(r"\D", "", matched_text)
                    if len(digits) < 13 or len(digits) > 19:
                        continue
                detections.append({
                    "type": name, "match": matched_text, "method": "regex",
                    "start": match.start(), "end": match.end(),
                })
        return detections

    def _spacy_detect(self, text: str, language: str = "en") -> List[Dict]:
        detections = []
        model_name = SPACY_MODELS.get(language, "en_core_web_sm")
        try:
            import spacy
            nlp = spacy.load(model_name, disable=["parser", "lemmatizer", "textcat"])
            doc = nlp(text[:100000])
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "GPE", "LOC", "ORG", "FAC"}:
                    detections.append({
                        "type": f"NER_{ent.label_}", "match": ent.text,
                        "method": "spacy", "start": ent.start_char, "end": ent.end_char,
                    })
        except Exception:
            pass
        return detections

    def _presidio_detect(self, text: str, language: str = "en") -> List[Dict]:
        detections = []
        try:
            from presidio_analyzer import AnalyzerEngine
            analyzer = AnalyzerEngine()
            results = analyzer.analyze(text=text, language=language, entities=[])
            for r in results:
                detections.append({
                    "type": f"PRESIDIO_{r.entity_type}", "match": text[r.start:r.end],
                    "method": "presidio", "start": r.start, "end": r.end,
                    "score": r.score,
                })
        except Exception:
            pass
        return detections

    def _ensure_spacy(self) -> bool:
        if not self._spacy_loaded:
            try:
                import spacy
                self._spacy_loaded = True
            except ImportError:
                return False
        return True

    def _ensure_presidio(self) -> bool:
        if not self._presidio_loaded:
            try:
                from presidio_analyzer import AnalyzerEngine
                _ = AnalyzerEngine()
                self._presidio_loaded = True
            except Exception:
                return False
        return True

    def get_stats(self) -> Dict:
        return {k: dict(v) if isinstance(v, dict) else v for k, v in self.stats.items()}

    def print_summary(self):
        s = self.stats
        print(f"\n{'='*60}\n PII DETECTION SUMMARY\n{'='*60}")
        print(f"  Documents scanned    : {s['total_documents']:,}")
        print(f"  Documents with PII   : {s['documents_with_pii']:,}")
        print(f"  Total redactions    : {s['total_redactions']:,}")
        for ptype, count in sorted(s.get("pii_detections", {}).items(), key=lambda x: -x[1]):
            print(f"    {ptype:<25s}: {count:>8,}")
        print("=" * 60)
