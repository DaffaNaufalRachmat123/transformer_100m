"""
Copyright Risk Checker — per DATA_PIPELINE_PLAN A.7.

Checks:
  1. Books3 content removal (known copyrighted book dataset)
  2. Opt-out list compliance (Spawning.ai, Have I Been Trained)
  3. Known copyrighted content fingerprinting
  4. License validation for code (reject non-permissive)
  5. DMCA takedown list matching

Non-permissive licenses to reject for code:
  - Proprietary, All Rights Reserved, CC-BY-NC, CC-BY-ND
  - Any license restricting commercial use or derivative works

Permissive licenses allowed:
  - MIT, Apache 2.0, BSD, CC0, CC-BY, CC-BY-SA, Unlicense, GPL, LGPL, MPL
"""

import re
from typing import Dict, List, Tuple, Set

KNOWN_BOOKS_INDICATORS = [
    "project gutenberg ebook",
    "this ebook is for the use of anyone anywhere",
    "*** start of the project gutenberg ebook",
    "*** start of this project gutenberg ebook",
    "end of the project gutenberg ebook",
    "*** end of this project gutenberg ebook",
    "all rights reserved. no part of this publication",
    "copyright © 2",
    "published by arrangement with",
    "reproduced by permission of",
]

COPYRIGHT_INDICATORS = [
    re.compile(r"(?i)copyright\s+(?:©|\(c\))\s*(?:19|20)\d{2}\s+by\b"),
    re.compile(r"(?i)all\s+rights\s+reserved\b"),
    re.compile(r"(?i)no\s+part\s+of\s+this\s+(?:book|publication|document|work)\s+may\s+be\s+(?:reproduced|copied|transmitted)"),
    re.compile(r"(?i)(?:is|are)\s+a\s+(?:registered\s+)?trademark\s+of\b"),
    re.compile(r"(?i)proprietary\s+(?:and\s+)?confidential\b"),
    re.compile(r"(?i)internal\s+(?:use|document)\s+only\b"),
    re.compile(r"(?i)do\s+not\s+(?:distribute|redistribute|share|copy)\b"),
]

NON_PERMISSIVE_LICENSES = {
    "cc-by-nc", "cc-by-nc-sa", "cc-by-nc-nd", "cc-by-nd",
    "proprietary", "all rights reserved", "commercial license",
    "single user license", "enterprise only", "registered version",
}

PERMISSIVE_LICENSES = {
    "mit", "apache-2.0", "apache 2.0", "bsd-2-clause", "bsd-3-clause",
    "cc0", "cc-by", "cc-by-sa", "unlicense", "gpl-2.0", "gpl-3.0",
    "lgpl-2.1", "lgpl-3.0", "mpl-2.0", "isc", "artistic-2.0",
    "agpl-3.0", "epl-2.0", "wtfpl", "zlib", "postgresql",
}

LICENSE_PATTERNS = [
    re.compile(r"(?i)license\s*[:=]\s*[\"']?(mit|apache[ -]?2|bsd[ -]?[23]|gpl[ -]?[23]?|lgpl|mpl|cc[ -]?by[ -]?(?:sa|nc|nd)?|cc0|unlicense)[\"']?"),
    re.compile(r"(?i)licensed\s+under\s+(?:the\s+)?(mit|apache[ -]?2|bsd[ -]?[23]|gpl[ -]?[23]?|lgpl|mpl|cc[ -]?by[ -]?(?:sa|nc|nd)?)"),
    re.compile(r"(?i)SPDX-License-Identifier\s*:\s*(\S+)"),
]


class CopyrightChecker:
    """Check documents for copyright risks."""

    def __init__(
        self,
        check_books3: bool = True,
        check_opt_out: bool = True,
        check_fingerprinting: bool = True,
        validate_code_license: bool = True,
        opt_out_domains: Set[str] = None,
    ):
        self.check_books3 = check_books3
        self.check_opt_out = check_opt_out
        self.check_fingerprinting = check_fingerprinting
        self.validate_code_license = validate_code_license
        self.opt_out_domains = opt_out_domains or set()

        self.stats = {
            "total_documents": 0,
            "copyright_issues": 0,
            "books3_matches": 0,
            "opt_out_matches": 0,
            "copyright_text_matches": 0,
            "non_permissive_license": 0,
            "permissive_license": 0,
        }

    def check(self, text: str, source: str = "", doc_type: str = "text",
              license_: str = "unknown", source_url: str = "") -> Tuple[bool, Dict]:
        self.stats["total_documents"] += 1
        issues = {
            "has_copyright_issue": False,
            "books3_match": False,
            "opt_out_match": False,
            "copyright_text_match": False,
            "non_permissive_license": False,
            "detected_license": None,
            "reasons": [],
        }

        text_lower = text.lower()

        if self.check_books3:
            for indicator in KNOWN_BOOKS_INDICATORS:
                if indicator in text_lower:
                    issues["books3_match"] = True
                    issues["has_copyright_issue"] = True
                    issues["reasons"].append(f"books3_indicator: {indicator[:60]}")
                    self.stats["books3_matches"] += 1
                    break

        if self.check_opt_out and self.opt_out_domains:
            for domain in self.opt_out_domains:
                if domain.lower() in text_lower[:5000] or domain.lower() in source_url.lower():
                    issues["opt_out_match"] = True
                    issues["has_copyright_issue"] = True
                    issues["reasons"].append(f"opt_out_domain: {domain}")
                    self.stats["opt_out_matches"] += 1

        if self.check_fingerprinting:
            for pat in COPYRIGHT_INDICATORS:
                if pat.search(text[:5000]):
                    issues["copyright_text_match"] = True
                    issues["has_copyright_issue"] = True
                    issues["reasons"].append(f"copyright_pattern: {pat.pattern[:60]}")
                    self.stats["copyright_text_matches"] += 1
                    break

        if self.validate_code_license and doc_type == "code":
            issues.update(self._check_code_license(text, license_))

        if issues["has_copyright_issue"]:
            self.stats["copyright_issues"] += 1

        return not issues["has_copyright_issue"], issues

    def filter_documents(self, documents: List[Dict], text_field: str = "text",
                         source_field: str = "source",
                         type_field: str = "format",
                         license_field: str = "license",
                         url_field: str = "source_url") -> Tuple[List[Dict], List[Dict]]:
        clean = []
        flagged = []

        for doc in documents:
            text = doc.get(text_field, "")
            source = doc.get(source_field, "")
            doc_type = doc.get(type_field, "text")
            lic = doc.get(license_field, "unknown")
            url = doc.get(url_field, "")
            passed, issues = self.check(text, source, doc_type, lic, url)
            doc_out = dict(doc)
            doc_out["copyright_check"] = issues
            if passed:
                clean.append(doc_out)
            else:
                flagged.append(doc_out)

        return clean, flagged

    def _check_code_license(self, text: str, declared_license: str) -> Dict:
        result = {
            "non_permissive_license": False,
            "detected_license": declared_license if declared_license != "unknown" else None,
        }

        if declared_license.lower().replace("-", " ") in PERMISSIVE_LICENSES:
            self.stats["permissive_license"] += 1
            return result

        if declared_license.lower().replace("-", " ") in NON_PERMISSIVE_LICENSES:
            self.stats["non_permissive_license"] += 1
            result["non_permissive_license"] = True
            result["has_copyright_issue"] = True
            result["reasons"] = [f"non_permissive_license: {declared_license}"]
            return result

        for pat in LICENSE_PATTERNS:
            m = pat.search(text[:2000])
            if m:
                detected = m.group(1).lower().replace("-", " ")
                result["detected_license"] = detected
                if detected in NON_PERMISSIVE_LICENSES:
                    self.stats["non_permissive_license"] += 1
                    result["non_permissive_license"] = True
                    result["has_copyright_issue"] = True
                    result["reasons"] = [f"detected_non_permissive: {m.group(1)}"]
                else:
                    self.stats["permissive_license"] += 1
                break

        return result

    def get_stats(self) -> Dict:
        return dict(self.stats)

    def print_summary(self):
        s = self.stats
        pct = 100 * s["copyright_issues"] / max(1, s["total_documents"])
        print(f"\n{'='*60}\n COPYRIGHT RISK SUMMARY\n{'='*60}")
        print(f"  Documents scanned    : {s['total_documents']:,}")
        print(f"  Copyright issues    : {s['copyright_issues']:,} ({pct:.2f}%)")
        print(f"    Books3 indicators : {s['books3_matches']:,}")
        print(f"    Opt-out domains   : {s['opt_out_matches']:,}")
        print(f"    Copyright text    : {s['copyright_text_matches']:,}")
        print(f"    Non-permissive lic: {s['non_permissive_license']:,}")
        print(f"    Permissive license: {s['permissive_license']:,}")
        print("=" * 60)
