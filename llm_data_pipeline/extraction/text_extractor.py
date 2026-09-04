"""
Text Extractor — per DATA_PIPELINE_PLAN A.3.

Extraction chain (in accuracy order, per DCLM findings):
  1. Trafilatura (primary — best boilerplate removal)
  2. readability-lxml (fallback #1)
  3. jusText (fallback #2)
  4. BeautifulSoup (fallback #3, legacy)

Format-specific parsers:
  - PDF: Apache Tika fallback → pdfminer (layout-aware)
  - Markdown: strip metadata, preserve structure
  - LaTeX: preserve math ($...$, \begin{equation}...)
  - Code: preserve with language tags
  - JSON/JSONL: extract text fields

Output: clean plaintext with preserved structure (tables, code blocks, math).
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple


FALLBACK_CHAIN = ["trafilatura", "readability_lxml", "jusText", "beautifulsoup"]


class TextExtractor:
    """Multi-format text extraction with fallback chain."""

    def __init__(
        self,
        primary_extractor: str = "trafilatura",
        fallback_chain: Optional[List[str]] = None,
    ):
        self.primary = primary_extractor
        self.fallback_chain = fallback_chain or FALLBACK_CHAIN
        self.stats = {
            "total_documents": 0,
            "extracted": 0,
            "extraction_failed": 0,
            "extraction_by_method": {},
            "format_counts": {},
        }

    def extract(self, text: str, format_hint: str = "html",
                url: str = "") -> Tuple[Optional[str], str, Dict]:
        """Extract clean text from raw content. Returns (extracted_text, method_used, metadata).

        format_hint: html, markdown, pdf, latex, code, json, or text
        """
        self.stats["total_documents"] += 1
        self.stats["format_counts"][format_hint] = self.stats["format_counts"].get(format_hint, 0) + 1

        if not text or not text.strip():
            self.stats["extraction_failed"] += 1
            return None, "empty", {}

        # Route to format-specific parser
        if format_hint == "html":
            result, method, meta = self._extract_html(text, url)
        elif format_hint == "markdown":
            result, method, meta = self._extract_markdown(text)
        elif format_hint == "pdf":
            result, method, meta = self._extract_pdf(text)
        elif format_hint == "latex":
            result, method, meta = self._extract_latex(text)
        elif format_hint == "code":
            result, method, meta = self._extract_code(text)
        elif format_hint == "json":
            result, method, meta = self._extract_json(text)
        else:  # plain text
            result, method, meta = text, "plain_text", {}

        if result and result.strip():
            self.stats["extracted"] += 1
            self.stats["extraction_by_method"][method] = self.stats["extraction_by_method"].get(method, 0) + 1
        else:
            self.stats["extraction_failed"] += 1
            result = None

        return result, method, meta

    def extract_batch(self, documents: List[Dict],
                      text_field: str = "text",
                      format_field: str = "format",
                      url_field: str = "source_url") -> List[Dict]:
        """Extract text from a batch of documents."""
        results = []
        for doc in documents:
            raw = doc.get(text_field, "")
            fmt = doc.get(format_field, "html")
            url = doc.get(url_field, "")
            clean, method, meta = self.extract(raw, fmt, url)
            doc_out = dict(doc)
            doc_out["extracted_text"] = clean or ""
            doc_out["extraction_method"] = method
            doc_out["extraction_meta"] = meta
            if clean is None:
                doc_out["extraction_status"] = "failed"
            else:
                doc_out["extraction_status"] = "ok"
            results.append(doc_out)
        return results

    def _extract_html(self, text: str, url: str = "") -> Tuple[Optional[str], str, Dict]:
        """Extract main content from HTML using fallback chain."""
        meta = {}

        # Determine if this is actually HTML
        has_html_tags = bool(re.search(r"<\s*(html|body|div|p|article|section)", text, re.IGNORECASE))
        if not has_html_tags:
            return text, "plain_text", meta

        # 1. Trafilatura (primary)
        try:
            import trafilatura
            result = trafilatura.extract(
                text,
                include_comments=False,
                include_tables=True,
                include_images=False,
                include_links=False,
                output_format="markdown",
                url=url,
            )
            if result and len(result.strip()) > 50:
                return result.strip(), "trafilatura", meta
        except ImportError:
            pass
        except Exception as e:
            meta["trafilatura_error"] = str(e)

        # 2. readability-lxml
        try:
            from readability import Document
            doc = Document(text)
            summary_html = doc.summary()
            title = doc.title()
            result = self._html_to_text(summary_html)
            if result and len(result.strip()) > 50:
                return result.strip(), "readability_lxml", meta
        except ImportError:
            pass
        except Exception as e:
            meta["readability_error"] = str(e)

        # 3. jusText
        try:
            import justext
            paragraphs = justext.justext(text, justext.get_stoplist("English"))
            result = "\n\n".join(
                p.text for p in paragraphs
                if not p.is_boilerplate and len(p.text.strip()) > 10
            )
            if result and len(result.strip()) > 50:
                return result.strip(), "jusText", meta
        except ImportError:
            pass
        except Exception as e:
            meta["justext_error"] = str(e)

        # 4. BeautifulSoup (last resort)
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, "lxml")
            # Remove script, style, nav, footer, header
            for tag in soup(["script", "style", "nav", "footer", "header",
                            "noscript", "iframe", "form"]):
                tag.decompose()
            result = soup.get_text(separator="\n", strip=True)
            if result and len(result.strip()) > 50:
                return result.strip(), "beautifulsoup", meta
        except ImportError:
            pass
        except Exception as e:
            meta["bs4_error"] = str(e)

        # All fallbacks failed — return original with HTML stripped
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) > 50:
            return clean, "regex_strip", meta

        return None, "all_failed", meta

    def _extract_markdown(self, text: str) -> Tuple[Optional[str], str, Dict]:
        """Clean and preserve Markdown structure."""
        # Remove YAML front matter
        text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
        # Remove HTML comments
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        text = text.strip()
        return (text if len(text) > 20 else None), "markdown_preserved", {}

    def _extract_pdf(self, text: str) -> Tuple[Optional[str], str, Dict]:
        """Extract text from PDF content (assumes pre-extracted with pdfminer/Tika)."""
        # PDF text is already extracted — just clean layout artifacts
        lines = text.split("\n")
        cleaned = []
        prev_empty = False
        for line in lines:
            stripped = line.strip()
            # Remove page numbers, headers, footers
            if re.match(r"^\d{1,4}$", stripped):
                continue
            if not stripped:
                if not prev_empty:
                    cleaned.append("")
                prev_empty = True
            else:
                prev_empty = False
                cleaned.append(stripped)
        result = "\n".join(cleaned)
        return (result if len(result) > 50 else None), "pdf_extracted", {}

    def _extract_latex(self, text: str) -> Tuple[Optional[str], str, Dict]:
        """Preserve LaTeX math while extracting readable text."""
        # Keep math environments intact
        math_envs = []
        def _save_math(match):
            math_envs.append(match.group(0))
            return f"__MATH_{len(math_envs) - 1}__"

        # Save display math
        text = re.sub(r"\$\$.*?\$\$", _save_math, text, flags=re.DOTALL)
        text = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", _save_math, text, flags=re.DOTALL)
        text = re.sub(r"\\begin\{align\}.*?\\end\{align\}", _save_math, text, flags=re.DOTALL)
        # Save inline math
        text = re.sub(r"\$[^$]+\$", _save_math, text)

        # Remove LaTeX commands but keep content
        text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
        text = re.sub(r"\\[a-zA-Z]+", "", text)

        # Restore math
        for i, env in enumerate(math_envs):
            text = text.replace(f"__MATH_{i}__", f"\n{env}\n")

        text = re.sub(r"\n{3,}", "\n\n", text)
        return (text.strip() if len(text.strip()) > 20 else None), "latex_preserved", {}

    def _extract_code(self, text: str) -> Tuple[Optional[str], str, Dict]:
        """Preserve code with language detection."""
        if len(text.strip()) < 10:
            return None, "code_empty", {}
        return text, "code_preserved", {"format": "code"}

    def _extract_json(self, text: str) -> Tuple[Optional[str], str, Dict]:
        """Extract text fields from JSON content."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Not valid JSON — treat as plain text
            return text, "json_fallback_text", {}

        # Recursively extract all text values
        extracted = self._json_text_values(data)
        return (extracted if len(extracted) > 20 else None), "json_extracted", {}

    def _json_text_values(self, obj, depth: int = 0) -> str:
        """Recursively extract string values from a JSON object."""
        if depth > 10:
            return ""
        texts = []
        if isinstance(obj, str) and len(obj) > 3:
            texts.append(obj)
        elif isinstance(obj, dict):
            for key in ["text", "content", "body", "description", "summary",
                       "title", "abstract", "article", "message"]:
                if key in obj and isinstance(obj[key], str):
                    texts.append(obj[key])
            for v in obj.values():
                texts.append(self._json_text_values(v, depth + 1))
        elif isinstance(obj, list):
            for item in obj:
                texts.append(self._json_text_values(item, depth + 1))
        return "\n".join(t for t in texts if t)

    def _html_to_text(self, html: str) -> str:
        """Convert HTML fragment to text."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            return soup.get_text(separator="\n", strip=True)
        except Exception:
            clean = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", clean).strip()

    def get_stats(self) -> Dict:
        """Return extraction statistics."""
        return dict(self.stats)

    def print_summary(self):
        """Print extraction summary."""
        s = self.stats
        print("\n" + "=" * 60)
        print(" TEXT EXTRACTION SUMMARY ")
        print("=" * 60)
        print(f"  Total documents      : {s['total_documents']:,}")
        print(f"  Successfully extracted: {s['extracted']:,} ({100*s['extracted']/max(1,s['total_documents']):.1f}%)")
        print(f"  Extraction failed    : {s['extraction_failed']:,}")
        print("-" * 60)
        print("  By method:")
        for method, count in sorted(s.get("extraction_by_method", {}).items(), key=lambda x: -x[1]):
            print(f"    {method:<25s} {count:>10,}")
        print("=" * 60)
