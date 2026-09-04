"""
Web Artifact Removal — per DATA_PIPELINE_PLAN A.5.

Removes residual web artifacts from extracted text:
  - HTML tags, CSS fragments, JavaScript snippets
  - Navigation menus, breadcrumbs
  - Footer/header boilerplate (copyright, contact info, site name)
  - Cookie consent banners
  - Advertisement text blocks
  - Social media widgets
  - Tracking/analytics code fragments
  - Comment sections (optional)
"""

import re
from typing import Dict, List


# Common footer patterns
FOOTER_PATTERNS = [
    re.compile(r"(?i)all rights reserved\.?\s*\d{4}(?:-\d{4})?"),
    re.compile(r"(?i)copyright\s+(?:©|\(c\))\s*\d{4}(?:-\d{4})?\s+[\w\s]+"),
    re.compile(r"(?i)terms\s+(?:of\s+)?(?:service|use)\s*\|\s*privacy\s+policy"),
    re.compile(r"(?i)this\s+(?:site|page)\s+(?:is|was)\s+(?:last\s+)?(?:updated|modified)"),
    re.compile(r"(?i)contact\s+us\s*(?:\||·)\s*(?:about|faq|help|support)"),
    re.compile(r"(?i)©\s*\d{4}(?:-\d{4})?\s+[\w\s\.]+"),
    re.compile(r"(?i)subscribe\s+(?:to\s+)?(?:our\s+)?newsletter"),
    re.compile(r"(?i)(?:sign|log)\s+(?:in|up|out)\s*(?:\||·|•)"),
]

# Cookie consent patterns
COOKIE_PATTERNS = [
    re.compile(r"(?i)this\s+(?:website|site)\s+uses\s+cookies"),
    re.compile(r"(?i)we\s+use\s+cookies\s+(?:to|for)"),
    re.compile(r"(?i)by\s+(?:continuing|using)\s+(?:to\s+use\s+)?(?:this|our)\s+(?:site|website).*cookies"),
    re.compile(r"(?i)(?:accept|reject|manage)\s+(?:all\s+)?(?:cookies|cookie\s+settings)"),
    re.compile(r"(?i)cookie\s+(?:policy|notice|consent|settings|preferences)"),
    re.compile(r"(?i)gdpr\s+(?:consent|cookie|compliance)"),
]

# Social media / sharing patterns
SOCIAL_PATTERNS = [
    re.compile(r"(?i)share\s+(?:on|this\s+(?:on\s+)?)\s*(?:twitter|facebook|linkedin|instagram|pinterest|reddit|whatsapp|telegram)"),
    re.compile(r"(?i)follow\s+(?:us\s+)?(?:on\s+)\s*(?:twitter|facebook|instagram|linkedin|youtube|tiktok)"),
    re.compile(r"(?i)(?:facebook|twitter|instagram|linkedin)\s+(?:share|like|follow|tweet)"),
    re.compile(r"(?i)share\s+this\s+(?:article|post|story|page)"),
    re.compile(r"(?i)add\s+(?:a\s+)?comment"),
    re.compile(r"(?i)join\s+(?:the\s+)?conversation"),
]

# Advertisement patterns
AD_PATTERNS = [
    re.compile(r"(?i)(?:sponsored|promoted)\s+(?:by|content|post|story|article)"),
    re.compile(r"(?i)(?:advertisement|advertorial)\s*(?:-|:|\n)"),
    re.compile(r"(?i)click\s+(?:here|below)\s+(?:to|for)\s+(?:buy|purchase|shop|order|download|subscribe|sign\s+up)"),
    re.compile(r"(?i)(?:limited|special|exclusive)\s+offer"),
    re.compile(r"(?i)(?:\d{1,3}%\s+(?:off|discount))"),
    re.compile(r"(?i)(?:buy|shop|purchase)\s+(?:now|today|online)"),
]

# Navigation patterns
NAV_PATTERNS = [
    re.compile(r"(?i)^(?:home|about|contact|blog|news|shop|store|login|sign\s+up|register)(?:\s*\|\s*(?:home|about|contact|blog|news|shop|store|login|sign\s+up|register))+$"),
    re.compile(r"(?i)^(?:menu|navigation|site\s+map|table\s+of\s+contents)\s*$"),
    re.compile(r"(?i)^(?:previous|next)\s*(?:page|post|article|chapter|section)"),
    re.compile(r"(?i)^(?:back\s+to\s+top|scroll\s+to\s+top|go\s+to\s+top)"),
]


class WebArtifactRemover:
    """Remove residual web artifacts from extracted text."""

    def __init__(
        self,
        remove_html_tags: bool = True,
        remove_css: bool = True,
        remove_javascript: bool = True,
        remove_navigation: bool = True,
        remove_footer: bool = True,
        remove_header: bool = True,
        remove_cookie_banners: bool = True,
        remove_ads: bool = True,
        remove_social_widgets: bool = True,
        remove_tracking: bool = True,
        remove_comments: bool = False,
    ):
        self.remove_html_tags = remove_html_tags
        self.remove_css = remove_css
        self.remove_js = remove_javascript
        self.remove_nav = remove_navigation
        self.remove_footer = remove_footer
        self.remove_header = remove_header
        self.remove_cookie = remove_cookie_banners
        self.remove_ads = remove_ads
        self.remove_social = remove_social_widgets
        self.remove_tracking = remove_tracking
        self.remove_comments = remove_comments

        self.stats = {
            "total_documents": 0,
            "lines_removed_total": 0,
            "lines_removed_by_type": {},
        }

    def clean(self, text: str) -> str:
        """Clean a single text of web artifacts."""
        if not text or not text.strip():
            return ""

        self.stats["total_documents"] += 1
        original_lines = text.count("\n") + 1

        # Split into lines for line-by-line filtering
        lines = text.split("\n")
        cleaned_lines = []
        removed_counts = {}

        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append(line)
                continue

            removed = False
            reason = None

            # Check each pattern category
            if self.remove_html_tags and self._has_html_tags(stripped):
                reason = "html_tags"
            elif self.remove_js and self._is_javascript(stripped):
                reason = "javascript"
            elif self.remove_css and self._is_css(stripped):
                reason = "css"
            elif self.remove_cookie and self._is_cookie(stripped):
                reason = "cookie"
            elif self.remove_ads and self._is_ad(stripped):
                reason = "ad"
            elif self.remove_social and self._is_social(stripped):
                reason = "social"
            elif self.remove_footer and self._is_footer(stripped):
                reason = "footer"
            elif self.remove_nav and self._is_navigation(stripped):
                reason = "navigation"
            elif self.remove_tracking and self._is_tracking(stripped):
                reason = "tracking"

            if reason:
                removed_counts[reason] = removed_counts.get(reason, 0) + 1
            else:
                cleaned_lines.append(line)

        # Update stats
        lines_removed = original_lines - len(cleaned_lines)
        self.stats["lines_removed_total"] += lines_removed
        for k, v in removed_counts.items():
            self.stats["lines_removed_by_type"][k] = self.stats["lines_removed_by_type"].get(k, 0) + v

        result = "\n".join(cleaned_lines)
        return re.sub(r"\n{3,}", "\n\n", result).strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts."""
        return [self.clean(t) for t in texts]

    def clean_documents(
        self, documents: List[Dict], text_field: str = "text"
    ) -> List[Dict]:
        """Clean web artifacts from a batch of documents."""
        results = []
        for doc in documents:
            text = doc.get(text_field, doc.get("normalized_text", ""))
            cleaned = self.clean(text)
            doc_out = dict(doc)
            doc_out["text"] = cleaned
            doc_out["cleaned_text"] = cleaned
            results.append(doc_out)
        return results

    def _has_html_tags(self, text: str) -> bool:
        """Check if text contains residual HTML tags."""
        return bool(re.search(r"<\s*/?\s*[a-zA-Z][a-zA-Z0-9]*\s*[^>]*>", text))

    def _is_javascript(self, text: str) -> bool:
        """Check if text looks like JavaScript code."""
        js_patterns = [
            r"\bfunction\s*\(.*\)\s*\{",
            r"\bvar\s+\w+\s*=",
            r"\bconst\s+\w+\s*=",
            r"\blet\s+\w+\s*=",
            r"\.addEventListener\(",
            r"document\.\w+",
            r"window\.\w+",
            r"\bconsole\.(log|error|warn)\(",
            r"\$\s*\(",
        ]
        return any(re.search(p, text) for p in js_patterns)

    def _is_css(self, text: str) -> bool:
        """Check if text looks like CSS."""
        css_patterns = [
            r"^\s*[.#]?\w[\w-]*\s*\{[^}]*\}",
            r"^\s*(margin|padding|border|color|font|background|display|position|width|height)\s*:",
            r"@media\s",
            r"@import\s",
        ]
        return any(re.search(p, text) for p in css_patterns)

    def _is_cookie(self, text: str) -> bool:
        return any(p.search(text) for p in COOKIE_PATTERNS)

    def _is_ad(self, text: str) -> bool:
        return any(p.search(text) for p in AD_PATTERNS)

    def _is_social(self, text: str) -> bool:
        return any(p.search(text) for p in SOCIAL_PATTERNS)

    def _is_footer(self, text: str) -> bool:
        return any(p.search(text) for p in FOOTER_PATTERNS)

    def _is_navigation(self, text: str) -> bool:
        """Check if line looks like a navigation element."""
        # Short line with many pipe/separator characters
        if len(text) < 200 and text.count("|") >= 2:
            return True
        return any(p.match(text) for p in NAV_PATTERNS)

    def _is_tracking(self, text: str) -> bool:
        """Check for tracking/analytics code fragments."""
        tracking_patterns = [
            r"gtag\s*\(|ga\s*\(",
            r"analytics\.(track|page|identify|group)",
            r"fbq\s*\(",
            r"_gaq\.push",
            r"googletag\.(cmd|pubads|defineSlot|enableServices)",
        ]
        return any(re.search(p, text) for p in tracking_patterns)

    def get_stats(self) -> Dict:
        """Return cleaning statistics."""
        return dict(self.stats)

    def print_summary(self):
        """Print artifact removal summary."""
        s = self.stats
        print("\n" + "=" * 60)
        print(" WEB ARTIFACT REMOVAL SUMMARY ")
        print("=" * 60)
        print(f"  Documents processed   : {s['total_documents']:,}")
        print(f"  Total lines removed   : {s['lines_removed_total']:,}")
        print("-" * 60)
        for remove_type, count in sorted(
            s.get("lines_removed_by_type", {}).items(), key=lambda x: -x[1]
        ):
            print(f"  {remove_type:<20s} : {count:>10,}")
        print("=" * 60)
