"""
Data Acquisition — download raw data from HuggingFace and other sources.

Each source connector yields documents in a standardized format:
  {doc_id, source, source_url, crawl_date, language, license, text}

Sources are defined in configs/pretrain.yaml under acquisition.sources.
"""

import json
import os
import sys
import time
from typing import Dict, Iterator, List, Optional

from ..common import ensure_dir, generate_doc_id, write_jsonl


TARGET_LANGUAGES = {"en", "id", "ja", "ko"}

SOURCE_REGISTRY = {}


def register_source(name: str):
    """Decorator to register acquisition source connectors."""
    def decorator(cls):
        SOURCE_REGISTRY[name] = cls
        return cls
    return decorator


class DataAcquisition:
    """Orchestrate data acquisition from all configured sources."""

    def __init__(self, config: Dict, output_dir: str = "data/raw/acquisition"):
        self.config = config
        self.output_dir = output_dir
        self.source_configs = config.get("sources", {})
        self.provenance_enabled = config.get("provenance", {}).get("enabled", True)
        ensure_dir(output_dir)

    def download_all(self):
        """Download from all enabled sources."""
        sources = self.source_configs
        for name, cfg in sources.items():
            if not cfg.get("enabled", False):
                print(f"[acquisition] Skipping disabled source: {name}")
                continue
            try:
                self._download_source(name, cfg)
            except Exception as e:
                print(f"[acquisition] Error downloading {name}: {e}")

    def _download_source(self, name: str, cfg: Dict):
        """Download from a single source."""
        print(f"[acquisition] Downloading {name}...")
        t0 = time.time()

        connector = self._get_connector(name)
        records = connector.download(cfg, self.output_dir)

        out_path = os.path.join(self.output_dir, f"{name}.jsonl")
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for doc in records:
                if not doc.get("text", "").strip():
                    continue
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                count += 1

        elapsed = time.time() - t0
        size_mb = os.path.getsize(out_path) / (1024 * 1024) if os.path.exists(out_path) else 0
        print(f"[acquisition] {name}: {count:,} docs, {size_mb:.1f} MB in {elapsed:.1f}s")

    def _get_connector(self, name: str):
        """Get or create the appropriate source connector."""
        if name == "fineweb_edu":
            return FineWebEduConnector()
        elif name == "dclm_baseline":
            return DCLMConnector()
        elif name == "wikipedia":
            return WikipediaConnector()
        elif name == "scientific":
            return ScientificConnector()
        elif name == "news":
            return NewsConnector()
        elif name == "the_stack_v2":
            return TheStackV2Connector()
        elif name == "math":
            return MathConnector()
        elif name == "cc100":
            return CC100Connector()
        elif name == "culturax":
            return CulturaXConnector()
        elif name == "mc4":
            return MC4Connector()
        elif name == "medical":
            return MedicalConnector()
        else:
            raise ValueError(f"Unknown source: {name}")


class BaseConnector:
    """Base class for data source connectors."""

    source_name = "base"

    def download(self, cfg: Dict, output_dir: str) -> List[Dict]:
        raise NotImplementedError

    def _make_doc(self, text: str, source: str, language: str = "en",
                  source_url: str = "", license_: str = "unknown") -> Dict:
        return {
            "doc_id": generate_doc_id(),
            "source": source,
            "source_url": source_url,
            "crawl_date": "",
            "language": language,
            "license": license_,
            "text": text.strip(),
        }

    def _hf_stream(self, dataset: str, subset: Optional[str] = None,
                   split: str = "train", **kwargs):
        """Stream from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

        load_kwargs = {"path": dataset, "split": split, "streaming": True}
        if subset:
            load_kwargs["name"] = subset
        load_kwargs.update(kwargs)
        return load_dataset(**load_kwargs)


class FineWebEduConnector(BaseConnector):
    source_name = "fineweb_edu"

    def download(self, cfg: Dict, output_dir: str) -> List[Dict]:
        dataset = cfg.get("dataset", "HuggingFaceFW/fineweb-edu")
        subset = cfg.get("subset", "sample-10BT")
        target_tokens = cfg.get("target_tokens", 1_500_000_000)

        docs = []
        chars = 0
        target_chars = target_tokens * 4  # ~4 chars per token for English

        try:
            ds = self._hf_stream(dataset, subset)
            for row in ds:
                text = row.get("text", "").strip()
                if len(text) < 100:
                    continue
                docs.append(self._make_doc(text, "fineweb_edu", "en",
                                           source_url=row.get("url", "")))
                chars += len(text)
                if chars >= target_chars:
                    break
        except Exception as e:
            print(f"[fineweb_edu] Stream error: {e}")

        return docs


class DCLMConnector(BaseConnector):
    source_name = "dclm_baseline"

    def download(self, cfg: Dict, output_dir: str) -> List[Dict]:
        dataset = cfg.get("dataset", "mlfoundations/dclm-baseline-1.0")
        target_tokens = cfg.get("target_tokens", 1_000_000_000)

        docs = []
        chars = 0
        target_chars = target_tokens * 4

        try:
            ds = self._hf_stream(dataset)
            for row in ds:
                text = row.get("text", "").strip()
                if len(text) < 100:
                    continue
                docs.append(self._make_doc(text, "dclm_baseline", "en"))
                chars += len(text)
                if chars >= target_chars:
                    break
        except Exception as e:
            print(f"[dclm] Stream error: {e}")

        return docs


class WikipediaConnector(BaseConnector):
    source_name = "wikipedia"

    def download(self, cfg: Dict, output_dir: str) -> List[Dict]:
        languages = cfg.get("languages", ["en", "id", "ja", "ko"])
        date = cfg.get("date", "20240501")
        docs = []

        for lang in languages:
            try:
                ds = self._hf_stream("wikimedia/wikipedia", f"{date}.{lang}")
                for row in ds:
                    text = row.get("text", "").strip()
                    title = row.get("title", "")
                    if len(text) < 200:
                        continue
                    docs.append(self._make_doc(
                        text, "wikipedia", lang,
                        source_url=f"https://{lang}.wikipedia.org/wiki/{title}",
                        license_="CC-BY-SA-3.0",
                    ))
            except Exception as e:
                print(f"[wikipedia] {lang}: {e}")

        return docs


class ScientificConnector(BaseConnector):
    source_name = "scientific"

    def download(self, cfg: Dict, output_dir: str) -> List[Dict]:
        sources = cfg.get("sources", ["arxiv", "pubmed_oa"])
        target_tokens = cfg.get("target_tokens", 500_000_000)
        docs = []
        chars = 0
        target_chars = target_tokens * 4

        if "arxiv" in sources:
            try:
                ds = self._hf_stream("scientific_papers", "arxiv", "train")
                for row in ds:
                    text = row.get("article", "").strip()
                    if len(text) < 500:
                        continue
                    docs.append(self._make_doc(text, "arxiv", "en"))
                    chars += len(text)
                    if chars >= target_chars:
                        break
            except Exception as e:
                print(f"[arxiv] {e}")

        return docs


class NewsConnector(BaseConnector):
    source_name = "news"

    def download(self, cfg: Dict, output_dir: str) -> List[Dict]:
        target_tokens = cfg.get("target_tokens", 300_000_000)
        docs = []
        chars = 0
        target_chars = target_tokens * 4

        try:
            ds = self._hf_stream("cc_news")
            for row in ds:
                text = row.get("text", "").strip()
                if len(text) < 200:
                    continue
                docs.append(self._make_doc(text, "cc_news", "en",
                                           source_url=row.get("url", "")))
                chars += len(text)
                if chars >= target_chars:
                    break
        except Exception as e:
            print(f"[news] {e}")

        return docs


class TheStackV2Connector(BaseConnector):
    source_name = "the_stack_v2"

    def download(self, cfg: Dict, output_dir: str) -> List[Dict]:
        languages = cfg.get("languages", ["python", "javascript", "java", "typescript", "rust", "go"])
        target_tokens = cfg.get("target_tokens", 800_000_000)
        docs = []
        chars = 0
        target_chars = target_tokens * 4

        for lang in languages:
            if chars >= target_chars:
                break
            try:
                ds = self._hf_stream("bigcode/the-stack-v2-train-smol-ids",
                                     data_dir=f"data/{lang}")
                for row in ds:
                    content = row.get("content", "").strip()
                    if len(content) < 50 or len(content) > 10000:
                        continue
                    repo = row.get("max_stars_repo_name", "")
                    docs.append(self._make_doc(
                        content, f"the_stack_v2/{lang}", "code",
                        source_url=repo, license_="permissive_only",
                    ))
                    chars += len(content)
                    if chars >= target_chars:
                        break
            except Exception as e:
                print(f"[the_stack_v2] {lang}: {e}")

        return docs


class MathConnector(BaseConnector):
    source_name = "math"

    def download(self, cfg: Dict, output_dir: str) -> List[Dict]:
        target_tokens = cfg.get("target_tokens", 500_000_000)
        docs = []
        chars = 0
        target_chars = target_tokens * 4

        # MathPile
        try:
            ds = self._hf_stream("GAIR/MathPile")
            for row in ds:
                text = row.get("text", "").strip()
                if len(text) < 50:
                    continue
                docs.append(self._make_doc(text, "mathpile", "en"))
                chars += len(text)
                if chars >= target_chars:
                    break
        except Exception as e:
            print(f"[mathpile] {e}")

        # OpenWebMath
        if chars < target_chars:
            try:
                ds = self._hf_stream("open-web-math/open-web-math")
                for row in ds:
                    text = row.get("text", "").strip()
                    if len(text) < 100:
                        continue
                    docs.append(self._make_doc(text, "openwebmath", "en"))
                    chars += len(text)
                    if chars >= target_chars:
                        break
            except Exception as e:
                print(f"[openwebmath] {e}")

        return docs


class CC100Connector(BaseConnector):
    source_name = "cc100"

    def download(self, cfg: Dict, output_dir: str) -> List[Dict]:
        languages = cfg.get("languages", ["id", "ja", "ko"])
        token_target = cfg.get("target_tokens_per_lang", 300_000_000)
        docs = []

        for lang in languages:
            chars_per_token = 4 if lang == "id" else 2  # CJK = more bytes/token
            target_chars = token_target * chars_per_token
            chars = 0

            try:
                ds = self._hf_stream("cc100", lang)
                for row in ds:
                    text = row.get("text", "").strip()
                    if len(text) < 50:
                        continue
                    docs.append(self._make_doc(text, "cc100", lang))
                    chars += len(text)
                    if chars >= target_chars:
                        break
            except Exception as e:
                print(f"[cc100] {lang}: {e}")

        return docs


class CulturaXConnector(BaseConnector):
    source_name = "culturax"

    def download(self, cfg: Dict, output_dir: str) -> List[Dict]:
        languages = cfg.get("languages", ["id", "ja", "ko"])
        token_target = cfg.get("target_tokens_per_lang", 200_000_000)
        docs = []

        for lang in languages:
            chars_per_token = 4 if lang == "id" else 2
            target_chars = token_target * chars_per_token
            chars = 0

            try:
                ds = self._hf_stream("uonlp/CulturaX", lang)
                for row in ds:
                    text = row.get("text", "").strip()
                    if len(text) < 50:
                        continue
                    docs.append(self._make_doc(text, "culturax", lang))
                    chars += len(text)
                    if chars >= target_chars:
                        break
            except Exception as e:
                print(f"[culturax] {lang}: {e}")

        return docs


class MC4Connector(BaseConnector):
    source_name = "mc4"

    def download(self, cfg: Dict, output_dir: str) -> List[Dict]:
        languages = cfg.get("languages", ["id", "ja", "ko"])
        token_target = cfg.get("target_tokens_per_lang", 200_000_000)
        docs = []

        for lang in languages:
            chars_per_token = 4 if lang == "id" else 2
            target_chars = token_target * chars_per_token
            chars = 0

            try:
                ds = self._hf_stream("mc4", lang)
                for row in ds:
                    text = row.get("text", "").strip()
                    if len(text) < 50:
                        continue
                    docs.append(self._make_doc(text, "mc4", lang))
                    chars += len(text)
                    if chars >= target_chars:
                        break
            except Exception as e:
                print(f"[mc4] {lang}: {e}")

        return docs


class MedicalConnector(BaseConnector):
    source_name = "medical"

    def download(self, cfg: Dict, output_dir: str) -> List[Dict]:
        target_tokens = cfg.get("target_tokens", 200_000_000)
        docs = []
        chars = 0
        target_chars = target_tokens * 4

        # PubMed abstracts
        try:
            ds = self._hf_stream("pubmed_qa", "pqa_labeled", "train")
            for row in ds:
                question = row.get("question", "")
                context = " ".join(row.get("context", {}).get("contexts", []))
                answer = row.get("final_decision", "")
                text = f"Q: {question}\nContext: {context}\nA: {answer}"
                if len(text) < 100:
                    continue
                docs.append(self._make_doc(text, "pubmed_qa", "en"))
                chars += len(text)
                if chars >= target_chars // 2:
                    break
        except Exception as e:
            print(f"[pubmed] {e}")

        return docs
