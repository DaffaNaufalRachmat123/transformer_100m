#!/usr/bin/env python3
"""
Pipeline Runner — orchestrate all pre-training data pipeline stages.

Usage:
    python llm_data_pipeline/pipeline_runner.py --config configs/pretrain.yaml
    python llm_data_pipeline/pipeline_runner.py --config configs/pretrain.yaml --stage acquisition
    python llm_data_pipeline/pipeline_runner.py --config configs/pretrain.yaml --smoke-test
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_data_pipeline.common import ensure_dir, load_jsonl, write_jsonl
from llm_data_pipeline.monitoring.stage_metrics import PipelineMonitor, StageMetrics

STAGE_ORDER = [
    "acquisition", "validation", "extraction", "normalization",
    "language_id", "cleaning", "quality_scoring", "safety",
    "deduplication", "contamination", "filtering", "mixing",
    "tokenization", "binary_dataset",
]


class PipelineRunner:
    """Run the full pre-training data pipeline."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.pipeline_cfg = self.config.get("pipeline", {})
        self.output_root = self.pipeline_cfg.get("output_root", "data/corpus_final")
        self.data_root = self.pipeline_cfg.get("data_root", "data/raw")
        self.smoke_test = self.pipeline_cfg.get("smoke_test", False)
        self.smoke_docs = self.pipeline_cfg.get("smoke_test_docs", 1000)
        self.seed = self.pipeline_cfg.get("seed", 42)
        self.num_workers = self.pipeline_cfg.get("num_workers", 8)

        self.monitor = PipelineMonitor(log_dir="logs")
        self._current_docs: List[Dict] = []

        ensure_dir(self.output_root)
        ensure_dir(self.data_root)
        ensure_dir("logs")

    def run(self, stages: Optional[List[str]] = None):
        """Run the pipeline. If stages is None, run all."""
        stages_to_run = stages or STAGE_ORDER
        print(f"\n{'#'*70}\n# PRETRAINING DATA PIPELINE\n# Version: {self.pipeline_cfg.get('name','v1')}\n# Stages: {stages_to_run}\n{'#'*70}")

        for stage_name in stages_to_run:
            if stage_name not in STAGE_ORDER:
                print(f"[runner] Skipping unknown stage: {stage_name}")
                continue

            t_start = time.time()
            print(f"\n{'='*60}\n STAGE: {stage_name.upper()}\n{'='*60}")

            try:
                self._run_stage(stage_name)
                elapsed = time.time() - t_start
                self.monitor.record_stage(
                    stage_name,
                    docs_in=len(self._current_docs),
                    docs_out=len(self._current_docs),
                    bytes_processed=sum(len(d.get("text", "")) for d in self._current_docs),
                )
                print(f"[runner] {stage_name} completed in {elapsed:.1f}s, {len(self._current_docs):,} docs")
            except Exception as e:
                print(f"[runner] ERROR in {stage_name}: {e}")
                import traceback
                traceback.print_exc()
                self.monitor.record_stage(stage_name, 0, 0)
                if self._stop_on_error(stage_name):
                    raise

        self.monitor.print_summary()
        self.monitor.save_report()
        print(f"\n[runner] Pipeline complete. Output: {self.output_root}")

    def _run_stage(self, stage_name: str):
        """Dispatch to the appropriate stage handler."""
        if stage_name == "acquisition":
            self._run_acquisition()
        elif stage_name == "validation":
            self._run_validation()
        elif stage_name == "extraction":
            self._run_extraction()
        elif stage_name == "normalization":
            self._run_normalization()
        elif stage_name == "language_id":
            self._run_language_id()
        elif stage_name == "cleaning":
            self._run_cleaning()
        elif stage_name == "quality_scoring":
            self._run_quality_scoring()
        elif stage_name == "safety":
            self._run_safety()
        elif stage_name == "deduplication":
            self._run_deduplication()
        elif stage_name == "contamination":
            self._run_contamination()
        elif stage_name == "filtering":
            self._run_filtering()
        elif stage_name == "mixing":
            self._run_mixing()
        elif stage_name == "tokenization":
            self._run_tokenization()
        elif stage_name == "binary_dataset":
            self._run_binary_dataset()

    def _load_docs(self, path: str) -> List[Dict]:
        """Load documents from JSONL file."""
        if not os.path.exists(path):
            print(f"[runner] No data found: {path}")
            return []
        return list(load_jsonl(path))

    def _save_docs(self, docs: List[Dict], stage_name: str):
        """Save documents to JSONL file for the next stage."""
        path = os.path.join(self.data_root, f"{stage_name}.jsonl")
        write_jsonl(path, docs)
        self._current_docs = docs
        print(f"[runner] Saved {len(docs):,} docs → {stage_name}.jsonl")

    def _run_acquisition(self):
        cfg = self.config.get("acquisition", {})
        from llm_data_pipeline.acquisition.sources import DataAcquisition
        from llm_data_pipeline.acquisition.provenance import ProvenanceTracker

        prov = ProvenanceTracker(
            db_path=cfg.get("provenance", {}).get("log_file", "data/provenance.db"),
        )
        prov.open()

        acquire = DataAcquisition(cfg, os.path.join(self.data_root, "acquisition"))
        acquire.download_all()

        source_files = []
        for root, _, files in os.walk(os.path.join(self.data_root, "acquisition")):
            for f in files:
                if f.endswith(".jsonl"):
                    source_files.append(os.path.join(root, f))

        all_docs = []
        for sf in source_files:
            docs = list(load_jsonl(sf))
            all_docs.extend(docs)

        if self.smoke_test:
            import random
            random.seed(self.seed)
            all_docs = random.sample(all_docs, min(self.smoke_docs, len(all_docs)))

        prov.register_batch(all_docs)
        prov.close()
        self._save_docs(all_docs, "acquisition")

    def _run_validation(self):
        cfg = self.config.get("validation", {})
        from llm_data_pipeline.extraction.validator import FileValidator

        docs = self._current_docs
        validator = FileValidator(
            encoding=cfg.get("encoding", "utf-8"),
            reject_non_utf8=cfg.get("reject_non_utf8", True),
            bom_strip=cfg.get("bom_strip", True),
            checksum_algorithm=cfg.get("checksum_algorithm", "blake3"),
            quarantine_dir=cfg.get("quarantine_dir", "data/raw/quarantine"),
            min_file_size_bytes=cfg.get("min_file_size_bytes", 100),
            remove_empty=cfg.get("remove_empty", True),
        )

        valid_docs = []
        rejected = 0
        for doc in docs:
            text = doc.get("text", "")
            is_valid, error = validator.validate_text(text)
            if is_valid:
                valid_docs.append(doc)
            else:
                rejected += 1

        validator.print_summary()
        self._save_docs(valid_docs, "validation")

    def _run_extraction(self):
        cfg = self.config.get("extraction", {})
        from llm_data_pipeline.extraction.text_extractor import TextExtractor

        docs = self._current_docs
        extractor = TextExtractor(
            primary_extractor=cfg.get("primary_extractor", "trafilatura"),
            fallback_chain=cfg.get("fallback_chain", None),
        )

        results = extractor.extract_batch(docs)
        valid_docs = [r for r in results if r.get("extraction_status") == "ok" and r.get("extracted_text", "").strip()]
        for r in valid_docs:
            r["text"] = r.pop("extracted_text", "")

        extractor.print_summary()
        self._save_docs(valid_docs, "extraction")

    def _run_normalization(self):
        cfg = self.config.get("extraction", {}).get("normalization", {})
        from llm_data_pipeline.extraction.normalizer import TextNormalizer

        docs = self._current_docs
        normalizer = TextNormalizer(
            unicode_form=cfg.get("unicode_form", "NFC"),
            ftfy_enabled=cfg.get("ftfy_enabled", True),
            collapse_whitespace=cfg.get("collapse_whitespace", True),
            line_ending=cfg.get("line_ending", "\n"),
            remove_zero_width=cfg.get("remove_zero_width", True),
            remove_control_chars=cfg.get("remove_control_chars", True),
            preserve_math=cfg.get("preserve_math", True),
            preserve_code_blocks=cfg.get("preserve_code_blocks", True),
        )
        results = normalizer.normalize_documents(docs)
        self._save_docs(results, "normalization")

    def _run_language_id(self):
        cfg = self.config.get("language_id", {})
        from llm_data_pipeline.language_id.classifier import LanguageClassifier

        docs = self._current_docs
        target_langs = set(cfg.get("target_languages", ["en", "id", "ja", "ko"]))

        classifier = LanguageClassifier(
            fasttext_model_path=cfg.get("primary_model", None),
            primary_threshold=cfg.get("primary_threshold", 0.65),
            secondary_threshold=cfg.get("secondary_threshold", 0.5),
            target_languages=target_langs,
            paragraph_level=cfg.get("paragraph_level", True),
        )

        kept, rejected = classifier.classify_documents(docs)
        classifier.print_summary()
        self._save_docs(kept, "language_id")

    def _run_cleaning(self):
        cfg = self.config.get("cleaning", {})
        from llm_data_pipeline.cleaning.web_artifact_removal import WebArtifactRemover
        from llm_data_pipeline.cleaning.text_repair import TextRepairer
        from llm_data_pipeline.cleaning.boilerplate_detector import BoilerplateDetector

        docs = self._current_docs

        wa_cfg = cfg.get("web_artifact_removal", {})
        remover = WebArtifactRemover(
            remove_html_tags=wa_cfg.get("remove_html_tags", True),
            remove_css=wa_cfg.get("remove_css", True),
            remove_javascript=wa_cfg.get("remove_javascript", True),
            remove_navigation=wa_cfg.get("remove_navigation", True),
            remove_footer=wa_cfg.get("remove_footer", True),
            remove_header=wa_cfg.get("remove_header", True),
            remove_cookie_banners=wa_cfg.get("remove_cookie_banners", True),
            remove_ads=wa_cfg.get("remove_ads", True),
            remove_social_widgets=wa_cfg.get("remove_social_widgets", True),
            remove_tracking=wa_cfg.get("remove_tracking", True),
        )
        docs = remover.clean_documents(docs)
        remover.print_summary()

        tr_cfg = cfg.get("text_repair", {})
        repairer = TextRepairer(
            ftfy_enabled=tr_cfg.get("ftfy_enabled", True),
            emoji_normalize=tr_cfg.get("emoji_normalize", True),
            url_normalize=tr_cfg.get("url_normalize", True),
            email_normalize=tr_cfg.get("email_normalize", True),
            phone_normalize=tr_cfg.get("phone_normalize", True),
            excessive_punctuation=tr_cfg.get("excessive_punctuation", True),
            excessive_whitespace=tr_cfg.get("excessive_whitespace", True),
        )
        docs = repairer.repair_documents(docs)

        bd_cfg = cfg.get("boilerplate_detection", {})
        detector = BoilerplateDetector(
            main_content_ratio_threshold=bd_cfg.get("main_content_ratio_threshold", 0.3),
        )
        docs, rejected = detector.filter_documents(docs)
        detector.print_summary()

        self._save_docs(docs, "cleaning")

    def _run_quality_scoring(self):
        cfg = self.config.get("quality_scoring", {})
        from llm_data_pipeline.quality_scoring.heuristic import HeuristicQualityScorer
        from llm_data_pipeline.quality_scoring.composite import CompositeQualityScorer

        docs = self._current_docs

        heur_cfg = cfg.get("heuristic", {})
        heuristic = HeuristicQualityScorer(
            perplexity_threshold=heur_cfg.get("perplexity", {}).get("threshold", 10000),
            entropy_min=heur_cfg.get("entropy", {}).get("min_entropy", 2.0),
            entropy_max=heur_cfg.get("entropy", {}).get("max_entropy", 7.0),
            compression_min=heur_cfg.get("compression_ratio", {}).get("min_ratio", 1.5),
            compression_max=heur_cfg.get("compression_ratio", {}).get("max_ratio", 12.0),
        )

        comp_cfg = cfg.get("composite", {})
        weights = comp_cfg.get("weights", {})
        composite = CompositeQualityScorer(
            weights=weights if weights else None,
            min_final_score=comp_cfg.get("min_final_score", 0.3),
        )

        model_cfg = cfg.get("model_based", {})
        score_using_model = model_cfg.get("enabled", True)
        model_scorer = None
        edu_scorer = None

        scored_docs = []
        for doc in docs:
            text = doc.get("text", "")
            lang = doc.get("language", "en")
            h_scores = heuristic.score(text, lang)
            m_score = 0.5
            e_score = 0.0

            if model_scorer:
                m_score = model_scorer.score(text)
            if edu_scorer:
                e_score = edu_scorer.score(text)

            final, all_scores = composite.score_document(text, heuristic, model_scorer, edu_scorer, lang)
            scored_docs.append({**doc, **all_scores})

        passing = [d for d in scored_docs if d.get("quality_final", 0) >= composite.min_score]
        composite.print_summary([d for d in scored_docs if "quality_final" in d])
        self._save_docs(passing, "quality_scoring")

    def _run_safety(self):
        cfg = self.config.get("safety", {})
        if not any(sub.get("enabled", False) for sub in cfg.values() if isinstance(sub, dict)):
            print("[runner] Safety filtering disabled — skipping")
            return

        docs = self._current_docs

        if cfg.get("pii_detection", {}).get("enabled", False):
            from llm_data_pipeline.safety.pii_detector import PIIDetector
            pii = PIIDetector()
            docs = pii.redact_documents(docs)
            pii.print_summary()

        if cfg.get("toxic_detection", {}).get("enabled", False):
            from llm_data_pipeline.safety.toxic_classifier import ToxicClassifier
            tox = ToxicClassifier()
            docs, _ = tox.filter_documents(docs)
            tox.print_summary()

        if cfg.get("secrets_detection", {}).get("enabled", False):
            from llm_data_pipeline.safety.secrets_detector import SecretsDetector
            sec = SecretsDetector()
            docs, _ = sec.filter_documents(docs)
            sec.print_summary()

        if cfg.get("copyright_check", {}).get("enabled", False):
            from llm_data_pipeline.safety.copyright_checker import CopyrightChecker
            cop = CopyrightChecker()
            docs, _ = cop.filter_documents(docs)
            cop.print_summary()

        self._save_docs(docs, "safety")

    def _run_deduplication(self):
        cfg = self.config.get("deduplication", {})
        docs = self._current_docs
        total_before = len(docs)

        if cfg.get("exact", {}).get("enabled", False):
            from llm_data_pipeline.deduplication.exact_dedup import ExactDeduplicator
            dd = ExactDeduplicator()
            docs = dd.deduplicate(docs)
            dd.print_summary()

        if cfg.get("minhash", {}).get("enabled", False):
            from llm_data_pipeline.deduplication.minhash_dedup import MinHashDeduplicator
            md = MinHashDeduplicator()
            docs = md.deduplicate(docs)
            md.print_summary()

        if cfg.get("suffix_array", {}).get("enabled", False):
            from llm_data_pipeline.deduplication.suffix_array_dedup import SuffixArrayDeduplicator
            sd = SuffixArrayDeduplicator()
            docs = sd.deduplicate_streaming(docs, window_size=100000)
            sd.print_summary()

        if cfg.get("paragraph", {}).get("enabled", False):
            from llm_data_pipeline.deduplication.paragraph_dedup import ParagraphDeduplicator
            pd = ParagraphDeduplicator()
            docs = pd.deduplicate(docs)
            pd.print_summary()

        removed = total_before - len(docs)
        print(f"[runner] Dedup removed {removed:,} docs ({100*removed/max(1,total_before):.1f}%)")
        self._save_docs(docs, "deduplication")

    def _run_contamination(self):
        cfg = self.config.get("contamination", {})
        from llm_data_pipeline.contamination.benchmark_loader import BenchmarkLoader
        from llm_data_pipeline.contamination.ngram_detector import NgramContaminationDetector
        from llm_data_pipeline.contamination.contamination_report import ContaminationReporter

        bench_names = cfg.get("benchmarks", ["mmlu", "arc_easy", "arc_challenge", "hellaswag",
                                              "winogrande", "gsm8k", "humaneval", "mbpp",
                                              "truthfulqa", "bbh", "math", "ifeval"])
        detect_cfg = cfg.get("detection", {})
        ngram_size = detect_cfg.get("ngram_size", 13)

        loader = BenchmarkLoader(cache_dir="data/benchmarks", benchmarks=bench_names)
        benchmarks = loader.load_cached()
        if not benchmarks:
            print("[runner] No cached benchmarks — run benchmark_loader.load_all(download=True) first")
            print("[runner] Skipping contamination check")
            return

        detector = NgramContaminationDetector(ngram_size=ngram_size)
        all_bench_texts = loader.extract_all_texts(benchmarks)
        detector.build_index(all_bench_texts)

        docs = self._current_docs
        clean, contaminated = detector.check_corpus(docs)

        reporter = ContaminationReporter(
            report_path=cfg.get("report_file", "data/contamination_report.json"),
        )
        report = reporter.generate_report(
            len(clean), len(contaminated), detector.get_stats(),
            contaminated, bench_names,
        )

        quarantine_dir = cfg.get("quarantine_dir", "data/raw/contaminated")
        ensure_dir(quarantine_dir)
        if contaminated:
            path = os.path.join(quarantine_dir, "contaminated_docs.jsonl")
            write_jsonl(path, contaminated)

        detector.print_summary()
        reporter.print_report(report)
        reporter.save_report(report)

        print(f"[runner] Contamination: {len(clean):,} clean, {len(contaminated):,} quarantined")
        self._save_docs(clean, "contamination")

    def _run_filtering(self):
        cfg = self.config.get("filtering", {})
        from llm_data_pipeline.filtering.length_filter import LengthFilter
        from llm_data_pipeline.filtering.repetition_filter import RepetitionFilter
        from llm_data_pipeline.filtering.composition_filter import CompositionFilter

        docs = self._current_docs

        len_cfg = cfg.get("length", {})
        lf = LengthFilter(
            min_tokens=len_cfg.get("min_tokens", 50),
            max_tokens=len_cfg.get("max_tokens", 100000),
            min_sentences=3,
            min_chars=200,
            min_words=20,
        )
        docs, _ = lf.filter_documents(docs)
        lf.print_summary()

        rep_cfg = cfg.get("repetition", {})
        rf = RepetitionFilter(
            line_dup_max=rep_cfg.get("line_dup_max", 0.3),
            paragraph_dup_max=rep_cfg.get("paragraph_dup_max", 0.2),
            ngram_max_repeat=rep_cfg.get("ngram_max_repeat", 10),
            ngram_size=rep_cfg.get("ngram_size", 10),
            char_max_ratio=rep_cfg.get("char_max_ratio", 0.2),
        )
        docs, _ = rf.filter_documents(docs)
        rf.print_summary()

        comp_cfg = cfg.get("composition", {})
        cf = CompositionFilter(
            alpha_min_ratio=comp_cfg.get("alpha_min_ratio", 0.6),
            numeric_max_ratio=comp_cfg.get("numeric_max_ratio", 0.2),
            symbol_max_ratio=comp_cfg.get("symbol_max_ratio", 0.1),
            non_printable_max=comp_cfg.get("non_printable_max", 0.01),
            mean_word_length_min=comp_cfg.get("mean_word_length_min", 3),
            mean_word_length_max=comp_cfg.get("mean_word_length_max", 15),
        )
        docs, _ = cf.filter_documents(docs)
        cf.print_summary()

        self._save_docs(docs, "filtering")

    def _run_mixing(self):
        cfg = self.config.get("mixing", {})
        from llm_data_pipeline.mixing.domain_proportions import DomainProportionMixer
        from llm_data_pipeline.mixing.upsampling import DomainUpsampler
        from llm_data_pipeline.mixing.curriculum import CurriculumScheduler

        docs = self._current_docs

        domain_cfg = cfg.get("domain_proportions", {})
        lang_cfg = cfg.get("language_proportions", {})
        mixer = DomainProportionMixer(
            domain_weights=domain_cfg if domain_cfg else None,
            language_weights=lang_cfg if lang_cfg else None,
        )
        docs = mixer.assign_weights(docs)
        mixer.print_distribution(docs)

        up_cfg = cfg.get("upsampling", {})
        upsampler = DomainUpsampler(
            upsample_factors={
                "code": up_cfg.get("code", 3.0),
                "math": up_cfg.get("math", 5.0),
                "wikipedia": up_cfg.get("wikipedia", 3.0),
                "educational": up_cfg.get("educational", 2.0),
            },
            max_epochs=cfg.get("max_epochs", 5),
        )
        docs = upsampler.apply(docs)
        upsampler.print_upsampling_plan(docs)

        total_target = cfg.get("target_tokens", 10_000_000_000)
        curriculum = CurriculumScheduler(total_target_tokens=total_target)
        curriculum.print_plan()

        self._save_docs(docs, "mixing")

    def _run_tokenization(self):
        cfg = self.config.get("tokenization", {})
        from llm_data_pipeline.tokenization.encoder import TokenizationEncoder
        from llm_data_pipeline.tokenization.sequence_packer import SequencePacker

        docs = self._current_docs

        encoder = TokenizationEncoder(
            vocab_path=cfg.get("vocab_dir", "vocab_6") + "/vocab.txt",
            merges_path=cfg.get("vocab_dir", "vocab_6") + "/merges.txt",
            max_seq_len=cfg.get("max_seq_len", 2048),
        )
        encoder.load_vocab()

        docs = encoder.encode_documents(docs)
        encoder.print_summary()

        packer = SequencePacker(
            seq_len=cfg.get("max_seq_len", 256),
            packing_algorithm=cfg.get("packing_algorithm", "first_fit_decreasing"),
            efficiency_target=cfg.get("packing_efficiency_target", 0.995),
        )
        sequences, attn_masks = packer.pack(docs)
        packer.print_summary()

        packed_docs = []
        for i, (seq, mask) in enumerate(zip(sequences, attn_masks or [])):
            packed_docs.append({
                "doc_id": f"packed_{i:010d}",
                "token_ids": seq,
                "attention_mask": mask if attn_masks else None,
                "seq_len": len(seq),
            })

        self._save_docs(packed_docs, "tokenization")

    def _run_binary_dataset(self):
        cfg = self.config.get("binary_dataset", {})
        from llm_data_pipeline.binary_dataset.shard_writer import ShardWriter
        from llm_data_pipeline.binary_dataset.index_builder import IndexBuilder

        docs = self._current_docs

        docs_by_lang: Dict[str, List[Dict]] = {}
        for doc in docs:
            lang = doc.get("language", "unknown")
            if lang not in docs_by_lang:
                docs_by_lang[lang] = []
            docs_by_lang[lang].append(doc)

        shard_size_mb = cfg.get("shard_size_mb", 512)

        seq_len = docs[0].get("seq_len", 256) if docs else 256
        writer = ShardWriter(
            output_dir=self.output_root,
            shard_size_mb=shard_size_mb,
            seq_len=seq_len,
            mmap_enabled=cfg.get("mmap_enabled", True),
        )

        shard_paths = writer.write_all_languages(docs_by_lang)
        writer.print_summary()

        if cfg.get("index_enabled", True):
            builder = IndexBuilder(
                output_dir=self.output_root,
                version_prefix=cfg.get("version_prefix", "pretrain_v1"),
            )
            builder.build(shard_paths, writer, self.config)

        print(f"\n[runner] Binary dataset written to {self.output_root}")

    def _stop_on_error(self, stage_name: str) -> bool:
        return stage_name in {"acquisition", "binary_dataset"}


def main():
    parser = argparse.ArgumentParser(description="Pre-training data pipeline runner")
    parser.add_argument("--config", required=True, help="Path to pretrain.yaml")
    parser.add_argument("--stage", nargs="+", help="Run specific stage(s)")
    parser.add_argument("--smoke-test", action="store_true", help="Run on 1000 docs only")
    parser.add_argument("--list-stages", action="store_true", help="List all stages")
    args = parser.parse_args()

    if args.list_stages:
        for i, s in enumerate(STAGE_ORDER):
            print(f"  {i+1:>2}. {s}")
        return

    runner = PipelineRunner(args.config)
    if args.smoke_test:
        runner.pipeline_cfg["smoke_test"] = True
        runner.smoke_test = True

    runner.run(args.stage)


if __name__ == "__main__":
    main()
