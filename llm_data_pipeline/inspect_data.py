#!/usr/bin/env python3
"""
Data Inspector — comprehensive analysis of existing corpus and SFT data.

Usage:
    # Inspect everything (corpus + SFT + DPO)
    python llm_data_pipeline/inspect_data.py --mode all

    # Inspect only pretraining corpus
    python llm_data_pipeline/inspect_data.py --mode corpus

    # Inspect only SFT data
    python llm_data_pipeline/inspect_data.py --mode sft

    # Inspect only DPO data
    python llm_data_pipeline/inspect_data.py --mode dpo

    # With sampling and custom output
    python llm_data_pipeline/inspect_data.py --mode all --sample 3000 --output my_report.json

    # Inspect smoke test data
    python llm_data_pipeline/inspect_data.py --mode corpus --corpus-dir data/corpus_smoke
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_data_pipeline.inspection.corpus_inspector import CorpusInspector
from llm_data_pipeline.inspection.sft_inspector import SFTInspector
from llm_data_pipeline.inspection.dpo_inspector import DPOInspector
from llm_data_pipeline.inspection.report_writer import ReportWriter


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive data inspection tool for transformer_490m_mobile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_data_pipeline/inspect_data.py --mode all
  python llm_data_pipeline/inspect_data.py --mode corpus --sample 5000
  python llm_data_pipeline/inspect_data.py --mode sft --output sft_report.json
  python llm_data_pipeline/inspect_data.py --mode corpus --corpus-dir data/corpus_smoke
        """,
    )

    parser.add_argument(
        "--mode", "-m",
        required=True,
        choices=["corpus", "sft", "dpo", "all"],
        help="Inspection mode: corpus, sft, dpo, or all",
    )
    parser.add_argument(
        "--corpus-dir", "-c",
        nargs="+",
        default=["data/corpus_final", "data/corpus_extra"],
        help="Corpus directories to inspect (default: data/corpus_final data/corpus_extra)",
    )
    parser.add_argument(
        "--sft-dir", "-s",
        nargs="+",
        default=["sft_data/sft", "sft_data/sft_clean"],
        help="SFT directories to inspect (default: sft_data/sft sft_data/sft_clean)",
    )
    parser.add_argument(
        "--dpo-dir", "-d",
        nargs="+",
        default=["sft_data/dpo"],
        help="DPO directories to inspect (default: sft_data/dpo)",
    )
    parser.add_argument(
        "--sample", "-n",
        type=int,
        default=5000,
        help="Sample size for quality/content checks (default: 5000)",
    )
    parser.add_argument(
        "--output", "-o",
        default="inspection_report.json",
        help="Output report path (default: inspection_report.json)",
    )
    parser.add_argument(
        "--no-contamination",
        action="store_true",
        help="Skip contamination scan (faster)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output JSON only, no terminal report",
    )

    args = parser.parse_args()

    print("\n" + "#" * 70)
    print(f"# DATA INSPECTOR — transformer_490m_mobile")
    print(f"# Mode: {args.mode} | Sample: {args.sample}")
    print("#" * 70)

    t_start = time.time()
    corpus_results = None
    sft_results = None
    dpo_results = None

    # ─── Corpus Inspection ──────────────────────────────────────────────────
    if args.mode in ("corpus", "all"):
        inspector = CorpusInspector(
            corpus_dirs=args.corpus_dir,
            sample_size=args.sample,
        )
        if args.no_contamination:
            inspector._check_contamination = lambda: None
        corpus_results = inspector.run()

    # ─── SFT Inspection ─────────────────────────────────────────────────────
    if args.mode in ("sft", "all"):
        inspector = SFTInspector(
            sft_dirs=args.sft_dir,
            sample_conversations=args.sample,
        )
        sft_results = inspector.run()

    # ─── DPO Inspection ─────────────────────────────────────────────────────
    if args.mode in ("dpo", "all"):
        inspector = DPOInspector(
            dpo_dirs=args.dpo_dir,
            sample_pairs=args.sample,
        )
        dpo_results = inspector.run()

    # ─── Write Report ───────────────────────────────────────────────────────
    writer = ReportWriter(output_path=args.output)
    report_path = writer.write(corpus_results, sft_results, dpo_results)

    if not args.json_only:
        report = {
            "summary": {},
            "alerts": [],
            "corpus": corpus_results.get("_totals", {}) if corpus_results else None,
            "sft": sft_results.get("_totals", {}) if sft_results else None,
            "dpo": dpo_results.get("_totals", {}) if dpo_results else None,
        }
        all_alerts = []
        if corpus_results:
            all_alerts.extend(corpus_results.get("alerts", []))
            report["summary"]["corpus"] = corpus_results.get("_totals", {})
        if sft_results:
            all_alerts.extend(sft_results.get("alerts", []))
            report["summary"]["sft"] = sft_results.get("_totals", {})
        if dpo_results:
            all_alerts.extend(dpo_results.get("alerts", []))
            report["summary"]["dpo"] = dpo_results.get("_totals", {})
        report["alerts"] = all_alerts

        writer.print_terminal_report(report)

    elapsed = time.time() - t_start
    print(f"\n[inspect] Total time: {elapsed:.1f}s")
    print(f"[inspect] Report saved to: {os.path.abspath(report_path)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
