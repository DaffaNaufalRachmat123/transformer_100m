"""
Report writer — generate comprehensive JSON reports from inspection results.
"""

import json
import os
import time
from typing import Dict


class ReportWriter:
    def __init__(self, output_path: str = "inspection_report.json"):
        self.output_path = output_path

    def write(self, corpus_results: Dict = None, sft_results: Dict = None, dpo_results: Dict = None) -> str:
        report = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "pipeline_version": "pretrain_v1",
            },
            "alerts": [],
        }

        if corpus_results:
            report["corpus"] = self._summarize_corpus(corpus_results)
            report["alerts"].extend(corpus_results.get("alerts", []))

        if sft_results:
            report["sft"] = self._summarize_sft(sft_results)
            report["alerts"].extend(sft_results.get("alerts", []))

        if dpo_results:
            report["dpo"] = self._summarize_dpo(dpo_results)
            report["alerts"].extend(dpo_results.get("alerts", []))

        report["summary"] = self._build_summary(report)
        report["alerts"] = list(dict.fromkeys(report["alerts"]))

        os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else ".", exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return self.output_path

    def _summarize_corpus(self, r: Dict) -> Dict:
        lang_s = r.get("language_summary", {})
        quality_l = r.get("quality_by_language", {})
        tok_l = r.get("tokenization_by_language", {})

        return {
            "totals": r.get("_totals", {}),
            "languages": lang_s,
            "quality_by_language": quality_l,
            "tokenization_by_language": tok_l,
            "files": {k: {
                "language": v["language"], "size_mb": v["size_mb"],
                "lines": v["line_count"], "estimated_tokens": v["estimated_tokens"]
            } for k, v in r.get("files", {}).items()},
            "contamination": {
                "status": r.get("contamination", {}).get("status"),
                "total_contaminated": r.get("contamination", {}).get("total_contaminated_samples", 0),
            },
            "language_verification": r.get("language_verification", {}),
            "cross_dedup_count": r.get("cross_dedup", {}).get("duplicates_found", 0),
        }

    def _summarize_sft(self, r: Dict) -> Dict:
        return {
            "totals": r.get("_totals", {}),
            "language_distribution": r.get("language_distribution", {}),
            "task_distribution": r.get("task_distribution", {}).get("global", {}),
            "template_validation": r.get("template_validation", {}).get("validity_pct"),
            "turn_statistics": r.get("turn_statistics", {}),
            "tokenization": r.get("tokenization", {}),
            "cross_dedup": r.get("cross_dedup", {}),
            "response_style": r.get("response_style", {}),
            "files": {k: {"conversations": v["conversations"], "avg_turns": v["avg_turns"]} for k, v in r.get("files", {}).items()},
        }

    def _summarize_dpo(self, r: Dict) -> Dict:
        return {
            "totals": r.get("_totals", {}),
            "language_distribution": r.get("language_distribution", {}),
            "length_analysis": r.get("length_analysis", {}),
            "gap_analysis": r.get("gap_analysis", {}),
            "quality_flags": r.get("quality_flags", {}),
            "files": {k: {"pairs": v["pairs"], "languages": v["languages"]} for k, v in r.get("files", {}).items()},
        }

    def _build_summary(self, report: Dict) -> Dict:
        s = {"flags": []}

        c = report.get("corpus", {})
        if c:
            s["corpus"] = {
                "total_files": c.get("totals", {}).get("total_files", 0),
                "total_size_gb": round(c.get("totals", {}).get("total_size_mb", 0) / 1024, 2),
                "total_lines": c.get("totals", {}).get("total_lines", 0),
                "estimated_tokens": c.get("totals", {}).get("estimated_tokens", 0),
                "languages": list(c.get("languages", {}).keys()),
                "contamination": c.get("contamination", {}).get("total_contaminated", 0),
            }
            if c.get("contamination", {}).get("total_contaminated", 0) > 0:
                s["flags"].append("CORPUS_CONTAMINATION_DETECTED")

        sft = report.get("sft", {})
        if sft:
            s["sft"] = {
                "total_files": sft.get("totals", {}).get("total_files", 0),
                "total_conversations": sft.get("totals", {}).get("total_conversations", 0),
                "template_validity_pct": sft.get("template_validation"),
                "estimated_tokens": sft.get("tokenization", {}).get("estimated_total_sft_tokens", 0),
            }
            if sft.get("cross_dedup", {}).get("duplicates_found", 0) > 100:
                s["flags"].append("SFT_HIGH_DUPLICATION")

        dpo = report.get("dpo", {})
        if dpo:
            s["dpo"] = {
                "total_files": dpo.get("totals", {}).get("total_files", 0),
                "total_pairs": dpo.get("totals", {}).get("total_pairs", 0),
                "chosen_longer_pct": dpo.get("length_analysis", {}).get("chosen_longer_pct"),
                "trivial_pairs": dpo.get("gap_analysis", {}).get("trivial_pairs", 0),
            }

        return s

    def print_terminal_report(self, report: Dict):
        """Print a human-readable report to terminal."""
        s = report.get("summary", {})
        alerts = report.get("alerts", [])

        print("\n" + "=" * 70)
        print(" INSPECTION REPORT SUMMARY")
        print("=" * 70)

        if c := s.get("corpus"):
            total_mb = c.get("total_size_mb", c.get("total_size_gb", 0) * 1024)
            print(f"\n  [CORPUS]")
            print(f"    Files      : {c.get('total_files', '?')}")
            print(f"    Total size : {total_mb/1024:.1f} GB")
            print(f"    Lines      : {c.get('total_lines', c.get('estimated_tokens', 0)//100):,}")
            print(f"    Est.tokens : ~{c.get('estimated_tokens', 0)//1_000_000}M")
            print(f"    Contam.    : {c.get('contamination', '?')}")

        if sft := s.get("sft"):
            print(f"\n  [SFT]")
            print(f"    Files      : {sft.get('total_files', '?')}")
            print(f"    Conv.      : {sft.get('total_conversations', 0):,}")
            print(f"    Est.tokens : ~{sft.get('estimated_tokens', 0)//1_000_000}M")

        if dpo := s.get("dpo"):
            print(f"\n  [DPO]")
            print(f"    Files      : {dpo.get('total_files', '?')}")
            print(f"    Pairs      : {dpo.get('total_pairs', 0):,}")
            print(f"    Trivial    : {dpo.get('trivial_pairs', '?')}")

        if s.get("flags"):
            print(f"\n  [FLAGS]")
            for f in s["flags"]:
                print(f"    ! {f}")

        if alerts:
            print(f"\n  [ALERTS ({len(alerts)})]")
            for a in alerts[:10]:
                print(f"    ! {a}")
            if len(alerts) > 10:
                print(f"    ... and {len(alerts)-10} more")

        print("\n" + "=" * 70)
        print(f" Full report: {self.output_path}")
        print("=" * 70)
