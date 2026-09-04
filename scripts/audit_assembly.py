#!/usr/bin/env python3
"""audit_assembly.py — Gate E: final corpus assembly audit.

Verifies the complete pretrain assembly before training:
  1. Language/bucket inventory with token estimates (bytes/4.2 for EN/ID,
     bytes/3.3 for JA — calibrated against the actual BPE tokenizer)
  2. Epoch math: budget tokens / unique tokens per language <= 4.0
  3. Weight table sums to 1.0 and every bucket dir exists and is non-empty
  4. Disk usage report

Usage:
  python3 scripts/audit_assembly.py \
      --corpus_dir data/corpus_1b \
      --weights '{"english":0.35,"japanese":0.25,"indonesian":0.20,"code":0.09,"math":0.03,"mixed":0.08}' \
      --budget 21000000000
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# calibrated bytes/token (from BPE port measurements on the actual corpora)
BYTES_PER_TOKEN = {"english": 4.2, "japanese": 3.3, "indonesian": 4.5,
                   "code": 3.8, "math": 4.0, "mixed": 4.2}
MAX_EPOCH = 4.0
# Buckets whose contents are INTENTIONALLY repeated many times (fact
# injection needs 100-1000 exposures per fact; parallel pairs benefit from
# multiple passes). The epoch ceiling only guards *natural* corpora against
# diminishing-returns over-repetition.
REPEAT_BY_DESIGN = {"mixed"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", default="data/corpus_1b")
    ap.add_argument("--weights", default='{"english":0.35,"japanese":0.25,"indonesian":0.20,"code":0.09,"math":0.03,"mixed":0.08}')
    ap.add_argument("--budget", type=int, default=21_000_000_000)
    ap.add_argument("--report", default="logs/gate_e_audit.json")
    args = ap.parse_args()

    weights = json.loads(args.weights)
    corpus = ROOT / args.corpus_dir

    report = {"budget_tokens": args.budget, "buckets": {}, "errors": []}

    total_unique = 0
    for bucket, weight in weights.items():
        bdir = corpus / bucket
        if not bdir.exists():
            report["errors"].append(f"missing bucket dir: {bucket}")
            continue
        files = sorted(bdir.glob("*.txt"))
        files = [f for f in files if not f.name.endswith("_raw.txt")]
        if not files:
            report["errors"].append(f"empty bucket: {bucket}")
            continue
        bytes_total = sum(f.stat().st_size for f in files)
        bpt = BYTES_PER_TOKEN.get(bucket, 4.2)
        unique_tokens = bytes_total / bpt
        budget_tokens = args.budget * weight
        epochs = budget_tokens / unique_tokens if unique_tokens > 0 else 0
        total_unique += unique_tokens
        ok_epoch = epochs <= MAX_EPOCH or bucket in REPEAT_BY_DESIGN
        if not ok_epoch:
            report["errors"].append(
                f"{bucket}: epoch {epochs:.1f} > {MAX_EPOCH} "
                f"(unique {unique_tokens/1e9:.2f}B, budget {budget_tokens/1e9:.2f}B)")
        report["buckets"][bucket] = {
            "files": [f.name for f in files],
            "bytes": bytes_total,
            "est_unique_tokens": int(unique_tokens),
            "weight": weight,
            "budget_tokens": int(budget_tokens),
            "epochs": round(epochs, 2),
            "epoch_ok": ok_epoch,
            "repeat_by_design": bucket in REPEAT_BY_DESIGN,
        }
        tag = "OK (repeat-by-design)" if bucket in REPEAT_BY_DESIGN else ("OK" if ok_epoch else "OVER")
        print(f"{bucket:<12} {bytes_total/1e9:5.2f}GB  ~{unique_tokens/1e9:5.2f}B unik  "
              f"budget {budget_tokens/1e9:5.2f}B  epoch {epochs:4.1f}  {tag}")

    wsum = sum(weights.values())
    if abs(wsum - 1.0) > 1e-6:
        report["errors"].append(f"weights sum {wsum} != 1.0")

    report["total_unique_tokens"] = int(total_unique)
    report["overall_epochs"] = round(args.budget / total_unique, 2) if total_unique else 0
    ok = not report["errors"]
    report["verdict"] = "PASS" if ok else "FAIL"

    rp = ROOT / args.report
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    print(f"\ntotal unique ~{total_unique/1e9:.2f}B tokens | "
          f"overall epochs {report['overall_epochs']}")
    print(f"Gate E: {'PASS' if ok else 'FAIL'} -> {rp}")
    for e in report["errors"]:
        print(f"  ERROR: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
