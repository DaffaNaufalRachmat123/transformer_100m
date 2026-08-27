#!/usr/bin/env python3
"""prepare_pretrain_corpus.py — Select and clean corpus for continued pretraining.

Selects EN/JA files from transformer_mobile_claude, excludes contaminated files,
and runs the cleaner on files that need it.

Usage:
    python3 scripts/prepare_pretrain_corpus.py \
        --source ../transformer_mobile_claude/data/corpus_final \
        --output data/corpus_final \
        [--clean_only openwebtext.txt]

If --clean_only is given, only that file is cleaned (others are symlinked).
Without --clean_only, all files are cleaned.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CLEAN_FILES = [
    "english/fineweb_edu_extra.txt",
    "english/fineweb.txt",
    "english/fineweb_edu.txt",
    "english/wiki_en.txt",
    "japanese/wiki_ja.txt",
]

CLEAN_NEEDED = {
    "english/openwebtext.txt",
}

EXCLUDE_FILES = {
    "japanese/mc4_ja.txt",
    "japanese/mc4_ja_epoch2.txt",
}

SKIP_SUFFIXES = {".DS_Store", "_epoch2.txt", "_epoch3.txt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Source corpus root (mobile_claude)")
    ap.add_argument("--output", default="data/corpus_final", help="Output corpus root")
    ap.add_argument("--clean_only", default=None, help="Only clean this file, symlink rest")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        print(f"[ERROR] Source not found: {source}")
        return 1

    all_targets = CLEAN_FILES + list(CLEAN_NEEDED)

    print("=" * 60)
    print("Corpus Preparation for Continued Pretraining (+2.5B)")
    print("=" * 60)
    print(f"  Source: {source}")
    print(f"  Output: {output}")
    print(f"  Files:  {len(all_targets)}")
    print()

    for rel in all_targets:
        src = source / rel
        dst = output / rel

        if not src.exists():
            print(f"  [SKIP] {rel} — not found in source")
            continue

        if src.stat().st_size == 0:
            print(f"  [SKIP] {rel} — empty (symlink)")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)

        needs_clean = rel in CLEAN_NEEDED or args.clean_only is None

        if args.dry_run:
            size_mb = src.stat().st_size / 1024 / 1024
            action = "CLEAN+COPY" if needs_clean else "SYMLINK"
            print(f"  [{action}] {rel} ({size_mb:.0f} MB)")
            continue

        if needs_clean:
            print(f"  [CLEAN] {rel} ...")
            cmd = [
                sys.executable, str(ROOT / "scripts" / "clean_pretrain_corpus.py"),
                "--input", str(source / Path(rel).parent),
                "--output", str(dst.parent),
                "--resume",
            ]
            result = subprocess.run(cmd, cwd=str(ROOT))
            if result.returncode != 0:
                print(f"  [WARN] cleaner returned {result.returncode} for {rel}")
            else:
                dst_cleaned = dst.parent / src.name
                if dst_cleaned.exists():
                    print(f"  [OK] {rel} cleaned -> {dst_cleaned}")
                else:
                    print(f"  [WARN] cleaned file not found: {dst_cleaned}")
        else:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            import os
            os.symlink(src, dst)
            size_mb = src.stat().st_size / 1024 / 1024
            print(f"  [LINK] {rel} -> {src} ({size_mb:.0f} MB)")

    print()
    print("Done. Verify with:")
    print(f"  du -sh {output}/*/")
    print(f"  wc -l {output}/*/*.txt | tail -5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
