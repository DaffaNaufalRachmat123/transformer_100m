#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TEXT_SUFFIXES = {".txt", ".jsonl", ".tsv"}
TEXT_BUCKETS = {"english", "indonesian", "japanese", "korean", "mixed"}
CODE_BUCKETS = {"code"}
MATH_BUCKETS = {"math"}

WHITESPACE_RE = re.compile(r"\s+")
HTML_RE = re.compile(r"</?[A-Za-z][A-Za-z0-9:_-]{0,63}(?:\s[^>]*)?>|<!--.*?-->", re.IGNORECASE)
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
REPEAT_CHAR_RE = re.compile(r"(.)\1{11,}")
MONTH_TIME_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b.*\bat\s+\d{1,2}:\d{2}\b",
    re.IGNORECASE,
)
DISCUSSION_RE = re.compile(
    r"\b(?:asked|answered|commented|replied|score\s+\d+|upvote|downvote)\b",
    re.IGNORECASE,
)
COOKIE_RE = re.compile(
    r"cookie policy|privacy policy|terms of service|all rights reserved|"
    r"enable javascript|enable cookies|sign in|log in|shopping cart|add to cart|"
    r"powered by|newsletter|subscribe now|contact us|comments are closed|"
    r"accept cookies|sale price|regular price|out of stock",
    re.IGNORECASE,
)
CODE_SIGNAL_RE = re.compile(
    r"#include\b|\bimport\b|\bfrom\b|\bclass\b|\bdef\b|\bfunction\b|\breturn\b|"
    r"\bconst\b|\blet\b|\bvar\b|\bpublic\b|\bprivate\b|\bstatic\b|\bvoid\b|"
    r"\bif\b|\belse\b|\bfor\b|\bwhile\b|=>|::|</?[A-Za-z]",
    re.IGNORECASE,
)
SPAM_KEYWORD_RE = re.compile(
    r"(?i)\b(?:"
    r"casino|gambling|poker|slot\s*machine|sports?\s*bet|betting\s*site|"
    r"judi\s*online|casino\s*bonus|free\s*spins|blackjack|roulette|"
    r"baccarat|slot\s*gacor|situsslot|"
    r"porn|xxx|sex\s*video|adult\s*video|escort\s*service|"
    r"call\s*girl|massage\s*parlor|出会い系|エロ|セックス|"
    r"風俗|デリヘル|콜걸|출장안마|조건만남|모텔|"
    r"viagra|cialis|pharmacy\s*online|diet\s*pill|weight\s*loss\s*pill|"
    r"steroid\s*for\s*sale|buy\s*adderall|"
    r"loan\s*approval|payday\s*loan|mortgage\s*refinance|"
    r"click\s*here\s*to\s*win|free\s*offer\s*limited|"
    r"work\s*from\s*home\s*earn|make\s*money\s*fast"
    r")\b"
)
JS_BLOCK_RE = re.compile(
    r"(?:function\s+\w+\s*\([^)]*\)\s*\{|"
    r"new\s+Ajax\.\w+\s*\(|"
    r"document\.(?:get|write|create)|"
    r"window\.(?:location|open)|"
    r"var\s+\w+\s*=\s*document\.)",
    re.IGNORECASE,
)
REPLACEMENT_CHAR_RE = re.compile(r"\ufffd{2,}")


@dataclass(frozen=True)
class BucketRules:
    min_chars: int
    max_urls: int
    min_alnum_ratio: float
    max_symbol_ratio: float


@dataclass
class ProgressState:
    total_files: int
    total_bytes: int
    report_every_seconds: float
    started_at: float
    processed_bytes: int = 0
    last_report_at: float = 0.0


RULES = {
    "text": BucketRules(min_chars=20, max_urls=1, min_alnum_ratio=0.55, max_symbol_ratio=0.38),
    "code": BucketRules(min_chars=1, max_urls=2, min_alnum_ratio=0.18, max_symbol_ratio=0.65),
    "math": BucketRules(min_chars=4, max_urls=1, min_alnum_ratio=0.18, max_symbol_ratio=0.72),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservative cleaner for pretraining corpora.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/corpus_trimmed"),
        help="Source corpus root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/corpus_cleaned"),
        help="Destination corpus root.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("logs/pretrain_clean_report.json"),
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze only, do not write cleaned files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional limit for quick validation runs.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Disable exact-line deduplication within each file.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=5.0,
        help="How often to print progress logs while processing a file.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous interrupted cleaning run from the state file.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Where to store resumable cleaner state. Defaults next to the report file.",
    )
    return parser.parse_args()


def default_state_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}.state.json")


def partial_output_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.name}.partial")


def compute_summary(file_reports: list[dict]) -> tuple[Counter, Counter[str]]:
    totals = Counter()
    drop_reasons: Counter[str] = Counter()
    for result in file_reports:
        totals["files"] += 1
        totals["lines_total"] += result["total"]
        totals["lines_kept"] += result["kept"]
        totals["lines_dropped"] += result["dropped"]
        drop_reasons.update(result["reasons"])
    return totals, drop_reasons


def write_summary(report_path: Path, input_root: Path, output_root: Path, dry_run: bool, file_reports: list[dict]) -> None:
    totals, drop_reasons = compute_summary(file_reports)
    summary = {
        "input": input_root.as_posix(),
        "output": output_root.as_posix(),
        "dry_run": dry_run,
        "files": file_reports,
        "totals": {
            **totals,
            "drop_rate": round(
                (totals["lines_dropped"] / totals["lines_total"])
                if totals["lines_total"]
                else 0.0,
                6,
            ),
        },
        "drop_reasons": dict(drop_reasons.most_common()),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def validate_resume_state(
    state: dict,
    *,
    input_root: Path,
    output_root: Path,
    report_path: Path,
    dry_run: bool,
    keep_duplicates: bool,
    max_files: int,
) -> None:
    expected = {
        "input": input_root.as_posix(),
        "output": output_root.as_posix(),
        "report": report_path.as_posix(),
        "dry_run": dry_run,
        "keep_duplicates": keep_duplicates,
        "max_files": max_files,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise SystemExit(
                f"[ERR] Resume state mismatch for {key}: state={state.get(key)!r} current={value!r}"
            )


def write_resume_state(
    state_path: Path,
    *,
    input_root: Path,
    output_root: Path,
    report_path: Path,
    dry_run: bool,
    keep_duplicates: bool,
    max_files: int,
    file_reports: list[dict],
    in_progress_file: str | None,
) -> None:
    payload = {
        "version": 1,
        "input": input_root.as_posix(),
        "output": output_root.as_posix(),
        "report": report_path.as_posix(),
        "dry_run": dry_run,
        "keep_duplicates": keep_duplicates,
        "max_files": max_files,
        "file_reports": file_reports,
        "in_progress_file": in_progress_file,
        "updated_at": int(time.time()),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_name(f"{state_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    tmp_path.replace(state_path)


def load_resume_state(state_path: Path) -> dict:
    with state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def emit_progress(
    progress: ProgressState,
    rel_path: Path,
    file_index: int,
    file_size: int,
    file_bytes_read: int,
    total_lines: int,
    kept: int,
    dropped: int,
    *,
    force: bool = False,
) -> None:
    now = time.monotonic()
    if not force and progress.last_report_at and now - progress.last_report_at < progress.report_every_seconds:
        return

    overall_done = progress.processed_bytes + file_bytes_read
    overall_pct = (overall_done / progress.total_bytes * 100.0) if progress.total_bytes else 100.0
    file_pct = (file_bytes_read / file_size * 100.0) if file_size else 100.0
    elapsed = max(now - progress.started_at, 0.001)
    rate = overall_done / elapsed
    remaining = max(progress.total_bytes - overall_done, 0)
    eta = (remaining / rate) if rate > 0 else 0.0

    print(
        f"[PROGRESS] [{file_index:02d}/{progress.total_files:02d}] {rel_path.as_posix()} | "
        f"file {file_pct:5.1f}% ({format_bytes(file_bytes_read)}/{format_bytes(file_size)}) | "
        f"overall {overall_pct:5.1f}% ({format_bytes(overall_done)}/{format_bytes(progress.total_bytes)}) | "
        f"lines {total_lines:,} kept {kept:,} dropped {dropped:,} | "
        f"speed {format_bytes(int(rate))}/s | eta {format_duration(eta)}",
        flush=True,
    )
    progress.last_report_at = now


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.replace("\x00", " "))
    return WHITESPACE_RE.sub(" ", text).strip()


def classify_bucket(path: Path) -> str:
    bucket = path.parent.name.lower()
    if bucket in CODE_BUCKETS:
        return "code"
    if bucket in MATH_BUCKETS:
        return "math"
    return "text"


def symbol_ratio(text: str) -> float:
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return 1.0
    symbols = 0
    for ch in visible:
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("S"):
            symbols += 1
    return symbols / len(visible)


def alnum_ratio(text: str) -> float:
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return 0.0
    return sum(ch.isalnum() for ch in visible) / len(visible)


def repeated_token_ratio(text: str) -> float:
    tokens = [token for token in re.split(r"\s+", text.lower()) if token]
    if len(tokens) < 8:
        return 1.0
    unique = len(set(tokens))
    return unique / len(tokens)


def looks_like_discussion_fragment(text: str) -> bool:
    lowered = text.lower()
    if text.startswith(("•", "- ", "* ")) and (MONTH_TIME_RE.search(lowered) or DISCUSSION_RE.search(lowered)):
        return True
    return bool(MONTH_TIME_RE.search(lowered) and DISCUSSION_RE.search(lowered))


def reject_reason(text: str, bucket: str) -> str | None:
    rules = RULES[bucket]
    stripped = text.strip()
    lowered = stripped.lower()

    if not stripped:
        return "empty"
    if len(stripped) < rules.min_chars:
        return "too_short"
    if HTML_RE.search(stripped):
        return "markup"
    if COOKIE_RE.search(lowered):
        return "web_boilerplate"
    if URL_RE.findall(stripped) and len(URL_RE.findall(stripped)) > rules.max_urls:
        return "too_many_urls"
    if EMAIL_RE.search(stripped) and bucket != "code":
        return "contact_fragment"
    if REPEAT_CHAR_RE.search(stripped):
        return "char_repetition"
    if REPLACEMENT_CHAR_RE.search(stripped):
        return "replacement_chars"
    if SPAM_KEYWORD_RE.search(stripped):
        return "spam_keyword"
    if bucket != "code" and JS_BLOCK_RE.search(stripped):
        return "js_leak"

    alpha_num = alnum_ratio(stripped)
    symbols = symbol_ratio(stripped)
    if bucket != "code" and alpha_num < rules.min_alnum_ratio and symbols > rules.max_symbol_ratio:
        return "symbol_heavy"

    if repeated_token_ratio(stripped) < 0.35:
        return "token_repetition"

    if bucket == "text":
        if looks_like_discussion_fragment(stripped):
            return "discussion_fragment"
    elif bucket == "code":
        if len(stripped) <= 4:
            return None
        if not CODE_SIGNAL_RE.search(stripped) and alpha_num < 0.30:
            return "low_code_signal"
    elif bucket == "math":
        if looks_like_discussion_fragment(stripped):
            return "discussion_fragment"
        if stripped.startswith(("•", "- ", "* ")) and "score" in lowered:
            return "discussion_fragment"

    return None


def cleaned_line(raw_line: str, suffix: str) -> str:
    raw = raw_line.rstrip("\n\r")
    if suffix == ".txt":
        return normalize_text(raw)
    return raw.strip()


def iter_corpus_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return files


def process_file(
    path: Path,
    input_root: Path,
    output_root: Path,
    dry_run: bool,
    dedup: bool,
    progress: ProgressState,
    file_index: int,
) -> dict:
    bucket = classify_bucket(path)
    rel_path = path.relative_to(input_root)
    out_path = output_root / rel_path
    suffix = path.suffix.lower()
    file_size = path.stat().st_size

    kept = 0
    dropped = 0
    total = 0
    file_bytes_read = 0
    reason_counts: Counter[str] = Counter()
    seen: set[str] = set()

    writer = None
    temp_out_path = partial_output_path(out_path)
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if temp_out_path.exists():
            temp_out_path.unlink()
        writer = temp_out_path.open("w", encoding="utf-8")

    print(
        f"[START] [{file_index:02d}/{progress.total_files:02d}] {rel_path.as_posix()} "
        f"({format_bytes(file_size)})",
        flush=True,
    )

    try:
        with path.open("rb") as handle:
            for raw_bytes in handle:
                total += 1
                file_bytes_read += len(raw_bytes)
                raw_line = raw_bytes.decode("utf-8", errors="ignore")
                line = cleaned_line(raw_line, suffix)
                reason = reject_reason(line, bucket)
                if reason is None and dedup:
                    dedup_key = normalize_text(line).lower()
                    if dedup_key in seen:
                        reason = "duplicate"
                    else:
                        seen.add(dedup_key)
                if reason is not None:
                    dropped += 1
                    reason_counts[reason] += 1
                    emit_progress(
                        progress=progress,
                        rel_path=rel_path,
                        file_index=file_index,
                        file_size=file_size,
                        file_bytes_read=file_bytes_read,
                        total_lines=total,
                        kept=kept,
                        dropped=dropped,
                    )
                    continue
                kept += 1
                if writer is not None:
                    writer.write(line)
                    writer.write("\n")
                emit_progress(
                    progress=progress,
                    rel_path=rel_path,
                    file_index=file_index,
                    file_size=file_size,
                    file_bytes_read=file_bytes_read,
                    total_lines=total,
                    kept=kept,
                    dropped=dropped,
                )
    finally:
        if writer is not None:
            writer.close()

    if not dry_run:
        if kept == 0:
            if temp_out_path.exists():
                temp_out_path.unlink()
            if out_path.exists():
                out_path.unlink()
        else:
            temp_out_path.replace(out_path)

    emit_progress(
        progress=progress,
        rel_path=rel_path,
        file_index=file_index,
        file_size=file_size,
        file_bytes_read=file_bytes_read,
        total_lines=total,
        kept=kept,
        dropped=dropped,
        force=True,
    )
    progress.processed_bytes += file_bytes_read

    return {
        "file": rel_path.as_posix(),
        "bucket": bucket,
        "total": total,
        "kept": kept,
        "dropped": dropped,
        "drop_rate": round((dropped / total) if total else 0.0, 6),
        "reasons": dict(reason_counts.most_common()),
    }


def main() -> int:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    report_path = args.report.resolve()
    state_path = (args.state_file.resolve() if args.state_file else default_state_path(report_path))

    if not input_root.exists():
        raise SystemExit(f"[ERR] Input corpus root not found: {input_root}")

    files = iter_corpus_files(input_root)
    if args.max_files > 0:
        files = files[:args.max_files]

    file_reports: list[dict] = []
    completed_by_file: dict[str, dict] = {}
    in_progress_file: str | None = None

    if args.resume:
        if not state_path.exists():
            raise SystemExit(f"[ERR] Resume state not found: {state_path}")
        state = load_resume_state(state_path)
        validate_resume_state(
            state,
            input_root=input_root,
            output_root=output_root,
            report_path=report_path,
            dry_run=args.dry_run,
            keep_duplicates=args.keep_duplicates,
            max_files=args.max_files,
        )
        file_reports = list(state.get("file_reports", []))
        completed_by_file = {item["file"]: item for item in file_reports}
        in_progress_file = state.get("in_progress_file")
        if in_progress_file:
            print(
                f"[RESUME] Reprocessing interrupted file from start: {in_progress_file}",
                flush=True,
            )
            if not args.dry_run:
                partial_path = partial_output_path(output_root / in_progress_file)
                if partial_path.exists():
                    partial_path.unlink()

    pending_files = [path for path in files if path.relative_to(input_root).as_posix() not in completed_by_file]

    total_bytes = sum(path.stat().st_size for path in files)
    progress = ProgressState(
        total_files=len(files),
        total_bytes=total_bytes,
        report_every_seconds=max(args.progress_seconds, 0.1),
        started_at=time.monotonic(),
        processed_bytes=sum(path.stat().st_size for path in files if path.relative_to(input_root).as_posix() in completed_by_file),
    )

    for path in pending_files:
        rel_path = path.relative_to(input_root).as_posix()
        file_index = next(index for index, candidate in enumerate(files, start=1) if candidate == path)
        write_resume_state(
            state_path,
            input_root=input_root,
            output_root=output_root,
            report_path=report_path,
            dry_run=args.dry_run,
            keep_duplicates=args.keep_duplicates,
            max_files=args.max_files,
            file_reports=file_reports,
            in_progress_file=rel_path,
        )
        result = process_file(
            path=path,
            input_root=input_root,
            output_root=output_root,
            dry_run=args.dry_run,
            dedup=not args.keep_duplicates,
            progress=progress,
            file_index=file_index,
        )
        file_reports.append(result)
        completed_by_file[result["file"]] = result
        write_summary(report_path, input_root, output_root, args.dry_run, file_reports)
        write_resume_state(
            state_path,
            input_root=input_root,
            output_root=output_root,
            report_path=report_path,
            dry_run=args.dry_run,
            keep_duplicates=args.keep_duplicates,
            max_files=args.max_files,
            file_reports=file_reports,
            in_progress_file=None,
        )
        print(
            f"[{file_index:02d}/{len(files):02d}] {result['file']}: "
            f"kept {result['kept']:,} / {result['total']:,}",
            flush=True,
        )

    write_summary(report_path, input_root, output_root, args.dry_run, file_reports)
    if state_path.exists():
        state_path.unlink()

    totals, _ = compute_summary(file_reports)

    print(
        "[OK] Cleanup summary: "
        f"kept {totals['lines_kept']:,} / {totals['lines_total']:,} lines, "
        f"report={report_path}",
        flush=True,
    )
    if not args.dry_run:
        print(f"[OK] Clean corpus written to: {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())