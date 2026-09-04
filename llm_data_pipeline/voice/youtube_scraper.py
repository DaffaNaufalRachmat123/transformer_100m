#!/usr/bin/env python3
# =============================================================================
# voice/youtube_scraper.py — YouTube Auto-Caption Scraper for ASR Data
# =============================================================================
# Scrapes YouTube auto-captioned videos for Japanese, Korean, and Indonesian.
# Uses yt-dlp for downloading audio + captions, ffmpeg for audio extraction.
#
# Prerequisites:
#   pip install yt-dlp
#   apt install ffmpeg
#
# Usage:
#   python youtube_scraper.py --languages ja ko id --max-hours 12000 --workers 8

import argparse
import subprocess
import json
import re
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# =============================================================================
# LANGUAGE CONFIGURATIONS
# =============================================================================
LANG_CONFIGS = {
    "ja": {
        "name": "Japanese",
        "yt_lang": "ja",
        "queries": [
            "\"ニュース\"", "\"解説\"", "\"講義\"", "\"インタビュー\"",
            "\"ポッドキャスト\"", "\"ドキュメンタリー\"", "\"会議\"",
            "\"プレゼンテーション\"", "\"討論\"", "\"講演\"",
            "\"朗読\"", "\"対談\"", "\"報道\"",
        ],
        "target_hours": 6000,
        "confidence_threshold": 0.80,
        "max_duration": 600,  # 10 min
        "min_duration": 30,   # 30 sec
    },
    "ko": {
        "name": "Korean",
        "yt_lang": "ko",
        "queries": [
            "\"뉴스\"", "\"강의\"", "\"인터뷰\"", "\"팟캐스트\"",
            "\"설명\"", "\"다큐멘터리\"", "\"토론\"",
            "\"발표\"", "\"강연\"", "\"대담\"",
        ],
        "target_hours": 4000,
        "confidence_threshold": 0.80,
        "max_duration": 600,
        "min_duration": 30,
    },
    "id": {
        "name": "Indonesian",
        "yt_lang": "id",
        "queries": [
            "\"podcast indonesia\"", "\"berita terkini\"", "\"kuliah umum\"",
            "\"ceramah\"", "\"diskusi publik\"", "\"wawancara eksklusif\"",
            "\"review\"", "\"tutorial\"", "\"dokumenter\"",
        ],
        "target_hours": 2000,
        "confidence_threshold": 0.75,
        "max_duration": 600,
        "min_duration": 30,
    },
}

# =============================================================================
# YOUTUBE SEARCH & DOWNLOAD
# =============================================================================
def search_videos(query, lang, max_results=500):
    """Search YouTube for videos with a query."""
    try:
        cmd = [
            "yt-dlp",
            f"ytsearch{max_results}:{query}",
            "--flat-playlist",
            "--dump-json",
            "--no-warnings",
            "--ignore-errors",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        videos = []
        for line in result.stdout.strip().split("\n"):
            if line:
                try:
                    vid = json.loads(line)
                    duration = vid.get("duration", 0) or 0
                    videos.append({
                        "id": vid.get("id"),
                        "title": vid.get("title", ""),
                        "duration": duration,
                        "url": f"https://youtube.com/watch?v={vid.get('id')}",
                    })
                except json.JSONDecodeError:
                    continue
        return videos
    except Exception as e:
        print(f"[SEARCH ERR] {query}: {e}")
        return []

def download_video(video_url, lang, output_dir, auto_subs=True):
    """Download a single video's audio + auto-captions."""
    try:
        output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
        cmd = [
            "yt-dlp",
            video_url,
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", output_template,
            "--no-warnings",
            "--ignore-errors",
            "--no-playlist",
            "--max-filesize", "500M",
        ]
        if auto_subs:
            cmd.extend([
                "--write-auto-subs",
                f"--sub-lang", lang,
                "--convert-subs", "srt",
            ])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception as e:
        return False

# =============================================================================
# PROCESS AND FILTER
# =============================================================================
def parse_caption_srt(srt_path):
    """Parse SRT caption file to segments with confidence."""
    segments = []
    if not os.path.exists(srt_path):
        return segments

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse SRT blocks
    blocks = re.split(r"\n\n+", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            time_line = lines[1] if len(lines) > 1 else ""
            text_lines = lines[2:] if len(lines) > 2 else []
            text = " ".join(text_lines).strip()

            # Remove HTML tags / formatting
            text = re.sub(r"<[^>]+>", "", text)

            if not text:
                continue

            # Estimate confidence (YouTube doesn't expose it in SRT)
            # Use heuristics: short text, common words = higher confidence
            word_count = len(text.split())
            has_proper_case = any(c.isupper() for c in text if c.isalpha())

            if word_count >= 2:
                segments.append({
                    "text": text,
                    "word_count": word_count,
                })
        except Exception:
            continue

    return segments

def compute_confidence(segments):
    """Compute average caption quality score."""
    if not segments:
        return 0.0
    total_words = sum(s["word_count"] for s in segments)
    if total_words == 0:
        return 0.0
    # Higher word count = better confidence (proxy)
    return min(1.0, total_words / (len(segments) * 10.0))

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="YouTube auto-caption scraper")
    parser.add_argument("--languages", nargs="+", default=["ja", "ko", "id"])
    parser.add_argument("--output", default="../../data/raw/voice")
    parser.add_argument("--max-hours", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output_base = Path(args.output)
    total_videos = 0
    total_seconds = 0

    for lang in args.languages:
        cfg = LANG_CONFIGS[lang]
        lang_dir = output_base / lang
        os.makedirs(lang_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  {cfg['name']} (target: {cfg['target_hours']:,} jam)")
        print(f"{'='*60}")

        lang_videos = 0
        lang_seconds = 0
        seen_ids = set()

        for query in cfg["queries"]:
            if lang_seconds / 3600 >= cfg["target_hours"]:
                break

            print(f"\n[SEARCH] {query}")
            videos = search_videos(query, cfg["yt_lang"], max_results=200)

            for v in videos:
                if v["id"] in seen_ids:
                    continue
                if v["duration"] < cfg["min_duration"] or v["duration"] > cfg["max_duration"]:
                    continue

                seen_ids.add(v["id"])
                download_dir = os.path.join(lang_dir, v["id"])
                os.makedirs(download_dir, exist_ok=True)

                success = download_video(v["url"], cfg["yt_lang"], download_dir)
                if success:
                    # Check caption quality
                    srt_files = list(Path(download_dir).glob("*.srt"))
                    if srt_files:
                        segments = parse_caption_srt(str(srt_files[0]))
                        conf = compute_confidence(segments)
                        if conf >= cfg["confidence_threshold"]:
                            lang_videos += 1
                            lang_seconds += v["duration"]
                        else:
                            # Low confidence, clean up
                            for f in Path(download_dir).iterdir():
                                try:
                                    os.remove(f)
                                except Exception:
                                    pass
                            try:
                                os.rmdir(download_dir)
                            except Exception:
                                pass

                if lang_seconds / 3600 >= cfg["target_hours"]:
                    break

        hours = lang_seconds / 3600
        print(f"[{lang.upper()}] Downloaded: {lang_videos:,} videos, {hours:.1f} hours")

        total_videos += lang_videos
        total_seconds += lang_seconds

        # Register
        try:
            from data_registry import register_dataset, update_download_status
            register_dataset(f"youtube_{lang}_captions", "voice", "asr",
                             source_type="youtube", language=lang)
            update_download_status(f"youtube_{lang}_captions", "complete",
                                   hour_count=int(hours))
        except ImportError:
            pass

    total_hours = total_seconds / 3600
    print(f"\n[DONE] Total: {total_videos:,} videos, {total_hours:.1f} hours across {len(args.languages)} languages")

if __name__ == "__main__":
    main()
