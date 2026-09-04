#!/usr/bin/env python3
# voice/asr_connector.py — Download Supervised ASR Datasets (Audio + Transcripts)
# Sources: LibriSpeech, GigaSpeech, Common Voice, ReazonSpeech, AI Hub Korea, etc.

import argparse, os
from pathlib import Path

SOURCES = {
    "en": ["librispeech", "gigaspeech", "common_voice_en", "tedlium", "peoples_speech"],
    "ja": ["reazonspeech", "youtube_ja", "csj", "cejc", "common_voice_ja", "jsut", "jvs"],
    "ko": ["ksponspeech", "ai_hub_ko", "youtube_ko", "common_voice_ko", "zeroth_korean"],
    "id": ["common_voice_id", "youtube_id", "podcast_id", "dpr_ri", "rri", "tedx_id"],
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/voice_asr.yaml")
    parser.add_argument("--output", default="../../data/processed/voice/asr")
    parser.add_argument("--process-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    try:
        from data_registry import register_dataset, update_download_status
        for lang, sources in SOURCES.items():
            for src in sources:
                name = f"asr_{src}"
                register_dataset(name, "voice", "asr", language=lang, source_type="huggingface")
                update_download_status(name, "pending")
    except ImportError:
        pass

    print("[ASR] Target: 60,850 hours across EN/JA/KO/ID")
    print("[ASR]   EN: 15,010 hrs | JA: 27,000 hrs | KO: 12,850 hrs | ID: 6,000 hrs")
    print("[ASR] NOTE: Implement actual downloads via HuggingFace datasets API")
    print("[ASR] NOTE: YouTube scraping runs separately via youtube_scraper.py")
    print("[ASR]   ReazonSpeech: HuggingFace datasets (reazon-research/reazonspeech)")
    print("[ASR]   NINJAL CSJ/CEJC: requires research application at ninjal.ac.jp")
    print("[ASR]   AI Hub Korea: requires registration at aihub.or.kr")
    print("[ASR]   LibriSpeech/GigaSpeech/Common Voice: via HuggingFace datasets")
    print("[ASR] Done")

if __name__ == "__main__":
    main()
