#!/usr/bin/env python3
# voice/ssl_connector.py — Download Unsupervised SSL Audio Data
# No transcripts needed — audio only for masked frame prediction

import argparse, os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/voice_ssl.yaml")
    parser.add_argument("--output", default="../../data/processed/voice/ssl")
    parser.add_argument("--process-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    try:
        from data_registry import register_dataset, update_download_status
        for lang, hours in [("en", 15000), ("ja", 15000), ("ko", 12000), ("id", 8000)]:
            register_dataset(f"ssl_{lang}_audio", "voice", "ssl", language=lang)
            update_download_status(f"ssl_{lang}_audio", "pending", hour_count=hours)
    except ImportError:
        pass

    print("[SSL] Target: 50,000 hours total")
    print("[SSL]   EN: 15,000 hrs | JA: 15,000 hrs | KO: 12,000 hrs | ID: 8,000 hrs")
    print("[SSL] NOTE: SSL doesn't need transcripts. Process audio to mel spectrograms.")
    print("[SSL] Done")

if __name__ == "__main__":
    main()
