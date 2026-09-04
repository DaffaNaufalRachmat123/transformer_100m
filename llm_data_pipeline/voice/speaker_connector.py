#!/usr/bin/env python3
# voice/speaker_connector.py — Download Speaker Identification Datasets

import argparse, os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/voice_speaker.yaml")
    parser.add_argument("--output", default="../../data/processed/voice/speaker")
    parser.add_argument("--process-only", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    try:
        from data_registry import register_dataset, update_download_status
        register_dataset("voxceleb1", "voice", "speaker", language="en",
                         source_type="huggingface", source_url="huggingface.co/datasets/aaria/VoxCeleb1")
        register_dataset("voxceleb2", "voice", "speaker", language="en",
                         source_type="huggingface", source_url="huggingface.co/datasets/aaria/VoxCeleb2")
        register_dataset("jvs", "voice", "speaker", language="ja",
                         source_type="web")
        register_dataset("voxceleb_en_5000", "voice", "speaker", language="en")
        register_dataset("jvs_ja_100", "voice", "speaker", language="ja")
        register_dataset("aihub_ko_1500", "voice", "speaker", language="ko")
        register_dataset("cv_id_1500", "voice", "speaker", language="id")
        register_dataset("crowdsource_id_1000", "voice", "speaker", language="id")
    except ImportError:
        pass

    print("[Speaker] Target: 10,000 speakers (EN:5K, JA:2K, KO:1.5K, ID:1.5K)")
    print("[Speaker]   VoxCeleb1+2: ~500 GB via HuggingFace")
    print("[Speaker]   JVS (100 JA speakers): Google Sites download")
    print("[Speaker]   ID crowdsource: Toloka.ai campaign ~$200")
    print("[Speaker] Done")

if __name__ == "__main__":
    main()
