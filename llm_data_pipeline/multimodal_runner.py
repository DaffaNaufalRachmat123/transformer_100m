#!/usr/bin/env python3
# =============================================================================
# multimodal_runner.py — Unified orchestrator for ALL data pipelines
# =============================================================================
# Modes: download, llm, vision, voice, sft, dpo, all
# Example:
#   python multimodal_runner.py --mode download     # Download all data
#   python multimodal_runner.py --mode llm          # LLM text pipeline only
#   python multimodal_runner.py --mode all          # Everything

import argparse
import subprocess
import sys
import os
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent

def run_cmd(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=PIPELINE_DIR)
    if result.returncode != 0:
        print(f"[ERR] {desc} FAILED with code {result.returncode}")
    return result.returncode == 0

def mode_download():
    """Download all datasets. Resumable via data_registry.py."""
    print("\n=== DOWNLOAD PHASE ===")
    print("Downloading all LLM, Vision, and Voice datasets...\n")

    # LLM text (via existing HuggingFace connectors)
    run_cmd([sys.executable, "pipeline_runner.py", "--config", "configs/pretrain.yaml",
             "--stage", "acquisition"], "LLM Text: HuggingFace acquisition")

    # Vision: SigLIP pairs
    run_cmd([sys.executable, "vision/siglip_connector.py",
             "--config", "configs/vision_siglip.yaml"], "Vision: SigLIP download")

    # Vision: Detection datasets
    run_cmd([sys.executable, "vision/detection_connector.py",
             "--config", "configs/vision_detection.yaml"], "Vision: Detection download")

    # Vision: OCR datasets
    run_cmd([sys.executable, "vision/ocr_connector.py",
             "--config", "configs/vision_ocr.yaml"], "Vision: OCR download")

    # Vision: Generate synthetic OCR data
    run_cmd([sys.executable, "vision/synthtext_cjk.py",
             "--config", "configs/vision_ocr.yaml"], "Vision: Synthetic OCR generation")

    # Voice: ASR datasets
    run_cmd([sys.executable, "voice/asr_connector.py",
             "--config", "configs/voice_asr.yaml"], "Voice: ASR download")

    # Voice: Speaker datasets
    run_cmd([sys.executable, "voice/speaker_connector.py",
             "--config", "configs/voice_speaker.yaml"], "Voice: Speaker download")

    # Voice: SSL audio
    run_cmd([sys.executable, "voice/ssl_connector.py",
             "--config", "configs/voice_ssl.yaml"], "Voice: SSL download")

    # Voice: YouTube scraping (JA + KO + ID)
    run_cmd([sys.executable, "voice/youtube_scraper.py",
             "--languages", "ja", "ko", "id",
             "--max-hours", "12000"], "Voice: YouTube scraping")

    print("\n=== DOWNLOAD COMPLETE ===")
    run_cmd([sys.executable, "data_registry.py", "--status"])

def mode_llm():
    """Run full LLM text pretraining pipeline (14 stages)."""
    run_cmd([sys.executable, "pipeline_runner.py",
             "--config", "configs/pretrain.yaml"], "LLM Text Pipeline (full)")

def mode_vision():
    """Process Vision data into training-ready format."""
    run_cmd([sys.executable, "vision/siglip_connector.py",
             "--config", "configs/vision_siglip.yaml", "--process-only"],
            "Vision: Processing SigLIP pairs")
    run_cmd([sys.executable, "vision/detection_connector.py",
             "--config", "configs/vision_detection.yaml", "--process-only"],
            "Vision: Processing Detection data")
    run_cmd([sys.executable, "vision/ocr_connector.py",
             "--config", "configs/vision_ocr.yaml", "--process-only"],
            "Vision: Processing OCR data")

def mode_voice():
    """Process Voice data into training-ready format."""
    run_cmd([sys.executable, "voice/asr_connector.py",
             "--config", "configs/voice_asr.yaml", "--process-only"],
            "Voice: Processing ASR data")
    run_cmd([sys.executable, "voice/speaker_connector.py",
             "--config", "configs/voice_speaker.yaml", "--process-only"],
            "Voice: Processing Speaker data")
    run_cmd([sys.executable, "voice/ssl_connector.py",
             "--config", "configs/voice_ssl.yaml", "--process-only"],
            "Voice: Processing SSL data")

def mode_sft():
    """Build SFT dataset (800K examples)."""
    run_cmd([sys.executable, "sft/sft_builder.py",
             "--config", "configs/sft.yaml"], "SFT Builder")

def mode_dpo():
    """Build DPO preference pairs (200K pairs)."""
    run_cmd([sys.executable, "dpo/dpo_builder.py",
             "--config", "configs/dpo.yaml"], "DPO Builder")

def main():
    parser = argparse.ArgumentParser(description="Multimodal Data Pipeline Runner")
    parser.add_argument("--mode", required=True,
                        choices=["download", "llm", "vision", "voice", "sft", "dpo", "all"],
                        help="Pipeline mode to run")

    # Init data registry if needed
    from data_registry import init_db
    init_db()

    args = parser.parse_args()
    mode = args.mode

    if mode == "download":
        mode_download()
    elif mode == "llm":
        mode_llm()
    elif mode == "vision":
        mode_vision()
    elif mode == "voice":
        mode_voice()
    elif mode == "sft":
        mode_sft()
    elif mode == "dpo":
        mode_dpo()
    elif mode == "all":
        success = True
        success &= mode_download() or True
        success &= mode_vision()
        success &= mode_voice()
        success &= mode_llm()
        success &= mode_sft()
        success &= mode_dpo()
        if success:
            print("\n[DONE] All pipelines completed successfully!")
        else:
            print("\n[WARN] Some pipelines had errors. Check logs above.")

if __name__ == "__main__":
    main()
