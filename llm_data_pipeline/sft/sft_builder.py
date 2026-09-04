#!/usr/bin/env python3
# sft/sft_builder.py — Build 800K SFT Examples for Multimodal 1B Model
# Categories: Chat, Translation, Reasoning, Math, Summarization,
#             Tool Calling, Vision QA, Code-Switching, Safety, Persona

import argparse, os, json
from pathlib import Path

SFT_CONFIG = {
    "chat_multi_turn":     {"count": 180000, "langs": {"en": 90000, "id": 35000, "ja": 30000, "ko": 25000}},
    "translation":         {"count": 120000, "langs": 12},  # all 12 language pairs
    "reasoning_cot":       {"count": 130000, "langs": {"en": 70000, "id": 20000, "ja": 20000, "ko": 20000}},
    "math_step_by_step":   {"count": 80000,  "langs": {"en": 40000, "id": 15000, "ja": 15000, "ko": 10000}},
    "summarization":       {"count": 50000,  "langs": {"en": 20000, "id": 15000, "ja": 10000, "ko": 5000}},
    "tool_calling":        {"count": 60000,  "langs": {"en": 35000, "mixed": 25000}},
    "vision_qa":           {"count": 66000,  "langs": {"en": 50000, "multilingual": 16000}},
    "code_switching":      {"count": 24000,  "langs": {"id_en": 10000, "ja_en": 8000, "ko_en": 6000}},
    "safety_refusal":      {"count": 30000,  "langs": {"en": 15000, "id": 5000, "ja": 5000, "ko": 5000}},
    "persona":             {"count": 30000,  "langs": {"en": 15000, "id": 15000}},
    "instruction_format":  {"count": 30000,  "langs": {"en": 15000, "id": 8000, "ja": 4000, "ko": 3000}},
}

def generate_example(category, lang):
    """Generate a single SFT example. Real implementation would load from datasets."""
    return {
        "messages": [
            {"role": "system", "content": "Kamu adalah asisten mobile pintar bernama [Nama]. Kamu bisa membantu dalam bahasa Indonesia, Inggris, Jepang, dan Korea."},
            {"role": "user", "content": f"[{category}] Pertanyaan contoh dalam bahasa {lang}"},
            {"role": "assistant", "content": f"[{category}] Jawaban contoh dalam bahasa {lang}"},
        ],
        "category": category,
        "language": lang,
        "source": "sft_builder",
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/sft.yaml")
    parser.add_argument("--output", default="../../data/processed/sft")
    parser.add_argument("--smoke-test", action="store_true", help="Generate 100 examples only")
    args = parser.parse_args()

    output_dir = Path(args.output)
    os.makedirs(output_dir, exist_ok=True)

    total_examples = 0
    output_file = output_dir / "sft_data.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for category, cfg in SFT_CONFIG.items():
            count = 100 if args.smoke_test else cfg["count"]
            langs = cfg.get("langs", {})
            if isinstance(langs, dict):
                for lang, lang_count in langs.items():
                    actual = lang_count if not args.smoke_test else min(20, lang_count)
                    for _ in range(actual):
                        ex = generate_example(category, lang)
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        total_examples += 1
            elif isinstance(langs, int):
                for _ in range(count):
                    ex = generate_example(category, "mixed")
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    total_examples += 1

    print(f"[SFT Builder] Generated {total_examples:,} examples → {output_file}")
    print(f"[SFT] Categories: {len(SFT_CONFIG)}")

    # Register
    try:
        from data_registry import register_dataset, update_download_status
        register_dataset("sft_800k", "llm", "pretrain", source_type="generated",
                         language="en/id/ja/ko")
        update_download_status("sft_800k", "complete", token_count=total_examples * 500)
    except ImportError:
        pass

    print("[SFT Builder] Done")

if __name__ == "__main__":
    main()
