#!/usr/bin/env python3
"""
Prepare a translation-focused SFT dataset by:
1. Taking ALL translation pairs from sft_translation.jsonl + sft_translation_ja.jsonl + sft_translation_enja.jsonl
2. Adding a small portion of general JA/EN chat to prevent forgetting
3. Oversampling shorter/simpler pairs for curriculum effect

Output: sft_data/sft_translation_focused/train.jsonl

Usage on Kaggle (after uploading):
    ./build/train_sft_cuda --stage 2 \
        --base_model checkpoints/ckpt_sft_final.safetensors \
        --sft_dir sft_data/sft_translation_focused \
        --lr 5e-6 --epochs 3 --batch_size 4 --max_seq 256 \
        --warmup 50 --save_every 500 --log_every 10
"""
import json
import os
import random
from pathlib import Path

def main():
    rng = random.Random(42)
    out_dir = Path("sft_data/sft_translation_focused")
    out_dir.mkdir(parents=True, exist_ok=True)

    translation_samples = []
    general_samples = []

    # 1. Load ONLY clean translation data (exclude noisy subtitle sft_translation.jsonl)
    for fname in ["sft_translation_ja.jsonl", "sft_translation_enja.jsonl"]:
        path = Path("sft_data/sft_clean") / fname
        if path.exists():
            with open(path) as f:
                for line in f:
                    translation_samples.append(line.strip())
            print(f"  Loaded {fname}: +{sum(1 for _ in open(path))} samples")
    # NOTE: sft_translation.jsonl excluded — contains noisy movie subtitle pairs

    # 2. Oversample: repeat each clean pair with different prompt templates
    augmented = []
    import random as _rng
    _rng.seed(123)
    sys_en2ja = [
        "You are a professional English-to-Japanese translator.",
        "You are a translator. Translate accurately and naturally.",
        "You are a multilingual translation assistant.",
        "Translate the following text accurately.",
    ]
    sys_ja2en = [
        "You are a professional Japanese-to-English translator.",
        "You are a translator. Translate accurately and naturally.",
        "You are a multilingual translation assistant.",
        "あなたはプロの翻訳者です。",
    ]
    tmpl_en2ja = [
        "Translate to Japanese: {t}", "日本語に翻訳してください: {t}",
        "Please translate this to Japanese: {t}", "Japanese translation: {t}",
        "Translate: {t}", "日本語に翻訳: {t}",
    ]
    tmpl_ja2en = [
        "Translate to English: {t}", "英語に翻訳してください: {t}",
        "英語に翻訳: {t}", "Please translate this to English: {t}",
        "English translation: {t}", "Translate: {t}",
    ]
    for sample_line in translation_samples:
        d = json.loads(sample_line)
        text = d["text"]
        if "<|user|>" not in text or "<|assistant|>" not in text:
            continue
        user_part = text.split("<|user|>")[1].split("<|assistant|>")[0]
        asst_part = text.split("<|assistant|>")[1].split("<|end|>")[0]
        # Detect direction from user part
        is_en2ja = any(k in user_part for k in ["Translate to Japanese", "日本語に翻訳", "Japanese translation"])
        is_ja2en = any(k in user_part for k in ["Translate to English", "英語に翻訳", "English translation"])
        if not is_en2ja and not is_ja2en:
            augmented.append(sample_line)
            continue
        # Extract source text (after the colon/template)
        for sep in [": ", "：", ": "]:
            if sep in user_part:
                src_text = user_part.split(sep, 1)[1].strip()
                break
        else:
            augmented.append(sample_line)
            continue
        # Generate 5 augmented versions with different templates
        if is_en2ja:
            for _ in range(5):
                sys_p = _rng.choice(sys_en2ja)
                usr_t = _rng.choice(tmpl_en2ja).format(t=src_text)
                aug_text = f"<|system|>{sys_p}<|user|>{usr_t}<|assistant|>{asst_part}<|end|>"
                augmented.append(json.dumps({"text": aug_text, "lang": "en-ja", "task": "translation"}, ensure_ascii=False))
        else:
            for _ in range(5):
                sys_p = _rng.choice(sys_ja2en)
                usr_t = _rng.choice(tmpl_ja2en).format(t=src_text)
                aug_text = f"<|system|>{sys_p}<|user|>{usr_t}<|assistant|>{asst_part}<|end|>"
                augmented.append(json.dumps({"text": aug_text, "lang": "ja-en", "task": "translation"}, ensure_ascii=False))
    translation_samples = augmented
    print(f"  After augmentation: {len(translation_samples)} samples")

    # 3. Load small portion of general EN/JA for anti-forgetting (10% of total)
    target_general = len(translation_samples) // 9  # ~10% of final mix
    en_samples = []
    ja_samples = []

    if Path("sft_data/sft_clean/sft_en.jsonl").exists():
        with open("sft_data/sft_clean/sft_en.jsonl") as f:
            en_samples = [l.strip() for l in f]
    if Path("sft_data/sft_clean/sft_ja.jsonl").exists():
        with open("sft_data/sft_clean/sft_ja.jsonl") as f:
            ja_samples = [l.strip() for l in f]

    rng.shuffle(en_samples)
    rng.shuffle(ja_samples)
    general_samples = en_samples[:target_general//2] + ja_samples[:target_general//2]
    print(f"  General anti-forgetting samples: {len(general_samples)}")

    # 4. Combine: translation + general
    all_samples = translation_samples + general_samples
    rng.shuffle(all_samples)

    # 5. Write output
    out_path = out_dir / "train.jsonl"
    with open(out_path, 'w') as f:
        for sample in all_samples:
            f.write(sample + '\n')

    print(f"\n{'='*60}")
    print(f"Translation-focused SFT dataset ready:")
    print(f"  Translation samples: {len(translation_samples)}")
    print(f"  General (anti-forget): {len(general_samples)}")
    print(f"  Total: {len(all_samples)}")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")
    print(f"""
Kaggle training command:

    python3 scripts/train_pipeline.py \\
        --stages sft \\
        --checkpoint_dir /kaggle/working/checkpoints \\
        --sft_dir sft_data/sft_translation_focused \\
        --sft_lr 5e-6 \\
        --sft_epochs 3 \\
        --sft_batch_size 4 \\
        --sft_max_seq 256 \\
        --sft_warmup 50 \\
        --sft_save_every 500 \\
        --log_every 10 \\
        --bpe bpe_v6.bin \\
        --vocab_dir vocab_6 \\
        --ddp --num_gpus 2 \\
        --resume

NOTE: Make sure to rename/remove ckpt_sft_final.safetensors first
      so the pipeline doesn't skip the SFT stage!
      Or use --sft_base_model to override.
""")

if __name__ == "__main__":
    main()
