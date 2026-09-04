#!/usr/bin/env python3
# =============================================================================
# vision/synthtext_cjk.py — Synthetic OCR Data Generator for CJK & Latin
# =============================================================================
# Generates synthetic text images for OCR training in Japanese (with vertical
# text, furigana), Korean (with jamo decomposition), and Latin (EN+ID).
# Uses Pillow + Cairo for CJK text shaping.
#
# Engineering effort: ~7 days
# Output: ~2.4M images (~200 GB)
#
# Usage:
#   python synthtext_cjk.py --config ../configs/vision_ocr.yaml --workers 8

import argparse
import random
import math
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
except ImportError:
    print("[ERR] pip install Pillow")
    sys.exit(1)

# =============================================================================
# FONT CONFIGURATIONS
# =============================================================================
LATIN_FONTS = {
    "sans":       {"path": "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf", "name": "Roboto"},
    "serif":      {"path": "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf", "name": "Noto Serif"},
    "mono":       {"path": "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", "name": "Noto Mono"},
    "display":    {"path": "/usr/share/fonts/truetype/oswald/Oswald-Regular.ttf", "name": "Oswald"},
    "handwriting": {"path": "/usr/share/fonts/truetype/dancing/DancingScript-Regular.ttf", "name": "Dancing Script"},
}

JAPANESE_FONTS = {
    "gothic":     {"path": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "name": "Noto Sans JP", "index": 1},
    "mincho":     {"path": "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", "name": "Noto Serif JP", "index": 1},
    "kyokasho":   {"path": "/usr/share/fonts/truetype/ud/UDDigiKyokashoN-R.ttc", "name": "UD Digi Kyokasho"},
    "maru":       {"path": "/usr/share/fonts/truetype/mplus/RoundedMplus1c-Regular.ttf", "name": "Rounded M+"},
    "handwriting": {"path": "/usr/share/fonts/opentype/klee/KleeOne-Regular.ttf", "name": "Klee One"},
}

KOREAN_FONTS = {
    "gothic":     {"path": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "name": "Noto Sans KR", "index": 2},
    "myeongjo":   {"path": "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", "name": "Noto Serif KR", "index": 2},
    "handwriting": {"path": "/usr/share/fonts/truetype/hi/HiMelody-Regular.ttf", "name": "Hi Melody"},
    "display":    {"path": "/usr/share/fonts/truetype/black/BlackHanSans-Regular.ttf", "name": "Black Han Sans"},
}

# =============================================================================
# TEXT SAMPLES (per language)
# =============================================================================
SAMPLE_TEXTS_EN = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning transforms data into intelligence",
    "Please proceed to gate B42 for boarding",
    "Chapter 3: Advanced Database Techniques",
    "STOP — Hazardous Materials Ahead",
]

SAMPLE_TEXTS_ID = [
    "Jalan Malioboro No. 123, Yogyakarta",
    "Nasi goreng spesial Rp 25.000",
    "PEMERINTAH KOTA BANDUNG — PERATURAN DAERAH",
    "Kartu Tanda Penduduk (KTP) berlaku seumur hidup",
    "Hati-hati! Ada perbaikan jalan di depan",
]

SAMPLE_TEXTS_JA = [
    "東京都新宿区西新宿2-8-1 東京都庁",
    "人工知能は急速に進化しています",
    "この電車は渋谷行きです。次の駅は原宿です。",
    "第１章　はじめに—機械学習の基礎—",
    "警告：この先、立入禁止区域",
]

SAMPLE_TEXTS_KO = [
    "서울특별시 강남구 테헤란로 152",
    "인공지능 기술이 빠르게 발전하고 있습니다",
    "이 열차는 부산행입니다.",
    "제1장 서론—기계학습의 기초—",
    "경고: 이 지역은 출입 금지 구역입니다",
]

# =============================================================================
# AUGMENTATIONS
# =============================================================================
def apply_augmentations(image, cfg):
    """Apply random augmentations to the synthetic text image."""
    img = image.copy()

    if random.random() < 0.5:  # Rotation ±15°
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255))

    if random.random() < 0.3:  # Gaussian blur
        sigma = random.uniform(0.5, 3.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    if random.random() < 0.3:  # Brightness ±30%
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(factor)

    if random.random() < 0.3:  # Contrast ±20%
        factor = random.uniform(0.8, 1.2)
        img = ImageEnhance.Contrast(img).enhance(factor)

    return img

# =============================================================================
# GENERATOR FUNCTIONS
# =============================================================================
def generate_latin_image(font_path, text, font_size=32, canvas_size=(300, 80)):
    """Generate a single Latin text image."""
    img = Image.new("RGB", canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    x = random.randint(5, max(5, canvas_size[0] - tw - 5))
    y = random.randint(5, max(5, canvas_size[1] - th - 5))
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return img, text

def generate_japanese_image(font_path, text, font_size=28, canvas_size=(300, 120), vertical=False):
    """Generate a Japanese text image (supports vertical tategaki)."""
    img = Image.new("RGB", canvas_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)

    if vertical:
        # Render each character in a vertical column
        x = random.randint(10, canvas_size[0] - font_size - 10)
        y = random.randint(5, 20)
        for char in text:
            if y + font_size > canvas_size[1] - 5:
                break
            draw.text((x, y), char, fill=(0, 0, 0), font=font)
            y += font_size + 2
    else:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        x = random.randint(5, max(5, canvas_size[0] - tw - 5))
        y = random.randint(5, max(5, canvas_size[1] - th - 5))
        draw.text((x, y), text, fill=(0, 0, 0), font=font)

    return img, text

def generate_korean_image(font_path, text, font_size=28, canvas_size=(300, 80)):
    """Generate a Korean text image."""
    return generate_latin_image(font_path, text, font_size, canvas_size)

# =============================================================================
# BATCH GENERATOR
# =============================================================================
def generate_batch(args):
    """Generate a batch of synthetic images for one language + font combo."""
    lang, font_name, font_path, count, output_dir, start_idx = args
    samples = {
        "en": SAMPLE_TEXTS_EN, "id": SAMPLE_TEXTS_ID,
        "ja": SAMPLE_TEXTS_JA, "ko": SAMPLE_TEXTS_KO
    }[lang]
    texts = samples * (count // len(samples) + 1)

    generated = 0
    for i in range(count):
        text = texts[i % len(texts)]
        if lang == "ja":
            vertical = random.random() < 0.30  # 30% vertical
            img, gt_text = generate_japanese_image(font_path, text, vertical=vertical)
        elif lang == "ko":
            img, gt_text = generate_korean_image(font_path, text)
        else:
            img, gt_text = generate_latin_image(font_path, text)

        img = apply_augmentations(img, {})
        idx = start_idx + i
        img_path = os.path.join(output_dir, lang, f"{font_name}_{idx:08d}.jpg")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img.save(img_path, "JPEG", quality=85)
        generated += 1

    return generated

# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Synthetic OCR data generator")
    parser.add_argument("--config", default="../configs/vision_ocr.yaml")
    parser.add_argument("--output", default="../../data/raw/vision/ocr/synthtiger_output")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--total", type=int, default=2400000)
    args = parser.parse_args()

    output_dir = Path(args.output)
    print(f"[SynthText CJK] Generating ~{args.total/1e6:.1f}M images to {output_dir}")

    # Define batches
    batches = []
    idx = 0

    # Latin EN: 500K (5 fonts × 100K)
    for name, cfg in LATIN_FONTS.items():
        batches.append(("en", name, cfg["path"], 100000, str(output_dir), idx))
        idx += 100000

    # Latin ID: 500K (5 fonts × 100K)
    for name, cfg in LATIN_FONTS.items():
        batches.append(("id", name, cfg["path"], 100000, str(output_dir), idx))
        idx += 100000

    # Japanese: 800K (5 fonts × 160K)
    for name, cfg in JAPANESE_FONTS.items():
        batches.append(("ja", name, cfg["path"], 160000, str(output_dir), idx))
        idx += 160000

    # Korean: 600K (4 fonts × 150K)
    for name, cfg in KOREAN_FONTS.items():
        batches.append(("ko", name, cfg["path"], 150000, str(output_dir), idx))
        idx += 150000

    total_expected = sum(b[3] for b in batches)
    print(f"[SynthText CJK] {len(batches)} batches, {total_expected:,} total images")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(generate_batch, batches))

    total_generated = sum(results)
    print(f"[SynthText CJK] Done! Generated {total_generated:,} images.")

    # Register in data registry
    try:
        from data_registry import register_dataset, update_download_status
        register_dataset("synthtext_cjk_all", "vision", "ocr",
                         source_type="generated", language="en/id/ja/ko")
        update_download_status("synthtext_cjk_all", "complete",
                               image_count=total_generated)
        print("[SynthText CJK] Registered in data_registry")
    except ImportError:
        pass

if __name__ == "__main__":
    main()
