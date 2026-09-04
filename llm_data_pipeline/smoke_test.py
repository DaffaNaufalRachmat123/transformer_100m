#!/usr/bin/env python3
"""
Smoke Test — run the full pipeline on a small subset to verify correctness.

Usage:
    python llm_data_pipeline/smoke_test.py --num-docs 1000
"""

import argparse
import os
import random
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_synthetic_docs(n: int = 100) -> list:
    """Generate synthetic documents for smoke testing the pipeline."""
    random.seed(42)

    en_texts = [
        "The quick brown fox jumps over the lazy dog. This is a normal English sentence about animals and nature. It contains multiple words and has proper grammatical structure.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data without being explicitly programmed. It has revolutionized many industries.",
        "Python is a high-level programming language known for its readability and versatility. It was created by Guido van Rossum and first released in 1991 as a successor to the ABC language.",
        "The theory of relativity revolutionized our understanding of space, time, and gravity. Albert Einstein published his special theory in 1905 and the general theory in 1915.",
        "Climate change poses significant challenges to ecosystems and human societies worldwide. Rising temperatures, extreme weather events, and sea level rise are all consequences.",
        "Quantum computing leverages the principles of superposition and entanglement to perform calculations that would be infeasible on classical computers.",
    ]

    id_texts = [
        "Bahasa Indonesia adalah bahasa resmi negara Indonesia yang digunakan oleh lebih dari 270 juta orang di seluruh nusantara. Bahasa ini memiliki banyak dialek dan variasi regional yang unik.",
        "Pembelajaran mesin merupakan cabang dari kecerdasan buatan yang memungkinkan sistem belajar dari data. Teknologi ini telah mengubah cara kita berinteraksi dengan komputer.",
        "Masakan Indonesia sangat beragam dan kaya akan rempah-rempah dari berbagai daerah. Setiap daerah memiliki hidangan khas yang mencerminkan budaya dan tradisi setempat.",
    ]

    ja_texts = [
        "日本語は日本で話されている言語で、約1億2500万人が使用しています。日本語にはひらがな、カタカナ、漢字の三つの文字体系があります。",
        "機械学習は、データから学習することを可能にする人工知能の一分野です。近年、深層学習の発展により多くの分野で革新的な成果が生まれています。",
        "東京は日本の首都であり、世界で最も人口の多い都市圏の一つです。歴史的な寺院や神社と最新のテクノロジーが共存しています。",
    ]

    ko_texts = [
        "한국어는 한반도와 그 주변 지역에서 약 8천만 명이 사용하는 언어입니다. 한국어는 독특한 문자 체계인 한글로 작성되며 과학적인 구조를 가지고 있습니다.",
        "기계 학습은 데이터로부터 학습할 수 있게 하는 인공지능의 한 분야입니다. 이 기술은 의료 진단부터 자율 주행 차량까지 다양한 분야에서 활용되고 있습니다.",
        "서울은 대한민국의 수도이며 세계에서 가장 큰 도시 중 하나입니다. 전통과 현대가 조화롭게 공존하는 역동적인 도시입니다.",
    ]

    code_texts = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# This function calculates the nth Fibonacci number using recursion\nresult = fibonacci(10)\nprint(f'The 10th Fibonacci number is {result}')",
        "function quickSort(arr) {\n    if (arr.length <= 1) return arr;\n    const pivot = arr[0];\n    const left = arr.slice(1).filter(x => x < pivot);\n    const right = arr.slice(1).filter(x => x >= pivot);\n    return [...quickSort(left), pivot, ...quickSort(right)];\n}\n\nconst sorted = quickSort([3, 1, 4, 1, 5, 9, 2, 6, 5, 3]);\nconsole.log(sorted);",
        "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n        System.out.println(\"This is a Java program\");\n    }\n}",
    ]

    math_texts = [
        "Theorem: For any real numbers a and b, (a + b)^2 = a^2 + 2ab + b^2. This is known as the binomial theorem for the square of a sum. It can be proven by expanding the product.",
        "Solve: $$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$. This is the Gaussian integral, a fundamental result in probability theory and mathematical physics.",
        "The derivative of f(x) = x^n is f'(x) = n * x^(n-1) for any real number n. This is the power rule of differentiation and is widely used in calculus.",
    ]

    boilerplate_text = "Copyright (c) 2024 All rights reserved. Terms of Service | Privacy Policy | Contact Us. This website is protected by copyright law. Unauthorized reproduction is prohibited."

    docs = []
    for i in range(n):
        lang = random.choice(["en", "en", "en", "en", "id", "ja", "ko"])
        source = random.choice(["fineweb_edu", "wikipedia", "the_stack_v2/python", "mathpile", "cc_news"])

        if lang == "en":
            texts = en_texts
            is_boilerplate = i % 20 == 0
        elif lang == "id":
            texts = id_texts
            is_boilerplate = i % 15 == 0
        elif lang == "ja":
            texts = ja_texts
            is_boilerplate = i % 15 == 0
        elif lang == "ko":
            texts = ko_texts
            is_boilerplate = i % 15 == 0
        else:
            texts = en_texts
            is_boilerplate = False

        if source in ("the_stack_v2/python", "the_stack_v2/javascript", "the_stack_v2/java"):
            text = random.choice(code_texts)
        elif source == "mathpile":
            text = random.choice(math_texts)
        elif is_boilerplate:
            text = boilerplate_text
        else:
            text = random.choice(texts)

        # Occasionally create duplicate
        if i % 25 == 0 and i > 0:
            text = docs[i - 5]["text"]

        docs.append({
            "doc_id": str(uuid.uuid4()),
            "source": source,
            "source_url": f"https://example.com/doc-{i}",
            "language": lang,
            "license": "CC-BY-4.0" if lang != "code" else "MIT",
            "text": text,
        })

    return docs


def run_smoke_test(num_docs: int = 100):
    """Run the full pipeline on synthetic data to verify correctness."""

    print(f"\n{'#'*60}")
    print(f"# SMOKE TEST — {num_docs} synthetic documents")
    print(f"{'#'*60}")

    # Generate synthetic documents
    docs = generate_synthetic_docs(num_docs)
    print(f"\nGenerated {len(docs)} synthetic documents")

    # Stage 1: Validation
    from llm_data_pipeline.extraction.validator import FileValidator
    validator = FileValidator(min_file_size_bytes=10)  # low threshold for short synth texts
    passed = 0
    for doc in docs:
        is_valid, _ = validator.validate_text(doc["text"])
        if is_valid:
            passed += 1
    validator.print_summary()
    assert passed == len(docs), f"All docs should pass, got {passed}/{len(docs)}"

    # Stage 2: Normalization
    from llm_data_pipeline.extraction.normalizer import TextNormalizer
    normalizer = TextNormalizer()
    docs = normalizer.normalize_documents(docs)
    for doc in docs:
        assert "normalized_text" in doc and doc["normalized_text"].strip(), f"Empty normalized text"
    print("[SMOKE] Normalization: PASSED")

    # Stage 3: Language ID
    from llm_data_pipeline.language_id.classifier import LanguageClassifier
    classifier = LanguageClassifier(
        primary_threshold=0.5,  # lower for smoke test with short texts
        target_languages={"en", "id", "ja", "ko"},
        paragraph_level=False,
    )
    kept, rejected = classifier.classify_documents(docs)
    print(f"[SMOKE] Language ID: {len(kept)} kept, {len(rejected)} rejected")
    assert len(kept) > 0, "Should keep some documents"
    docs = kept

    # Stage 4: Cleaning
    from llm_data_pipeline.cleaning.web_artifact_removal import WebArtifactRemover
    from llm_data_pipeline.cleaning.text_repair import TextRepairer
    from llm_data_pipeline.cleaning.boilerplate_detector import BoilerplateDetector

    remover = WebArtifactRemover()
    docs = remover.clean_documents(docs)
    repairer = TextRepairer()
    docs = repairer.repair_documents(docs)
    detector = BoilerplateDetector(main_content_ratio_threshold=0.3)
    docs, rejected = detector.filter_documents(docs)
    print(f"[SMOKE] Cleaning: {len(docs)} kept, {len(rejected)} rejected")
    assert len(docs) > 0, "Should keep some documents after cleaning"

    # Stage 5: Quality Scoring
    from llm_data_pipeline.quality_scoring.heuristic import HeuristicQualityScorer
    heuristic = HeuristicQualityScorer()
    scores = [heuristic.score(d["text"], d.get("language", "en")) for d in docs]
    assert len(scores) == len(docs), "All docs should have scores"
    print(f"[SMOKE] Quality Scoring: {len(scores)} scored")
    for s in scores:
        assert s["perplexity"] > 0, "Perplexity should be > 0"
        assert 0 <= s["readability"] <= 1, "Readability should be 0-1"

    # Stage 6: Safety
    from llm_data_pipeline.safety.pii_detector import PIIDetector
    from llm_data_pipeline.safety.toxic_classifier import ToxicClassifier
    from llm_data_pipeline.safety.secrets_detector import SecretsDetector

    pii = PIIDetector()
    docs = pii.redact_documents(docs)
    tox = ToxicClassifier()
    docs, _ = tox.filter_documents(docs)
    sec = SecretsDetector()
    docs, _ = sec.filter_documents(docs)
    print(f"[SMOKE] Safety: {len(docs)} docs passed")

    # Stage 7: Dedup
    from llm_data_pipeline.deduplication.exact_dedup import ExactDeduplicator
    dedup = ExactDeduplicator()
    docs = dedup.deduplicate(docs)
    print(f"[SMOKE] Dedup: {len(docs)} unique docs")

    # Stage 8: Contamination
    from llm_data_pipeline.contamination.ngram_detector import NgramContaminationDetector
    detector = NgramContaminationDetector(ngram_size=5)
    detector.build_index(["hello world this is a benchmark test"], "test")
    clean, contaminated = detector.check_corpus(docs)
    print(f"[SMOKE] Contamination: {len(clean)} clean, {len(contaminated)} contaminated")
    assert len(clean) == len(docs), "No synthetic docs should match benchmark"

    # Stage 9: Filtering
    from llm_data_pipeline.filtering.length_filter import LengthFilter
    from llm_data_pipeline.filtering.repetition_filter import RepetitionFilter
    from llm_data_pipeline.filtering.composition_filter import CompositionFilter

    lf = LengthFilter(min_tokens=3, max_tokens=5000)
    docs, _ = lf.filter_documents(docs)
    rf = RepetitionFilter()
    docs, _ = rf.filter_documents(docs)
    cf = CompositionFilter(alpha_min_ratio=0.3, require_stopwords=False)
    docs, _ = cf.filter_documents(docs)
    print(f"[SMOKE] Filtering: {len(docs)} docs passed")

    # Stage 10: Mixing
    from llm_data_pipeline.mixing.domain_proportions import DomainProportionMixer
    mixer = DomainProportionMixer()
    docs = mixer.assign_weights(docs)
    for doc in docs:
        assert "sampling_weight" in doc, "Should have sampling_weight"
        assert doc["sampling_weight"] > 0, "Weight should be > 0"
    print(f"[SMOKE] Mixing: {len(docs)} docs weighted")

    print(f"\n{'='*60}")
    print(f" ALL SMOKE TESTS PASSED ({len(docs)} docs)")
    print(f"{'='*60}")

    return docs


def main():
    parser = argparse.ArgumentParser(description="Smoke test for data pipeline")
    parser.add_argument("--num-docs", type=int, default=100, help="Number of synthetic docs")
    args = parser.parse_args()

    run_smoke_test(num_docs=args.num_docs)


if __name__ == "__main__":
    main()
