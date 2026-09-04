#!/usr/bin/env python3
"""kaggle_translation_eval.py — COMET + round-trip eval for translation
checkpoints (T4 tier metrics beyond chrF).

Inputs (upload as Kaggle datasets):
  - generations.jsonl: {"lang": "en-id", "src": ..., "ref": ..., "hyp": ...}
    (produced locally by eval_translation.py detail dump or infer run)

Outputs:
  - comet_scores.json: mean COMET (wmt22-comet-da) per direction
  - roundtrip.json:   back-translation similarity (chrF) per direction using
                      NLLB-3.3B as the pivot

Usage:
  !python kaggle_translation_eval.py --gens /kaggle/input/gens/generations.jsonl
"""

import argparse
import json
from pathlib import Path


def chrf_char(ref, hyp, n=6):
    from collections import Counter
    r, h = list(ref), list(hyp)
    if not r or not h:
        return 0.0
    ref_ng = Counter("".join(r[i:i + n]) for i in range(len(r) - n + 1)) if len(r) >= n else Counter(r)
    hyp_ng = Counter("".join(h[i:i + n]) for i in range(len(h) - n + 1)) if len(h) >= n else Counter(h)
    if not hyp_ng:
        return 0.0
    m = sum(min(ref_ng.get(g, 0), c) for g, c in hyp_ng.items())
    p, rc = m / sum(hyp_ng.values()), m / sum(ref_ng.values())
    return 2 * p * rc / (p + rc) if p + rc else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--out_dir", default="/kaggle/working")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    import torch

    rows = [json.loads(l) for l in open(args.gens, encoding="utf-8")]
    print(f"{len(rows)} generations loaded")

    # ── 1. COMET (wmt22-comet-da)
    from comet import download_model, load_from_checkpoint
    ckpt = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(ckpt)
    data = [{"src": r["src"], "mt": r["hyp"], "ref": r["ref"]} for r in rows]
    scores = comet_model.predict(data, batch_size=args.batch, gpus=1 if torch.cuda.is_available() else 0)
    comet_out = {}
    for r, s in zip(rows, scores.scores):
        comet_out.setdefault(r["lang"], []).append(s)
    comet_summary = {k: round(sum(v) / len(v), 4) for k, v in comet_out.items()}

    # ── 2. Round-trip via NLLB-3.3B (hyp in target lang -> back to src lang)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    nllb = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-3.3B", torch_dtype=torch.float16).to(device).eval()

    NLLB_CODE = {"en": "eng_Latn", "id": "ind_Latn", "ja": "jpn_Jpan"}

    def translate_batch(texts, to_lang):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to(device)
        gen = nllb.generate(**enc,
                            forced_bos_token_id=tok.convert_tokens_to_ids(NLLB_CODE[to_lang]),
                            max_new_tokens=128, num_beams=1)
        return tok.batch_decode(gen, skip_special_tokens=True)

    rt_out = {}
    B = args.batch
    for i in range(0, len(rows), B):
        chunk = rows[i:i + B]
        tgts = [r["lang"].split("-")[1] for r in chunk]
        srcs = [r["lang"].split("-")[0] for r in chunk]
        # group by target lang (forced_bos differs per language)
        by_lang = {}
        for j, r in enumerate(chunk):
            by_lang.setdefault((tgts[j], srcs[j]), []).append((j, r["hyp"]))
        for (tgt, src), items in by_lang.items():
            backs = translate_batch([h for _, h in items], src)
            for (j, hyp), back in zip(items, backs):
                rt_out.setdefault(chunk[j]["lang"], []).append(
                    chrf_char(chunk[j]["src"], back))
        if (i // B) % 20 == 0:
            print(f"roundtrip {i}/{len(rows)}", flush=True)

    rt_summary = {k: round(sum(v) / len(v), 4) for k, v in rt_out.items()}

    out = {"comet": comet_summary, "roundtrip_chrf": rt_summary,
           "n": len(rows)}
    with open(Path(args.out_dir) / "translation_comet.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
