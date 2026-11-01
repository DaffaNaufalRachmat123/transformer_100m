# LaBSE + train_bpe — Google Colab bundle

Self-contained package for two stages that were blocked on local hardware:

1. **LaBSE filter** — score EN↔JA parallel pairs (`en\tja` TSV) and drop the
   misaligned ones. Uses `sentence-transformers/LaBSE` on the Colab GPU.
2. **train_bpe** — the C++ BPE trainer that regenerates `vocab.txt` +
   `merges.txt` (+ a packed `.bin`) from a pretraining corpus.

## Contents

```
labse_bpe_colab/
├── filter_parallel_labse.py   # main LaBSE filter (sentence-transformers)
├── kaggle_labse_filter.py     # alt filter (transformers/torch, 3-col TSV)
├── pair_ok.py                 # heuristic EN/JA alignment checks (stdlib only)
├── bpe/{bpe_tokenizer.h,.cpp} # BPE trainer library
├── tools/train_bpe.cpp        # trainer CLI
├── requirements.txt
├── setup.sh
└── README.md
```

## Setup (Colab)

Upload this folder to Colab (or the Files sidebar), then in a cell:

```
!bash setup.sh
```

It installs `sentence-transformers`/`torch` and compiles `train_bpe` into the
folder. If you already have a GPU runtime, LaBSE runs on CUDA automatically.

## 1. LaBSE filter

```
!python filter_parallel_labse.py \
    --src /content/par_ccmatrix.txt \
    --out /content/par_ccmatrix_labse.txt \
    --threshold 0.70
```

- Input: 2-column TSV `english<TAB>japanese`, one pair per line.
- `--device cuda` (default auto-detects), `--fp32` to disable fp16, `--batch`
  and `--max_seq` to tune throughput.
- Threshold 0.70 keeps genuine pairs (0.75–0.95) and drops garbage (0.15–0.17).

The **alternative** filter (`kaggle_labse_filter.py`) takes a 3-column TSV
`a<TAB>b<TAB>source` and also writes a JSON rejection report:

```
!python kaggle_labse_filter.py --raw /content/wikimatrix.tsv \
    --pair en-ja --out /content/wikimatrix_labse.tsv
```

## 2. train_bpe

```
!./train_bpe \
    --out_dir vocab_8 --vocab_size 49152 --min_freq 15 \
    --corpus data/corpus_pretrain_v8/japanese:2000 \
    --corpus data/corpus_pretrain_v8/english:1300 \
    --corpus data/corpus_pretrain_v8/code:1100 \
    --corpus data/corpus_pretrain_v8/math:600 \
    --binary bpe_v8.bin --checkpoint bpe_checkpoints/bpe_v8_ckpt.bin
```

- `--corpus <dir|file[:MB]>` — repeatable; `:MB` caps each file in that source.
- `--resume` continues from `--checkpoint`.
- Outputs `vocab.txt` + `merges.txt` into `--out_dir`, plus the packed binary.