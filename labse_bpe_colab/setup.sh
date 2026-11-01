#!/bin/bash
# setup.sh — sets up LaBSE filter + train_bpe on Google Colab (GPU T4).
#
# Run once in a Colab cell:
#     !bash setup.sh
set -e
cd "$(dirname "$0")"

echo "=== [1/2] Python deps (sentence-transformers + torch) ==="
pip install -q -r requirements.txt

echo ""
echo "=== [2/2] Compiling train_bpe (C++17) ==="
c++ -std=c++17 -O3 -DNDEBUG -pthread -I. -Ibpe \
    -o train_bpe tools/train_bpe.cpp bpe/bpe_tokenizer.cpp

echo ""
echo "=== Ready ==="
echo "  LaBSE filter : python filter_parallel_labse.py --help"
echo "  alt filter   : python kaggle_labse_filter.py --help"
echo "  BPE trainer  : ./train_bpe --help"