#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  finalize_v9.sh — everything that was blocked on the LaBSE filter.
#
#    1. wait for scripts/filter_parallel_labse.py
#    2. rebuild the EN<->JA parallel corpus from the filtered pairs
#    3. rebuild data/corpus_pretrain_v8 (picks up the new openwebmath math data)
#    4. train vocab_9 + bpe_v9.bin   (digits are now one token each)
#    5. re-measure BYTES_PER_TOKEN   (digit splitting changes every bucket)
#    6. preflight
#    7. repoint every operational vocab_8 reference at vocab_9
#
#  Run detached; it takes several hours:
#    nohup bash scripts/finalize_v9.sh > logs/finalize_v9.log 2>&1 &
# ═══════════════════════════════════════════════════════════════════════════
set -u
cd "$(dirname "$0")/.."
say() { echo ""; echo "═══ $* ═══"; date '+%H:%M:%S'; }

say "1/7 waiting for the LaBSE filter"
while pgrep -f "filter_parallel_labse.py" > /dev/null; do sleep 120; done
FILTERED=data/parallel_raw/en-ja/par_ccmatrix_labse.txt
if [ ! -s "$FILTERED" ]; then
    echo "[ERR] $FILTERED missing or empty — aborting"; exit 1
fi
echo "filtered pairs: $(wc -l < "$FILTERED")"

say "2/7 rebuilding the parallel corpus"
python3 -u scripts/build_mixed_bucket.py || exit 1

say "3/7 rebuilding the pretraining corpus"
rm -rf data/corpus_pretrain_v8
python3 -u scripts/build_pretrain_corpus.py \
    --out data/corpus_pretrain_v8 --progress-secs 60 || exit 1

say "4/7 training vocab_9"
# Rebuild the trainer first. The installed binary has silently lagged the source
# twice already, and a stale one here would train vocab_9 without the digit
# split — the entire reason for this rebuild — with no visible symptom.
c++ -std=c++17 -O3 -DNDEBUG -pthread -I. -Ibpe \
    -o build_infer/train_bpe tools/train_bpe.cpp bpe/bpe_tokenizer.cpp || exit 1
# mixed/ stays out on purpose: its facts repeat 1000x and would buy merges for
# template phrases instead of language.
mkdir -p bpe_checkpoints
./build_infer/train_bpe \
    --out_dir vocab_9 --vocab_size 49152 --min_freq 15 \
    --corpus data/corpus_pretrain_v8/japanese:2000 \
    --corpus data/corpus_pretrain_v8/english:1300 \
    --corpus data/corpus_pretrain_v8/code:1100 \
    --corpus data/corpus_pretrain_v8/math:600 \
    --binary bpe_v9.bin --checkpoint bpe_checkpoints/bpe_v9_ckpt.bin || exit 1

say "5/7 re-measuring bytes/token for vocab_9"
python3 -u scripts/measure_bytes_per_token.py --vocab_dir vocab_9 --write || exit 1

say "6/7 preflight"
python3 -u scripts/preflight_pretrain.py \
    --vocab_dir vocab_9 --bpe bpe_v9.bin \
    --data_dir data/corpus_pretrain_v8 --max_seq 512 || exit 1

say "7/7 repointing vocab_8 -> vocab_9"
# Only operational references. Comments that record what vocab_7/vocab_8
# measured are history and stay as they are.
for f in running_commands/*.md scripts/run_bpe_after_corpus.sh tools/train_bpe.cpp; do
    [ -f "$f" ] || continue
    perl -0pi -e "
      s/'vocab_8'/'vocab_9'/g;
      s/'bpe_v8\.bin'/'bpe_v9.bin'/g;
      s/--vocab_dir vocab_8/--vocab_dir vocab_9/g;
      s/--bpe bpe_v8\.bin/--bpe bpe_v9.bin/g;
      s/--out_dir vocab_8/--out_dir vocab_9/g;
      s/--binary bpe_v8\.bin/--binary bpe_v9.bin/g;
      s/bpe_v8_ckpt\.bin/bpe_v9_ckpt.bin/g;
      s/\\\$SRC\/vocab_8/\\\$SRC\/vocab_9/g;
      s/\\\$SRC\/bpe_v8\.bin/\\\$SRC\/bpe_v9.bin/g;
      s/ls -d vocab_8 bpe_v8\.bin/ls -d vocab_9 bpe_v9.bin/g;
      s/corpus_pretrain_v8\/japanese:2000/corpus_pretrain_v8\/japanese:2000/g;
    " "$f"
done
perl -0pi -e 's/matching bpe_v8\.bin or vocab_8 files/matching bpe_v9.bin or vocab_9 files/g' \
    transformer/train_cuda_unified.cu
perl -0pi -e 's/# Measured on data\/corpus_pretrain_v8 with vocab_8\./# Measured on data\/corpus_pretrain_v8 with vocab_9./' \
    scripts/build_pretrain_corpus.py

echo ""
echo "remaining operational vocab_8 refs (should be none):"
grep -rn "vocab_8\|bpe_v8" running_commands/ scripts/*.sh tools/train_bpe.cpp 2>/dev/null \
  | grep -vE "came out with|are both 49152|lopsided" || echo "  none"

say "FINALIZE V9 DONE"
du -sh data/corpus_pretrain_v8/*/ 2>/dev/null
