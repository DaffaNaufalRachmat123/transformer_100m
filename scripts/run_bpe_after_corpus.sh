#!/bin/bash
# Stage 2 of the unattended chain: waits for run_corpus_rebuild.sh to finish,
# then trains the tokenizer on the finished corpus and preflights the result.
#
# BPE has to come after the corpus, not before: code is 1100 of the 5000 MB
# sample, and the code bucket is being rebuilt from 95%-Python to a mix that
# includes Java, Go, PHP, Ruby, Kotlin, C++ and JS. Merges learned from the old
# bucket would carry none of those idioms, and re-training the tokenizer later
# invalidates every checkpoint trained against it.
cd "/Users/daffanaufalrachmat/Documents/Real Projects/TransformerSample/transformer_100m_cuda" || exit 1

echo "=== waiting for corpus build ==="
while ! grep -q "CORPUS BUILD DONE" logs/corpus_rebuild.log 2>/dev/null; do
  if ! pgrep -f "run_corpus_rebuild.sh" > /dev/null; then
    echo "[ERR] rebuild script exited without finishing — aborting"
    tail -20 logs/corpus_rebuild.log
    exit 1
  fi
  sleep 120
done
echo "corpus ready at $(date '+%H:%M:%S')"
echo

for b in japanese english code math mixed; do
  printf "  %-9s %s\n" "$b" "$(du -sh "data/corpus_pretrain_v8/$b" 2>/dev/null | cut -f1)"
done
echo

echo "=== training BPE -> vocab_8 ==="
mkdir -p bpe_checkpoints
# mixed/ is excluded on purpose: its facts are repeated 1000x each, so a
# frequency-driven merge learner would spend slots on template phrases.
/tmp/train_bpe \
  --out_dir vocab_8 \
  --vocab_size 49152 \
  --min_freq 15 \
  --corpus data/corpus_pretrain_v8/japanese:2000 \
  --corpus data/corpus_pretrain_v8/english:1300 \
  --corpus data/corpus_pretrain_v8/code:1100 \
  --corpus data/corpus_pretrain_v8/math:600 \
  --binary bpe_v8.bin \
  --checkpoint bpe_checkpoints/bpe_v8_ckpt.bin \
  --resume
rc=$?
echo "BPE exit=$rc at $(date '+%H:%M:%S')"
[ $rc -ne 0 ] && exit $rc

echo
echo "=== preflight ==="
python3 scripts/preflight_pretrain.py \
  --vocab_dir vocab_8 --bpe bpe_v8.bin \
  --data_dir data/corpus_pretrain_v8 --max_seq 512
echo "PIPELINE DONE at $(date '+%H:%M:%S')"
