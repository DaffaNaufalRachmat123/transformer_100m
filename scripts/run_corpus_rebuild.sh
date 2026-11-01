#!/bin/bash
# Waits for the in-flight fetch jobs, then rebuilds the pretraining corpus.
# Written as a script rather than a chain of interactive commands because the
# build takes hours and must not die with the shell.
cd "/Users/daffanaufalrachmat/Documents/Real Projects/TransformerSample/transformer_100m_cuda" || exit 1

echo "=== waiting for fetch jobs ==="
while pgrep -f "fetch_corpus.py --dataset" > /dev/null; do
  sleep 60
done
echo "all fetches finished at $(date '+%H:%M:%S')"
echo
echo "=== code_raw inventory ==="
for f in data/code_raw/*.txt; do
  sep=$(head -c 2000000 "$f" | grep -c '^<|endoftext|>$')
  printf "  %-24s %6s MB  separators in first 2MB: %s\n" \
         "$(basename "$f")" "$(( $(stat -f%z "$f") / 1000000 ))" "$sep"
done
echo
echo "=== rebuilding corpus ==="
rm -rf data/corpus_pretrain_v8
python3 -u scripts/build_pretrain_corpus.py \
  --out data/corpus_pretrain_v8 --progress-secs 45
echo
echo "CORPUS BUILD DONE at $(date '+%H:%M:%S')"
du -sh data/corpus_pretrain_v8/*/
