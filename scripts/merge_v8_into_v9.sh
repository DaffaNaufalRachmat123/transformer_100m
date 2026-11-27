#!/usr/bin/env bash
# =============================================================================
#  merge_v8_into_v9.sh — OPTIONAL helper for LATER when you are ready to run
#  the full 50B pipeline (Pretrain + SFT + DPO + GRPO) using the v9 folders!
#
#  Combines v8 + v9 inside the *_v9* directories via 0-byte APFS/Linux hardlinks
#  WITHOUT ever modifying any v8 folder:
#    - data/corpus_pretrain_v8   -> data/corpus_pretrain_v9
#    - sft_data/sft_mixed_enja   -> sft_data/sft_v9_50b/v8_base_train.jsonl
#    - sft_data/dpo_ceiling_enja -> sft_data/dpo_v9_50b/v8_base_train.jsonl
#    - rl_data/ceiling_enja      -> rl_data/rl_v9_50b/v8_base_train.jsonl
#
#  Usage (ONLY when ready to run 50B training):
#    bash scripts/merge_v8_into_v9.sh --hardlink-v8-into-v9
#    bash scripts/merge_v8_into_v9.sh --remove-v8-links-from-v9
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

V8_CORPUS="$PROJECT_DIR/data/corpus_pretrain_v8"
V9_CORPUS="$PROJECT_DIR/data/corpus_pretrain_v9"

MODE="${1:-}"
case "$MODE" in
  --hardlink-v8-into-v9)
    count=0
    for bucket in japanese english code math mixed; do
      mkdir -p "$V9_CORPUS/$bucket"
      for f in "$V8_CORPUS/$bucket"/*.txt; do
        [ -f "$f" ] || continue
        base="$(basename "$f")"
        [[ "$base" == v9_* ]] && continue
        ln -f "$f" "$V9_CORPUS/$bucket/$base"
        count=$((count + 1))
      done
    done
    [ -f "$PROJECT_DIR/sft_data/sft_mixed_enja/train.jsonl" ] && \
      ln -f "$PROJECT_DIR/sft_data/sft_mixed_enja/train.jsonl" "$PROJECT_DIR/sft_data/sft_v9_50b/v8_base_train.jsonl"
    [ -f "$PROJECT_DIR/sft_data/dpo_ceiling_enja/train.jsonl" ] && \
      ln -f "$PROJECT_DIR/sft_data/dpo_ceiling_enja/train.jsonl" "$PROJECT_DIR/sft_data/dpo_v9_50b/v8_base_train.jsonl"
    [ -f "$PROJECT_DIR/rl_data/ceiling_enja/train.jsonl" ] && \
      ln -f "$PROJECT_DIR/rl_data/ceiling_enja/train.jsonl" "$PROJECT_DIR/rl_data/rl_v9_50b/v8_base_train.jsonl"
    echo "[OK] Linked $count pretrain v8 shards + SFT/DPO/RL v8 base sets into v9 folders (0 extra bytes; all v8 folders untouched)."
    ;;

  --remove-v8-links-from-v9)
    count=0
    for bucket in japanese english code math mixed; do
      for f in "$V9_CORPUS/$bucket"/*.txt; do
        [ -e "$f" ] || continue
        base="$(basename "$f")"
        [[ "$base" == v9_* ]] && continue
        rm -f "$f"
        count=$((count + 1))
      done
    done
    rm -f "$PROJECT_DIR/sft_data/sft_v9_50b/v8_base_train.jsonl" \
          "$PROJECT_DIR/sft_data/dpo_v9_50b/v8_base_train.jsonl" \
          "$PROJECT_DIR/rl_data/rl_v9_50b/v8_base_train.jsonl"
    echo "[OK] Removed $count pretrain v8 links + SFT/DPO/RL v8 links from v9 folders (pure v9 add-ons remain)."
    ;;

  *)
    echo "Usage: $0 [--hardlink-v8-into-v9 | --remove-v8-links-from-v9]"
    exit 1
    ;;
esac
