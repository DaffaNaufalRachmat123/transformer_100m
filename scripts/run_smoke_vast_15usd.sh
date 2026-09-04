#!/usr/bin/env bash
# ============================================================================
# run_smoke_vast_15usd.sh — $15 SMOKE TEST: 892M MODEL END-TO-END (vast.ai)
#
# WHAT THIS RUN PROVES (in priority order):
#   G0  ENGINEERING: 892M dims (D=1280/L=34/GQA 20:5/FFN 5120) compile,
#       train, checkpoint, SFT-load, infer — no NaN, VRAM fits.
#       -> de-risks the $400-700 full 21B run
#   G0b MEASURED tok/s at 892M -> precise cost extrapolation for the 4B
#       validation run and the full 21B run (no more estimates).
#   G1  pretrain ema_loss <= 4.35 @ ~250M tokens and still falling
#       (892M @ 250M is heavily undertrained by design — this gate only
#        proves optimization works, not quality)
#   G2  FACTS RECIPE (post-SFT, honest eval 3 langs) — the real gate:
#         value     >= 40%   (previous smaller-model baselines were 20/40)
#         attr      >= 50%   (previous baselines were 20/60)
#         lang      = 100%
#         JA subj-bind >= 50% (previous baseline was 10%)
#       Pass => injection+bank recipe works AT 892M; full budget will scale it.
#   G3  translation: T1 compliance >= 95%, T2 bank exact >= 35%
#
# GPU CHOICE (vast.ai console filters, on-demand):
#   1x A100 PCIe 40GB, $/hr <= 0.85, reliability >= 0.99, CUDA 12.x,
#   disk >= 60GB. NOTE: A100 PCIe 40GB = HALF the 80GB/SXM compute
#   (156 TFLOPS BF16 dense) + no NVLink — fine for single-GPU smoke, but
#   measure real tok/s in P1 to size the full run.
#   Memory @ seq 2048: static 14.3GB + logits(B4x2048x65k) 1.1GB + activations
#   ~2.5GB ≈ 18GB — 40GB holds batch 4-8 comfortably.
#
# BUDGET LEDGER (1x A100-40GB PCIe @ $0.80/hr, $15 = 18.75 h):
#   P0 build (arch auto) + checks     0.5h   $0.40
#   P1 pretrain 892M ~250M tok        15.5h  $12.40  (~4.5K tok/s expected)
#   P2 SFT 1 epoch (75M tok)          2.0h   $1.60
#   P3 DPO 55K rows                   0.4h   $0.32
#   P4 evals                          0.3h   $0.24
#   TOTAL                            18.7h  ~$14.96  (buffer ~$0)
# The script auto-computes P1's token budget from money actually remaining —
# stop it any time; checkpoints save every 500 steps.
#
# VRAM LADDER (on OOM halve batch, restart, resume):
#   pretrain: --batch_size 6 --max_seq 2048  -> OOM? 4/2048 -> OOM? 2/2048
#   SFT:      --batch_size 2                 -> OOM? 1
#   DPO:      --batch_size 2                 -> OOM? 1
#
# LOCAL PREP (laptop, before renting — data/ below must be uploaded):
#   bash scripts/make_smoke_corpus.sh                      # data/corpus_smoke 6.4GB
#   python3 scripts/prepare_sft_smoke_mix.py --vocab6      # sft_smoke_v6 1.29M rows
#   rsync -aP data/corpus_smoke  vast:/root/project/data/
#   rsync -aP sft_data/sft_smoke_v6 sft_data/sft_facts_v11 \
#             sft_data/sft_translation_tri sft_data/sft_translation_bank \
#             sft_data/dpo_tri vocab_6 bpe transformer cuda tools scripts \
#             CMakeLists.txt config_pretrain.json vast:/root/project/
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

HOURLY=${HOURLY:-0.80}          # set to your instance's $/hr
BUDGET=${BUDGET:-15.0}
SPENT_FILE=.smoke_spent
touch $SPENT_FILE
SPENT=$(cat $SPENT_FILE 2>/dev/null || echo 0)
T0=$(date +%s)
mkdir -p ckpt/smoke results logs

phase_cost () {  # phase_cost <name>
  local now=$(date +%s)
  local c=$(python3 -c "print(($now-$T0)/3600*$HOURLY)")
  T0=$now
  SPENT=$(python3 -c "print($SPENT+$c)")
  echo "$SPENT" > $SPENT_FILE
  echo "[$1] +\$${c}  total=\$$SPENT / \$$BUDGET"
  python3 -c "exit(0 if $SPENT < $BUDGET else 1)" || { echo "BUDGET EXCEEDED — stopping (checkpoints saved)."; exit 2; }
}
hours_left () { python3 -c "print(max(0, ($BUDGET-$SPENT)/$HOURLY))"; }

# ── P0: build (auto-detects sm_80/sm_89 from the GPU) ─────────────────────
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
bash scripts/build_cuda.sh 2>&1 | tail -5
phase_cost "P0 build"

# ── P1: PRETRAIN 892M — token budget from money remaining ─────────────────
# Remaining hours minus SFT(2.0) + DPO(0.4) + eval(0.3) reservations.
# Expected throughput (this codebase, grad-ckpt ~7.14 GFLOP/tok, seq 2048):
#   A100 PCIe 40GB @ 15-22% MFU (156 TFLOPS dense) -> 3.5-4.8K tok/s.
#   WATCH the first tok/s log line: it sizes the real 21B run cost.
TPS=${TPS:-4200}                                  # conservative A100-40G estimate
PRETRAIN_HOURS=$(python3 -c "print(max(1.0, $(hours_left) - 2.7))")
MAX_TOKENS=$(python3 -c "print(int($PRETRAIN_HOURS*3600*$TPS))")
echo "[P1] pretrain budget: ${PRETRAIN_HOURS}h -> ${MAX_TOKENS} tokens @ ${TPS} tok/s"

# Smoke weights: code/math zeroed (vocab_6 strips newlines / math source gone),
# mixed boosted to 0.22 -> injection corpus gets ~130-250 exposures per fact
# rendering (above the ~100x memorization threshold).
export TW_W_EN=0.33 TW_W_JA=0.25 TW_W_ID=0.20 TW_W_CODE=0.00 TW_W_MATH=0.00 TW_W_MIXED=0.22

# LR: 1e-4 (scaled down for 892M vs the older smaller model's 1.5e-4).
# The trainer's auto-decay safety net catches drift if this is still hot.
./build/train_pretrain_cuda \
  --data_dir data/corpus_smoke \
  --vocab_dir vocab_6 \
  --checkpoint_dir ckpt/smoke \
  --batch_size 6 --max_seq 2048 --lr 1e-4 \
  --max_tokens ${MAX_TOKENS}
# single-GPU instance: no DDP flags needed.
# OOM? drop --batch_size to 4 then 2 (see VRAM LADDER).
unset TW_W_EN TW_W_JA TW_W_ID TW_W_CODE TW_W_MATH TW_W_MIXED
phase_cost "P1 pretrain"

# ── P2: SFT smoke mix (1.29M rows, vocab6-filtered, 1 epoch) ──────────────
./build/train_sft_cuda --stage 2 \
  --base_model ckpt/smoke/ckpt_pretrain_final.safetensors \
  --sft_dir sft_data/sft_smoke_v6 \
  --batch_size 4 --epochs 1 --lr 1e-4
phase_cost "P2 SFT"

# ── P3: DPO (translation preferences, 55K) ────────────────────────────────
./build/train_sft_cuda --stage 3 \
  --base_model ckpt/smoke/ckpt_sft.safetensors \
  --dpo_dir sft_data/dpo_tri \
  --batch_size 2 --epochs 1 --lr 5e-6
phase_cost "P3 DPO"

# ── P4: EVALS — the decision numbers ──────────────────────────────────────
# G2: facts (honest, 3 langs, 150 samples)
python3 scripts/eval_facts.py \
  --model ckpt/smoke/ckpt_sft_final.safetensors \
  --config config_pretrain.json --vocab vocab_6 \
  --eval sft_data/sft_facts_v11/eval.jsonl \
  --max_test 150 --infer build_infer/infer \
  --out results/facts_eval_smoke_892m.json

# G3: translation tiers
python3 scripts/eval_translation.py \
  --model ckpt/smoke/ckpt_sft_final.safetensors \
  --config config_pretrain.json --vocab vocab_6 \
  --eval sft_data/sft_translation_tri/eval.jsonl \
  --bank_eval sft_data/sft_translation_bank/eval.jsonl \
  --max_test 120 --max_bank 100 \
  --infer build_infer/infer \
  --out results/translation_eval_smoke_892m.json
phase_cost "P4 evals"

# ── Report ─────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "892M SMOKE DONE — spent \$$SPENT of \$$BUDGET"
echo ""
echo "Measured tok/s (from P1 log): \$TPS -> full 21B run cost projection:"
python3 - <<EOF
tps_real = ${TPS}   # REPLACE with the value observed in the P1 log line
for cfg, n, rate in [("1x A100-40G", 1, 0.80), ("2x A100-40G", 2, 1.60), ("4x A100-40G", 4, 3.20), ("4x A100-80G", 4, 5.00), ("4x H100", 4, 11.96)]:
    # scale: measured single-GPU tok/s x count (DDP near-linear)
    hours = 21e9 / (tps_real * n) / 3600
    print(f"  {cfg:<14} {hours:5.1f}h  ~\${hours*rate:,.0f}")
EOF
echo ""
echo "GATES: G1 loss<=4.35 | G2 value>=40 attr>=50 lang=100 JA-bind>=50"
echo "      | G3 T1>=95 T2>=35"
echo "All pass -> next: 4B-token 892M validation run (vocab_7), then 21B."
echo "=================================================================="
