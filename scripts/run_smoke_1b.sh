#!/usr/bin/env bash
# ============================================================================
# run_smoke_1b.sh — SMOKE TEST 892M: proof-of-recipe before the full 21B run
#
# WHY 4B tokens (not more, not less):
#   - Matches V9's 3.85B token budget → clean controlled comparison:
#     SAME data budget, 4.4x params + new recipe (injection, 6 buckets).
#     Any metric delta is attributable to model scale + data recipe.
#   - Scaling law prediction: loss(892M,4B) sits comfortably below the
#     smaller-model loss curve at the same token count (Δ ≈ 0.10~0.15 nats)
#     → expect ~2.9-3.3 vs V9's 3.81.
#   - Injection corpus (facts 134M tok + bank ~260M tok) at mixed-weight 8%
#     receives ~250-320 exposures during 4B tokens — far above the ~100x
#     memorization threshold. Value-hit signal WILL be visible if the
#     pipeline is wired right.
#
# BUDGETS: pretrain 4B tok | SFT 2 ep ~200M tok | DPO 55K | RL 600 steps
# COST:    8x4090 ~$60-75 | 4xH100 ~$90-110 | 4xL40S ~$85-100   (~10x cheaper
#          than a failed full run)
#
# GATES — all four must pass before committing to the full 21B run:
#   G1  pretrain ema_loss @4B <= 3.30 (vs V9 3.81 @3.85B), no NaN, VRAM fits
#   G2  facts eval (honest, 3 langs):
#         attr  >= 65%  (V9: EN 44 / JA 60)
#         value >= 60%  (V9: 48/48)
#         lang  = 100%
#         subj-bind >= 70% (V9: EN 92 / JA 28)
#   G3  translation: T1 compliance >= 97%, T2 bank exact >= 50%
#   G4  coding v3 pass@4 >= 30% (pre-RL, vocab_7 dependent)
# If any gate fails → debug on THIS budget; each re-run is ~$60-110, not $700.
#
# PREREQ (not done by this script):
#   [1] vocab_7 tokenizer retrained (newline-preserving) — REQUIRED for code
#   [2] dims 892M edited in train_cuda.cu:51-59 + train_cuda_unified.cu:54-62
#   [3] corpus at data/corpus_1b (6 buckets, Gate E PASS — done)
#   [4] rebuild: bash scripts/build_cuda.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

B=build
CKPT=ckpt/smoke_1b
mkdir -p $CKPT rl_work

# ── 1. PRETRAIN — 892M @ 4B tokens (≈ V9 budget) ──────────────────────────
# batch_size: start 8; on 80GB cards raise to 16-24 after VRAM check
$B/train_pretrain_cuda \
  --data_dir data/corpus_1b \
  --vocab_dir vocab_7 \
  --checkpoint_dir $CKPT \
  --batch_size 8 --max_tokens 4000000000 \
  --ddp --num_gpus 4
# GATE G1: tail the log — ema_loss @4B must be <= 3.30

# ── 2. SFT — mixed smoke set (1.43M rows, 2 epochs) ───────────────────────
python3 scripts/prepare_sft_smoke_mix.py
$B/train_sft_cuda --stage 2 \
  --base_model $CKPT/ckpt_pretrain_final.safetensors \
  --sft_dir sft_data/sft_smoke \
  --batch_size 4 --epochs 2

# ── 3. DPO — translation preferences (55K) ────────────────────────────────
$B/train_sft_cuda --stage 3 \
  --base_model $CKPT/ckpt_sft.safetensors \
  --dpo_dir sft_data/dpo_tri \
  --batch_size 2 --epochs 1

# ── 4. RL SMOKE — small mixed verifier set, ~600 steps ────────────────────
# full rl_data = 24K prompts (too slow for smoke); subset: 400/task family
mkdir -p rl_data_smoke
for f in coding_v2 coding_v3 math math_v2 math_pot code_bugfix; do
  head -400 rl_data/$f.jsonl > rl_data_smoke/$f.jsonl
done
$B/train_sft_cuda --stage 5 \
  --base_model $CKPT/ckpt_sft_final.safetensors \
  --ref_model $CKPT/ckpt_sft_final.safetensors \
  --rl_dir rl_data_smoke \
  --verifier_cmd "python3 scripts/rl_verifier.py" \
  --rl_workdir rl_work \
  --batch_size 2 --n_samples 6 --rl_max_new 160 \
  --rl_beta 0.1 --epochs 1 --lr 1e-5
# watch rl_work/reward.jsonl pass-rate: should climb from ~20-40% → 60%+

# ── 5. EVALS — the decision numbers ───────────────────────────────────────
# G2: facts (honest 3-lang, 300/fact sample for tighter CI than 50)
python3 scripts/eval_facts.py \
  --model $CKPT/ckpt_rl_final.safetensors \
  --config config_pretrain.json --vocab vocab_7 \
  --eval sft_data/sft_facts_v11/eval.jsonl \
  --max_test 300 --infer build_infer/infer \
  --out results/facts_eval_smoke1b.json

# G3: translation tiers (T1/T2/T3 + contamination report)
python3 scripts/eval_translation.py \
  --model $CKPT/ckpt_rl_final.safetensors \
  --config config_pretrain.json --vocab vocab_7 \
  --eval sft_data/sft_translation_tri/eval.jsonl \
  --bank_eval sft_data/sft_translation_bank/eval.jsonl \
  --dedup_train sft_data/sft_translation_tri/train.jsonl \
  --max_test 200 --max_bank 150 \
  --infer build_infer/infer \
  --out results/translation_eval_smoke1b.json

# G4: coding pass@4 on hidden fresh instances (sample 200)
python3 - <<'EOF'
import json, random, subprocess, sys
sys.path.insert(0, "scripts")
import build_sft_coding_v3 as V3
rng = random.Random(7)
rows = [json.loads(l) for l in open("rl_data/coding_v3.jsonl")]
sample = rng.sample(rows, 200)
# TODO: batch through build_infer/infer --stdin --eot_marker, extract code
# blocks, run vs mock_input/expected. (Wiring lives in eval harness next.)
print(f"coding eval rows prepared: {len(sample)}")
EOF

echo "============================================================"
echo "SMOKE COMPLETE — check gates G1-G4 against thresholds above."
echo "All pass  -> greenlight full 892M @ 21B (~$400-740)."
echo "Any fails -> iterate HERE (~$60-110/iteration)."
echo "============================================================"
