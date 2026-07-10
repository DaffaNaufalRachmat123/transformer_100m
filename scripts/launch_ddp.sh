#!/bin/bash
# =============================================================================
# launch_ddp.sh — Launch DDP training (N processes, 1 GPU each)
#
# Usage:
#   ./scripts/launch_ddp.sh ./build/train_pretrain_cuda [args...]
#   ./scripts/launch_ddp.sh ./build/train_sft_cuda --stage 2 [args...]
#   NUM_GPUS=4 ./scripts/launch_ddp.sh ./build/train_pretrain_cuda [args...]
#
# Auto-detects available GPUs if NUM_GPUS is not set.
# Each process gets RANK, WORLD_SIZE, LOCAL_RANK environment variables.
# NCCL uses shared memory for inter-GPU communication (fastest on same node).
# =============================================================================
set -e

# Auto-detect GPU count if not specified
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -lt 1 ]; then
        echo "[ERROR] No GPUs detected. Set NUM_GPUS manually."
        exit 1
    fi
fi

BINARY="$1"
shift  # Remove binary from args

if [ ! -f "$BINARY" ]; then
    echo "[ERROR] Binary not found: $BINARY"
    echo "Usage: $0 <binary> [training args...]"
    echo "  Set NUM_GPUS=N to override GPU count (default: auto-detect)"
    exit 1
fi

echo "═══════════════════════════════════════════════════════"
echo " DDP Launch: $NUM_GPUS GPUs"
echo " Binary: $BINARY"
echo " Args: $@"
echo "═══════════════════════════════════════════════════════"

# NCCL optimizations for same-node multi-GPU
export NCCL_IB_DISABLE=1          # No InfiniBand on Kaggle
export NCCL_P2P_DISABLE=0         # Enable peer-to-peer (PCIe/NVLink)
export NCCL_SHM_DISABLE=0         # Enable shared memory
export NCCL_SOCKET_IFNAME=lo      # Use loopback for control

# Generate unique NCCL ID file
export NCCL_ID_FILE="/tmp/nccl_id_transformer_100m"

# Array to hold background PIDs
declare -a PIDS=()

# Launch ranks 1..N-1 in background
for ((r=1; r<NUM_GPUS; r++)); do
    RANK=$r WORLD_SIZE=$NUM_GPUS LOCAL_RANK=$r \
        CUDA_VISIBLE_DEVICES=$r \
        "$BINARY" "$@" --ddp --num_gpus $NUM_GPUS &
    PIDS+=($!)
done

# Small delay to let workers start and write NCCL ID
sleep 0.5

# Launch rank 0 (foreground)
RANK=0 WORLD_SIZE=$NUM_GPUS LOCAL_RANK=0 \
    CUDA_VISIBLE_DEVICES=0 \
    "$BINARY" "$@" --ddp --num_gpus $NUM_GPUS
EXIT_CODE=$?

# Wait for all background ranks
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# Cleanup
rm -f "$NCCL_ID_FILE"

echo ""
echo "═══════════════════════════════════════════════════════"
echo " DDP Training complete (exit code: $EXIT_CODE)"
echo "═══════════════════════════════════════════════════════"
exit $EXIT_CODE
