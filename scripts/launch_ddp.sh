#!/bin/bash
# =============================================================================
# launch_ddp.sh — Launch DDP training (2 processes, 1 GPU each)
#
# Usage:
#   ./scripts/launch_ddp.sh ./build/train_pretrain_cuda [args...]
#   ./scripts/launch_ddp.sh ./build/train_sft_cuda --stage 2 [args...]
#
# Each process gets RANK, WORLD_SIZE, LOCAL_RANK environment variables.
# NCCL uses shared memory for inter-GPU communication (fastest on same node).
# =============================================================================
set -e

NUM_GPUS=2
BINARY="$1"
shift  # Remove binary from args

if [ ! -f "$BINARY" ]; then
    echo "[ERROR] Binary not found: $BINARY"
    echo "Usage: $0 <binary> [training args...]"
    exit 1
fi

echo "═══════════════════════════════════════════════════════"
echo " DDP Launch: $NUM_GPUS GPUs"
echo " Binary: $BINARY"
echo " Args: $@"
echo "═══════════════════════════════════════════════════════"

# NCCL optimizations for same-node T4 x2
export NCCL_IB_DISABLE=1          # No InfiniBand on Kaggle
export NCCL_P2P_DISABLE=0         # Enable peer-to-peer (PCIe)
export NCCL_SHM_DISABLE=0         # Enable shared memory
export NCCL_SOCKET_IFNAME=lo      # Use loopback for control

# Generate unique NCCL ID file
export NCCL_ID_FILE="/tmp/nccl_id_transformer_100m"

# Launch rank 1 (background)
RANK=1 WORLD_SIZE=$NUM_GPUS LOCAL_RANK=1 \
    CUDA_VISIBLE_DEVICES=1 \
    "$BINARY" "$@" --ddp --num_gpus $NUM_GPUS &
PID_RANK1=$!

# Small delay to let rank 1 start
sleep 0.5

# Launch rank 0 (foreground)
RANK=0 WORLD_SIZE=$NUM_GPUS LOCAL_RANK=0 \
    CUDA_VISIBLE_DEVICES=0 \
    "$BINARY" "$@" --ddp --num_gpus $NUM_GPUS
EXIT_CODE=$?

# Wait for rank 1
wait $PID_RANK1 2>/dev/null || true

# Cleanup
rm -f "$NCCL_ID_FILE"

echo ""
echo "═══════════════════════════════════════════════════════"
echo " DDP Training complete (exit code: $EXIT_CODE)"
echo "═══════════════════════════════════════════════════════"
exit $EXIT_CODE
