#pragma once
// =============================================================================
// cuda_ddp.h — NCCL-based Data-Parallel (multi-process, one GPU per process)
//
// Supports two launch modes:
//   1. Multi-process: each process owns one GPU (env RANK/WORLD_SIZE/LOCAL_RANK)
//      Launch via: scripts/launch_ddp.sh
//   2. Single-process fallback: forks internally (simpler for Kaggle notebooks)
//
// After init(), each rank:
//   - Computes forward/backward on its batch slice
//   - AllReduces gradients via NCCL
//   - Runs optimizer locally (weights stay in sync)
//   - Only rank 0 saves checkpoints
// =============================================================================

#ifdef USE_DDP

#include <nccl.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include "cuda_tensor.h"

namespace cuda {
namespace ddp {

// ─── Global DDP state ──────────────────────────────────────────────────────
struct DDPState {
    int world_size = 1;
    int rank = 0;
    int local_rank = 0;
    ncclComm_t comm = nullptr;
    cudaStream_t nccl_stream = nullptr;
    bool initialized = false;
    bool is_forked = false;     // true if we forked internally
    pid_t child_pid = -1;       // for single-process launcher
};

inline DDPState& state() {
    static DDPState s;
    return s;
}

inline int rank() { return state().rank; }
inline int world_size() { return state().world_size; }
inline bool is_main() { return state().rank == 0; }

// ─── Initialize from environment (multi-process mode) ──────────────────────
// Expects: RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
// Set by launch_ddp.sh or mpirun
inline bool init_from_env() {
    auto& s = state();
    const char* rank_str = getenv("RANK");
    const char* world_str = getenv("WORLD_SIZE");
    const char* local_str = getenv("LOCAL_RANK");

    if (!rank_str || !world_str) return false;

    s.rank = atoi(rank_str);
    s.world_size = atoi(world_str);
    s.local_rank = local_str ? atoi(local_str) : s.rank;
    return true;
}

// ─── Initialize NCCL (called after CUDA context is set) ────────────────────
inline bool init(int num_gpus) {
    auto& s = state();
    if (s.initialized) return true;

    // ── STEP 1: establish rank/world WITHOUT touching CUDA ──────────────────
    // CUDA does NOT survive fork(), so the fork must happen before any CUDA
    // call. This function must be invoked BEFORE cuda::ctx.init().
    if (!init_from_env()) {
        if (num_gpus >= 2) {
            // Fork N-1 children for ranks 1..N-1
            s.world_size = num_gpus;
            s.is_forked = true;
            s.rank = 0;
            s.local_rank = 0;

            for (int r = 1; r < num_gpus; r++) {
                pid_t pid = fork();
                if (pid < 0) {
                    fprintf(stderr, "[DDP] fork() failed for rank %d\n", r);
                    return false;
                }
                if (pid == 0) {
                    // Child: I am rank r
                    s.rank = r;
                    s.local_rank = r;
                    s.child_pid = -1;
                    break;  // Children don't fork further
                }
                // Parent: record first child PID (for legacy compat), continue forking
                if (r == 1) s.child_pid = pid;
            }
        } else {
            s.rank = 0; s.local_rank = 0; s.world_size = 1;
        }
    }
    return true;  // NCCL comm is created later in init_nccl()
}

// ─── local device id for this rank (call before ctx.init) ──────────────────
inline int device_for_rank() {
    auto& s = state();
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    // If both GPUs are visible, map rank->physical GPU. If CUDA_VISIBLE_DEVICES
    // pins one GPU per process, device 0 is correct.
    return (device_count > 1) ? s.local_rank : 0;
}

// ─── STEP 2: create the NCCL communicator (call AFTER cuda::ctx.init) ───────
inline bool init_nccl() {
    auto& s = state();
    if (s.initialized) return true;

    ncclUniqueId nccl_id;
    if (s.rank == 0) ncclGetUniqueId(&nccl_id);

    if (s.is_forked) {
        const char* id_file = "/tmp/nccl_id_transformer_100m";
        if (s.rank == 0) {
            FILE* f = fopen(id_file, "wb");
            if (f) { fwrite(&nccl_id, sizeof(nccl_id), 1, f); fclose(f); }
            // Wait enough for all children to read
            usleep(200000 + (s.world_size - 2) * 50000);
        } else {
            // Children wait for rank 0 to write the ID file
            usleep(50000 * s.rank);  // stagger reads to reduce contention
            FILE* f = fopen(id_file, "rb");
            while (!f) { usleep(10000); f = fopen(id_file, "rb"); }
            (void)fread(&nccl_id, sizeof(nccl_id), 1, f); fclose(f);
        }
    }

    ncclResult_t res = ncclCommInitRank(&s.comm, s.world_size, nccl_id, s.rank);
    if (res != ncclSuccess) {
        fprintf(stderr, "[DDP] ncclCommInitRank failed (rank=%d): %s\n",
                s.rank, ncclGetErrorString(res));
        return false;
    }
    cudaStreamCreate(&s.nccl_stream);
    s.initialized = true;
    printf("[DDP] Rank %d/%d on GPU %d — NCCL ready\n", s.rank, s.world_size, s.local_rank);
    return true;
}

// ─── Shutdown ──────────────────────────────────────────────────────────────
inline void shutdown() {
    auto& s = state();
    if (!s.initialized) return;

    ncclCommDestroy(s.comm);
    cudaStreamDestroy(s.nccl_stream);

    // If we forked, parent waits for all children
    if (s.is_forked && s.rank == 0) {
        // Wait for all child processes (world_size - 1 children)
        for (int i = 0; i < s.world_size - 1; i++) {
            int wstatus;
            wait(&wstatus);
        }
        unlink("/tmp/nccl_id_transformer_100m");
    }

    // Children exit after shutdown
    if (s.is_forked && s.rank != 0) {
        _exit(0);
    }

    s.initialized = false;
}

// ─── AllReduce FP16 (in-place, SUM) on the NCCL stream ────────────────────
inline void allreduce_fp16(void* data, size_t count) {
    auto& s = state();
    ncclAllReduce(data, data, count, ncclFloat16, ncclSum,
                  s.comm, s.nccl_stream);
}

// ─── AllReduce FP32 (in-place, SUM) ───────────────────────────────────────
inline void allreduce_fp32(void* data, size_t count) {
    auto& s = state();
    ncclAllReduce(data, data, count, ncclFloat, ncclSum,
                  s.comm, s.nccl_stream);
}

// ─── AllReduce all gradients (grouped for bandwidth) ───────────────────────
inline void allreduce_grads(std::vector<CudaTensor*>& grad_tensors) {
    auto& s = state();
    ncclGroupStart();
    for (auto* g : grad_tensors) {
        if (!g || !g->data || g->numel() == 0) continue;
        ncclDataType_t dtype = (g->dtype == DType::FP16) ? ncclFloat16 : ncclFloat;
        ncclAllReduce(g->data, g->data, g->numel(), dtype, ncclSum,
                      s.comm, s.nccl_stream);
    }
    ncclGroupEnd();
}

// ─── Synchronize NCCL stream ──────────────────────────────────────────────
inline void sync() {
    cudaStreamSynchronize(state().nccl_stream);
}

// ─── Scale FP16 buffer by 1/world_size (average after SUM) ────────────────
inline void average_grad(CudaTensor& grad) {
    if (grad.numel() == 0 || !grad.data) return;
    float scale = 1.0f / (float)state().world_size;
    // scale_fp16 is in cuda:: namespace (one level up)
    ::cuda::scale_fp16(grad, scale);
}

// ─── Broadcast tensor from rank 0 ─────────────────────────────────────────
inline void broadcast(void* data, size_t count, ncclDataType_t dtype) {
    auto& s = state();
    ncclBroadcast(data, data, count, dtype, 0, s.comm, s.nccl_stream);
}

} // namespace ddp
} // namespace cuda

#endif // USE_DDP
