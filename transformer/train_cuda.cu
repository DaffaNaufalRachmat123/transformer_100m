// =============================================================================
// train_cuda.cu — 100M Transformer Pretraining (CUDA GPU)
//
// Architecture: GQA + RoPE + RMSNorm + SwiGLU + Weight Tying
//   D=576, L=13, NQH=9, NKV=3, DH=64, FFN=2304, V=65536
//
// Key features:
//   - Mixed precision FP16 (T4 compatible, sm_75)
//   - Gradient checkpointing (store only layer inputs)
//   - cuDNN fused attention
//   - Fused AdamW optimizer kernel
//   - DDP via NCCL (--ddp flag)
//   - Adaptive checkpointing (overwrite mode for Kaggle)
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <csignal>
#include <random>
#include <algorithm>

#include "cuda/cuda_tensor.h"
#include "cuda/cuda_context.h"
#include "cuda/cuda_ops.h"
#include "cuda/cuda_kernels.cuh"
#include "cuda/cuda_monitor.h"
#include "cuda/cuda_autograd.h"
#include "bpe/bpe_tokenizer.h"
#include "transformer/model_config.h"
#include "transformer/pretrain_dataset.h"
#include "transformer/safetensors.h"
#include "transformer/train_step.cuh"
#include "cuda/ce_embed_kernels.cuh"

#ifdef USE_DDP
#include "cuda/cuda_ddp.h"
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

// ─── Architecture constants (100M model) ───────────────────────────────────
static constexpr int D_MODEL    = 576;
static constexpr int N_LAYERS   = 13;
static constexpr int N_Q_HEADS  = 9;
static constexpr int N_KV_HEADS = 3;
static constexpr int D_HEAD     = D_MODEL / N_Q_HEADS;  // 64
static constexpr int FFN_DIM    = 2304;
static constexpr int MODEL_VOCAB = 65536;
static constexpr float ROPE_THETA = 500000.0f;

// ─── CLI config ────────────────────────────────────────────────────────────
static std::string g_data_dir       = "data/corpus_final";
static std::string g_checkpoint_dir = "checkpoints";
static std::string g_vocab_dir      = "vocab_6";
static std::string g_bpe_path       = "";  // empty = use vocab_dir text format
static std::string g_optimizer      = "adamw";
static uint64_t g_max_tokens        = 2'000'000'000ULL;
static int g_batch_size             = 8;   // T4 16GB fits 8 easily for 100M
static int g_max_seq                = 256;
static bool g_resume                = false;
static bool g_use_ddp               = false;
static int  g_num_gpus              = 2;

// ─── AdamW hyperparams ─────────────────────────────────────────────────────
static float g_lr          = 3e-4f;
static float g_beta1       = 0.9f;
static float g_beta2       = 0.95f;
static float g_eps         = 1e-8f;
static float g_wd          = 0.1f;
static float g_grad_clip   = 1.0f;

// ─── Loss scaling (prevents FP16 gradient underflow) ───────────────────────
static float g_loss_scale  = 1.0f;   // 1.0 = no scaling; 1024 recommended for FP16
static int   g_debug_steps = 0;      // 0 = disabled; N = dump diag every N steps

// ─── Checkpoint config ─────────────────────────────────────────────────────
static int g_save_every    = 500;
static int g_log_every     = 10;
static bool g_overwrite_ckpt = true;  // Kaggle mode: single ckpt overwritten

// ─── Global state ──────────────────────────────────────────────────────────
static volatile bool g_stop_requested = false;
static uint64_t g_tokens_processed = 0;
static int g_current_phase = 0;

void signal_handler(int) { g_stop_requested = true; }

// ─── Helpers ───────────────────────────────────────────────────────────────
static float get_rss_mb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("VmRSS:") == 0) {
            long kb = std::stol(line.substr(6));
            return kb / 1024.0f;
        }
    }
    return 0.0f;
}

std::string fmt_tok(uint64_t n) {
    char buf[32];
    if (n >= 1'000'000'000) snprintf(buf, sizeof(buf), "%.2fB", n / 1e9);
    else if (n >= 1'000'000) snprintf(buf, sizeof(buf), "%.1fM", n / 1e6);
    else snprintf(buf, sizeof(buf), "%.0fK", n / 1e3);
    return buf;
}

// Token count with fixed 2-decimal M/B (for "x/y" progress display)
static std::string fmt_tok2(uint64_t n) {
    char buf[32];
    if (n >= 1'000'000'000ULL) snprintf(buf, sizeof(buf), "%.2fB", n / 1e9);
    else                        snprintf(buf, sizeof(buf), "%.2fM", n / 1e6);
    return buf;
}

// Human duration: 1d13h06m44s / 34m19s / 12s
static std::string fmt_dur(double seconds) {
    if (seconds < 0 || !std::isfinite(seconds)) return "--";
    uint64_t s = (uint64_t)(seconds + 0.5);
    uint64_t d = s / 86400; s %= 86400;
    uint64_t h = s / 3600;  s %= 3600;
    uint64_t m = s / 60;    s %= 60;
    char buf[48];
    if (d > 0)      snprintf(buf, sizeof(buf), "%llud%lluh%02llum%02llus",
                             (unsigned long long)d,(unsigned long long)h,(unsigned long long)m,(unsigned long long)s);
    else if (h > 0) snprintf(buf, sizeof(buf), "%lluh%02llum%02llus",
                             (unsigned long long)h,(unsigned long long)m,(unsigned long long)s);
    else if (m > 0) snprintf(buf, sizeof(buf), "%llum%02llus",
                             (unsigned long long)m,(unsigned long long)s);
    else            snprintf(buf, sizeof(buf), "%llus",(unsigned long long)s);
    return buf;
}

// ASCII progress bar: [>...] width chars, filled by fraction
static std::string fmt_bar(double frac, int width = 20) {
    if (frac < 0) frac = 0; if (frac > 1) frac = 1;
    int filled = (int)(frac * width);
    std::string b = "[";
    for (int i = 0; i < width; i++) {
        if (i < filled) b += '=';
        else if (i == filled) b += '>';
        else b += '.';
    }
    b += "]";
    return b;
}

// ─── Diagnostic: compute stats of FP16 tensor on GPU ───────────────────────
struct TensorStats {
    float min_val, max_val, rms_val;
    int nan_count, inf_count;
};

static TensorStats compute_stats_fp16(const cuda::CudaTensor& t) {
    TensorStats s = {0, 0, 0, 0, 0};
    if (!t.data || t.numel() == 0) return s;
    int n = (int)t.numel();
    int sample_n = std::min(n, 65536);
    std::vector<__half> buf(sample_n);
    cudaMemcpy(buf.data(), t.data, sample_n * sizeof(__half), cudaMemcpyDeviceToHost);
    s.min_val = 1e30f; s.max_val = -1e30f;
    double sq_sum = 0.0;
    for (int i = 0; i < sample_n; i++) {
        float v = __half2float(buf[i]);
        if (std::isnan(v)) { s.nan_count++; continue; }
        if (std::isinf(v)) { s.inf_count++; continue; }
        if (v < s.min_val) s.min_val = v;
        if (v > s.max_val) s.max_val = v;
        sq_sum += (double)v * v;
    }
    int valid = sample_n - s.nan_count - s.inf_count;
    s.rms_val = valid > 0 ? sqrtf((float)(sq_sum / valid)) : 0.0f;
    if (valid == 0) { s.min_val = 0; s.max_val = 0; }
    return s;
}

static float absmax_fp16(const cuda::CudaTensor& t) {
    auto s = compute_stats_fp16(t);
    return std::max(fabsf(s.min_val), fabsf(s.max_val));
}

// ─── Full training diagnostic dump (matches train_diag format) ─────────────
static void dump_train_diag(FILE* f, int step, float loss, float prev_loss,
                            float lr, float gnorm, float clip_coeff,
                            const std::vector<cuda::CudaTensor>& params,
                            const ts::FwdCache& cache,
                            const ts::Grads& grads) {
    float delta = loss - prev_loss;
    fprintf(f, "STEP %d loss=%.6f delta=%+.4f lr=%.2e gnorm=%.4f clip=%.6f\n",
            step, loss, delta, lr, gnorm, clip_coeff);

    // FWD: forward activation stats per layer
    fprintf(f, "FWD ");
    {
        auto s = compute_stats_fp16(cache.embed_out);
        fprintf(f, "embed:%.2f/%.2f/%.4f/%d/%d ", s.min_val, s.max_val, s.rms_val, s.nan_count, s.inf_count);
    }
    for (int l = 0; l < (int)cache.layers.size(); l++) {
        auto& lc = cache.layers[l];
        auto sq = compute_stats_fp16(lc.Q);
        auto sff = compute_stats_fp16(lc.ff);
        fprintf(f, "L%dQ:%.2f/%.2f/%.4f/%d/%d L%dFF:%.2f/%.2f/%.4f/%d/%d ",
                l, sq.min_val, sq.max_val, sq.rms_val, sq.nan_count, sq.inf_count,
                l, sff.min_val, sff.max_val, sff.rms_val, sff.nan_count, sff.inf_count);
    }
    fprintf(f, "\n");

    // GRAD: gradient stats per layer
    fprintf(f, "GRAD ");
    {
        auto s = compute_stats_fp16(grads.d_embed);
        fprintf(f, "emb:%.4e/%.4e/%.4e/%d/%d ", s.min_val, s.max_val, s.rms_val, s.nan_count, s.inf_count);
    }
    for (int l = 0; l < N_LAYERS; l++) {
        auto& lg = grads.layers[l];
        auto sq = compute_stats_fp16(lg.dwq);
        auto s1 = compute_stats_fp16(lg.dw1);
        auto s2 = compute_stats_fp16(lg.dw2);
        fprintf(f, "L%ddWQ:%.4e/%.4e/%.4e/%d/%d L%ddW1:%.4e/%.4e/%.4e/%d/%d L%ddW2:%.4e/%.4e/%.4e/%d/%d ",
                l, sq.min_val, sq.max_val, sq.rms_val, sq.nan_count, sq.inf_count,
                l, s1.min_val, s1.max_val, s1.rms_val, s1.nan_count, s1.inf_count,
                l, s2.min_val, s2.max_val, s2.rms_val, s2.nan_count, s2.inf_count);
    }
    fprintf(f, "\n");

    // PARAM: first 20 parameter tensors min/max
    fprintf(f, "PARAM ");
    int np = std::min((int)params.size(), 20);
    for (int i = 0; i < np; i++) {
        auto s = compute_stats_fp16(params[i]);
        fprintf(f, "p%d:%.4f/%.4f/%d/%d ", i, s.min_val, s.max_val, s.nan_count, s.inf_count);
    }
    fprintf(f, "\n---\n");
    fflush(f);
}

// ─── Full backward diagnostic dump (matches backward_diag format) ──────────
static void dump_backward_diag(FILE* f, int step,
                               const cuda::CudaTensor& dx_final,
                               const ts::Grads& grads) {
    auto dxf = compute_stats_fp16(dx_final);
    fprintf(f, "STEP %d dx_final:%.4e/%.4e\n",
            step, std::max(fabsf(dxf.min_val), fabsf(dxf.max_val)), dxf.rms_val);
    for (int l = N_LAYERS - 1; l >= 0; l--) {
        auto& lg = grads.layers[l];
        fprintf(f, "  L%d dQ:%.4e dK:%.4e dV:%.4e dgate:%.4e dup:%.4e dW2:%.4e dWQ:%.4e\n",
                l, absmax_fp16(lg.dwq), absmax_fp16(lg.dwk), absmax_fp16(lg.dwv),
                absmax_fp16(lg.dw1), absmax_fp16(lg.dw3), absmax_fp16(lg.dw2),
                absmax_fp16(lg.dwq));
    }
    fprintf(f, "---\n");
    fflush(f);
}

// ─── Phase configuration (curriculum learning) ─────────────────────────────
struct PhaseConfig {
    int seq_len;
    uint64_t tokens;
};

static std::vector<PhaseConfig> compute_phases(uint64_t total_tokens) {
    std::vector<PhaseConfig> phases;
    uint64_t full_9b = 9'000'000'000ULL;
    uint64_t p1 = total_tokens * 2'200'000'000ULL / full_9b;
    uint64_t p2 = total_tokens * 3'200'000'000ULL / full_9b;
    uint64_t p3 = total_tokens * 1'000'000'000ULL / full_9b;
    uint64_t p4 = total_tokens - p1 - p2 - p3;

    int seqs[] = {256, 512, 1024, 2048};
    uint64_t toks[] = {p1, p2, p3, p4};
    for (int i = 0; i < 4; i++) {
        if (toks[i] >= 1536)
            phases.push_back({seqs[i], toks[i]});
    }
    if (phases.empty())
        phases.push_back({256, total_tokens});
    return phases;
}

uint64_t get_seq_len_for_tokens(uint64_t processed, const std::vector<PhaseConfig>& phases) {
    if (phases.empty()) return 256;
    uint64_t cum = 0;
    for (auto& p : phases) {
        cum += p.tokens;
        if (processed < cum) return p.seq_len;
    }
    return phases.back().seq_len;
}

// ─── Cosine LR schedule with warmup ───────────────────────────────────────
static float cosine_lr(int step, int warmup, float max_lr, float min_lr, int total_steps) {
    if (step < warmup) return max_lr * step / warmup;
    float progress = (float)(step - warmup) / (float)(total_steps - warmup);
    return min_lr + 0.5f * (max_lr - min_lr) * (1.0f + cosf(M_PI * progress));
}

// ─── Checkpoint save (overwrite mode for Kaggle) ───────────────────────────
void save_checkpoint(const std::string& dir,
                     const std::vector<cuda::CudaTensor>& params,
                     const std::vector<cuda::CudaTensor>& optimizer_m,
                     const std::vector<cuda::CudaTensor>& optimizer_v,
                     uint64_t tokens_done, int step,
                     const std::string& suffix = "") {
    fs::create_directories(dir);

    std::string model_path = dir + "/ckpt_pretrain" + suffix + ".safetensors";
    SafeTensorsWriter w;

    // IMPORTANT: SafeTensorsWriter stores only POINTERS to the tensor data and
    // writes them lazily in finalize(). So every host buffer must stay alive
    // until finalize() is called. We keep them all in this persistent vector.
    std::vector<std::vector<__half>> host_bufs;
    host_bufs.reserve(1 + N_LAYERS * 9 + 1);

    // Embedding [V, D]
    {
        host_bufs.emplace_back(params[0].numel());
        auto& host = host_bufs.back();
        cudaMemcpy(host.data(), params[0].data, params[0].nbytes, cudaMemcpyDeviceToHost);
        w.add_tensor("embed.weight", STDtype::F16,
                     {(uint32_t)params[0].shape[0], (uint32_t)params[0].shape[1]}, host.data());
    }

    // Layers
    for (int l = 0; l < N_LAYERS; l++) {
        int base = 1 + l * 9;
        const char* names[] = {"wq","wk","wv","wo","w1","w2","w3","rms_attn","rms_ffn"};
        for (int i = 0; i < 9; i++) {
            auto& p = params[base + i];
            host_bufs.emplace_back(p.numel());
            auto& host = host_bufs.back();
            cudaMemcpy(host.data(), p.data, p.nbytes, cudaMemcpyDeviceToHost);
            std::string tname = "layer." + std::to_string(l) + "." + names[i] + ".weight";
            if (p.ndim == 1)
                w.add_tensor(tname, STDtype::F16, {(uint32_t)p.numel()}, host.data());
            else
                w.add_tensor(tname, STDtype::F16,
                             {(uint32_t)p.shape[0], (uint32_t)p.shape[1]}, host.data());
        }
    }

    // Final RMS
    {
        auto& p = params[1 + N_LAYERS * 9];
        host_bufs.emplace_back(p.numel());
        auto& host = host_bufs.back();
        cudaMemcpy(host.data(), p.data, p.nbytes, cudaMemcpyDeviceToHost);
        w.add_tensor("final_rms.weight", STDtype::F16, {(uint32_t)p.numel()}, host.data());
    }
    w.finalize(model_path);

    // Save training state
    std::string state_path = dir + "/train_state.txt";
    std::ofstream sf(state_path);
    sf << "tokens_processed=" << tokens_done << "\n";
    sf << "step=" << step << "\n";
    sf << "phase=" << g_current_phase << "\n";
    sf.close();

    // ─── Save optimizer states (FP32: m, v, master) ────────────────────────
    {
        std::string optim_path = dir + "/ckpt_optim" + suffix + ".safetensors";
        SafeTensorsWriter ow;
        std::vector<std::vector<float>> optim_bufs;
        optim_bufs.reserve(params.size() * 3);

        const char* layer_names[] = {"wq","wk","wv","wo","w1","w2","w3","rms_attn","rms_ffn"};

        auto save_f32_tensor = [&](const cuda::CudaTensor& t, const std::string& name) {
            optim_bufs.emplace_back(t.numel());
            auto& hbuf = optim_bufs.back();
            cudaMemcpy(hbuf.data(), t.data, t.numel() * sizeof(float), cudaMemcpyDeviceToHost);
            if (t.ndim == 1)
                ow.add_tensor(name, STDtype::F32, {(uint32_t)t.numel()}, hbuf.data());
            else
                ow.add_tensor(name, STDtype::F32,
                             {(uint32_t)t.shape[0], (uint32_t)t.shape[1]}, hbuf.data());
        };

        // Embedding optimizer states
        save_f32_tensor(optimizer_m[0], "m.embed");
        save_f32_tensor(optimizer_v[0], "v.embed");

        // Layer optimizer states
        for (int l = 0; l < N_LAYERS; l++) {
            int base = 1 + l * 9;
            for (int i = 0; i < 9; i++) {
                std::string prefix = "layer." + std::to_string(l) + "." + layer_names[i];
                save_f32_tensor(optimizer_m[base + i], "m." + prefix);
                save_f32_tensor(optimizer_v[base + i], "v." + prefix);
            }
        }

        // Final RMS optimizer states
        int frms_idx = 1 + N_LAYERS * 9;
        save_f32_tensor(optimizer_m[frms_idx], "m.final_rms");
        save_f32_tensor(optimizer_v[frms_idx], "v.final_rms");

        ow.finalize(optim_path);
        printf("[CKPT] Optimizer: %s (%.1f MB)\n",
               optim_path.c_str(), fs::file_size(optim_path) / 1e6);
    }

    printf("[CKPT] Saved: %s (%.1f MB) | %s tokens\n",
           model_path.c_str(), fs::file_size(model_path) / 1e6,
           fmt_tok(tokens_done).c_str());
}

// ─── RoPE table pre-computation ────────────────────────────────────────────
struct RoPETables {
    std::vector<float> cos_host, sin_host;
    float *cos_dev = nullptr, *sin_dev = nullptr;
    int max_seq, half_dim;

    void init(int max_seq_len, int d_head, float theta) {
        max_seq = max_seq_len;
        half_dim = d_head / 2;
        cos_host.resize(max_seq * half_dim);
        sin_host.resize(max_seq * half_dim);

        for (int pos = 0; pos < max_seq; pos++) {
            for (int j = 0; j < half_dim; j++) {
                float freq = pos / powf(theta, 2.0f * j / d_head);
                cos_host[pos * half_dim + j] = cosf(freq);
                sin_host[pos * half_dim + j] = sinf(freq);
            }
        }

        cudaMalloc(&cos_dev, cos_host.size() * sizeof(float));
        cudaMalloc(&sin_dev, sin_host.size() * sizeof(float));
        cudaMemcpy(cos_dev, cos_host.data(), cos_host.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(sin_dev, sin_host.data(), sin_host.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    ~RoPETables() {
        if (cos_dev) cudaFree(cos_dev);
        if (sin_dev) cudaFree(sin_dev);
    }
};

// ─── Init model parameters (Xavier/He init) ────────────────────────────────
std::vector<cuda::CudaTensor> init_params() {
    std::vector<cuda::CudaTensor> p;
    std::mt19937 rng(42);
    std::normal_distribution<float> norm(0.0f, 1.0f);

    auto make_weight = [&](int64_t rows, int64_t cols, float stddev) {
        size_t n = (size_t)(rows * cols);
        std::vector<__half> host_fp16(n);
        for (size_t i = 0; i < n; i++)
            host_fp16[i] = __float2half(norm(rng) * stddev);

        cuda::CudaTensor t({rows, cols}, cuda::DType::FP16);
        cudaMemcpy(t.data, host_fp16.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
        return t;
    };

    auto make_ones = [](int64_t d) {
        std::vector<__half> host(d);
        for (int64_t i = 0; i < d; i++) host[i] = __float2half(1.0f);
        cuda::CudaTensor t({d}, cuda::DType::FP16);
        cudaMemcpy(t.data, host.data(), d * sizeof(__half), cudaMemcpyHostToDevice);
        return t;
    };

    // Embedding [V, D]
    p.push_back(make_weight(MODEL_VOCAB, D_MODEL, 0.02f));

    // 13 layers × 9 params each
    for (int l = 0; l < N_LAYERS; l++) {
        float s = 1.0f / sqrtf((float)D_MODEL);
        float sf = 1.0f / sqrtf((float)FFN_DIM);
        p.push_back(make_weight(D_MODEL, D_MODEL, s));             // Wq [D, D]
        p.push_back(make_weight(D_MODEL, N_KV_HEADS * D_HEAD, s)); // Wk [D, KVD]
        p.push_back(make_weight(D_MODEL, N_KV_HEADS * D_HEAD, s)); // Wv [D, KVD]
        p.push_back(make_weight(D_MODEL, D_MODEL, s));             // Wo [D, D]
        p.push_back(make_weight(D_MODEL, FFN_DIM, s));             // W1 gate [D, FFN]
        p.push_back(make_weight(FFN_DIM, D_MODEL, sf));            // W2 down [FFN, D]
        p.push_back(make_weight(D_MODEL, FFN_DIM, s));             // W3 up [D, FFN]
        p.push_back(make_ones(D_MODEL));                           // rms_attn [D]
        p.push_back(make_ones(D_MODEL));                           // rms_ffn [D]
    }

    // Final RMSNorm gamma [D]
    p.push_back(make_ones(D_MODEL));

    size_t total_params = 0;
    for (auto& t : p) total_params += t.numel();
    printf("[INIT] %zu tensors | %.1fM parameters (FP16)\n",
           p.size(), total_params / 1e6);
    return p;
}

// ─── Checkpoint frequency (adaptive) ──────────────────────────────────────
bool should_checkpoint(uint64_t tokens, int step) {
    if (g_overwrite_ckpt) return (step % g_save_every == 0) && step > 0;
    uint64_t interval;
    if (g_max_seq <= 256)      interval = 2'000'000;
    else if (g_max_seq <= 512) interval = 1'000'000;
    else                       interval = 500'000;
    return (tokens % interval < (uint64_t)g_batch_size * g_max_seq);
}

// ─── Main ──────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);

    // ─── Parse CLI ─────────────────────────────────────────────────────────
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data_dir" && i+1 < argc)       g_data_dir = argv[++i];
        else if (arg == "--checkpoint_dir" && i+1 < argc) g_checkpoint_dir = argv[++i];
        else if (arg == "--vocab_dir" && i+1 < argc)  g_vocab_dir = argv[++i];
        else if (arg == "--bpe" && i+1 < argc)        g_bpe_path = argv[++i];
        else if (arg == "--max_tokens" && i+1 < argc) g_max_tokens = std::stoull(argv[++i]);
        else if (arg == "--batch_size" && i+1 < argc) g_batch_size = std::stoi(argv[++i]);
        else if (arg == "--max_seq" && i+1 < argc)    g_max_seq = std::stoi(argv[++i]);
        else if (arg == "--lr" && i+1 < argc)         g_lr = std::stof(argv[++i]);
        else if (arg == "--save_every" && i+1 < argc) g_save_every = std::stoi(argv[++i]);
        else if (arg == "--log_every" && i+1 < argc)  g_log_every = std::stoi(argv[++i]);
        else if (arg == "--resume")                   g_resume = true;
        else if (arg == "--ddp")                      g_use_ddp = true;
        else if (arg == "--num_gpus" && i+1 < argc)   g_num_gpus = std::stoi(argv[++i]);
        else if (arg == "--loss_scale" && i+1 < argc) g_loss_scale = std::stof(argv[++i]);
        else if (arg == "--debug_steps" && i+1 < argc) g_debug_steps = std::stoi(argv[++i]);
        else if (arg == "--help") {
            printf("Usage: train_pretrain_cuda [options]\n");
            printf("  --data_dir <path>         Corpus directory [data/corpus_final]\n");
            printf("  --checkpoint_dir <path>   Checkpoint output [checkpoints]\n");
            printf("  --vocab_dir <path>        Vocab directory [vocab_6]\n");
            printf("  --bpe <path>              BPE binary model path (optional)\n");
            printf("  --max_tokens <N>          Token budget [9B]\n");
            printf("  --batch_size <N>          Batch size per GPU [8]\n");
            printf("  --max_seq <N>             Initial max seq length [256]\n");
            printf("  --lr <float>              Peak learning rate [3e-4]\n");
            printf("  --save_every <N>          Checkpoint every N steps [500]\n");
            printf("  --log_every <N>           Log every N steps [10]\n");
            printf("  --resume                  Resume from checkpoint\n");
            printf("  --ddp                     Enable multi-GPU DDP (NCCL)\n");
            printf("  --num_gpus <N>            Number of GPUs for DDP [2]\n");
            printf("  --loss_scale <float>      Static loss scale for FP16 [1.0]\n");
            printf("  --debug_steps <N>         Dump gradient diag every N steps [0=off]\n");
            return 0;
        }
    }

    // ─── Init CUDA ─────────────────────────────────────────────────────────
    // For DDP we MUST fork BEFORE any CUDA call (CUDA context does not survive
    // fork). So: fork first -> ctx.init(local device) -> create NCCL comm.
#ifdef USE_DDP
    int ddp_device = 0;
    if (g_use_ddp) {
        if (!cuda::ddp::init(g_num_gpus)) {   // fork only, no CUDA yet
            fprintf(stderr, "[ERR] DDP fork failed\n");
            return 1;
        }
        ddp_device = cuda::ddp::device_for_rank();
    }
    cuda::ctx.init(ddp_device);
    cuda::nvml_init();
    if (g_use_ddp) {
        if (!cuda::ddp::init_nccl()) {         // NCCL comm after CUDA init
            fprintf(stderr, "[ERR] NCCL init failed\n");
            return 1;
        }
    }
#else
    cuda::ctx.init(0);
    cuda::nvml_init();
#endif

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  100M Transformer — CUDA Pre-training                    ║\n");
    printf("║  D=%d L=%d QH=%d KVH=%d DH=%d FFN=%d V=%d        ║\n",
           D_MODEL, N_LAYERS, N_Q_HEADS, N_KV_HEADS, D_HEAD, FFN_DIM, MODEL_VOCAB);
    printf("║  GPU: %s (sm_%d) | FP16 Mixed Precision | AdamW  ║\n",
           cuda::ctx.gpu_caps.name, cuda::ctx.gpu_caps.sm_version);
    printf("║  Tokens: %-10s | Batch: %-3d | Seq: %-4d            ║\n",
           fmt_tok(g_max_tokens).c_str(), g_batch_size, g_max_seq);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // ─── Auto-tune batch size based on available VRAM ──────────────────────
    if (g_batch_size <= 8) {
        // User didn't specify a large batch — auto-detect from VRAM
        float vram_gb = cuda::ctx.gpu_caps.vram_bytes / (1024.0f * 1024.0f * 1024.0f);
        int auto_batch = (int)((vram_gb - 1.5f) * 5.0f);  // ~5 samples per free GB at seq=256
        auto_batch = std::max(4, std::min(auto_batch, 500));
        // Scale for sequence length
        auto_batch = auto_batch * 256 / g_max_seq;
        auto_batch = std::max(4, auto_batch);
        if (auto_batch > g_batch_size) {
            printf("[AUTO] Batch size: %d → %d (detected %.1f GB VRAM)\n",
                   g_batch_size, auto_batch, vram_gb);
            g_batch_size = auto_batch;
        }
    }

    // ─── Load tokenizer ────────────────────────────────────────────────────
    BPEEncoder tok;
    if (!g_bpe_path.empty()) {
        if (!tok.load(g_bpe_path)) {
            fprintf(stderr, "[ERR] Failed to load BPE binary: %s\n", g_bpe_path.c_str());
            return 1;
        }
    } else {
        tok.load_vocab(g_vocab_dir);
    }
    printf("[TOK] Loaded: vocab_size=%zu\n", tok.vocab_size());

    int tok_bos = tok.bos_id(), tok_eos = tok.eos_id(), tok_pad = tok.pad_id();
    if (tok_pad < 0) tok_pad = tok_bos;

    // ─── Compute training phases ────────────────────────────────────────────
    auto phases = compute_phases(g_max_tokens);
    printf("[PHASE] %zu phases (curriculum):\n", phases.size());
    for (size_t i = 0; i < phases.size(); i++)
        printf("  Phase %zu: seq=%d, tokens=%s\n",
               i+1, phases[i].seq_len, fmt_tok(phases[i].tokens).c_str());

    // ─── Init model ────────────────────────────────────────────────────────
    auto params = init_params();

    // Master weights (FP32 for optimizer precision)
    std::vector<cuda::CudaTensor> master_params;
    for (auto& p : params) {
        master_params.push_back(cuda::astype(p, cuda::DType::F32));
    }

    // Optimizer states (FP32)
    std::vector<cuda::CudaTensor> adamw_m, adamw_v;
    for (auto& p : params) {
        adamw_m.push_back(cuda::zeros_like(cuda::CudaTensor(p.shape_vec(), cuda::DType::F32)));
        adamw_v.push_back(cuda::zeros_like(cuda::CudaTensor(p.shape_vec(), cuda::DType::F32)));
    }

    // ─── Resume checkpoint ─────────────────────────────────────────────────
    int start_step = 0;
    if (g_resume) {
        std::string ckpt_path = g_checkpoint_dir + "/ckpt_pretrain.safetensors";
        std::string state_path = g_checkpoint_dir + "/train_state.txt";
        if (fs::exists(ckpt_path)) {
            printf("[RESUME] Loading: %s\n", ckpt_path.c_str());
            SafeTensorsLoader loader(ckpt_path);

            // Load embedding
            std::vector<__half> buf(params[0].numel());
            loader.read_into_fp16("embed.weight", buf.data());
            cudaMemcpy(params[0].data, buf.data(), params[0].nbytes, cudaMemcpyHostToDevice);

            // Load layers
            for (int l = 0; l < N_LAYERS; l++) {
                int base = 1 + l * 9;
                const char* names[] = {"wq","wk","wv","wo","w1","w2","w3","rms_attn","rms_ffn"};
                for (int n = 0; n < 9; n++) {
                    std::string tname = "layer." + std::to_string(l) + "." + names[n] + ".weight";
                    auto& p = params[base + n];
                    std::vector<__half> pbuf(p.numel());
                    loader.read_into_fp16(tname, pbuf.data());
                    cudaMemcpy(p.data, pbuf.data(), p.nbytes, cudaMemcpyHostToDevice);
                }
            }

            // Load final RMS
            auto& frms = params[1 + N_LAYERS * 9];
            std::vector<__half> frms_buf(frms.numel());
            loader.read_into_fp16("final_rms.weight", frms_buf.data());
            cudaMemcpy(frms.data, frms_buf.data(), frms.nbytes, cudaMemcpyHostToDevice);

            // Restore FP32 master weights
            for (size_t i = 0; i < params.size(); i++) {
                master_params[i] = cuda::astype(params[i], cuda::DType::F32);
            }

            // ─── Load optimizer states if available ─────────────────────────
            std::string optim_path = g_checkpoint_dir + "/ckpt_optim.safetensors";
            if (fs::exists(optim_path)) {
                printf("[RESUME] Loading optimizer: %s\n", optim_path.c_str());
                SafeTensorsLoader opt_loader(optim_path);
                const char* layer_names[] = {"wq","wk","wv","wo","w1","w2","w3","rms_attn","rms_ffn"};

                auto load_f32_tensor = [&](cuda::CudaTensor& t, const std::string& name) {
                    if (!opt_loader.has_tensor(name)) return;
                    std::vector<float> hbuf(t.numel());
                    opt_loader.read_into_f32(name, hbuf.data());
                    cudaMemcpy(t.data, hbuf.data(), t.numel() * sizeof(float), cudaMemcpyHostToDevice);
                };

                // Embedding
                load_f32_tensor(adamw_m[0], "m.embed");
                load_f32_tensor(adamw_v[0], "v.embed");

                // Layers
                for (int l = 0; l < N_LAYERS; l++) {
                    int base = 1 + l * 9;
                    for (int i = 0; i < 9; i++) {
                        std::string prefix = "layer." + std::to_string(l) + "." + layer_names[i];
                        load_f32_tensor(adamw_m[base + i], "m." + prefix);
                        load_f32_tensor(adamw_v[base + i], "v." + prefix);
                    }
                }

                // Final RMS
                int frms_idx = 1 + N_LAYERS * 9;
                load_f32_tensor(adamw_m[frms_idx], "m.final_rms");
                load_f32_tensor(adamw_v[frms_idx], "v.final_rms");

                // Also load master weights from model (FP32 precision preserved)
                // The master_params were already set from FP16→FP32 above,
                // but if optimizer file exists, training was proper so FP16 cast is fine
                printf("[RESUME] Optimizer states restored (%zu tensors)\n", adamw_m.size() * 2);
            } else {
                printf("[RESUME] No optimizer checkpoint — cold start (m=0, v=0)\n");
            }

            // Restore training state
            if (fs::exists(state_path)) {
                std::ifstream sf(state_path);
                std::string line;
                while (std::getline(sf, line)) {
                    if (line.find("tokens_processed=") == 0)
                        g_tokens_processed = std::stoull(line.substr(17));
                    else if (line.find("step=") == 0)
                        start_step = std::stoi(line.substr(5));
                    else if (line.find("phase=") == 0)
                        g_current_phase = std::stoi(line.substr(6));
                }
            }
            printf("[RESUME] Restored at step=%d, tokens=%s\n",
                   start_step, fmt_tok(g_tokens_processed).c_str());
        } else {
            printf("[RESUME] No checkpoint found — starting fresh\n");
        }
    }

    // ─── Pre-compute RoPE tables ────────────────────────────────────────────
    RoPETables rope;
    rope.init(2048, D_HEAD, ROPE_THETA);
    printf("[RoPE] Tables pre-computed (max_seq=2048, d_head=%d)\n", D_HEAD);

    // ─── Init dataset ──────────────────────────────────────────────────────
    PretrainDatasetConfig ds_cfg;
    ds_cfg.corpus_root = g_data_dir;
    ds_cfg.seq_len = g_max_seq;
    ds_cfg.batch_size = g_batch_size;
    ds_cfg.seed = 42;
#ifdef USE_DDP
    // Each DDP rank reads a different data stream (otherwise both GPUs train on
    // identical batches, wasting the second GPU).
    if (g_use_ddp) ds_cfg.seed = 42 + 1000 * (uint64_t)cuda::ddp::rank();
#endif
    ds_cfg.weighted_sampling = true;

    PretrainDataset dataset(ds_cfg);
    dataset.set_tokenize_fn([&tok](const std::string& text, std::vector<uint32_t>& out) {
        auto ids = tok.encode(text, true, true);
        out.assign(ids.begin(), ids.end());
    });
    if (!dataset.init()) {
        fprintf(stderr, "[ERR] Failed to init dataset from %s\n", g_data_dir.c_str());
        return 1;
    }
    dataset.print_info();
    printf("[DATA] Streaming loader ready: %s\n", g_data_dir.c_str());

    // ─── Estimate total steps for LR schedule ──────────────────────────────
    const int cli_batch_size = g_batch_size;  // remember original for VRAM auto-scaling
    int est_total_steps = (int)(g_max_tokens / (g_batch_size * g_max_seq));
    int warmup_steps = std::min(1000, est_total_steps / 20);
    float min_lr = g_lr * 0.1f;

    // ─── Open diagnostic log files if debug mode ───────────────────────────
    FILE* f_train_diag = nullptr;
    FILE* f_bwd_diag = nullptr;
    if (g_debug_steps > 0) {
        std::string td_path = g_checkpoint_dir + "/train_diag.txt";
        std::string bd_path = g_checkpoint_dir + "/backward_diag.txt";
        f_train_diag = fopen(td_path.c_str(), "a");
        f_bwd_diag   = fopen(bd_path.c_str(), "a");
        if (f_train_diag) printf("[DIAG] train_diag → %s\n", td_path.c_str());
        if (f_bwd_diag)   printf("[DIAG] backward_diag → %s\n", bd_path.c_str());
    }
    if (g_loss_scale != 1.0f)
        printf("[SCALE] Loss scaling: %.0f (gradients computed at %gx, unscaled before optimizer)\n",
               g_loss_scale, g_loss_scale);

    // Tracking variables for diagnostics
    float prev_loss = 11.1f;   // ln(65536) ≈ 11.09, expected random init loss
    float last_gnorm = 0.0f;
    float last_clip_coeff = 1.0f;

    // ─── Training loop ──────────────────────────────────────────────────────
    auto t_start = Clock::now();
    auto t_last_log = t_start;
    uint64_t tokens_at_last_log = g_tokens_processed;
    int step = start_step;
    int step_at_last_log = step;
    float running_loss = 0.0f;
    int running_count = 0;

    // ─── Persistent gradient buffers (allocated once, reused every step) ────
    ts::ModelDims dims;
    dims.B = g_batch_size; dims.S = g_max_seq; dims.D = D_MODEL; dims.L = N_LAYERS;
    dims.HQ = N_Q_HEADS; dims.HKV = N_KV_HEADS; dims.DH = D_HEAD;
    dims.FFN = FFN_DIM; dims.V = MODEL_VOCAB; dims.rope_theta = ROPE_THETA;

    ts::Grads grads;
    grads.d_embed     = cuda::CudaTensor({MODEL_VOCAB, D_MODEL}, cuda::DType::FP16);
    grads.embed_accum = cuda::CudaTensor({MODEL_VOCAB, D_MODEL}, cuda::DType::F32);
    grads.d_final_rms = cuda::CudaTensor({D_MODEL}, cuda::DType::FP16);
    grads.layers.resize(N_LAYERS);
    for (int l = 0; l < N_LAYERS; l++) {
        int base = 1 + l * 9;
        auto& lg = grads.layers[l];
        lg.dwq = cuda::CudaTensor(params[base+0].shape_vec(), cuda::DType::FP16);
        lg.dwk = cuda::CudaTensor(params[base+1].shape_vec(), cuda::DType::FP16);
        lg.dwv = cuda::CudaTensor(params[base+2].shape_vec(), cuda::DType::FP16);
        lg.dwo = cuda::CudaTensor(params[base+3].shape_vec(), cuda::DType::FP16);
        lg.dw1 = cuda::CudaTensor(params[base+4].shape_vec(), cuda::DType::FP16);
        lg.dw2 = cuda::CudaTensor(params[base+5].shape_vec(), cuda::DType::FP16);
        lg.dw3 = cuda::CudaTensor(params[base+6].shape_vec(), cuda::DType::FP16);
        lg.drms_attn = cuda::CudaTensor(params[base+7].shape_vec(), cuda::DType::FP16);
        lg.drms_ffn  = cuda::CudaTensor(params[base+8].shape_vec(), cuda::DType::FP16);
    }
    cuda::sync_compute();

    while (g_tokens_processed < g_max_tokens && !g_stop_requested) {
        // Update seq_len based on curriculum phase
        int cur_seq = (int)get_seq_len_for_tokens(g_tokens_processed, phases);
        if (cur_seq != g_max_seq) {
            g_max_seq = cur_seq;
            // ─── Auto-adjust batch size to fit VRAM ────────────────────────
            // Target: keep B*S ≈ constant (≈ cli_batch_size * 256 tokens per GPU)
            // Calibrated from: B=70, S=256 uses 13.3GB / 14.6GB available
            int base_bs_product = cli_batch_size * 256;  // CLI batch_size is for seq=256
            g_batch_size = std::max(1, base_bs_product / g_max_seq);
            printf("[PHASE] Switching to seq_len=%d batch=%d at %s tokens\n",
                   g_max_seq, g_batch_size, fmt_tok(g_tokens_processed).c_str());
        }

        int B = g_batch_size, S = g_max_seq;

        // Fetch batch from streaming dataset
        std::vector<uint32_t> batch_raw(B * S);
        if (!dataset.next_batch(batch_raw.data(), B)) break;

        // Build targets (shift left by 1) and mask (1.0 for non-pad)
        std::vector<int> tokens_cpu(B * S), targets_cpu(B * S);
        std::vector<float> mask_cpu(B * S, 0.0f);
        for (int i = 0; i < B * S; i++) tokens_cpu[i] = (int)batch_raw[i];
        for (int b = 0; b < B; b++) {
            for (int s = 0; s < S - 1; s++) {
                targets_cpu[b * S + s] = (int)batch_raw[b * S + s + 1];
                if (batch_raw[b * S + s] != (uint32_t)tok_pad &&
                    batch_raw[b * S + s + 1] != (uint32_t)tok_pad)
                    mask_cpu[b * S + s] = 1.0f;
            }
            targets_cpu[b * S + S - 1] = tok_eos;
        }

        // Upload to GPU
        cuda::CudaTensor d_tokens = cuda::CudaTensor::from_host(tokens_cpu.data(), {B, S});
        cuda::CudaTensor d_targets = cuda::CudaTensor::from_host(targets_cpu.data(), {B, S});
        cuda::CudaTensor d_mask = cuda::CudaTensor::from_host(mask_cpu.data(), {B, S});

        // ─── LR schedule ───────────────────────────────────────────────────
        float lr = cosine_lr(step, warmup_steps, g_lr, min_lr, est_total_steps);

        // ─── FORWARD PASS (correct GQA attention + stored activations) ─────
        dims.S = S;
        ts::FwdCache cache;
        cuda::CudaTensor x_final = ts::forward(params, d_tokens, dims,
                                               rope.cos_dev, rope.sin_dev, cache);
        // x_final: [B*S, D] view into cache.x_final (owned by cache)

        // Logits (weight tying: x_final @ embed.T) → [B*S, V]
        auto logits = cuda::matmul_transb(x_final, params[0]);

        // ─── LOSS + d_logits (single fused kernel, no buffer overflow) ─────
        int BS = B * S;
        cuda::CudaTensor per_token_loss({(int64_t)BS}, cuda::DType::F32);
        cuda::CudaTensor d_logits({(int64_t)BS, (int64_t)MODEL_VOCAB}, cuda::DType::FP16);
        launch_ce_fwd_bwd((const __half*)logits.data, (const int*)d_targets.data,
                          (const float*)d_mask.data,
                          (float*)per_token_loss.data, (__half*)d_logits.data,
                          BS, MODEL_VOCAB, cuda::ctx.compute_stream);

        // Read loss + mask count (one sync per step for logging + grad scale)
        cuda::sync_compute();
        std::vector<float> h_loss(BS), h_mask = mask_cpu;
        cudaMemcpy(h_loss.data(), per_token_loss.data, BS * sizeof(float), cudaMemcpyDeviceToHost);
        double loss_sum = 0.0; int tok_count = 0;
        for (int i = 0; i < BS; i++) { loss_sum += h_loss[i]; if (mask_cpu[i] > 0.0f) tok_count++; }
        float loss_val = tok_count > 0 ? (float)(loss_sum / tok_count) : 0.0f;
        running_loss += loss_val;
        running_count++;

        // Scale d_logits by 1/tok_count (mean loss gradient)
        if (tok_count > 0)
            launch_scale_fp16((__half*)d_logits.data, 1.0f / tok_count,
                              d_logits.numel(), cuda::ctx.compute_stream);

        // ─── LOSS SCALING: multiply d_logits by loss_scale for FP16 stability ──
        if (g_loss_scale != 1.0f)
            launch_scale_fp16((__half*)d_logits.data, g_loss_scale,
                              d_logits.numel(), cuda::ctx.compute_stream);

        // ─── BACKWARD ──────────────────────────────────────────────────────
        // final rmsnorm output grad: dx_final = d_logits @ embed  → [B*S, D]
        auto dx_final = cuda::matmul(d_logits, params[0]);  // [B*S, D]

        // Weight-tie output-projection embedding grad: d_embed_out = d_logits^T @ x_final
        auto d_embed_out = cuda::matmul(d_logits, x_final, true, false); // [V, D]
        launch_f16_to_f32_seed((const __half*)d_embed_out.data,
                               (float*)grads.embed_accum.data,
                               (size_t)MODEL_VOCAB * D_MODEL, cuda::ctx.compute_stream);

        ts::backward(params, d_tokens, x_final, dx_final, dims,
                     rope.cos_dev, rope.sin_dev, cache, grads);

        // ─── DDP: AllReduce gradients across GPUs ──────────────────────────
#ifdef USE_DDP
        if (g_use_ddp) {
            std::vector<cuda::CudaTensor*> all_grads;
            all_grads.push_back(&grads.d_embed);
            all_grads.push_back(&grads.d_final_rms);
            for (auto& lg : grads.layers) {
                all_grads.push_back(&lg.dwq); all_grads.push_back(&lg.dwk);
                all_grads.push_back(&lg.dwv); all_grads.push_back(&lg.dwo);
                all_grads.push_back(&lg.dw1); all_grads.push_back(&lg.dw2);
                all_grads.push_back(&lg.dw3);
                all_grads.push_back(&lg.drms_attn); all_grads.push_back(&lg.drms_ffn);
            }
            cuda::ddp::allreduce_grads(all_grads);
            cuda::ddp::sync();
            for (auto* gp : all_grads)
                if (gp && gp->data && gp->numel() > 0) cuda::ddp::average_grad(*gp);
        }
#endif

        // ─── GRADIENT CLIPPING (global L2 norm) ────────────────────────────
        // First: unscale gradients if loss_scale > 1
        if (g_loss_scale != 1.0f) {
            float inv_scale = 1.0f / g_loss_scale;
            launch_scale_fp16((__half*)grads.d_embed.data, inv_scale, grads.d_embed.numel(), cuda::ctx.compute_stream);
            launch_scale_fp16((__half*)grads.d_final_rms.data, inv_scale, grads.d_final_rms.numel(), cuda::ctx.compute_stream);
            for (int l = 0; l < N_LAYERS; l++) {
                auto& lg = grads.layers[l];
                launch_scale_fp16((__half*)lg.dwq.data, inv_scale, lg.dwq.numel(), cuda::ctx.compute_stream);
                launch_scale_fp16((__half*)lg.dwk.data, inv_scale, lg.dwk.numel(), cuda::ctx.compute_stream);
                launch_scale_fp16((__half*)lg.dwv.data, inv_scale, lg.dwv.numel(), cuda::ctx.compute_stream);
                launch_scale_fp16((__half*)lg.dwo.data, inv_scale, lg.dwo.numel(), cuda::ctx.compute_stream);
                launch_scale_fp16((__half*)lg.dw1.data, inv_scale, lg.dw1.numel(), cuda::ctx.compute_stream);
                launch_scale_fp16((__half*)lg.dw2.data, inv_scale, lg.dw2.numel(), cuda::ctx.compute_stream);
                launch_scale_fp16((__half*)lg.dw3.data, inv_scale, lg.dw3.numel(), cuda::ctx.compute_stream);
                launch_scale_fp16((__half*)lg.drms_attn.data, inv_scale, lg.drms_attn.numel(), cuda::ctx.compute_stream);
                launch_scale_fp16((__half*)lg.drms_ffn.data, inv_scale, lg.drms_ffn.numel(), cuda::ctx.compute_stream);
            }
        }

        if (g_grad_clip > 0.0f) {
            float* d_norm_sq = nullptr;
            cudaMalloc(&d_norm_sq, sizeof(float));
            cudaMemsetAsync(d_norm_sq, 0, sizeof(float), cuda::ctx.compute_stream);

            // Accumulate squared norms of all gradient tensors
            launch_l2_norm_sq((const __half*)grads.d_embed.data, d_norm_sq,
                              grads.d_embed.numel(), cuda::ctx.compute_stream);
            launch_l2_norm_sq((const __half*)grads.d_final_rms.data, d_norm_sq,
                              grads.d_final_rms.numel(), cuda::ctx.compute_stream);
            for (int l = 0; l < N_LAYERS; l++) {
                auto& lg = grads.layers[l];
                launch_l2_norm_sq((const __half*)lg.dwq.data, d_norm_sq, lg.dwq.numel(), cuda::ctx.compute_stream);
                launch_l2_norm_sq((const __half*)lg.dwk.data, d_norm_sq, lg.dwk.numel(), cuda::ctx.compute_stream);
                launch_l2_norm_sq((const __half*)lg.dwv.data, d_norm_sq, lg.dwv.numel(), cuda::ctx.compute_stream);
                launch_l2_norm_sq((const __half*)lg.dwo.data, d_norm_sq, lg.dwo.numel(), cuda::ctx.compute_stream);
                launch_l2_norm_sq((const __half*)lg.dw1.data, d_norm_sq, lg.dw1.numel(), cuda::ctx.compute_stream);
                launch_l2_norm_sq((const __half*)lg.dw2.data, d_norm_sq, lg.dw2.numel(), cuda::ctx.compute_stream);
                launch_l2_norm_sq((const __half*)lg.dw3.data, d_norm_sq, lg.dw3.numel(), cuda::ctx.compute_stream);
                launch_l2_norm_sq((const __half*)lg.drms_attn.data, d_norm_sq, lg.drms_attn.numel(), cuda::ctx.compute_stream);
                launch_l2_norm_sq((const __half*)lg.drms_ffn.data, d_norm_sq, lg.drms_ffn.numel(), cuda::ctx.compute_stream);
            }

            // Read norm and clip if needed
            cuda::sync_compute();
            float h_norm_sq = 0.0f;
            cudaMemcpy(&h_norm_sq, d_norm_sq, sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_norm_sq);

            float global_norm = sqrtf(h_norm_sq);
            last_gnorm = global_norm;
            last_clip_coeff = 1.0f;
            if (global_norm > g_grad_clip) {
                float clip_coeff = g_grad_clip / (global_norm + 1e-6f);
                last_clip_coeff = clip_coeff;
                launch_scale_fp16((__half*)grads.d_embed.data, clip_coeff, grads.d_embed.numel(), cuda::ctx.compute_stream);
                launch_scale_fp16((__half*)grads.d_final_rms.data, clip_coeff, grads.d_final_rms.numel(), cuda::ctx.compute_stream);
                for (int l = 0; l < N_LAYERS; l++) {
                    auto& lg = grads.layers[l];
                    launch_scale_fp16((__half*)lg.dwq.data, clip_coeff, lg.dwq.numel(), cuda::ctx.compute_stream);
                    launch_scale_fp16((__half*)lg.dwk.data, clip_coeff, lg.dwk.numel(), cuda::ctx.compute_stream);
                    launch_scale_fp16((__half*)lg.dwv.data, clip_coeff, lg.dwv.numel(), cuda::ctx.compute_stream);
                    launch_scale_fp16((__half*)lg.dwo.data, clip_coeff, lg.dwo.numel(), cuda::ctx.compute_stream);
                    launch_scale_fp16((__half*)lg.dw1.data, clip_coeff, lg.dw1.numel(), cuda::ctx.compute_stream);
                    launch_scale_fp16((__half*)lg.dw2.data, clip_coeff, lg.dw2.numel(), cuda::ctx.compute_stream);
                    launch_scale_fp16((__half*)lg.dw3.data, clip_coeff, lg.dw3.numel(), cuda::ctx.compute_stream);
                    launch_scale_fp16((__half*)lg.drms_attn.data, clip_coeff, lg.drms_attn.numel(), cuda::ctx.compute_stream);
                    launch_scale_fp16((__half*)lg.drms_ffn.data, clip_coeff, lg.drms_ffn.numel(), cuda::ctx.compute_stream);
                }
            }
        }

        // ─── DIAGNOSTIC DUMP (after unscale + clip, before optimizer) ──────
        bool do_diag = (g_debug_steps > 0 && (step % g_debug_steps == 0));
        // Also dump on large loss spikes (delta > 0.5)
        if (g_debug_steps > 0 && (loss_val - prev_loss) > 0.5f) do_diag = true;
        if (do_diag) {
            cuda::sync_compute();
            if (f_train_diag)
                dump_train_diag(f_train_diag, step, loss_val, prev_loss,
                                lr, last_gnorm, last_clip_coeff,
                                params, cache, grads);
            if (f_bwd_diag)
                dump_backward_diag(f_bwd_diag, step, dx_final, grads);
        }
        prev_loss = loss_val;

        // ─── OPTIMIZER STEP ────────────────────────────────────────────────
        cuda::adamw_step(params[0], master_params[0], adamw_m[0], adamw_v[0],
                         grads.d_embed, lr, g_beta1, g_beta2, g_eps, g_wd, step + 1);

        for (int l = 0; l < N_LAYERS; l++) {
            int base = 1 + l * 9;
            auto& lg = grads.layers[l];
            cuda::CudaTensor* layer_grads[] = {
                &lg.dwq, &lg.dwk, &lg.dwv, &lg.dwo,
                &lg.dw1, &lg.dw2, &lg.dw3, &lg.drms_attn, &lg.drms_ffn
            };
            for (int i = 0; i < 9; i++) {
                cuda::adamw_step(params[base + i], master_params[base + i],
                                 adamw_m[base + i], adamw_v[base + i],
                                 *layer_grads[i], lr, g_beta1, g_beta2, g_eps, g_wd, step + 1);
            }
        }

        int frms_idx = 1 + N_LAYERS * 9;
        cuda::adamw_step(params[frms_idx], master_params[frms_idx],
                         adamw_m[frms_idx], adamw_v[frms_idx],
                         grads.d_final_rms, lr, g_beta1, g_beta2, g_eps, 0.0f, step + 1);

        // ─── UPDATE COUNTERS ──────────────────────────────────────────────
        // DDP: each GPU processes B samples, total = B * world_size
        int effective_world = 1;
#ifdef USE_DDP
        if (g_use_ddp) effective_world = g_num_gpus;
#endif
        g_tokens_processed += (uint64_t)(B * S * effective_world);
        step++;

        // ─── LOGGING ─────────────────────────────────────────────────────
        if (step % g_log_every == 0) {
#ifdef USE_DDP
            if (!g_use_ddp || cuda::ddp::is_main()) {
#endif
            auto t_now = Clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double interval = std::chrono::duration<double>(t_now - t_last_log).count();
            int steps_done = step - step_at_last_log;
            uint64_t toks_done = g_tokens_processed - tokens_at_last_log;

            float avg_loss = running_loss / std::max(running_count, 1);
            // instantaneous throughput over the last logging interval
            double inst_tok_s = interval > 0 ? toks_done / interval : 0.0;
            double s_per_step = steps_done > 0 ? interval / steps_done : 0.0;

            double frac = (double)g_tokens_processed / (double)g_max_tokens;
            double eta = inst_tok_s > 0
                       ? (double)(g_max_tokens - g_tokens_processed) / inst_tok_s
                       : -1.0;

            auto stats = cuda::get_gpu_stats(0);
            float ram_gb  = get_rss_mb() / 1024.0f;
            float vram_gb = stats.mem_used_bytes / (1024.0 * 1024.0 * 1024.0);

            // throughput as K/M tok/s
            char tps[24];
            if (inst_tok_s >= 1e6) snprintf(tps, sizeof(tps), "%.2fM", inst_tok_s / 1e6);
            else                   snprintf(tps, sizeof(tps), "%.1fK", inst_tok_s / 1e3);

            printf("[S%d] step=%-7d seq=%d %s %.2f%% | tok=%s/%s | "
                   "loss=%.4f lr=%.2e gn=%.3f cl=%.3f %s tok/s %.1fs/step elapsed=%s ETA %s | "
                   "RAM %.1fGB VRAM %.1fGB GPU(proc) %d%% GPU(sys) %d%%\n",
                   g_current_phase + 1, step, S,
                   fmt_bar(frac).c_str(), frac * 100.0,
                   fmt_tok2(g_tokens_processed).c_str(), fmt_tok2(g_max_tokens).c_str(),
                   avg_loss, lr, last_gnorm, last_clip_coeff, tps, s_per_step,
                   fmt_dur(elapsed).c_str(), fmt_dur(eta).c_str(),
                   ram_gb, vram_gb, stats.gpu_util_pct, stats.gpu_util_pct);

            running_loss = 0.0f;
            running_count = 0;
            t_last_log = t_now;
            tokens_at_last_log = g_tokens_processed;
            step_at_last_log = step;
#ifdef USE_DDP
            }
#endif
        }

        // ─── CHECKPOINT ──────────────────────────────────────────────────
        if (should_checkpoint(g_tokens_processed, step)) {
#ifdef USE_DDP
            if (!g_use_ddp || cuda::ddp::is_main())
#endif
            save_checkpoint(g_checkpoint_dir, params, adamw_m, adamw_v,
                           g_tokens_processed, step);
        }
    }

    // ─── Final checkpoint ─────────────────────────────────────────────────
#ifdef USE_DDP
    if (!g_use_ddp || cuda::ddp::is_main())
#endif
    save_checkpoint(g_checkpoint_dir, params, adamw_m, adamw_v,
                   g_tokens_processed, step, "_final");
    printf("\n[DONE] Training complete. %s tokens in %d steps.\n",
           fmt_tok(g_tokens_processed).c_str(), step);

    // Close diagnostic files
    if (f_train_diag) fclose(f_train_diag);
    if (f_bwd_diag)   fclose(f_bwd_diag);

#ifdef USE_DDP
    if (g_use_ddp) cuda::ddp::shutdown();
#endif
    cuda::nvml_shutdown();
    return 0;
}
