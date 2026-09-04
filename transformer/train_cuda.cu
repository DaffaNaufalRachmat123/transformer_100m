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
#include <sstream>
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
#include "transformer/arch_runtime.h"
#include "transformer/pretrain_dataset.h"
#include "transformer/safetensors.h"
#include "transformer/train_step.cuh"
#include "cuda/ce_embed_kernels.cuh"

#ifdef USE_DDP
#include "cuda/cuda_ddp.h"
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

// ─── Architecture (runtime-configurable via --arch_config) ────────────────
// Defaults = 492M EN/JA: D=1280, L=24, GQA 20Q:5KV (d_head 64), SwiGLU 2.8x.
// Kernels are dimension-agnostic (grad_check passed D=64..3072), so these are
// plain mutable globals refreshed from JSON before any allocation happens.
// The literals below are placeholders; sync_arch() always overwrites them from
// g_arch, whose real defaults live in transformer/model_config.h.
static ModelConfig g_arch;
static std::string g_arch_config = "";  // empty = keep defaults

static int   D_MODEL     = 1280;
static int   N_LAYERS    = 24;
static int   N_Q_HEADS   = 20;
static int   N_KV_HEADS  = 5;
static int   D_HEAD      = 64;
static int   FFN_DIM     = 3584;
static int   MODEL_VOCAB = 49152;
static int   MAX_SEQ_LEN = 2048;
static float ROPE_THETA  = 500000.0f;

static void sync_arch() {
    D_MODEL     = (int)g_arch.d_model;
    N_LAYERS    = (int)g_arch.n_layers;
    N_Q_HEADS   = (int)g_arch.n_q_heads;
    N_KV_HEADS  = (int)g_arch.n_kv_heads;
    D_HEAD      = (int)g_arch.d_head;
    FFN_DIM     = (int)g_arch.ffn_dim;
    MODEL_VOCAB = (int)g_arch.vocab_size;
    MAX_SEQ_LEN = (int)g_arch.max_seq_len;
    ROPE_THETA  = g_arch.rope_theta;
}

// ─── CLI config ────────────────────────────────────────────────────────────
static std::string g_data_dir       = "data/corpus_final";
static std::string g_checkpoint_dir = "checkpoints";
static std::string g_vocab_dir      = "vocab_6";
static std::string g_bpe_path       = "";  // empty = use vocab_dir text format
static std::string g_optimizer      = "adamw";
static uint64_t g_max_tokens        = 2'000'000'000ULL;
static int g_batch_size             = 8;   // T4 16GB fits 8 easily for 100M
static int g_max_seq                = 2048;  // constant 2048 (no curriculum)
static bool g_resume                = false;
static bool g_use_ddp               = false;
static int  g_num_gpus              = 0;    // 0 = auto-detect

// ─── AdamW hyperparams ─────────────────────────────────────────────────────
// LR lowered from 3e-4 -> 2e-4 -> 1.5e-4: at seq_len=512+ the model showed
// sustained slow-creep drift (EMA rising monotonically once LR hits its
// cosine-scheduled plateau) across 2 independent resume sessions from
// tokens ~724M. The cumulative nature of the drift (many small steps in a
// consistent wrong direction, detectable only over 30+ steps) indicates
// the LR is simply too high for this region of the data at this context
// length, not a single-batch spike event. The auto-decay mechanism below
// (Fix 3) provides an additional safety net for any remaining drift.
// ─── Safety mode ───────────────────────────────────────────────────────────
enum class SafetyMode { Legacy, Light, Adaptive };
static SafetyMode g_safety = SafetyMode::Light;
static const char* safety_mode_name(SafetyMode m) {
    if (m == SafetyMode::Legacy) return "legacy";
    if (m == SafetyMode::Adaptive) return "adaptive";
    return "light";
}

// ─── Learning rate ─────────────────────────────────────────────────────────
// Light mode default: 3e-4 (same as the 731M-token run that produced good
// results). Legacy mode: 1.5e-4 (was tightened after observing drift in the
// 256→512 phase transition with the old codebase).
// NOTE: both can be overridden with --lr.
static float g_lr          = 3e-4f;
static float g_beta1       = 0.9f;
static float g_beta2       = 0.95f;
static float g_eps         = 1e-8f;
static float g_wd          = 0.1f;
// Tightened from 1.0 -> 0.5: FP16 gradients clip too loosely at 1.0,
// letting occasional bad batches (e.g. corrupted-tokenization tails from
// long JA/KO lines) push large updates through before being caught.
static float g_grad_clip   = 0.5f;
// FP16 loss scaling. d_logits is scaled by (g_loss_scale / tok_count) and
// stored as FP16, so the effective survival threshold for a softmax entry is
//     p > 5.96e-8 * tok_count / g_loss_scale      (5.96e-8 = min FP16 subnormal)
// With the old default of 1.0 and B=35,S=256 (tok_count=8960) that threshold is
// p > 5.3e-4, while the median softmax probability for a 65536-vocab model is
// ~4.5e-6 — i.e. ~99.6% of the gradient silently flushes to zero. Default is
// therefore 4096, which puts the threshold at ~1.3e-7 and preserves the signal.
// Do not lower this without re-deriving the bound above.
static float g_loss_scale  = 4096.0f;

// ─── Checkpoint config ─────────────────────────────────────────────────────
static int g_save_every    = 500;
static int g_log_every     = 10;
static bool g_overwrite_ckpt = true;  // always overwrite checkpoint
static int g_upload_every  = 0;      // 0 = disabled; N = upload ckpt to GDrive every N steps
static int g_upload_prep_ahead = 20; // prepare zip N steps before upload_every
static std::string g_upload_script = ""; // path to upload script
static int g_debug_steps   = 0;      // 0 = disabled; N = dump diag every N steps

// ─── Eval-during-training config ────────────────────────────────────────────
static uint64_t g_eval_every_tokens = 0;       // 0=disabled; run inference every N tokens
static std::string g_eval_infer_bin = "./build_infer/infer";  // path to infer binary
static uint64_t g_next_eval_tokens  = 0;       // next milestone for eval

// ─── Global state ──────────────────────────────────────────────────────────
static volatile bool g_stop_requested = false;
static uint64_t g_tokens_processed = 0;
static int g_current_phase = 0;
static float gnorm_ema           = 0.0f;
static float lr_adaptive_factor  = 1.0f;

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

// ─── Phase configuration (curriculum learning) ─────────────────────────────
struct PhaseConfig {
    int seq_len;
    uint64_t tokens;
};

static std::vector<PhaseConfig> compute_phases(uint64_t total_tokens) {
    // Single-phase mode: train at the user-specified --max_seq only.
    // No curriculum transitions — the model gets full budget at one seq_len.
    // This is ideal for small models (100M) where long-context curriculum
    // adds complexity without proportional benefit.
    return {{g_max_seq, total_tokens}};
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

// ─── Cosine LR schedule with warmup (TOKEN-based) ─────────────────────────
// Progress is measured in TOKENS CONSUMED, not optimizer steps.
//
// The previous step-based version made the schedule shape depend on the
// hardware, because total_steps was derived as
//     max_tokens / (batch * seq * world)
// so halving the GPU count (or the batch size) doubled total_steps and
// stretched the cosine curve to twice its length. Concretely, at 763M/2B
// tokens the 2xT4 run had already decayed to lr=9.4e-05 (22.5% along its
// curve) while the 1xT4 run was still pinned near peak at 1.35e-04 (13.8%
// along its curve) — a 43% higher LR at the same point in the data, on top
// of the 2x smaller batch. That combination is what drove the instability.
//
// Token-based progress is invariant to world size, batch size and the
// curriculum's mid-run VRAM auto-scaling: 763M/2B tokens is 38.15% of the
// run on any configuration, so a checkpoint resumes onto exactly the same
// LR regardless of the machine it moves to.
//
// `progress` MUST be clamped to [0,1]. cos() is periodic, so an unclamped
// progress > 1 makes the LR climb back up instead of staying at min_lr:
//   progress=1.0 -> min_lr   progress=1.5 -> ~55% of peak   progress=2.0 -> peak
// That is reachable whenever --max_tokens is lowered on resume.
static float cosine_lr_tokens(uint64_t tokens_done, uint64_t warmup_tokens,
                              float max_lr, float min_lr, uint64_t total_tokens) {
    if (warmup_tokens > 0 && tokens_done < warmup_tokens)
        return max_lr * (float)tokens_done / (float)warmup_tokens;
    if (total_tokens <= warmup_tokens) return min_lr;   // degenerate schedule
    float progress = (float)(tokens_done - warmup_tokens) /
                     (float)(total_tokens - warmup_tokens);
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;   // hold at min_lr past the schedule
    return min_lr + 0.5f * (max_lr - min_lr) * (1.0f + cosf(M_PI * progress));
}

// ─── Forward declarations ──────────────────────────────────────────────────
void save_checkpoint(const std::string& dir,
                     const std::vector<cuda::CudaTensor>& params,
                     const std::vector<cuda::CudaTensor>& optimizer_m,
                     const std::vector<cuda::CudaTensor>& optimizer_v,
                     uint64_t tokens_done, int step,
                     float ema_loss_val, float ema_local_min_val, float lr_decay_val,
                     const std::string& suffix = "");

// ─── Eval-during-training: save ckpt to temp dir, run infer, print output ──
void run_eval(const std::string& ckpt_dir,
              const std::vector<cuda::CudaTensor>& params,
              const std::vector<cuda::CudaTensor>& optimizer_m,
              const std::vector<cuda::CudaTensor>& optimizer_v,
              uint64_t tokens_done, int step,
              float ema_loss_val, float ema_local_min_val, float lr_decay_val,
              float loss_val, float lr_val, float gnorm_val,
              int seq_len, int phase) {
    // Check if infer binary exists
    if (!fs::exists(g_eval_infer_bin)) {
        printf("\n");
        printf("╔══════════════════════════════════════════════════════════════════╗\n");
        printf("║  [EVAL] @ %s tokens | step=%d | phase=%d | seq=%d              ║\n",
               fmt_tok(tokens_done).c_str(), step, phase + 1, seq_len);
        printf("║  loss=%.4f  ema=%.4f  gnorm=%.2f  lr=%.2e  decay=%.2f       ║\n",
               loss_val, ema_loss_val, gnorm_val, lr_val, lr_decay_val);
        printf("║  [SKIP] infer binary not found: %s                             ║\n",
               g_eval_infer_bin.c_str());
        printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
        return;
    }

    // Save to a TEMP directory so the main checkpoint dir is untouched
    std::string tmp_dir = ckpt_dir + "/eval_tmp";
    fs::create_directories(tmp_dir);

    // Save model weights only (no optimizer, no train_state) to temp dir
    save_checkpoint(tmp_dir, params, optimizer_m, optimizer_v,
                    tokens_done, step, ema_loss_val, ema_local_min_val, lr_decay_val,
                    "");  // no suffix → saves as ckpt_pretrain.safetensors in tmp_dir

    // Write a temp config.json for the infer binary
    std::string cfg_path = tmp_dir + "/config_eval.json";
    {
        std::ofstream cf(cfg_path);
        cf << "{\n";
        cf << "  \"vocab_size\": " << MODEL_VOCAB << ",\n";
        cf << "  \"d_model\": " << D_MODEL << ",\n";
        cf << "  \"n_layers\": " << N_LAYERS << ",\n";
        cf << "  \"n_q_heads\": " << N_Q_HEADS << ",\n";
        cf << "  \"n_kv_heads\": " << N_KV_HEADS << ",\n";
        cf << "  \"d_head\": " << D_HEAD << ",\n";
        cf << "  \"ffn_dim\": " << FFN_DIM << ",\n";
        cf << "  \"max_seq_len\": " << MAX_SEQ_LEN << ",\n";
        cf << "  \"rope_theta\": " << ROPE_THETA << ",\n";
        cf << "  \"weight_tying\": true,\n";
        cf << "  \"pad_token_id\": " << g_arch.pad_token_id << ",\n";
        cf << "  \"eos_token_id\": " << g_arch.eos_token_id << ",\n";
        cf << "  \"bos_token_id\": " << g_arch.bos_token_id << "\n";
        cf << "}\n";
        cf.close();
    }

    std::string model_path = tmp_dir + "/ckpt_pretrain.safetensors";
    std::string vocab_path = g_vocab_dir;

    // Test prompts (EN + JA only, factual NTP)
    const char* prompts[] = {
        "The capital of Japan is",
        "the chemical formula of water is",
        "the largest planet in the solar system is",
        "the currency of the United States is",
        "the color of the sky is",
        "日本の首都は",
        "地球の衛星は",
        "水の化学式は",
        "日本の通貨は",
        "人間の足の数は",
    };
    const char* labels[] = {"EN","EN","EN","EN","EN","JA","JA","JA","JA","JA"};

    // Everything below is accumulated into `out` so the exact same text can be
    // (a) printed to stdout as before and (b) appended to eval_log.txt, without
    // duplicating every printf. Formatting is done with snprintf into a scratch
    // buffer to keep the printf-style format strings intact.
    std::ostringstream out;
    char lb[2048];
    #define EMIT(...) do { snprintf(lb, sizeof(lb), __VA_ARGS__); out << lb; } while (0)

    {
        std::time_t now = std::time(nullptr);
        char ts[64];
        std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
        EMIT("\n");
        EMIT("╔══════════════════════════════════════════════════════════════════╗\n");
        EMIT("║  [EVAL] @ %s tokens | step=%d | phase=%d | seq=%d              ║\n",
             fmt_tok(tokens_done).c_str(), step, phase + 1, seq_len);
        EMIT("║  %s\n", ts);
        EMIT("║  loss=%.4f  ema=%.4f  gnorm=%.2f  lr=%.2e  decay=%.2f       ║\n",
             loss_val, ema_loss_val, gnorm_val, lr_val, lr_decay_val);
        EMIT("║  perplexity = e^%.4f = %.1f                                     ║\n",
             loss_val, expf(loss_val));
        EMIT("╠══════════════════════════════════════════════════════════════════╣\n");
    }

    for (int i = 0; i < 10; i++) {
        std::string cmd = g_eval_infer_bin
            + " --model " + model_path
            + " --config " + cfg_path
            + " --vocab " + vocab_path
            + " --prompt \"" + prompts[i] + "\""
            + " --max_tokens 8 --temperature 0.8 --top_p 0.9 --rep_penalty 1.3"
            + " --show_probs --show_probs_topk 3"
            + " 2>&1";

        FILE* pipe = popen(cmd.c_str(), "r");
        std::string generated;
        std::vector<std::string> prob_lines;
        if (pipe) {
            char buf[1024];
            while (fgets(buf, sizeof(buf), pipe)) {
                std::string line(buf);
                // Probability lines from --show_probs (stderr, captured via 2>&1).
                // This MUST be checked BEFORE the metadata filter below: that
                // filter drops any line starting with two spaces, which would
                // otherwise swallow every "    [" probability line.
                if (line.rfind("    [", 0) == 0) {
                    prob_lines.push_back(line);
                    continue;
                }
                // Skip metadata/log lines
                if (line.empty() || line[0] == '[' ||
                    (line.size() >= 2 && line[0] == ' ' && line[1] == ' ') ||
                    line.rfind("SafeTensors:", 0) == 0 ||
                    line.rfind("TOTAL:", 0) == 0)
                    continue;
                generated += line;
            }
            (void)pclose(pipe);
        }
        // Trim whitespace
        while (!generated.empty() && (generated[0] == '\n' || generated[0] == '\r'))
            generated.erase(0, 1);
        while (!generated.empty() && (generated.back() == '\n' || generated.back() == '\r'))
            generated.pop_back();

        EMIT("║  [%s] %s\n", labels[i], prompts[i]);
        EMIT("║  → %s\n", generated.empty() ? "(no output)" : generated.c_str());
        if (!prob_lines.empty()) {
            EMIT("║  Top tokens:\n");
            for (size_t p = 0; p < prob_lines.size() && p < 4; p++) {
                // Strip leading spaces and newline
                std::string pl = prob_lines[p];
                while (!pl.empty() && (pl[0] == ' ' || pl[0] == '\n' || pl[0] == '\r'))
                    pl.erase(0, 1);
                while (!pl.empty() && (pl.back() == '\n' || pl.back() == '\r'))
                    pl.pop_back();
                EMIT("║    %s\n", pl.c_str());
            }
            if (prob_lines.size() > 4)
                EMIT("║    ... (%zu more steps)\n", prob_lines.size() - 4);
        }
        EMIT("║\n");
    }

    EMIT("╚══════════════════════════════════════════════════════════════════╝\n\n");
    #undef EMIT

    // ─── Print to stdout (unchanged behaviour) ─────────────────────────────
    const std::string report = out.str();
    fputs(report.c_str(), stdout);
    fflush(stdout);

    // ─── Persist to eval_log.txt, then upload ──────────────────────────────
    // Written to checkpoint_dir (NOT tmp_dir — that gets deleted below).
    // Appended so one file accumulates the full history across milestones.
    //
    // Every step here is best-effort: eval logging must NEVER interrupt
    // training. Any failure is reported and swallowed.
    try {
        std::string log_path = ckpt_dir + "/eval_log.txt";
        {
            std::ofstream lf(log_path, std::ios::app);
            if (lf) {
                lf << report;
                lf.flush();
            } else {
                printf("[EVAL] WARN: cannot open %s for append — skipping upload\n",
                       log_path.c_str());
                log_path.clear();
            }
        }

        if (!log_path.empty()) {
            std::string script = g_upload_script.empty()
                ? "python3 scripts/upload_gdrive.py"
                : g_upload_script;
            // Synchronous: the file is only a few KB, so this finishes in well
            // under a second. Running it async would race with the next eval's
            // append. Exit status is deliberately ignored.
            std::string cmd = script
                + " --eval_log " + log_path
                + " --step " + std::to_string(step)
                + " --tokens " + std::to_string(tokens_done);
            int rc = system(cmd.c_str());
            if (rc != 0)
                printf("[EVAL] WARN: eval_log upload failed (rc=%d) — training continues\n", rc);
        }
    } catch (const std::exception& e) {
        printf("[EVAL] WARN: eval_log write/upload error: %s — training continues\n", e.what());
    } catch (...) {
        printf("[EVAL] WARN: eval_log write/upload failed — training continues\n");
    }
    fflush(stdout);

    // Clean up temp directory
    try { fs::remove_all(tmp_dir); }
    catch (...) { /* best effort */ }
}

// ─── Checkpoint save (overwrite mode) ──────────────────────────────────────
void save_checkpoint(const std::string& dir,
                     const std::vector<cuda::CudaTensor>& params,
                     const std::vector<cuda::CudaTensor>& optimizer_m,
                     const std::vector<cuda::CudaTensor>& optimizer_v,
                     uint64_t tokens_done, int step,
                     float ema_loss_val, float ema_local_min_val, float lr_decay_val,
                     const std::string& suffix) {
    fs::create_directories(dir);

    std::string model_path = dir + "/ckpt_pretrain" + suffix + ".safetensors";
    std::string optim_path = dir + "/ckpt_optim"  + suffix + ".safetensors";
    std::string state_path = dir + "/train_state.txt";
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
    {
        std::ofstream sf(state_path);
        sf << "tokens_processed=" << tokens_done << "\n";
        sf << "step=" << step << "\n";
        sf << "phase=" << g_current_phase << "\n";
        sf << "ema_loss=" << ema_loss_val << "\n";
        sf << "ema_local_min=" << ema_local_min_val << "\n";
        sf << "lr_decay_factor=" << lr_decay_val << "\n";
        sf << "safety_mode=" << safety_mode_name(g_safety) << "\n";
        sf << "lr=" << g_lr << "\n";
        sf << "grad_clip=" << g_grad_clip << "\n";
        sf << "gnorm_ema=" << gnorm_ema << "\n";
        sf << "lr_adaptive_factor=" << lr_adaptive_factor << "\n";
        sf.close();
    }

    // ─── Save optimizer states (FP32: m, v, master) ────────────────────────
    {
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

// ─── Checkpoint frequency ──────────────────────────────────────────────────
bool should_checkpoint(int step) {
    return (step % g_save_every == 0) && step > 0;
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
        else if (arg == "--arch_config" && i+1 < argc) g_arch_config = argv[++i];
        else if (arg == "--vocab_dir" && i+1 < argc)  g_vocab_dir = argv[++i];
        else if (arg == "--bpe" && i+1 < argc)        g_bpe_path = argv[++i];
        else if (arg == "--max_tokens" && i+1 < argc) g_max_tokens = std::stoull(argv[++i]);
        else if (arg == "--batch_size" && i+1 < argc) g_batch_size = std::stoi(argv[++i]);
        else if (arg == "--max_seq" && i+1 < argc)    g_max_seq = std::stoi(argv[++i]);
        else if (arg == "--lr" && i+1 < argc)         g_lr = std::stof(argv[++i]);
        else if (arg == "--grad_clip" && i+1 < argc) g_grad_clip = std::stof(argv[++i]);
        else if (arg == "--safety" && i+1 < argc) {
            std::string sm = argv[++i];
            if (sm == "legacy")      g_safety = SafetyMode::Legacy;
            else if (sm == "light")  g_safety = SafetyMode::Light;
            else if (sm == "adaptive") g_safety = SafetyMode::Adaptive;
            else { fprintf(stderr, "[ERR] --safety must be 'legacy', 'light', or 'adaptive'\n"); return 1; }
        }
        else if (arg == "--save_every" && i+1 < argc) g_save_every = std::stoi(argv[++i]);
        else if (arg == "--log_every" && i+1 < argc)  g_log_every = std::stoi(argv[++i]);
        else if (arg == "--loss_scale" && i+1 < argc) g_loss_scale = std::stof(argv[++i]);
        else if (arg == "--upload_every" && i+1 < argc) g_upload_every = std::stoi(argv[++i]);
        else if (arg == "--upload_prep_ahead" && i+1 < argc) g_upload_prep_ahead = std::stoi(argv[++i]);
        else if (arg == "--upload_script" && i+1 < argc) g_upload_script = argv[++i];
        else if (arg == "--debug_steps" && i+1 < argc) g_debug_steps = std::stoi(argv[++i]);
        else if (arg == "--eval_every_tokens" && i+1 < argc) g_eval_every_tokens = std::stoull(argv[++i]);
        else if (arg == "--eval_infer_bin" && i+1 < argc)   g_eval_infer_bin = argv[++i];
        else if (arg == "--resume")                   g_resume = true;
        else if (arg == "--ddp")                      g_use_ddp = true;
        else if (arg == "--num_gpus" && i+1 < argc)   g_num_gpus = std::stoi(argv[++i]);
        else if (arg == "--help") {
            printf("Usage: train_pretrain_cuda [options]\n");
            printf("  --data_dir <path>         Corpus directory [data/corpus_final]\n");
            printf("  --checkpoint_dir <path>   Checkpoint output [checkpoints]\n");
            printf("  --arch_config <path>      Architecture JSON [built-in 492M]\n");
            printf("  --vocab_dir <path>        Vocab directory [vocab_6]\n");
            printf("  --bpe <path>              BPE binary model path (optional)\n");
            printf("  --max_tokens <N>          Token budget [9B]\n");
            printf("  --batch_size <N>          Batch size per GPU [8]\n");
            printf("  --max_seq <N>             Initial max seq length [256]\n");
            printf("  --lr <float>              Peak learning rate [3e-4 light / 1.5e-4 legacy]\n");
            printf("  --grad_clip <float>        Gradient clipping threshold [0.5]\n");
            printf("  --loss_scale <float>      FP16 loss scale [4096] (do not set below ~1024)\n");
            printf("  --safety <mode>           Safety: 'light', 'legacy', or 'adaptive' [light]\n");
            printf("    light  — no auto-decay, soft phase cooldown, resume-safe\n");
            printf("    legacy — full auto-decay + aggressive cooldown (original)\n");
            printf("  --save_every <N>          Checkpoint every N steps [500]\n");
            printf("  --eval_every_tokens <N>   Run inference eval every N tokens [0=off]\n");
            printf("  --eval_infer_bin <PATH>   Path to infer binary [./build_infer/infer]\n");
            printf("  --log_every <N>           Log every N steps [10]\n");
            printf("  --resume                  Resume from checkpoint\n");
            printf("  --ddp                     Enable multi-GPU DDP (NCCL)\n");
            printf("  --num_gpus <N>            Number of GPUs for DDP [2]\n");
            return 0;
        }
    }

    // ─── Init CUDA ─────────────────────────────────────────────────────────
    // Architecture must be resolved before any tensor is sized.
    if (!g_arch_config.empty() && !arch::load(g_arch_config, g_arch)) return 1;
    sync_arch();

    // RoPE tables are cut to MAX_SEQ_LEN; a longer --max_seq would index past them.
    if (g_max_seq > MAX_SEQ_LEN) {
        fprintf(stderr, "[ERR] --max_seq %d exceeds max_seq_len %d in the arch config\n",
                g_max_seq, MAX_SEQ_LEN);
        return 1;
    }

    // Auto-detect GPU count if not specified
    if (g_use_ddp && g_num_gpus == 0) {
        cudaGetDeviceCount(&g_num_gpus);
        if (g_num_gpus < 2) {
            printf("[DDP] Only %d GPU detected — disabling DDP, running single GPU\n", g_num_gpus);
            g_use_ddp = false;
            g_num_gpus = 1;
        } else {
            printf("[DDP] Auto-detected %d GPUs\n", g_num_gpus);
        }
    }
    if (!g_use_ddp) g_num_gpus = 1;

    // For DDP we MUST fork BEFORE any CUDA call (CUDA context does not survive
    // fork). So: fork first -> ctx.init(local device) -> create NCCL comm.
#ifdef USE_DDP
    int ddp_device = 0;
    if (g_use_ddp) {
        if (!cuda::ddp::init(g_num_gpus)) {
            fprintf(stderr, "[ERR] DDP fork failed\n");
            return 1;
        }
        ddp_device = cuda::ddp::device_for_rank();
    }
    cuda::ctx.init(ddp_device);
    cuda::nvml_init();
    if (g_use_ddp) {
        if (!cuda::ddp::init_nccl()) {
            fprintf(stderr, "[ERR] NCCL init failed\n");
            return 1;
        }
    }
#else
    cuda::ctx.init(0);
    cuda::nvml_init();
#endif

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Transformer — CUDA Pre-training (sm_75 / T4)           ║\n");
    printf("║  D=%d L=%d QH=%d KVH=%d DH=%d FFN=%d V=%d        ║\n",
           D_MODEL, N_LAYERS, N_Q_HEADS, N_KV_HEADS, D_HEAD, FFN_DIM, MODEL_VOCAB);
    printf("║  Params: %.1fM                                           ║\n",
           g_arch.param_count() / 1e6);
    printf("║  GPU: %s | FP16 Mixed Precision | AdamW       ║\n",
           g_use_ddp ? "T4 x2 (DDP)" : "Single T4   ");
    printf("║  Tokens: %-10s | Batch: %-3d | Seq: %-4d            ║\n",
           fmt_tok(g_max_tokens).c_str(), g_batch_size, g_max_seq);
printf("║  Safety: %-7s | LR: %-8.1e                             ║\n",
           safety_mode_name(g_safety), g_lr);
    printf("║  Grad clip: %-5.1f                                       ║\n",
           g_grad_clip);
    if (g_eval_every_tokens > 0)
        printf("║  Eval: every %s tokens                                  ║\n",
               fmt_tok(g_eval_every_tokens).c_str());
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    // Init eval milestone
    if (g_eval_every_tokens > 0) {
        g_next_eval_tokens = g_eval_every_tokens;
        // If resuming past a milestone, skip to next one
        while (g_next_eval_tokens <= g_tokens_processed)
            g_next_eval_tokens += g_eval_every_tokens;
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

    // ─── Spike-detection / auto-decay / cooldown state ─────────────────────
    // Declared before the resume block because train_state.txt parsing
    // (inside the resume block below) needs read/write access to these.
    float ema_loss = logf((float)MODEL_VOCAB);  // uniform-prediction loss at init
    bool  ema_initialized = false;
    static constexpr float kEmaBeta = 0.95f;
    float last_grad_norm = 0.0f;

    // ─── Cross-platform resume: detect LR/clip changes, trigger warmup ────
    float saved_lr = g_lr;           // default: same as current
    float saved_grad_clip = g_grad_clip;
    bool lr_changed = false;
    bool clip_changed = false;

    float ema_local_min = 11.1f;    // running local minimum of EMA
    float lr_decay_factor = 1.0f;   // LR multiplier (1.0 = no decay)
    int steps_since_last_decay_trigger = 0;
    bool last_decay_was_recovery = false;

    // ─── GLALR: gnorm-loss adaptive LR (adaptive mode only) ─────────────────
    static constexpr float kGnormEmaBeta       = 0.95f;
    static constexpr float kGnormEmaThreshold  = 5.0f;
    static constexpr int   kGnormHighSteps     = 10;
    static constexpr int   kGnormRecoverySteps = 50;
    static constexpr float kLrAdaptiveDecay    = 0.5f;
    static constexpr float kLrAdaptiveFloor    = 0.2f;
    // gnorm_ema & lr_adaptive_factor are file-scope (persist across ckpt)
    int   gnorm_high_cnt     = 0;
    int   gnorm_stable_cnt   = 0;
    std::vector<float> loss_history;
    int   loss_history_idx = 0;

    // ─── Phase cooldown config (mode-dependent) ──────────────────────────────
    // Light mode: softer cooldown, shorter ramp — designed for frequent resume
    // Legacy mode: aggressive cooldown — original behaviour
    const int   kPhaseWarmupSteps   = (g_safety == SafetyMode::Legacy) ? 200 : 100;
    const float kPhaseCooldownFloor = (g_safety == SafetyMode::Legacy) ? 0.3f : 0.5f;
    // Auto-decay constants (used only in legacy mode)
    static constexpr float kAutoDecayThreshold  = 0.08f;
    static constexpr float kAutoDecayFactor     = 0.7f;
    static constexpr float kMinLrDecayFactor    = 0.2f;
    static constexpr int   kAutoDecayCheckEvery  = 20;
    static constexpr float kRecoveryTolerance   = 0.02f;
    static constexpr int   kRecoveryStableSteps  = 100;
    int phase_switch_step = 0;    // set after resume block

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
                    auto parse_ull = [&](const char* field, size_t off) {
                        try { return std::stoull(line.substr(off)); }
                        catch (const std::exception& e) {
                            printf("[RESUME-ERR] stoull('%s') failed: %s — line='%s'\n",
                                   line.substr(off).c_str(), e.what(), line.c_str());
                            throw;
                        }
                    };
                    auto parse_int = [&](const char* field, size_t off) {
                        try { return std::stoi(line.substr(off)); }
                        catch (const std::exception& e) {
                            printf("[RESUME-ERR] stoi('%s') failed: %s — line='%s'\n",
                                   line.substr(off).c_str(), e.what(), line.c_str());
                            throw;
                        }
                    };
                    auto parse_flt = [&](const char* field, size_t off) {
                        try { return std::stof(line.substr(off)); }
                        catch (const std::exception& e) {
                            printf("[RESUME-ERR] stof('%s') failed: %s — line='%s'\n",
                                   line.substr(off).c_str(), e.what(), line.c_str());
                            throw;
                        }
                    };
                    if (line.find("tokens_processed=") == 0)
                        g_tokens_processed = parse_ull("tokens_processed", 17);
                    else if (line.find("step=") == 0)
                        start_step = parse_int("step", 5);
                    else if (line.find("phase=") == 0)
                        g_current_phase = parse_int("phase", 6);
                    else if (line.find("lr_decay_factor=") == 0) {
                        lr_decay_factor = parse_flt("lr_decay_factor", 16);
                        if (!std::isfinite(lr_decay_factor))
                            lr_decay_factor = 1.0f;
                        if (g_safety == SafetyMode::Light)
                            lr_decay_factor = 1.0f;
                    }
                    else if (line.find("ema_local_min=") == 0) {
                        ema_local_min = parse_flt("ema_local_min", 14);
                        if (std::isnan(ema_local_min) || std::isinf(ema_local_min))
                            ema_local_min = 11.1f;
                    }
                    else if (line.find("ema_loss=") == 0) {
                        ema_loss = parse_flt("ema_loss", 9);
                        ema_initialized = !std::isnan(ema_loss) && !std::isinf(ema_loss);
                        if (!ema_initialized)
                            printf("[RESUME] EMA was NaN/Inf — reinitializing\n");
                    }
                    else if (line.find("lr=") == 0) {
                        saved_lr = parse_flt("lr", 3);
                        if (!std::isfinite(saved_lr)) saved_lr = g_lr;
                    }
                    else if (line.find("grad_clip=") == 0) {
                        saved_grad_clip = parse_flt("grad_clip", 10);
                        if (!std::isfinite(saved_grad_clip)) saved_grad_clip = g_grad_clip;
                    }
                    else if (line.find("gnorm_ema=") == 0) {
                        gnorm_ema = parse_flt("gnorm_ema", 10);
                        if (!std::isfinite(gnorm_ema)) gnorm_ema = 0.0f;
                    }
                    else if (line.find("lr_adaptive_factor=") == 0) {
                        lr_adaptive_factor = parse_flt("lr_adaptive_factor", 19);
                        if (!std::isfinite(lr_adaptive_factor)) lr_adaptive_factor = 1.0f;
                    }
                }
            }
            // Force one warmup ramp at resume: the cooldown ramp scales LR
            // down to ~30% and linearly climbs back over kPhaseWarmupSteps
            // steps. This is deliberate — the model may have drifted since
            // the checkpoint was saved (e.g. if the optimizer state is from
            // a slightly different data pipeline), and easing back in is
            // safer than a cold jump to the full cosine-scheduled LR.
            // NOTE: Legacy mode only. Light mode skips this because it
            // assumes the user knows what they're doing with frequent resumes.
            if (g_safety == SafetyMode::Legacy)
                phase_switch_step = start_step;
            else
                phase_switch_step = start_step - 1'000'000;  // sentinel, no trigger

            // ─── Cross-platform resume: detect LR/clip changes ────────────
            lr_changed = (std::abs(g_lr - saved_lr) / std::max(saved_lr, 1e-8f) > 0.01f);
            clip_changed = (std::abs(g_grad_clip - saved_grad_clip) > 0.01f);
            if (lr_changed || clip_changed) {
                phase_switch_step = start_step;  // enable warmup ramp
                printf("[RESUME] Platform change detected:\n");
                if (lr_changed)
                    printf("  LR: %.1e → %.1e (%.1f%% change)\n",
                           saved_lr, g_lr, 100.0f * (g_lr - saved_lr) / saved_lr);
                if (clip_changed)
                    printf("  Grad clip: %.1f → %.1f\n", saved_grad_clip, g_grad_clip);
                printf("  Warmup enabled for %d steps to adapt optimizer state.\n",
                       kPhaseWarmupSteps);
                if (lr_changed && g_lr > saved_lr * 1.5f)
                    printf("  [WARN] LR increased >50%% — monitor gnorm for spikes.\n");
            }
            printf("[RESUME] Restored at step=%d, tokens=%s\n",
                   start_step, fmt_tok(g_tokens_processed).c_str());
        } else {
            printf("[RESUME] No checkpoint found — starting fresh\n");
        }
    }
    // For fresh start: sentinel far in the past so the cooldown ramp never
    // spuriously triggers early in phase 1. For resume it's already set
    // inside the if (g_resume) block above.
    if (!g_resume) phase_switch_step = start_step - 1'000'000;

    // ─── Pre-compute RoPE tables ────────────────────────────────────────────
    RoPETables rope;
    rope.init(MAX_SEQ_LEN, D_HEAD, ROPE_THETA);
    printf("[RoPE] Tables pre-computed (max_seq=%d, d_head=%d)\n", MAX_SEQ_LEN, D_HEAD);

    // ─── Init dataset ──────────────────────────────────────────────────────
    PretrainDatasetConfig ds_cfg;
    ds_cfg.corpus_root = g_data_dir;
    ds_cfg.seq_len = g_max_seq;
    ds_cfg.batch_size = g_batch_size;
    // Dataset RNG seed. This MUST advance with training progress.
    //
    // PretrainDataset seeds its RNG once in the constructor (rng_(cfg.seed)),
    // and that RNG picks both the file and the random byte offset for every
    // shard. train_state.txt does not persist any dataset position, so a
    // hardcoded seed meant every --resume replayed the exact same file/offset
    // sequence from the beginning. Symptom: identical prefetch lines across
    // sessions ("wiki_ja.txt lines=24196" twice) and a loss that dropped ~200x
    // faster than the token budget allows — memorisation, not learning.
    //
    // Deriving the seed from g_tokens_processed (already restored from the
    // checkpoint at this point — see the resume block above) gives each
    // session a different data stream at zero cost, with no train_state.txt
    // format change and no loss of checkpoint compatibility.
    //
    // The *1000 stride leaves room for the rank offset below without any
    // (tokens, rank) pair ever colliding with another.
    ds_cfg.seed = 42 + (g_tokens_processed / 1000000) * 1000;
#ifdef USE_DDP
    // Each DDP rank reads a different data stream (otherwise both GPUs train on
    // identical batches, wasting the second GPU).
    if (g_use_ddp) ds_cfg.seed += (uint64_t)cuda::ddp::rank();
#endif
    ds_cfg.weighted_sampling = true;

    printf("[DATA] dataset seed=%llu (from tokens=%s)\n",
           (unsigned long long)ds_cfg.seed, fmt_tok(g_tokens_processed).c_str());

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

    // ─── LR schedule (token-based, hardware-invariant) ─────────────────────
    const int cli_batch_size = g_batch_size;  // remember original for VRAM auto-scaling
    // Progress is measured in tokens, so world size / batch size / the
    // curriculum's VRAM auto-scaler cannot reshape the curve. A run is 38%
    // done at 763M/2B tokens on 1 GPU or 8, which means a checkpoint moved
    // between machines resumes onto exactly the same LR.
    //
    // Warmup is 1% of the budget (20M tokens at 2B), capped so tiny debug
    // runs still get a warmup that is short relative to the run.
    const uint64_t warmup_tokens = std::min<uint64_t>(g_max_tokens / 100,
                                                      20'000'000ULL);
    float min_lr = g_lr * 0.1f;
    // Steps are still reported for readability; they are no longer used by
    // the schedule itself.
    const int lr_world = (g_use_ddp && g_num_gpus > 1) ? g_num_gpus : 1;
    int est_total_steps = (int)std::max<uint64_t>(1,
        g_max_tokens / ((uint64_t)g_batch_size * (uint64_t)g_max_seq * (uint64_t)lr_world));
    printf("[LR] schedule: token-based | warmup=%s / %s tokens "
           "peak=%.2e min=%.2e (est_steps=%d world=%d)\n",
           fmt_tok(warmup_tokens).c_str(), fmt_tok(g_max_tokens).c_str(),
           g_lr, min_lr, est_total_steps, lr_world);

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

    // ─── Init CUDA memory pool (arena allocator) ────────────────────────────
    // All persistent allocations (params, optimizer, grads) are done above.
    // The arena handles all transient allocations in forward/backward (O(1) each).
    cuda::arena.init_auto(0.75f);  // 75% of remaining free VRAM

    // Pre-allocate gradient clipping scalar (avoid per-step cudaMalloc)
    float* d_norm_sq = nullptr;
    cudaMalloc(&d_norm_sq, sizeof(float));

    while (g_tokens_processed < g_max_tokens && !g_stop_requested) {
        // ─── Reset arena pool (reclaim all transient memory from last step) ─
        cuda::arena.reset();
        // Update seq_len based on curriculum phase
        int cur_seq = (int)get_seq_len_for_tokens(g_tokens_processed, phases);
        if (cur_seq != g_max_seq) {
            g_max_seq = cur_seq;
            // ─── Auto-adjust batch size to fit VRAM (conservative for T4) ──
            // Use lookup table with ~15% safety margin instead of formula
            if      (g_max_seq <= 256)  g_batch_size = cli_batch_size;
            else if (g_max_seq <= 512)  g_batch_size = std::max(1, (int)(cli_batch_size * 256 / g_max_seq * 0.85f));
            else if (g_max_seq <= 1024) g_batch_size = std::max(1, (int)(cli_batch_size * 256 / g_max_seq * 0.85f));
            else                        g_batch_size = std::max(1, (int)(cli_batch_size * 256 / g_max_seq * 0.85f));

            // Update dims for forward/backward pass
            dims.B = g_batch_size;
            dims.S = g_max_seq;

            // Update dataset to pack sequences with new seq_len
            dataset.set_seq_len((uint32_t)g_max_seq);

            // Advance phase counter for logging
            g_current_phase++;

            // Mark this step so the LR schedule below applies a short
            // cooldown+re-warmup ramp instead of jumping straight back to
            // the full cosine-scheduled LR (see kPhaseWarmupSteps below).
            // Only trigger for genuine mid-run phase transitions, not the
            // seq_len resync that happens on the very first iteration after
            // a resume (that is handled explicitly in the resume block).
            if (step > start_step)
                phase_switch_step = step;

            printf("[PHASE] Switching to seq_len=%d batch=%d at %s tokens (phase %d)\n",
                   g_max_seq, g_batch_size, fmt_tok(g_tokens_processed).c_str(),
                   g_current_phase + 1);
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
        float lr = cosine_lr_tokens(g_tokens_processed, warmup_tokens,
                                    g_lr, min_lr, g_max_tokens);
        // Phase-transition cooldown: for kPhaseWarmupSteps steps after a
        // curriculum seq_len switch, scale the LR down to kPhaseCooldownFloor
        // and linearly ramp back up to 1.0x. This gives the optimizer time
        // to adapt to the new sequence length / gradient noise profile
        // instead of taking full-sized steps immediately, which is what
        // triggered the earlier divergence at the 256->512 boundary.
        {
            int steps_since_switch = step - phase_switch_step;
            if (steps_since_switch >= 0 && steps_since_switch < kPhaseWarmupSteps) {
                float ramp = kPhaseCooldownFloor +
                             (1.0f - kPhaseCooldownFloor) *
                             ((float)steps_since_switch / (float)kPhaseWarmupSteps);
                lr *= ramp;
            }
        }
        // Auto-decay/recovery multiplier: lr_decay_factor drifts below 1.0
        // when a slow-creep EMA rise is detected, and back toward 1.0 when
        // the EMA proves stable over an extended window (see block after
        // the spike-detector above). Applied *after* the phase-cooldown ramp
        // so the two mechanisms compose cleanly.
        lr *= lr_decay_factor;
        // GLALR: adaptive LR gating (adaptive mode only)
        if (g_safety == SafetyMode::Adaptive)
            lr *= lr_adaptive_factor;

        // ─── FORWARD PASS (correct GQA attention + stored activations) ─────
        dims.B = B; dims.S = S;
        ts::FwdCache cache;
        cuda::CudaTensor x_final = ts::forward(params, d_tokens, dims,
                                               rope.cos_dev, rope.sin_dev, cache);

        // ─── OOM CHECK: detect CUDA out-of-memory before it corrupts state ───
        {
            cudaError_t err = cudaGetLastError();
            if (err == cudaErrorMemoryAllocation || err == cudaErrorLaunchFailure) {
                printf("\n[OOM] CUDA out-of-memory at step=%d seq=%d batch=%d!\n", step, S, B);
                printf("[OOM] Skipping checkpoint save to protect existing checkpoint.\n");
                printf("[OOM] Uploading last good checkpoint to Google Drive...\n");
                // Trigger emergency upload (sync, not async)
                std::string script = g_upload_script.empty()
                    ? "python3 scripts/upload_gdrive.py"
                    : g_upload_script;
                std::string cmd = script
                    + " --ckpt_dir " + g_checkpoint_dir
                    + " --step " + std::to_string(step)
                    + " --tokens " + std::to_string(g_tokens_processed)
                    + " --zip";
                printf("[OOM] Running: %s\n", cmd.c_str());
                (void)system(cmd.c_str());
                printf("[OOM] Exiting to prevent further damage.\n");
#ifdef USE_DDP
                if (g_use_ddp) cuda::ddp::shutdown();
#endif
                cuda::nvml_shutdown();
                return 2;  // exit code 2 = OOM
            }
        }
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

        // ─── SPIKE DETECTION: skip optimizer on catastrophic loss ───────────
        // Compare against a slow-moving EMA baseline instead of the raw
        // previous step's loss. The old check (loss > prev*1.5 && loss > 8.0)
        // almost never fired in practice for this model: normal loss sits in
        // the 3.5-4.7 range, so an 8.0 absolute floor meant real divergence
        // (e.g. 3.85 -> 4.6, a +19% sustained regression) went completely
        // undetected. Thresholds below are tuned to this model's loss scale.
        bool skip_this_step = false;
        if (!ema_initialized) {
            ema_loss = loss_val;
            ema_initialized = true;
        } else if (step > 10 && loss_val > ema_loss * 1.4f && loss_val > 5.5f) {
            printf("[SKIP] step=%d loss=%.4f (ema=%.4f, ratio=%.2f) — skipping optimizer\n",
                   step, loss_val, ema_loss, loss_val / ema_loss);
            skip_this_step = true;
        }
        // NaN/Inf loss: corrupted forward pass — skip entire step
        if (std::isnan(loss_val) || std::isinf(loss_val)) {
            printf("[NAN/INF] loss=%.4f at step %d — skipping optimizer\n", loss_val, step);
            skip_this_step = true;
        }
        // Fold the loss into the EMA baseline. Normal steps use the fast
        // beta so the baseline tracks real progress. Detected spikes use a
        // much slower beta (trickle) so a single bad batch doesn't drag the
        // baseline up, but a *sustained* regime shift (e.g. genuinely harder
        // data at a new curriculum phase) still eventually gets absorbed
        // instead of skipping every step forever.
        if (!skip_this_step)
            ema_loss = kEmaBeta * ema_loss + (1.0f - kEmaBeta) * loss_val;
        else if (!std::isnan(loss_val) && !std::isinf(loss_val))
            ema_loss = 0.999f * ema_loss + 0.001f * loss_val;
        // NaN/Inf loss: leave EMA unchanged to avoid permanent NaN corruption

        // ─── AUTO-DECAY/RECOVERY: drift-based LR gating (LEGACY ONLY) ─────────
        // Light mode: skip entirely. The auto-decay ratchet was the primary
        // cause of LR degradation across sessions. Light mode relies on
        // spike detection + gradient clipping for safety, which is sufficient
        // for a 100M model with AdamW and frequent checkpoints.
        //
        // Legacy mode: full auto-decay as originally implemented.
        // The spike-detector above only catches instantaneous jumps. Slow
        // drift (EMA rising ~1-2% per logging interval, sustained for 50+
        // steps) sneaks past it because the baseline itself drifts up with
        // the trend. This block samples EMA periodically and gates the LR
        // via a multiplicative factor that decays when drift is detected
        // and slowly recovers when the EMA proves stable.
        if (g_safety == SafetyMode::Legacy) {
            ema_local_min = std::min(ema_local_min, ema_loss);
            if (!skip_this_step)
                steps_since_last_decay_trigger++;

            if (step % kAutoDecayCheckEvery == 0) {
                if (ema_loss > ema_local_min * (1.0f + kAutoDecayThreshold)) {
                    lr_decay_factor = std::max(kMinLrDecayFactor,
                                               lr_decay_factor * kAutoDecayFactor);
                    printf("[AUTO-DECAY] EMA drift +%.1f%% | local-min=%.4f ema=%.4f "
                           "-> lr_decay_factor=%.2f\n",
                           100.0f * (ema_loss / ema_local_min - 1.0f),
                           ema_local_min, ema_loss, lr_decay_factor);
                    ema_local_min = ema_loss;
                    steps_since_last_decay_trigger = 0;
                    last_decay_was_recovery = false;
                } else if (lr_decay_factor < 1.0f &&
                           steps_since_last_decay_trigger >= kRecoveryStableSteps &&
                           ema_loss <= ema_local_min * (1.0f + kRecoveryTolerance)) {
                    lr_decay_factor = std::min(1.0f,
                                               lr_decay_factor / kAutoDecayFactor);
                    printf("[AUTO-RECOVER] EMA stable for %d steps (%.4f <= %.4f) "
                           "-> lr_decay_factor=%.2f\n",
                           steps_since_last_decay_trigger, ema_loss,
                           ema_local_min * (1.0f + kRecoveryTolerance),
                           lr_decay_factor);
                    steps_since_last_decay_trigger = 0;
                    last_decay_was_recovery = true;
                }
            }
        }

        // Scale d_logits by (loss_scale / tok_count) to amplify gradients in FP16
        if (!skip_this_step && tok_count > 0)
            launch_scale_fp16((__half*)d_logits.data, g_loss_scale / tok_count,
                              d_logits.numel(), cuda::ctx.compute_stream);

        // ─── BACKWARD ──────────────────────────────────────────────────────
        if (!skip_this_step) {
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

        // ─── UN-SCALE gradients (divide by loss_scale) ───────────────────────
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

        // ─── GRADIENT CLIPPING (global L2 norm) ────────────────────────────
        if (g_grad_clip > 0.0f) {
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

            float global_norm = sqrtf(h_norm_sq);
            last_grad_norm = global_norm;  // for logging

            // ─── GLALR: gnorm-loss adaptive LR (adaptive mode only) ─────────
            if (g_safety == SafetyMode::Adaptive) {
                // Update gnorm EMA
                if (gnorm_ema == 0.0f) gnorm_ema = global_norm;
                else gnorm_ema = kGnormEmaBeta * gnorm_ema
                               + (1.0f - kGnormEmaBeta) * global_norm;

                // Update loss history (circular buffer)
                if ((int)loss_history.size() < kGnormRecoverySteps)
                    loss_history.resize(kGnormRecoverySteps);
                loss_history[loss_history_idx % kGnormRecoverySteps] = ema_loss;
                loss_history_idx++;

                // Check: sustained gnorm elevation + loss rising
                if (gnorm_ema > kGnormEmaThreshold) {
                    gnorm_high_cnt++;
                    gnorm_stable_cnt = 0;
                } else {
                    gnorm_high_cnt = 0;
                    gnorm_stable_cnt++;
                }

                // Reduce: gnorm sustained high + loss is rising
                if (gnorm_high_cnt >= kGnormHighSteps
                    && loss_history_idx >= kGnormHighSteps) {
                    int prev_idx = (loss_history_idx - kGnormHighSteps)
                                   % kGnormRecoverySteps;
                    if (ema_loss > loss_history[prev_idx] * 1.02f) {
                        float old_factor = lr_adaptive_factor;
                        lr_adaptive_factor = std::max(kLrAdaptiveFloor,
                            lr_adaptive_factor * kLrAdaptiveDecay);
                        printf("[GLALR] REDUCE: gnorm_ema=%.1f > %.1f for %d steps, "
                               "loss ↑ %.1f%% → lr_adaptive %.2f→%.2f\n",
                               gnorm_ema, kGnormEmaThreshold, kGnormHighSteps,
                               100.0f*(ema_loss/loss_history[prev_idx] - 1.0f),
                               old_factor, lr_adaptive_factor);
                        gnorm_high_cnt = 0;
                    }
                }

                // Restore: gnorm stable + loss not rising
                if (gnorm_stable_cnt >= kGnormRecoverySteps
                    && lr_adaptive_factor < 1.0f
                    && loss_history_idx >= kGnormRecoverySteps) {
                    int prev_idx = (loss_history_idx - kGnormRecoverySteps)
                                   % kGnormRecoverySteps;
                    if (ema_loss <= loss_history[prev_idx]) {
                        float old_factor = lr_adaptive_factor;
                        lr_adaptive_factor = std::min(1.0f,
                            lr_adaptive_factor / kLrAdaptiveDecay);
                        printf("[GLALR] RESTORE: gnorm stable for %d steps, loss ↓ "
                               "→ lr_adaptive %.2f→%.2f\n",
                               kGnormRecoverySteps, old_factor, lr_adaptive_factor);
                        gnorm_stable_cnt = 0;
                    }
                }
            }
            // ─── END GLALR ──────────────────────────────────────────────────
            // NaN/Inf gradient norm: corrupted gradients — skip clipping+optimizer
            if (!std::isfinite(global_norm)) {
                printf("[NAN/INF] gnorm=%.2f at step %d — skipping clipping+optimizer\n",
                       global_norm, step);
                skip_this_step = true;
            }
            if (!skip_this_step && global_norm > g_grad_clip) {
                float clip_coeff = g_grad_clip / (global_norm + 1e-6f);
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

        // ─── OPTIMIZER STEP ────────────────────────────────────────────────
        if (!skip_this_step) {
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
        } // end if (!skip_this_step) optimizer guard

        } // end if (!skip_this_step)

        // ─── UPDATE COUNTERS ──────────────────────────────────────────────
        // DDP: each GPU processes B samples, total = B * world_size
        int effective_world = 1;
#ifdef USE_DDP
        if (g_use_ddp) effective_world = g_num_gpus;
#endif
        g_tokens_processed += (uint64_t)(B * S * effective_world);
        step++;

        // ─── EVAL MILESTONE CHECK ──────────────────────────────────────────
        if (g_eval_every_tokens > 0 && g_tokens_processed >= g_next_eval_tokens) {
#ifdef USE_DDP
            if (!g_use_ddp || cuda::ddp::is_main())
#endif
            {
                run_eval(g_checkpoint_dir, params, adamw_m, adamw_v,
                         g_tokens_processed, step,
                         ema_loss, ema_local_min, lr_decay_factor,
                         running_loss / std::max(running_count, 1),
                         lr, last_grad_norm, S, g_current_phase);
            }
            g_next_eval_tokens = ((g_tokens_processed / g_eval_every_tokens) + 1)
                                 * g_eval_every_tokens;
        }

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
                   "loss=%.4f ema=%.4f gnorm=%.2f decay=%.2f lr=%.2e %s tok/s %.1fs/step elapsed=%s ETA %s | "
                   "RAM %.1fGB VRAM %.1fGB GPU(proc) %d%% GPU(sys) %d%%\n",
                   g_current_phase + 1, step, S,
                   fmt_bar(frac).c_str(), frac * 100.0,
                   fmt_tok2(g_tokens_processed).c_str(), fmt_tok2(g_max_tokens).c_str(),
                   avg_loss, ema_loss, last_grad_norm, lr_decay_factor, lr, tps, s_per_step,
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
        if (should_checkpoint(step)) {
#ifdef USE_DDP
            if (!g_use_ddp || cuda::ddp::is_main())
#endif
            save_checkpoint(g_checkpoint_dir, params, adamw_m, adamw_v,
                           g_tokens_processed, step,
                           ema_loss, ema_local_min, lr_decay_factor);

            // ─── UPLOAD TO GOOGLE DRIVE (simple: snapshot+compress+upload) ──
            if (g_upload_every > 0 && step % g_upload_every == 0) {
#ifdef USE_DDP
                if (!g_use_ddp || cuda::ddp::is_main()) {
#endif
                std::string script = g_upload_script.empty()
                    ? "python3 scripts/upload_gdrive.py"
                    : g_upload_script;
                std::string cmd = script
                    + " --ckpt_dir " + g_checkpoint_dir
                    + " --step " + std::to_string(step)
                    + " --tokens " + std::to_string(g_tokens_processed)
                    + " --zip"
                    + " &";  // async — ~70s total, won't block training
                printf("[UPLOAD] Triggering upload (step=%d)...\n", step);
                (void)system(cmd.c_str());
#ifdef USE_DDP
                }
#endif
            }
        }
    }

    // ─── Final checkpoint ─────────────────────────────────────────────────
#ifdef USE_DDP
    if (!g_use_ddp || cuda::ddp::is_main())
#endif
    save_checkpoint(g_checkpoint_dir, params, adamw_m, adamw_v,
                   g_tokens_processed, step,
                   ema_loss, ema_local_min, lr_decay_factor, "_final");
    printf("\n[DONE] Training complete. %s tokens in %d steps.\n",
           fmt_tok(g_tokens_processed).c_str(), step);
    cuda::arena.print_stats();

    // ─── Final upload to Google Drive ─────────────────────────────────────
    {
        std::string script = g_upload_script.empty()
            ? "python3 scripts/upload_gdrive.py"
            : g_upload_script;
        std::string cmd = script
            + " --ckpt_dir " + g_checkpoint_dir
            + " --step " + std::to_string(step)
            + " --tokens " + std::to_string(g_tokens_processed)
            + " --stage pretrain --zip";
        printf("[UPLOAD] Final upload...\n");
        (void)system(cmd.c_str());
    }

    cudaFree(d_norm_sq);
#ifdef USE_DDP
    if (g_use_ddp) cuda::ddp::shutdown();
#endif
    cuda::nvml_shutdown();
    return 0;
}
