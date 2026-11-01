// =============================================================================
// train_cuda_unified.cu — SFT / DPO / Phone Training (CUDA)
//
// Single binary, stage selected via --stage flag:
//   --stage 2  = SFT (Supervised Fine-Tuning)
//   --stage 3  = DPO (Direct Preference Optimization)
//   --stage 4  = Phone Adapter
//
// Architecture: runtime-selected via --arch_config (default 492M EN/JA:
// D=1280, L=24, QH=20, KVH=5, DH=64, FFN=3584, vocab 49152)
// Target: Kaggle T4 x2 (sm_75, FP16)
//
// This file uses the SAME correct forward/backward as pretraining:
//   - ts::forward / ts::backward       (transformer/train_step.cuh)
//   - launch_gqa_attn_fwd/bwd          (cuda/attn_kernels.cuh)  — real attention
//   - launch_ce_fwd_bwd                (cuda/ce_embed_kernels.cuh) — safe CE
//   - weight-tied embedding gradient   (seed embed_accum before backward)
//   - global-L2 gradient clipping
// The old broken cuda::flash_attention / cuda::training_backward /
// cuda::cross_entropy_* paths are NO LONGER used here.
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#include <csignal>
#include <random>

#include "cuda/cuda_tensor.h"
#include "cuda/cuda_context.h"
#include "cuda/cuda_ops.h"
#include "cuda/cuda_kernels.cuh"
#include "cuda/cuda_monitor.h"
#include "bpe/bpe_tokenizer.h"
#include "transformer/model_config.h"
#include "transformer/arch_runtime.h"
#include "transformer/safetensors.h"
#include "transformer/train_step.cuh"
#include "cuda/ce_embed_kernels.cuh"

#ifdef USE_DDP
#include "cuda/cuda_ddp.h"
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

// ─── Architecture (runtime-configurable via --arch_config) ────────────────
// Defaults = 492M EN/JA, matching the pretrain trainer. Must match the
// checkpoint passed as --base_model, otherwise tensor shapes will not line up.
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
static int g_stage                     = 2;
static std::string g_sft_dir          = "sft_data/sft_clean";
static std::string g_dpo_dir          = "sft_data/dpo";
static std::string g_phone_dir        = "sft_data/phone";
static std::string g_checkpoint_dir   = "checkpoints";
static std::string g_base_model       = "checkpoints/ckpt_pretrain_final.safetensors";
static std::string g_ref_model        = "";
static std::string g_bpe_path         = "";
static std::string g_vocab_dir        = "vocab_6";
static int    g_max_steps             = -1;
static int    g_max_seq               = 512;
static int    g_batch_size            = 4;
static int    g_epochs                = 3;
static float  g_lr                    = 3e-5f;
static float  g_dpo_beta             = 0.1f;
static float  g_dpo_max_margin       = 5.0f;
static int    g_log_every             = 10;
static int    g_save_every            = 200;
static int    g_warmup                = 100;
static float  g_grad_clip             = 1.0f;
static bool   g_use_ddp               = false;
static int    g_num_gpus              = 2;
static bool   g_resume                = false;
static int    g_upload_every          = 0;      // 0 = disabled (fallback for all stages)
static int    g_sft_upload_every      = -1;     // -1 = use g_upload_every; >=0 overrides for SFT
static int    g_dpo_upload_every      = -1;     // -1 = use g_upload_every; >=0 overrides for DPO
static std::string g_upload_script    = "";     // path to upload_gdrive.py

static volatile bool g_stop = false;
void sig_handler(int) { g_stop = true; }

// ─── RL (Stage 5) config ───────────────────────────────────────────────────
static std::string g_rl_dir        = "rl_data";
static int    g_rl_n_samples       = 6;      // N completions sampled per prompt
static int    g_rl_max_new         = 80;     // max new tokens per completion
static float  g_rl_beta            = 0.1f;   // KL penalty coefficient
static float  g_rl_temp            = 0.8f;   // sampling temperature
static int    g_rl_top_k           = 50;     // top-k sampling
static std::string g_verifier_cmd  = "python3 scripts/rl_verifier.py";
static std::string g_rl_workdir    = "rl_work";

// ─── Token IDs ─────────────────────────────────────────────────────────────
static int TOK_BOS = -1, TOK_EOS = -1, TOK_PAD = -1;
static int TOK_SYSTEM = -1, TOK_USER = -1, TOK_ASSISTANT = -1, TOK_END = -1;

// ─── Load model from safetensors ───────────────────────────────────────────
std::vector<cuda::CudaTensor> load_model(const std::string& path) {
    std::vector<cuda::CudaTensor> params;
    SafeTensorsLoader loader(path);

    // Embedding
    cuda::CudaTensor embed({MODEL_VOCAB, D_MODEL}, cuda::DType::FP16);
    std::vector<__half> buf(MODEL_VOCAB * D_MODEL);
    loader.read_into_fp16("embed.weight", buf.data());
    cudaMemcpy(embed.data, buf.data(), embed.nbytes, cudaMemcpyHostToDevice);
    params.push_back(std::move(embed));

    // Layers
    for (int l = 0; l < N_LAYERS; l++) {
        std::string prefix = "layer." + std::to_string(l) + ".";
        const char* names[] = {"wq","wk","wv","wo","w1","w2","w3","rms_attn","rms_ffn"};
        int64_t shapes[][2] = {
            {D_MODEL, D_MODEL},              // wq
            {D_MODEL, N_KV_HEADS * D_HEAD},  // wk
            {D_MODEL, N_KV_HEADS * D_HEAD},  // wv
            {D_MODEL, D_MODEL},              // wo
            {D_MODEL, FFN_DIM},              // w1 (gate)
            {FFN_DIM, D_MODEL},              // w2 (down)
            {D_MODEL, FFN_DIM},              // w3 (up)
            {D_MODEL, 0},                    // rms_attn (1D)
            {D_MODEL, 0},                    // rms_ffn (1D)
        };

        for (int n = 0; n < 9; n++) {
            std::string tname = prefix + names[n] + ".weight";
            cuda::CudaTensor t;
            if (n >= 7) {  // RMS norms are 1D
                t = cuda::CudaTensor({(int64_t)D_MODEL}, cuda::DType::FP16);
            } else {
                t = cuda::CudaTensor({shapes[n][0], shapes[n][1]}, cuda::DType::FP16);
            }
            std::vector<__half> pbuf(t.numel());
            loader.read_into_fp16(tname, pbuf.data());
            cudaMemcpy(t.data, pbuf.data(), t.nbytes, cudaMemcpyHostToDevice);
            params.push_back(std::move(t));
        }
    }

    // Final RMS
    cuda::CudaTensor frms({(int64_t)D_MODEL}, cuda::DType::FP16);
    std::vector<__half> frms_buf(D_MODEL);
    loader.read_into_fp16("final_rms.weight", frms_buf.data());
    cudaMemcpy(frms.data, frms_buf.data(), frms.nbytes, cudaMemcpyHostToDevice);
    params.push_back(std::move(frms));

    printf("[MODEL] Loaded %zu params from %s\n", params.size(), path.c_str());
    return params;
}

// ─── JSON string extraction helper ─────────────────────────────────────────
static std::string extract_json_string(const std::string& line, const std::string& key) {
    std::string search = "\"" + key + "\":";
    auto pos = line.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size();
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) pos++;
    if (pos >= line.size() || line[pos] != '"') return "";
    pos++; // skip opening quote
    std::string result;
    for (; pos < line.size() && line[pos] != '"'; pos++) {
        if (line[pos] == '\\' && pos + 1 < line.size()) {
            pos++;
            if (line[pos] == 'n') result += '\n';
            else if (line[pos] == 't') result += '\t';
            else if (line[pos] == '"') result += '"';
            else if (line[pos] == '\\') result += '\\';
            else { result += '\\'; result += line[pos]; }
        } else {
            result += line[pos];
        }
    }
    return result;
}

// ─── SFT Data Loading (JSONL chat format) ──────────────────────────────────
struct SFTSample {
    std::vector<int> tokens;
    std::vector<float> loss_mask;  // 1.0 for assistant tokens, 0.0 otherwise
};

std::vector<SFTSample> load_sft_data(const std::string& dir, BPEEncoder& tok, int max_seq) {
    std::vector<SFTSample> samples;
    for (auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".jsonl") continue;
        std::ifstream f(entry.path());
        std::string line;
        while (std::getline(f, line)) {
            auto text_pos = line.find("\"text\":\"");
            if (text_pos == std::string::npos) {
                // Try with space: "text": "
                auto alt_pos = line.find("\"text\": \"");
                if (alt_pos == std::string::npos) continue;
                text_pos = alt_pos;
            }
            auto text_start = (line[text_pos + 7] == ' ') ? text_pos + 9 : text_pos + 8;
            // Find the closing quote scanning from text_start, unescaping JSON
            // escapes ("\n", "\t", "\"", "\\", ...) so multi-line responses
            // (e.g. code solutions) survive the naive line parser.
            std::string text;
            for (size_t i = text_start; i < line.size() && line[i] != '"'; i++) {
                if (line[i] == '\\' && i + 1 < line.size()) {
                    i++;
                    if (line[i] == 'n') text += '\n';
                    else if (line[i] == 't') text += '\t';
                    else if (line[i] == 'r') text += '\r';
                    else if (line[i] == '"') text += '"';
                    else if (line[i] == '\\') text += '\\';
                    else { text += '\\'; text += line[i]; }
                } else {
                    text += line[i];
                }
            }
            if (text.empty()) continue;

            auto ids = tok.encode(text, true, true);
            if ((int)ids.size() > max_seq) ids.resize(max_seq);

            SFTSample s;
            s.tokens = std::vector<int>(ids.begin(), ids.end());
            s.loss_mask.resize(s.tokens.size(), 0.0f);

            // Mask: only compute loss on assistant response tokens.
            // Mask <|assistant|> so model learns to predict the first response token.
            // Mask <|end|> so model learns when to stop generating.
            bool in_assistant = false;
            for (size_t i = 0; i < s.tokens.size(); i++) {
                if (s.tokens[i] == TOK_ASSISTANT) in_assistant = true;
                else if (s.tokens[i] == TOK_END) {
                    if (in_assistant) s.loss_mask[i] = 1.0f;
                    in_assistant = false;
                }
                if (in_assistant && s.tokens[i] != TOK_ASSISTANT) s.loss_mask[i] = 1.0f;
            }
            samples.push_back(std::move(s));
        }
    }
    printf("[SFT] Loaded %zu samples from %s\n", samples.size(), dir.c_str());
    return samples;
}

// ─── DPO Data Loading (JSONL preference format) ────────────────────────────
struct DPOSample {
    std::vector<int> chosen_tokens;
    std::vector<int> rejected_tokens;
    std::vector<float> chosen_mask;
    std::vector<float> rejected_mask;
};

std::vector<DPOSample> load_dpo_data(const std::string& dir, BPEEncoder& tok, int max_seq) {
    std::vector<DPOSample> samples;
    for (auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".jsonl") continue;
        std::ifstream f(entry.path());
        std::string line;
        while (std::getline(f, line)) {
            std::string system_txt = extract_json_string(line, "system");
            std::string prompt_txt = extract_json_string(line, "prompt");
            std::string chosen_txt = extract_json_string(line, "chosen");
            std::string rejected_txt = extract_json_string(line, "rejected");

            if (prompt_txt.empty() || chosen_txt.empty() || rejected_txt.empty()) continue;

            std::string prefix = "<|system|>" + system_txt + "<|user|>" + prompt_txt + "<|assistant|>";
            std::string chosen_full = prefix + chosen_txt + "<|end|>";
            std::string rejected_full = prefix + rejected_txt + "<|end|>";

            auto chosen_ids = tok.encode(chosen_full, true, true);
            auto rejected_ids = tok.encode(rejected_full, true, true);
            if ((int)chosen_ids.size() > max_seq) chosen_ids.resize(max_seq);
            if ((int)rejected_ids.size() > max_seq) rejected_ids.resize(max_seq);

            auto build_mask = [&](const std::vector<TokenID>& ids) {
                std::vector<float> mask(ids.size(), 0.0f);
                bool in_assistant = false;
                for (size_t i = 0; i < ids.size(); i++) {
                    if ((int)ids[i] == TOK_ASSISTANT) in_assistant = true;
                    else if ((int)ids[i] == TOK_END) { if (in_assistant) mask[i] = 1.0f; in_assistant = false; }
                    if (in_assistant && (int)ids[i] != TOK_ASSISTANT) mask[i] = 1.0f;
                }
                return mask;
            };

            DPOSample s;
            s.chosen_tokens = std::vector<int>(chosen_ids.begin(), chosen_ids.end());
            s.rejected_tokens = std::vector<int>(rejected_ids.begin(), rejected_ids.end());
            s.chosen_mask = build_mask(chosen_ids);
            s.rejected_mask = build_mask(rejected_ids);
            samples.push_back(std::move(s));
        }
    }
    printf("[DPO] Loaded %zu preference pairs from %s\n", samples.size(), dir.c_str());
    return samples;
}

// ─── RL Data Loading (JSONL, one verifiable coding task per line) ──────────
struct RLSample {
    std::string lang;                 // "en" | "ja"
    std::string prompt_text;          // human-readable instruction
    std::vector<int> prompt_tokens;   // encoded <|system|>..<|user|>..<|assistant|> prefix
    std::string mock_input;           // stdin content for the verifier
    std::string expected_output;      // expected stdout (stripped)
};

std::vector<RLSample> load_rl_data(const std::string& dir, BPEEncoder& tok, int max_seq) {
    std::vector<RLSample> samples;
    for (auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".jsonl") continue;
        std::ifstream f(entry.path());
        std::string line;
        while (std::getline(f, line)) {
            std::string lang = extract_json_string(line, "lang");
            std::string prompt_txt = extract_json_string(line, "prompt");
            std::string mock_input = extract_json_string(line, "mock_input");
            std::string expected   = extract_json_string(line, "expected_output");
            // Optional per-row system prompt (math rows use a math-tutor
            // system; default stays the coding assistant).
            std::string system     = extract_json_string(line, "system");
            if (prompt_txt.empty() || expected.empty()) continue;

            if (system.empty())
                system = "You are a coding assistant.";
            std::string prefix = "<|system|>" + system + "<|user|>"
                               + prompt_txt + "<|assistant|>";
            auto ids = tok.encode(prefix, true, true);
            if ((int)ids.size() > max_seq) ids.resize(max_seq);

            RLSample s;
            s.lang = lang;
            s.prompt_text = prompt_txt;
            s.prompt_tokens = std::vector<int>(ids.begin(), ids.end());
            s.mock_input = mock_input;
            s.expected_output = expected;
            samples.push_back(std::move(s));
        }
    }
    printf("[RL] Loaded %zu verifiable tasks from %s\n", samples.size(), dir.c_str());
    return samples;
}

// ─── CUDA kernels for RL sampling / scaling ───────────────────────────────
// Gather logits rows (one per sequence) from a [M, V] tensor into a [G, V]
// buffer, so only the positions we sample from are copied to host.
__global__ void gather_logits_rows_kernel(
    const __half* __restrict__ logits, const int* __restrict__ rows,
    __half* __restrict__ out, int V)
{
    int g = blockIdx.x;
    const size_t src = (size_t)rows[g] * V;
    const size_t dst = (size_t)g * V;
    for (int v = threadIdx.x; v < V; v += blockDim.x)
        out[dst + v] = logits[src + v];
}

static inline void launch_gather_logits_rows(
    const __half* logits, const int* rows, __half* out, int G, int V,
    cudaStream_t stream)
{
    gather_logits_rows_kernel<<<G, 256, 0, stream>>>(logits, rows, out, V);
}

// Scale each row of a [R, V] FP16 tensor by its own per-token scalar.
__global__ void scale_rows_fp16_kernel(
    __half* __restrict__ d, const float* __restrict__ scale, int R, int V)
{
    int r = blockIdx.x;
    float s = (r < R) ? scale[r] : 0.0f;
    size_t base = (size_t)r * V;
    for (int v = threadIdx.x; v < V; v += blockDim.x)
        d[base + v] = __float2half(__half2float(d[base + v]) * s);
}

static inline void launch_scale_rows_fp16(
    __half* d, const float* scale, int R, int V, cudaStream_t stream)
{
    scale_rows_fp16_kernel<<<R, 256, 0, stream>>>(d, scale, R, V);
}

// ─── Host-side top-k + temperature sampling over a [G, V] logits buffer ───
// Returns the sampled token per alive sequence; fills next_tok[g] (or -1 if
// the sequence is already dead). EOS stops the sequence.
static void sample_top_k_host(
    const std::vector<__half>& logits,   // [G, V], logits for current position
    const std::vector<int>& alive,       // alive[g] != 0
    std::vector<int>& next_tok,          // out
    int G, int V, float temp, int top_k,
    std::mt19937& rng)
{
    for (int g = 0; g < G; g++) {
        if (!alive[g]) { next_tok[g] = -1; continue; }
        const __half* row = &logits[(size_t)g * V];
        if (temp <= 0.0f) {
            int best = 0; float bestv = -1e30f;
            for (int v = 0; v < V; v++) {
                float f = __half2float(row[v]);
                if (f > bestv) { bestv = f; best = v; }
            }
            next_tok[g] = best;
            continue;
        }
        // Temperature softmax
        std::vector<float> probs(V);
        float max_l = -1e30f;
        for (int v = 0; v < V; v++) max_l = fmaxf(max_l, __half2float(row[v]));
        float sum = 0.0f;
        for (int v = 0; v < V; v++) {
            probs[v] = expf((__half2float(row[v]) - max_l) / temp);
            sum += probs[v];
        }
        if (sum < 1e-8f) { next_tok[g] = 0; continue; }
        for (int v = 0; v < V; v++) probs[v] /= sum;

        // Top-K filter
        if (top_k > 0 && top_k < V) {
            std::vector<int> idx(V);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                              [&](int a, int b){ return probs[a] > probs[b]; });
            float s_topk = 0.0f;
            for (int i = 0; i < top_k; i++) s_topk += probs[idx[i]];
            std::vector<float> topk_vals(top_k);
            for (int i = 0; i < top_k; i++) topk_vals[i] = probs[idx[i]];
            std::fill(probs.begin(), probs.end(), 0.0f);
            if (s_topk > 1e-8f)
                for (int i = 0; i < top_k; i++) probs[idx[i]] = topk_vals[i] / s_topk;
        }

        // Multinomial sample
        float total = 0.0f;
        for (float p : probs) total += p;
        if (total <= 1e-8f) { next_tok[g] = 0; continue; }
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        float r = u(rng) * total, cum = 0.0f;
        int tok = V - 1;
        for (int v = 0; v < V; v++) {
            cum += probs[v];
            if (r <= cum) { tok = v; break; }
        }
        next_tok[g] = tok;
    }
}

// ─── Shared training-state (params, master, optimizer, grads, dims) ─────────
struct TrainState {
    ts::ModelDims dims;
    ts::Grads grads;
    std::vector<cuda::CudaTensor>* params = nullptr;
    std::vector<cuda::CudaTensor>* master = nullptr;
    std::vector<cuda::CudaTensor>* opt_m = nullptr;
    std::vector<cuda::CudaTensor>* opt_v = nullptr;
    float* rope_cos = nullptr;
    float* rope_sin = nullptr;
};

// ─── Allocate persistent gradient buffers (once) ────────────────────────────
static void alloc_grads(ts::Grads& grads, const std::vector<cuda::CudaTensor>& params) {
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
}

// ─── Gather all gradient tensors into a flat pointer list ───────────────────
static std::vector<cuda::CudaTensor*> gather_grads(ts::Grads& grads) {
    std::vector<cuda::CudaTensor*> all;
    all.push_back(&grads.d_embed);
    all.push_back(&grads.d_final_rms);
    for (auto& lg : grads.layers) {
        all.push_back(&lg.dwq); all.push_back(&lg.dwk); all.push_back(&lg.dwv);
        all.push_back(&lg.dwo); all.push_back(&lg.dw1); all.push_back(&lg.dw2);
        all.push_back(&lg.dw3); all.push_back(&lg.drms_attn); all.push_back(&lg.drms_ffn);
    }
    return all;
}

// ─── Global-L2 gradient clipping (same as pretraining) ──────────────────────
static void clip_grads(ts::Grads& grads, float clip) {
    if (clip <= 0.0f) return;
    float* d_norm_sq = nullptr;
    cudaMalloc(&d_norm_sq, sizeof(float));
    cudaMemsetAsync(d_norm_sq, 0, sizeof(float), cuda::ctx.compute_stream);

    auto acc = gather_grads(grads);
    for (auto* g : acc)
        launch_l2_norm_sq((const __half*)g->data, d_norm_sq, g->numel(), cuda::ctx.compute_stream);

    cuda::sync_compute();
    float h_norm_sq = 0.0f;
    cudaMemcpy(&h_norm_sq, d_norm_sq, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_norm_sq);

    float global_norm = sqrtf(h_norm_sq);
    if (global_norm > clip) {
        float coeff = clip / (global_norm + 1e-6f);
        for (auto* g : acc)
            launch_scale_fp16((__half*)g->data, coeff, g->numel(), cuda::ctx.compute_stream);
    }
}

// ─── AdamW update over all params ───────────────────────────────────────────
static void adamw_all(std::vector<cuda::CudaTensor>& params,
                      std::vector<cuda::CudaTensor>& master,
                      std::vector<cuda::CudaTensor>& opt_m,
                      std::vector<cuda::CudaTensor>& opt_v,
                      ts::Grads& grads, float lr, float wd, int step) {
    cuda::adamw_step(params[0], master[0], opt_m[0], opt_v[0],
                     grads.d_embed, lr, 0.9f, 0.95f, 1e-8f, wd, step + 1);
    for (int l = 0; l < N_LAYERS; l++) {
        int base = 1 + l * 9;
        auto& lg = grads.layers[l];
        cuda::CudaTensor* lgrads[] = {&lg.dwq,&lg.dwk,&lg.dwv,&lg.dwo,
                                      &lg.dw1,&lg.dw2,&lg.dw3,&lg.drms_attn,&lg.drms_ffn};
        for (int i = 0; i < 9; i++)
            cuda::adamw_step(params[base+i], master[base+i], opt_m[base+i], opt_v[base+i],
                             *lgrads[i], lr, 0.9f, 0.95f, 1e-8f, (i < 7 ? wd : 0.0f), step + 1);
    }
    int frms_idx = 1 + N_LAYERS * 9;
    cuda::adamw_step(params[frms_idx], master[frms_idx], opt_m[frms_idx], opt_v[frms_idx],
                     grads.d_final_rms, lr, 0.9f, 0.95f, 1e-8f, 0.0f, step + 1);
}

#ifdef USE_DDP
static void ddp_allreduce(ts::Grads& grads) {
    if (!g_use_ddp) return;
    auto all = gather_grads(grads);
    cuda::ddp::allreduce_grads(all);
    cuda::ddp::sync();
    for (auto* g : all)
        if (g && g->data && g->numel() > 0) cuda::ddp::average_grad(*g);
}
#endif

// ─── One masked-CE training step (SFT / Phone) ──────────────────────────────
// Returns the mean loss over non-masked tokens. Runs forward → CE → backward →
// (DDP allreduce) → clip → AdamW. Weight-tying handled correctly.
static float train_step_ce(TrainState& st,
                           const std::vector<int>& tokens_cpu,
                           const std::vector<int>& targets_cpu,
                           const std::vector<float>& mask_cpu,
                           int B, int S, float lr, float wd, int step) {
    auto& params = *st.params;
    st.dims.B = B; st.dims.S = S;

    cuda::CudaTensor d_tokens  = cuda::CudaTensor::from_host(tokens_cpu.data(), {B, S});
    cuda::CudaTensor d_targets = cuda::CudaTensor::from_host(targets_cpu.data(), {B, S});
    cuda::CudaTensor d_mask    = cuda::CudaTensor::from_host(mask_cpu.data(), {B, S});

    // Forward
    ts::FwdCache cache;
    cuda::CudaTensor x_final = ts::forward(params, d_tokens, st.dims,
                                           st.rope_cos, st.rope_sin, cache);

    // Logits (weight-tied)
    auto logits = cuda::matmul_transb(x_final, params[0]);   // [B*S, V]

    int BS = B * S;
    cuda::CudaTensor per_token_loss({(int64_t)BS}, cuda::DType::F32);
    cuda::CudaTensor d_logits({(int64_t)BS, (int64_t)MODEL_VOCAB}, cuda::DType::FP16);
    launch_ce_fwd_bwd((const __half*)logits.data, (const int*)d_targets.data,
                      (const float*)d_mask.data,
                      (float*)per_token_loss.data, (__half*)d_logits.data,
                      BS, MODEL_VOCAB, cuda::ctx.compute_stream);

    cuda::sync_compute();
    std::vector<float> h_loss(BS);
    cudaMemcpy(h_loss.data(), per_token_loss.data, BS * sizeof(float), cudaMemcpyDeviceToHost);
    double loss_sum = 0.0; int tok_count = 0;
    for (int i = 0; i < BS; i++) { loss_sum += h_loss[i]; if (mask_cpu[i] > 0.0f) tok_count++; }
    float loss_val = tok_count > 0 ? (float)(loss_sum / tok_count) : 0.0f;

    if (tok_count > 0)
        launch_scale_fp16((__half*)d_logits.data, 1.0f / tok_count,
                          d_logits.numel(), cuda::ctx.compute_stream);

    // Backward
    auto dx_final = cuda::matmul(d_logits, params[0]);              // [B*S, D]
    auto d_embed_out = cuda::matmul(d_logits, x_final, true, false); // [V, D]
    launch_f16_to_f32_seed((const __half*)d_embed_out.data,
                           (float*)st.grads.embed_accum.data,
                           (size_t)MODEL_VOCAB * D_MODEL, cuda::ctx.compute_stream);
    ts::backward(params, d_tokens, x_final, dx_final, st.dims,
                 st.rope_cos, st.rope_sin, cache, st.grads);

#ifdef USE_DDP
    ddp_allreduce(st.grads);
#endif
    clip_grads(st.grads, g_grad_clip);
    adamw_all(params, *st.master, *st.opt_m, *st.opt_v, st.grads, lr, wd, step);
    return loss_val;
}

// ─── LR schedule (warmup + cosine) ──────────────────────────────────────────
// ─── DDP topology helpers ──────────────────────────────────────────────────
// Single source of truth for world size / rank, valid in both DDP and
// single-GPU builds. Stages 2-4 use these to shard data and to size the LR
// schedule, so that a run produces the same optimization trajectory whether
// it is launched on 1 GPU or N.
static inline int ddp_world() {
#ifdef USE_DDP
    if (g_use_ddp) return std::max(1, cuda::ddp::world_size());
#endif
    return 1;
}

static inline int ddp_rank() {
#ifdef USE_DDP
    if (g_use_ddp) return cuda::ddp::rank();
#endif
    return 0;
}

static float sched_lr(int step, int total_steps) {
    float lr = g_lr * std::min(1.0f, (float)(step + 1) / (float)std::max(1, g_warmup));
    if (step >= g_warmup && total_steps > g_warmup) {
        float progress = (float)(step - g_warmup) / (float)(total_steps - g_warmup);
        // Clamp: cos() is periodic, so progress > 1 (reachable when a resumed
        // run overshoots the recomputed total, or --max_steps is lowered)
        // would ramp the LR back UP toward peak instead of holding at the
        // floor. Same failure mode that was fixed in the pretrain schedule.
        if (progress < 0.0f) progress = 0.0f;
        if (progress > 1.0f) progress = 1.0f;
        lr = g_lr * 0.5f * (1.0f + cosf(3.14159265f * progress));
        lr = std::max(lr, g_lr * 0.1f);
    }
    return lr;
}

// ─── Main ──────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    signal(SIGTERM, sig_handler);
    signal(SIGINT, sig_handler);

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--stage" && i+1 < argc)          g_stage = std::stoi(argv[++i]);
        else if (arg == "--sft_dir" && i+1 < argc)   g_sft_dir = argv[++i];
        else if (arg == "--dpo_dir" && i+1 < argc)   g_dpo_dir = argv[++i];
        else if (arg == "--phone_dir" && i+1 < argc) g_phone_dir = argv[++i];
        else if (arg == "--checkpoint_dir" && i+1 < argc) g_checkpoint_dir = argv[++i];
        else if (arg == "--arch_config" && i+1 < argc) g_arch_config = argv[++i];
        else if (arg == "--base_model" && i+1 < argc) g_base_model = argv[++i];
        else if (arg == "--ref_model" && i+1 < argc) g_ref_model = argv[++i];
        else if (arg == "--vocab_dir" && i+1 < argc) g_vocab_dir = argv[++i];
        else if (arg == "--bpe" && i+1 < argc)       g_bpe_path = argv[++i];
        else if (arg == "--max_steps" && i+1 < argc) g_max_steps = std::stoi(argv[++i]);
        else if (arg == "--max_seq" && i+1 < argc)   g_max_seq = std::stoi(argv[++i]);
        else if (arg == "--batch_size" && i+1 < argc) g_batch_size = std::stoi(argv[++i]);
        else if (arg == "--epochs" && i+1 < argc)    g_epochs = std::stoi(argv[++i]);
        else if (arg == "--lr" && i+1 < argc)        g_lr = std::stof(argv[++i]);
        else if (arg == "--dpo_beta" && i+1 < argc)  g_dpo_beta = std::stof(argv[++i]);
        else if (arg == "--dpo_max_margin" && i+1 < argc) g_dpo_max_margin = std::stof(argv[++i]);
        else if (arg == "--grad_clip" && i+1 < argc) g_grad_clip = std::stof(argv[++i]);
        else if (arg == "--log_every" && i+1 < argc) g_log_every = std::stoi(argv[++i]);
        else if (arg == "--save_every" && i+1 < argc) g_save_every = std::stoi(argv[++i]);
        else if (arg == "--upload_every" && i+1 < argc) g_upload_every = std::stoi(argv[++i]);
        else if (arg == "--sft_upload_every" && i+1 < argc) g_sft_upload_every = std::stoi(argv[++i]);
        else if (arg == "--dpo_upload_every" && i+1 < argc) g_dpo_upload_every = std::stoi(argv[++i]);
        else if (arg == "--upload_script" && i+1 < argc) g_upload_script = argv[++i];
        else if (arg == "--warmup" && i+1 < argc)    g_warmup = std::stoi(argv[++i]);
        else if (arg == "--ddp")                     g_use_ddp = true;
        else if (arg == "--num_gpus" && i+1 < argc)  g_num_gpus = std::stoi(argv[++i]);
        else if (arg == "--resume")                  g_resume = true;
        else if (arg == "--rl_dir" && i+1 < argc)    g_rl_dir = argv[++i];
        else if (arg == "--n_samples" && i+1 < argc) g_rl_n_samples = std::stoi(argv[++i]);
        else if (arg == "--rl_max_new" && i+1 < argc) g_rl_max_new = std::stoi(argv[++i]);
        else if (arg == "--rl_beta" && i+1 < argc)   g_rl_beta = std::stof(argv[++i]);
        else if (arg == "--rl_temp" && i+1 < argc)   g_rl_temp = std::stof(argv[++i]);
        else if (arg == "--rl_top_k" && i+1 < argc)  g_rl_top_k = std::stoi(argv[++i]);
        else if (arg == "--verifier_cmd" && i+1 < argc) g_verifier_cmd = argv[++i];
        else if (arg == "--rl_workdir" && i+1 < argc) g_rl_workdir = argv[++i];
        else if (arg == "--help") {
            printf("Usage: train_sft_cuda --stage <2|3|4|5> [options]\n");
            printf("  --stage 2          SFT (Supervised Fine-Tuning)\n");
            printf("  --stage 3          DPO (Direct Preference Optimization)\n");
            printf("  --stage 4          Phone Adapter\n");
            printf("  --stage 5          RL (GRPO-lite, rule-based verifier)\n");
            printf("  --arch_config      Architecture JSON [built-in 492M]\n");
            printf("  --base_model       Pretrained model checkpoint\n");
            printf("  --sft_dir          SFT data directory\n");
            printf("  --dpo_dir          DPO data directory\n");
            printf("  --rl_dir           RL data directory (JSONL verifiable tasks)\n");
            printf("  --n_samples        RL completions sampled per prompt [6]\n");
            printf("  --rl_max_new       RL max new tokens per completion [80]\n");
            printf("  --rl_beta          RL KL penalty coefficient [0.1]\n");
            printf("  --rl_temp          RL sampling temperature [0.8]\n");
            printf("  --rl_top_k         RL top-k sampling [50]\n");
            printf("  --verifier_cmd     Verifier command (req.jsonl out.jsonl) [python3 scripts/rl_verifier.py]\n");
            printf("  --rl_workdir       RL scratch dir for verifier I/O [rl_work]\n");
            printf("  --batch_size       Batch size [4]\n");
            printf("  --epochs           Number of epochs [3]\n");
            printf("  --lr               Learning rate [3e-5]\n");
            printf("  --max_seq          Max sequence length [512]\n");
            printf("  --grad_clip        Global grad-norm clip [1.0]\n");
            printf("  --save_every       Save checkpoint every N steps [200]\n");
            printf("  --upload_every     Upload checkpoint to GDrive every N steps [0=off]\n");
            printf("  --upload_script    Path to upload_gdrive.py [scripts/upload_gdrive.py]\n");
            printf("  --ddp              Enable multi-GPU DDP (NCCL)\n");
            printf("  --resume           Resume from checkpoint (weights + optim + step)\n");
            return 0;
        }
    }

    // ─── Init CUDA (DDP: fork BEFORE any CUDA call) ─────────────────────────
    // Architecture must be resolved before any tensor is sized.
    if (!g_arch_config.empty() && !arch::load(g_arch_config, g_arch)) return 1;
    sync_arch();

    // RoPE tables are cut to MAX_SEQ_LEN; a longer --max_seq would index past them.
    if (g_max_seq > MAX_SEQ_LEN) {
        fprintf(stderr, "[ERR] --max_seq %d exceeds max_seq_len %d in the arch config\n",
                g_max_seq, MAX_SEQ_LEN);
        return 1;
    }

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

    const char* stage_names[] = {"", "", "SFT", "DPO", "Phone", "RL"};
    printf("\n[Stage %d: %s] %.1fM model | FP16 | batch=%d | seq=%d | lr=%.1e%s\n",
           g_stage, stage_names[g_stage], g_arch.param_count() / 1e6,
           g_batch_size, g_max_seq, g_lr,
           g_use_ddp ? " | DDP x2" : "");

    // Load tokenizer
    BPEEncoder tok;
    try {
        if (!g_bpe_path.empty()) {
            if (tok.load(g_bpe_path)) {
                fprintf(stderr, "[TOK] Loaded BPE binary: %s\n", g_bpe_path.c_str());
            } else if (!g_vocab_dir.empty()) {
                fprintf(stderr,
                        "[TOK] Failed to load BPE binary: %s; falling back to vocab dir: %s\n",
                        g_bpe_path.c_str(), g_vocab_dir.c_str());
                tok.load_vocab(g_vocab_dir);
            } else {
                fprintf(stderr, "[ERR] Failed to load BPE binary: %s\n", g_bpe_path.c_str());
                return 1;
            }
        } else if (!g_vocab_dir.empty()) {
            tok.load_vocab(g_vocab_dir);
            fprintf(stderr, "[TOK] Loaded vocab dir: %s\n", g_vocab_dir.c_str());
        } else {
            fprintf(stderr, "[ERR] No tokenizer source provided. Pass --bpe or --vocab_dir.\n");
            return 1;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[ERR] Tokenizer load failed: %s\n", e.what());
        return 1;
    }
    if (tok.vocab_size() == 0) {
        fprintf(stderr, "[ERR] Tokenizer loaded zero tokens.\n");
        return 1;
    }
    printf("[TOK] Loaded: vocab_size=%zu\n", tok.vocab_size());

    // Same probe as train_cuda.cu. vocab_7 and vocab_8 are both 49152 tokens,
    // so nothing else here can tell them apart, and SFT-ing a pretrained
    // checkpoint under the wrong tokenizer silently destroys it.
    {
        static const char* kProbe =
            "The capital of Japan is Tokyo.\n日本語のテキストです。コンピュータ\n"
            "def f(x):\n    return x + 1\n\\frac{1}{2} + \\sqrt{3} = 0";
        auto ids = tok.encode(std::string(kProbe), false, false);
        uint64_t h = 1469598103934665603ULL;
        for (auto id : ids) {
            uint64_t v = (uint64_t)id;
            for (int b = 0; b < 4; b++) { h ^= (v >> (8 * b)) & 0xFF; h *= 1099511628211ULL; }
        }
        printf("[TOK] fingerprint=%016llx (%zu probe tokens)\n",
               (unsigned long long)h, ids.size());
        std::ifstream sf(g_checkpoint_dir + "/train_state.txt");
        std::string line;
        while (sf && std::getline(sf, line)) {
            if (line.rfind("tok_fingerprint=", 0) != 0) continue;
            uint64_t saved = strtoull(line.substr(16).c_str(), nullptr, 10);
            if (saved != 0 && saved != h) {
                fprintf(stderr,
                    "\n[ERR] Tokenizer mismatch against the pretrained checkpoint.\n"
                    "      checkpoint fingerprint %016llx, this run %016llx\n"
                    "      Point --vocab_dir/--bpe at the tokenizer pretraining used.\n",
                    (unsigned long long)saved, (unsigned long long)h);
                return 1;
            }
            break;
        }
    }

    TOK_BOS = tok.bos_id(); TOK_EOS = tok.eos_id(); TOK_PAD = tok.pad_id();
    if (TOK_PAD < 0) TOK_PAD = TOK_BOS;
    TOK_SYSTEM = (int)tok.special_id(SpecialTokens::SYSTEM);
    TOK_USER = (int)tok.special_id(SpecialTokens::USER);
    TOK_ASSISTANT = (int)tok.special_id(SpecialTokens::ASSISTANT);
    TOK_END = (int)tok.special_id(SpecialTokens::END);
    auto bad_tok = [](int tok_id) { return tok_id < 0 || tok_id >= MODEL_VOCAB; };
    if (bad_tok(TOK_BOS) || bad_tok(TOK_EOS) || bad_tok(TOK_PAD) ||
        bad_tok(TOK_SYSTEM) || bad_tok(TOK_USER) ||
        bad_tok(TOK_ASSISTANT) || bad_tok(TOK_END)) {
        fprintf(stderr,
                "[ERR] Tokenizer special IDs invalid: bos=%d eos=%d pad=%d system=%d user=%d assistant=%d end=%d\n",
                TOK_BOS, TOK_EOS, TOK_PAD, TOK_SYSTEM, TOK_USER, TOK_ASSISTANT, TOK_END);
        fprintf(stderr,
                "[ERR] Ensure the runtime can load the matching bpe_v8.bin or vocab_8 files before Stage %d.\n",
                g_stage);
        return 1;
    }
    printf("[TOK] Special ids: bos=%d eos=%d pad=%d system=%d user=%d assistant=%d end=%d\n",
           TOK_BOS, TOK_EOS, TOK_PAD, TOK_SYSTEM, TOK_USER, TOK_ASSISTANT, TOK_END);

    // Load base model + master/optimizer
    auto params = load_model(g_base_model);
    std::vector<cuda::CudaTensor> master_params, opt_m, opt_v;
    for (auto& p : params) {
        master_params.push_back(cuda::astype(p, cuda::DType::F32));
        opt_m.push_back(cuda::zeros_like(cuda::CudaTensor(p.shape_vec(), cuda::DType::F32)));
        opt_v.push_back(cuda::zeros_like(cuda::CudaTensor(p.shape_vec(), cuda::DType::F32)));
    }

    // ─── Init untrained special-token embeddings from trained ones ────────
    // Tokens 1,2,3,15 were never seen during pretraining — their embeddings
    // are close to random init. Copy BOS(0) → SYSTEM(1), USER(2), ASSISTANT(3)
    // and EOS(4) → PAD(15) so SFT doesn't start from untrained embeddings.
    // NOTE: This is a temporary workaround until pretraining is re-run with
    // chat data mixed in via data/corpus_final/mixed/.
    {
        auto copy_embed_row = [&](int dst_id, int src_id, const char* dst_name, const char* src_name) {
            if (bad_tok(dst_id) || bad_tok(src_id)) {
                fprintf(stderr,
                        "[ERR] Cannot initialize %s from %s: dst=%d src=%d\n",
                        dst_name, src_name, dst_id, src_id);
                return false;
            }
            size_t row_bytes = (size_t)D_MODEL * sizeof(__half);
            cudaError_t err = cudaMemcpy((__half*)params[0].data + (size_t)dst_id * D_MODEL,
                                         (const __half*)params[0].data + (size_t)src_id * D_MODEL,
                                         row_bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "[ERR] Failed to copy %s embedding from %s: %s\n",
                        dst_name, src_name, cudaGetErrorString(err));
                return false;
            }
            size_t f32_row_bytes = (size_t)D_MODEL * sizeof(float);
            err = cudaMemcpy((float*)master_params[0].data + (size_t)dst_id * D_MODEL,
                             (const float*)master_params[0].data + (size_t)src_id * D_MODEL,
                             f32_row_bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "[ERR] Failed to copy master %s embedding from %s: %s\n",
                        dst_name, src_name, cudaGetErrorString(err));
                return false;
            }
            return true;
        };
        if (!copy_embed_row(TOK_SYSTEM, TOK_BOS, "SYSTEM", "BOS") ||
            !copy_embed_row(TOK_USER, TOK_BOS, "USER", "BOS") ||
            !copy_embed_row(TOK_ASSISTANT, TOK_BOS, "ASSISTANT", "BOS") ||
            !copy_embed_row(TOK_PAD, TOK_EOS, "PAD", "EOS")) {
            return 1;
        }
         printf("[INIT] Copied BOS(%d) embedding → SYSTEM(%d) USER(%d) ASSISTANT(%d)\n",
             TOK_BOS, TOK_SYSTEM, TOK_USER, TOK_ASSISTANT);
         printf("[INIT] Copied EOS(%d) embedding → PAD(%d)\n", TOK_EOS, TOK_PAD);
    }

    // RoPE tables
    float *rope_cos = nullptr, *rope_sin = nullptr;
    {
        int half_dim = D_HEAD / 2;
        std::vector<float> cos_h((size_t)MAX_SEQ_LEN * half_dim),
                           sin_h((size_t)MAX_SEQ_LEN * half_dim);
        for (int pos = 0; pos < MAX_SEQ_LEN; pos++)
            for (int j = 0; j < half_dim; j++) {
                float freq = pos / powf(ROPE_THETA, 2.0f * j / D_HEAD);
                cos_h[pos * half_dim + j] = cosf(freq);
                sin_h[pos * half_dim + j] = sinf(freq);
            }
        cudaMalloc(&rope_cos, cos_h.size() * sizeof(float));
        cudaMalloc(&rope_sin, sin_h.size() * sizeof(float));
        cudaMemcpy(rope_cos, cos_h.data(), cos_h.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(rope_sin, sin_h.data(), sin_h.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (g_stage == 4) g_lr *= 0.3f;  // Conservative for phone adapter

    // Shared training state
    TrainState st;
    st.params = &params; st.master = &master_params;
    st.opt_m = &opt_m; st.opt_v = &opt_v;
    st.rope_cos = rope_cos; st.rope_sin = rope_sin;
    st.dims.D = D_MODEL; st.dims.L = N_LAYERS;
    st.dims.HQ = N_Q_HEADS; st.dims.HKV = N_KV_HEADS; st.dims.DH = D_HEAD;
    st.dims.FFN = FFN_DIM; st.dims.V = MODEL_VOCAB; st.dims.rope_theta = ROPE_THETA;
    alloc_grads(st.grads, params);
    cuda::sync_compute();

    // ─── Resume state (populated by --resume block below) ─────────────────
    const char* stage_tag_early = (g_stage == 2) ? "sft" : (g_stage == 3) ? "dpo" : (g_stage == 4) ? "phone" : "rl";
    int   start_step   = 0;
    int   start_epoch  = 0;
    size_t start_sample_idx = 0;

    if (g_resume) {
        std::string sp = g_checkpoint_dir + "/train_state_" + stage_tag_early + ".txt";
        if (fs::exists(sp)) {
            std::string last_ckpt;
            std::ifstream sf(sp);
            std::string line;
            while (std::getline(sf, line)) {
                if      (line.rfind("step=", 0)        == 0) start_step        = std::stoi(line.substr(5));
                else if (line.rfind("epoch=", 0)       == 0) start_epoch       = std::stoi(line.substr(6));
                else if (line.rfind("sample_idx=", 0)  == 0) start_sample_idx  = std::stoull(line.substr(11));
                else if (line.rfind("last_ckpt=", 0)   == 0) last_ckpt         = line.substr(10);
            }
            sf.close();

            // Reload weights from the last saved checkpoint (overrides base_model).
            if (!last_ckpt.empty()) {
                std::string wpath = g_checkpoint_dir + "/" + last_ckpt;
                if (fs::exists(wpath)) {
                    printf("[RESUME] Loading weights: %s\n", wpath.c_str());
                    SafeTensorsLoader loader(wpath);
                    std::vector<__half> buf(params[0].numel());
                    loader.read_into_fp16("embed.weight", buf.data());
                    cudaMemcpy(params[0].data, buf.data(), params[0].nbytes, cudaMemcpyHostToDevice);
                    const char* names[] = {"wq","wk","wv","wo","w1","w2","w3","rms_attn","rms_ffn"};
                    for (int l = 0; l < N_LAYERS; l++) {
                        int base = 1 + l * 9;
                        for (int n = 0; n < 9; n++) {
                            std::string tn = "layer." + std::to_string(l) + "." + names[n] + ".weight";
                            auto& p = params[base + n];
                            std::vector<__half> pbuf(p.numel());
                            loader.read_into_fp16(tn, pbuf.data());
                            cudaMemcpy(p.data, pbuf.data(), p.nbytes, cudaMemcpyHostToDevice);
                        }
                    }
                    auto& frms = params[1 + N_LAYERS * 9];
                    std::vector<__half> fb(frms.numel());
                    loader.read_into_fp16("final_rms.weight", fb.data());
                    cudaMemcpy(frms.data, fb.data(), frms.nbytes, cudaMemcpyHostToDevice);
                    // Refresh master (FP32) from reloaded FP16 weights.
                    for (size_t i = 0; i < params.size(); i++)
                        master_params[i] = cuda::astype(params[i], cuda::DType::F32);
                }
            }

            // Load optimizer state (m, v FP32).
            std::string opath = g_checkpoint_dir + "/ckpt_" + stage_tag_early + "_optim.safetensors";
            if (fs::exists(opath)) {
                printf("[RESUME] Loading optimizer: %s\n", opath.c_str());
                SafeTensorsLoader ol(opath);
                const char* layer_names[] = {"wq","wk","wv","wo","w1","w2","w3","rms_attn","rms_ffn"};
                auto load_f32 = [&](cuda::CudaTensor& t, const std::string& name) {
                    if (!ol.has_tensor(name)) return;
                    std::vector<float> hbuf(t.numel());
                    ol.read_into_f32(name, hbuf.data());
                    cudaMemcpy(t.data, hbuf.data(), t.numel() * sizeof(float), cudaMemcpyHostToDevice);
                };
                load_f32(opt_m[0], "m.embed");
                load_f32(opt_v[0], "v.embed");
                for (int l = 0; l < N_LAYERS; l++) {
                    int base = 1 + l * 9;
                    for (int i = 0; i < 9; i++) {
                        std::string pfx = "layer." + std::to_string(l) + "." + layer_names[i];
                        load_f32(opt_m[base + i], "m." + pfx);
                        load_f32(opt_v[base + i], "v." + pfx);
                    }
                }
                int frms_idx = 1 + N_LAYERS * 9;
                load_f32(opt_m[frms_idx], "m.final_rms");
                load_f32(opt_v[frms_idx], "v.final_rms");
                printf("[RESUME] Optimizer states restored (%zu tensors)\n", opt_m.size() * 2);
            } else {
                printf("[RESUME] No optimizer file — cold optimizer (m=0, v=0)\n");
            }

            printf("[RESUME] Restored at step=%d, epoch=%d, sample_idx=%zu\n",
                   start_step, start_epoch, start_sample_idx);
        } else {
            printf("[RESUME] No train_state found for stage '%s' — starting fresh\n",
                   stage_tag_early);
        }
    }

    // ─── Save checkpoint utility (weights + optimizer + train_state) ──────
    // Layout on disk:
    //   ckpt<suffix>.safetensors     — FP16 model weights
    //   ckpt_<tag>_optim.safetensors — FP32 optimizer m,v (single file, overwritten)
    //   train_state_<tag>.txt        — step, epoch, sample_idx (single file, overwritten)
    // where <tag> = sft | dpo | phone (derived from stage).
    const char* stage_tag = (g_stage == 2) ? "sft" : (g_stage == 3) ? "dpo" : (g_stage == 4) ? "phone" : "rl";

    auto save_stage_checkpoint = [&](const std::string& suffix,
                                     int cur_step = -1, int cur_epoch = -1,
                                     size_t cur_sample_idx = 0) {
#ifdef USE_DDP
        if (g_use_ddp && !cuda::ddp::is_main()) return;
#endif
        fs::create_directories(g_checkpoint_dir);
        std::string path = g_checkpoint_dir + "/ckpt" + suffix + ".safetensors";
        SafeTensorsWriter w;

        // Keep all host buffers alive until finalize() (writer stores pointers).
        std::vector<std::vector<__half>> host_bufs;
        host_bufs.reserve(1 + N_LAYERS * 9 + 1);

        {
            host_bufs.emplace_back(params[0].numel());
            auto& host = host_bufs.back();
            cudaMemcpy(host.data(), params[0].data, params[0].nbytes, cudaMemcpyDeviceToHost);
            w.add_tensor("embed.weight", STDtype::F16,
                         {(uint32_t)params[0].shape[0], (uint32_t)params[0].shape[1]}, host.data());
        }
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
        {
            auto& frms = params[1 + N_LAYERS * 9];
            host_bufs.emplace_back(frms.numel());
            auto& host = host_bufs.back();
            cudaMemcpy(host.data(), frms.data, frms.nbytes, cudaMemcpyDeviceToHost);
            w.add_tensor("final_rms.weight", STDtype::F16, {(uint32_t)frms.numel()}, host.data());
        }
        w.finalize(path);
        printf("[CKPT] Saved FP16: %s (%.1f MB)\n", path.c_str(), (float)fs::file_size(path) / 1e6f);

        // ── Save optimizer state (m, v FP32) — overwrite single file ──
        {
            std::string opt_path = g_checkpoint_dir + "/ckpt_" + stage_tag + "_optim.safetensors";
            SafeTensorsWriter ow;
            std::vector<std::vector<float>> obufs;
            obufs.reserve(params.size() * 2);
            const char* layer_names[] = {"wq","wk","wv","wo","w1","w2","w3","rms_attn","rms_ffn"};
            auto save_f32 = [&](const cuda::CudaTensor& t, const std::string& name) {
                obufs.emplace_back(t.numel());
                auto& hbuf = obufs.back();
                cudaMemcpy(hbuf.data(), t.data, t.numel() * sizeof(float), cudaMemcpyDeviceToHost);
                if (t.ndim == 1)
                    ow.add_tensor(name, STDtype::F32, {(uint32_t)t.numel()}, hbuf.data());
                else
                    ow.add_tensor(name, STDtype::F32,
                                  {(uint32_t)t.shape[0], (uint32_t)t.shape[1]}, hbuf.data());
            };
            save_f32(opt_m[0], "m.embed");
            save_f32(opt_v[0], "v.embed");
            for (int l = 0; l < N_LAYERS; l++) {
                int base = 1 + l * 9;
                for (int i = 0; i < 9; i++) {
                    std::string pfx = "layer." + std::to_string(l) + "." + layer_names[i];
                    save_f32(opt_m[base + i], "m." + pfx);
                    save_f32(opt_v[base + i], "v." + pfx);
                }
            }
            int frms_idx = 1 + N_LAYERS * 9;
            save_f32(opt_m[frms_idx], "m.final_rms");
            save_f32(opt_v[frms_idx], "v.final_rms");
            ow.finalize(opt_path);
            printf("[CKPT] Optimizer: %s (%.1f MB)\n", opt_path.c_str(),
                   (float)fs::file_size(opt_path) / 1e6f);
        }

        // ── Save train_state (step/epoch/sample_idx + last-ckpt suffix) ──
        if (cur_step >= 0) {
            std::string sp = g_checkpoint_dir + "/train_state_" + stage_tag + ".txt";
            std::ofstream sf(sp);
            sf << "step=" << cur_step << "\n";
            sf << "epoch=" << cur_epoch << "\n";
            sf << "sample_idx=" << cur_sample_idx << "\n";
            sf << "last_ckpt=ckpt" << suffix << ".safetensors\n";
            sf.close();
        }
    };

    // ─── Upload to Google Drive (async: snapshot+compress+upload) ──────────
    // Resolve effective upload interval per stage
    int sft_upload_interval  = (g_sft_upload_every >= 0) ? g_sft_upload_every : g_upload_every;
    int dpo_upload_interval  = (g_dpo_upload_every >= 0) ? g_dpo_upload_every : g_upload_every;

    auto trigger_upload = [&](int cur_step, const char* stage) {
#ifdef USE_DDP
        if (g_use_ddp && !cuda::ddp::is_main()) return;
#endif
        std::string script = g_upload_script.empty()
            ? "python3 scripts/upload_gdrive.py"
            : g_upload_script;
        std::string cmd = script
            + " --ckpt_dir " + g_checkpoint_dir
            + " --step " + std::to_string(cur_step)
            + " --stage " + stage
            + " --zip"
            + " &";
        printf("[UPLOAD] Triggering upload (step=%d, stage=%s)...\n", cur_step, stage);
        (void)system(cmd.c_str());
    };

    // ═══════════════════════════════════════════════════════════════════════
    // Stage 2 (SFT) and Stage 4 (Phone) — masked next-token CE
    // ═══════════════════════════════════════════════════════════════════════
    if (g_stage == 2 || g_stage == 4) {
        const std::string& data_dir = (g_stage == 2) ? g_sft_dir : g_phone_dir;
        const char* tag = (g_stage == 2) ? "SFT" : "Phone";
        auto samples = load_sft_data(data_dir, tok, g_max_seq);
        if (samples.empty()) {
            fprintf(stderr, "[ERR] No %s data found in %s\n", tag, data_dir.c_str());
            return 1;
        }

        // ─── DDP data sharding ─────────────────────────────────────────────
        // Every rank runs this same loop, so without an explicit stride all
        // GPUs would load the IDENTICAL batch, allreduce identical gradients
        // and produce exactly the single-GPU result at N times the cost (and
        // an N-times-too-short schedule, since `step` advances once per rank
        // per global batch). Give each rank its own slice: rank r takes the
        // batch starting at (bi + r*B), and the loop strides by world*B.
        const int   ddp_w      = ddp_world();
        const int   ddp_r      = ddp_rank();
        const size_t global_bs = (size_t)g_batch_size * (size_t)ddp_w;

        // total_steps counts GLOBAL batches: one optimizer step consumes
        // global_bs samples, so the cosine schedule spans the real number of
        // updates and has the same shape on 1 GPU or N.
        int total_steps = (int)((size_t)g_epochs * samples.size() / global_bs);
        if (total_steps < 1) total_steps = 1;
        if (g_max_steps > 0) total_steps = std::min(total_steps, g_max_steps);

        auto t0 = Clock::now();
        int step = start_step;
        float running_loss = 0;

        for (int epoch = start_epoch; epoch < g_epochs && !g_stop; epoch++) {
            // Same seed on every rank => same permutation => the disjoint
            // rank slices below together cover the epoch exactly once.
            std::mt19937 rng(42 + epoch);
            std::shuffle(samples.begin(), samples.end(), rng);

            // On the first (resumed) epoch, skip already-processed samples.
            size_t bi_start = (epoch == start_epoch) ? start_sample_idx : 0;
            // Align the resume point to a global-batch boundary so the rank
            // slices stay disjoint after a restart.
            bi_start = (bi_start / global_bs) * global_bs;

            for (size_t bi = bi_start; bi + global_bs <= samples.size() && !g_stop;
                 bi += global_bs) {
                if (g_max_steps > 0 && step >= g_max_steps) break;

                // This rank's slice of the global batch.
                const size_t base = bi + (size_t)ddp_r * (size_t)g_batch_size;

                int S = g_max_seq, B = g_batch_size;
                std::vector<int> batch_tokens(B * S, TOK_PAD);
                std::vector<int> batch_targets(B * S, TOK_PAD);
                std::vector<float> batch_mask(B * S, 0.0f);

                for (int b = 0; b < B; b++) {
                    auto& sample = samples[base + b];
                    int len = std::min((int)sample.tokens.size(), S - 1);
                    for (int t = 0; t < len; t++) {
                        batch_tokens[b * S + t] = sample.tokens[t];
                        batch_targets[b * S + t] = (t + 1 < (int)sample.tokens.size())
                                                    ? sample.tokens[t + 1] : TOK_EOS;
                        batch_mask[b * S + t] = (t < (int)sample.loss_mask.size())
                                                 ? sample.loss_mask[t] : 0.0f;
                    }
                }

                float lr = sched_lr(step, total_steps);
                float loss_val = train_step_ce(st, batch_tokens, batch_targets, batch_mask,
                                                B, S, lr, 0.01f, step);
                running_loss += loss_val;
                step++;

                if (step % g_log_every == 0) {
#ifdef USE_DDP
                    if (!g_use_ddp || cuda::ddp::is_main())
#endif
                    {
                        double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
                        printf("[%s] ep=%d step=%d/%d loss=%.4f lr=%.2e %.1f tok/s\n",
                               tag, epoch + 1, step, total_steps, running_loss / g_log_every, lr,
                               (double)(step - start_step) * (double)global_bs * S / elapsed);
                    }
                    running_loss = 0;
                }
                if (step % g_save_every == 0)
                    save_stage_checkpoint(std::string("_") + (g_stage == 2 ? "sft" : "phone"),
                                          step, epoch, bi + global_bs);
                if (sft_upload_interval > 0 && step % sft_upload_interval == 0)
                    trigger_upload(step, (g_stage == 2) ? "sft" : "phone");
            }
        }
        save_stage_checkpoint(std::string("_") + (g_stage == 2 ? "sft" : "phone") + "_final",
                              step, g_epochs, 0);
        printf("\n[DONE] %s complete. %d steps.\n", tag, step);
        trigger_upload(step, (g_stage == 2) ? "sft" : "phone");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Stage 3: DPO Training Loop
    //
    // DPO loss = -log(sigmoid(beta * ((lp_c - ref_lp_c) - (lp_r - ref_lp_r))))
    // Gradient wrt policy logits (per DPO): the standard CE gradient of each
    // response, scaled by the DPO coefficient. We backprop the CHOSEN branch
    // (dominant signal) with the correct ts::backward pass.
    // ═══════════════════════════════════════════════════════════════════════
    else if (g_stage == 3) {
        auto dpo_samples = load_dpo_data(g_dpo_dir, tok, g_max_seq);
        if (dpo_samples.empty()) {
            fprintf(stderr, "[ERR] No DPO data found in %s\n", g_dpo_dir.c_str());
            return 1;
        }

        // Reference model (frozen copy of base)
        std::string ref_path = g_ref_model.empty() ? g_base_model : g_ref_model;
        auto ref_params = load_model(ref_path);
        printf("[DPO] Reference model loaded from: %s\n", ref_path.c_str());

        // ─── DDP data sharding (see the SFT loop for the rationale) ────────
        const int    ddp_w      = ddp_world();
        const int    ddp_r      = ddp_rank();
        const size_t global_bs  = (size_t)g_batch_size * (size_t)ddp_w;

        int total_steps = (int)((size_t)g_epochs * dpo_samples.size() / global_bs);
        if (total_steps < 1) total_steps = 1;
        if (g_max_steps > 0) total_steps = std::min(total_steps, g_max_steps);

        auto t0 = Clock::now();
        int step = start_step;
        float running_loss = 0, running_margin = 0;

        // Helper: forward + logits only (no grad) for reference/log-prob use.
        auto forward_logits = [&](std::vector<cuda::CudaTensor>& p,
                                  const cuda::CudaTensor& d_tok, int B, int S) {
            ts::FwdCache cache;
            st.dims.B = B; st.dims.S = S;
            cuda::CudaTensor x = ts::forward(p, d_tok, st.dims, rope_cos, rope_sin, cache);
            cuda::CudaTensor logits = cuda::matmul_transb(x, p[0]);  // [B*S, V]
            // Ensure the matmul (which reads x, a view into cache) completes
            // before 'cache' is destroyed at end of this scope.
            cuda::sync_compute();
            return logits;
        };

        for (int epoch = start_epoch; epoch < g_epochs && !g_stop; epoch++) {
            // Same seed on every rank => same permutation => disjoint slices.
            std::mt19937 rng(42 + epoch);
            std::shuffle(dpo_samples.begin(), dpo_samples.end(), rng);

            size_t bi_start = (epoch == start_epoch) ? start_sample_idx : 0;
            bi_start = (bi_start / global_bs) * global_bs;

            for (size_t bi = bi_start; bi + global_bs <= dpo_samples.size() && !g_stop;
                 bi += global_bs) {
                if (g_max_steps > 0 && step >= g_max_steps) break;
                int S = g_max_seq, B = g_batch_size;

                // This rank's slice of the global batch.
                const size_t base = bi + (size_t)ddp_r * (size_t)g_batch_size;

                std::vector<int> chosen_tok(B * S, TOK_PAD), rejected_tok(B * S, TOK_PAD);
                std::vector<int> chosen_tgt(B * S, TOK_PAD), rejected_tgt(B * S, TOK_PAD);
                std::vector<float> chosen_mask(B * S, 0.0f), rejected_mask(B * S, 0.0f);

                for (int b = 0; b < B; b++) {
                    auto& sample = dpo_samples[base + b];
                    int clen = std::min((int)sample.chosen_tokens.size(), S - 1);
                    for (int t = 0; t < clen; t++) {
                        chosen_tok[b * S + t] = sample.chosen_tokens[t];
                        chosen_tgt[b * S + t] = (t + 1 < (int)sample.chosen_tokens.size())
                                                 ? sample.chosen_tokens[t + 1] : TOK_EOS;
                        chosen_mask[b * S + t] = (t < (int)sample.chosen_mask.size())
                                                  ? sample.chosen_mask[t] : 0.0f;
                    }
                    int rlen = std::min((int)sample.rejected_tokens.size(), S - 1);
                    for (int t = 0; t < rlen; t++) {
                        rejected_tok[b * S + t] = sample.rejected_tokens[t];
                        rejected_tgt[b * S + t] = (t + 1 < (int)sample.rejected_tokens.size())
                                                   ? sample.rejected_tokens[t + 1] : TOK_EOS;
                        rejected_mask[b * S + t] = (t < (int)sample.rejected_mask.size())
                                                    ? sample.rejected_mask[t] : 0.0f;
                    }
                }

                auto d_chosen_tok   = cuda::CudaTensor::from_host(chosen_tok.data(), {B, S});
                auto d_rejected_tok = cuda::CudaTensor::from_host(rejected_tok.data(), {B, S});
                auto d_chosen_tgt   = cuda::CudaTensor::from_host(chosen_tgt.data(), {B, S});
                auto d_rejected_tgt = cuda::CudaTensor::from_host(rejected_tgt.data(), {B, S});
                auto d_chosen_mask  = cuda::CudaTensor::from_host(chosen_mask.data(), {B, S});
                auto d_rejected_mask= cuda::CudaTensor::from_host(rejected_mask.data(), {B, S});

                // Reference (frozen) and policy log-probs for the DPO logit
                auto ref_lc = forward_logits(ref_params, d_chosen_tok, B, S);
                auto ref_lr = forward_logits(ref_params, d_rejected_tok, B, S);
                auto pol_lc = forward_logits(params, d_chosen_tok, B, S);
                auto pol_lr = forward_logits(params, d_rejected_tok, B, S);

                float lp_c  = cuda::masked_log_probs(pol_lc, d_chosen_tgt, d_chosen_mask).item_f32();
                float lp_r  = cuda::masked_log_probs(pol_lr, d_rejected_tgt, d_rejected_mask).item_f32();
                float rlp_c = cuda::masked_log_probs(ref_lc, d_chosen_tgt, d_chosen_mask).item_f32();
                float rlp_r = cuda::masked_log_probs(ref_lr, d_rejected_tgt, d_rejected_mask).item_f32();

                float raw_margin = (lp_c - rlp_c) - (lp_r - rlp_r);
                float margin = fmaxf(fminf(raw_margin, g_dpo_max_margin), -g_dpo_max_margin);
                float logit_val = g_dpo_beta * margin;
                float dpo_loss  = logf(1.0f + expf(-logit_val));
                running_loss   += dpo_loss;
                running_margin += margin;

                // sigma(-r) = 1/(1+exp(r)); numerically stable for large positive r
                float sig_neg = (logit_val >= 0.0f)
                    ? expf(-logit_val) / (1.0f + expf(-logit_val))
                    : 1.0f / (1.0f + expf(logit_val));
                // dL/d(logits_c[t]) = +beta*sigma(-r)/N * CE_grad[t] → chosen log-prob goes up
                float coeff          =  g_dpo_beta * sig_neg;
                float coeff_rejected = -g_dpo_beta * sig_neg;  // rejected log-prob goes down

                // ── Backward through the CHOSEN branch (increase chosen log-prob) ──
                st.dims.B = B; st.dims.S = S;
                ts::FwdCache cache_c;
                cuda::CudaTensor x_final_c = ts::forward(params, d_chosen_tok, st.dims,
                                                         rope_cos, rope_sin, cache_c);
                auto logits_c = cuda::matmul_transb(x_final_c, params[0]);

                int BS = B * S;
                cuda::CudaTensor per_token_loss({(int64_t)BS}, cuda::DType::F32);
                cuda::CudaTensor d_logits_c({(int64_t)BS, (int64_t)MODEL_VOCAB}, cuda::DType::FP16);
                launch_ce_fwd_bwd((const __half*)logits_c.data, (const int*)d_chosen_tgt.data,
                                  (const float*)d_chosen_mask.data,
                                  (float*)per_token_loss.data, (__half*)d_logits_c.data,
                                  BS, MODEL_VOCAB, cuda::ctx.compute_stream);

                cuda::sync_compute();
                int tok_count_c = 0;
                for (int i = 0; i < BS; i++) if (chosen_mask[i] > 0.0f) tok_count_c++;
                float norm_c = (tok_count_c > 0) ? (coeff / tok_count_c) : 0.0f;
                launch_scale_fp16((__half*)d_logits_c.data, norm_c,
                                  d_logits_c.numel(), cuda::ctx.compute_stream);

                auto dx_final_c = cuda::matmul(d_logits_c, params[0]);
                auto d_embed_c = cuda::matmul(d_logits_c, x_final_c, true, false);
                launch_f16_to_f32_seed((const __half*)d_embed_c.data,
                                       (float*)st.grads.embed_accum.data,
                                       (size_t)MODEL_VOCAB * D_MODEL, cuda::ctx.compute_stream);
                ts::backward(params, d_chosen_tok, x_final_c, dx_final_c, st.dims,
                             rope_cos, rope_sin, cache_c, st.grads);

                // ── Backward through the REJECTED branch (decrease rejected log-prob) ──
                {
                    ts::Grads r_grads;
                    alloc_grads(r_grads, params);

                    ts::FwdCache cache_r;
                    cuda::CudaTensor x_final_r = ts::forward(params, d_rejected_tok, st.dims,
                                                             rope_cos, rope_sin, cache_r);
                    auto logits_r = cuda::matmul_transb(x_final_r, params[0]);

                    cuda::CudaTensor d_logits_r({(int64_t)BS, (int64_t)MODEL_VOCAB}, cuda::DType::FP16);
                    {
                        cuda::CudaTensor tloss({(int64_t)BS}, cuda::DType::F32);
                        launch_ce_fwd_bwd((const __half*)logits_r.data, (const int*)d_rejected_tgt.data,
                                          (const float*)d_rejected_mask.data,
                                          (float*)tloss.data, (__half*)d_logits_r.data,
                                          BS, MODEL_VOCAB, cuda::ctx.compute_stream);
                    }

                    cuda::sync_compute();
                    int tok_count_r = 0;
                    for (int i = 0; i < BS; i++) if (rejected_mask[i] > 0.0f) tok_count_r++;
                    float norm_r = (tok_count_r > 0) ? (coeff_rejected / tok_count_r) : 0.0f;
                    launch_scale_fp16((__half*)d_logits_r.data, norm_r,
                                      d_logits_r.numel(), cuda::ctx.compute_stream);

                    auto dx_final_r = cuda::matmul(d_logits_r, params[0]);
                    auto d_embed_r = cuda::matmul(d_logits_r, x_final_r, true, false);
                    launch_f16_to_f32_seed((const __half*)d_embed_r.data,
                                           (float*)r_grads.embed_accum.data,
                                           (size_t)MODEL_VOCAB * D_MODEL, cuda::ctx.compute_stream);
                    ts::backward(params, d_rejected_tok, x_final_r, dx_final_r, st.dims,
                                 rope_cos, rope_sin, cache_r, r_grads);

                    // Accumulate rejected gradients into main grads
                    elementwise_add_fp16((__half*)st.grads.d_embed.data,
                                         (__half*)r_grads.d_embed.data,
                                         (__half*)st.grads.d_embed.data,
                                         st.grads.d_embed.numel(), cuda::ctx.compute_stream);
                    elementwise_add_fp16((__half*)st.grads.d_final_rms.data,
                                         (__half*)r_grads.d_final_rms.data,
                                         (__half*)st.grads.d_final_rms.data,
                                         st.grads.d_final_rms.numel(), cuda::ctx.compute_stream);
                    launch_f32_add((const float*)r_grads.embed_accum.data,
                                   (float*)st.grads.embed_accum.data,
                                   (size_t)MODEL_VOCAB * D_MODEL, cuda::ctx.compute_stream);
                    for (int l = 0; l < N_LAYERS; l++) {
                        auto& lg = st.grads.layers[l];
                        auto& rg = r_grads.layers[l];
                        cuda::CudaTensor* lga[] = {&lg.dwq,&lg.dwk,&lg.dwv,&lg.dwo,&lg.dw1,&lg.dw2,&lg.dw3,&lg.drms_attn,&lg.drms_ffn};
                        cuda::CudaTensor* rga[] = {&rg.dwq,&rg.dwk,&rg.dwv,&rg.dwo,&rg.dw1,&rg.dw2,&rg.dw3,&rg.drms_attn,&rg.drms_ffn};
                        for (int i = 0; i < 9; i++)
                            elementwise_add_fp16((__half*)lga[i]->data, (__half*)rga[i]->data,
                                                 (__half*)lga[i]->data, lga[i]->numel(),
                                                 cuda::ctx.compute_stream);
                    }
                }

#ifdef USE_DDP
                ddp_allreduce(st.grads);
#endif
                clip_grads(st.grads, g_grad_clip);
                float lr = sched_lr(step, total_steps);
                adamw_all(params, master_params, opt_m, opt_v, st.grads, lr, 0.005f, step);

                step++;
                if (step % g_log_every == 0) {
#ifdef USE_DDP
                    if (!g_use_ddp || cuda::ddp::is_main())
#endif
                    {
                        double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
                        printf("[DPO] ep=%d step=%d/%d loss=%.4f margin=%.3f lr=%.2e %.1f tok/s\n",
                               epoch + 1, step, total_steps,
                               running_loss / g_log_every, running_margin / g_log_every,
                               lr, (double)(step - start_step) * (double)global_bs * S * 2 / elapsed);
                    }
                    running_loss = 0; running_margin = 0;
                }
                if (step % g_save_every == 0)
                    save_stage_checkpoint("_dpo",
                                          step, epoch, bi + global_bs);
                if (dpo_upload_interval > 0 && step % dpo_upload_interval == 0)
                    trigger_upload(step, "dpo");
            }
        }
        save_stage_checkpoint("_dpo_final", step, g_epochs, 0);
        printf("\n[DONE] DPO complete. %d steps.\n", step);
        trigger_upload(step, "dpo");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Stage 5: RL — GRPO-lite with rule-based verifier (coding)
    //
    // Per training step:
    //   1. Sample N completions per prompt from the policy (top-k + temp).
    //   2. Decode code, run rule-based verifier (executes code, compares stdout).
    //   3. Group-normalised advantage A[g] = (r - mean)/(std+eps) per prompt.
    //   4. Per-token KL penalty vs reference: kl_t = loss_pol - loss_ref.
    //   5. Backprop CE gradient scaled by (A[g] - beta*kl_t)/Nt[g].
    // ═══════════════════════════════════════════════════════════════════════
    else if (g_stage == 5) {
        auto rl_samples = load_rl_data(g_rl_dir, tok, g_max_seq);
        if (rl_samples.empty()) {
            fprintf(stderr, "[ERR] No RL data found in %s\n", g_rl_dir.c_str());
            return 1;
        }

        // Reference model (frozen) for the KL penalty.
        std::string ref_path = g_ref_model.empty() ? g_base_model : g_ref_model;
        auto ref_params = load_model(ref_path);
        printf("[RL] Reference model: %s\n", ref_path.c_str());

        const int ddp_w = ddp_world();
        const int ddp_r = ddp_rank();
        const size_t global_bs = (size_t)g_batch_size * (size_t)ddp_w;

        int total_steps = (int)((size_t)g_epochs * rl_samples.size() / global_bs);
        if (total_steps < 1) total_steps = 1;
        if (g_max_steps > 0) total_steps = std::min(total_steps, g_max_steps);

        fs::create_directories(g_rl_workdir);

        auto t0 = Clock::now();
        int step = start_step;
        float running_loss = 0.0f, running_rew = 0.0f;

        // Forward + logits (no grad) for reference log-prob use.
        auto forward_logits = [&](std::vector<cuda::CudaTensor>& p,
                                  const cuda::CudaTensor& d_tok_in, int B, int S) {
            ts::FwdCache cache;
            st.dims.B = B; st.dims.S = S;
            cuda::CudaTensor x = ts::forward(p, d_tok_in, st.dims,
                                             rope_cos, rope_sin, cache);
            cuda::CudaTensor logits = cuda::matmul_transb(x, p[0]);
            cuda::sync_compute();
            return logits;
        };

        for (int epoch = start_epoch; epoch < g_epochs && !g_stop; epoch++) {
            // Same seed on every rank => same shuffle => disjoint slices.
            std::mt19937 rng(42 + epoch);
            std::shuffle(rl_samples.begin(), rl_samples.end(), rng);

            size_t bi_start = (epoch == start_epoch) ? start_sample_idx : 0;
            bi_start = (bi_start / global_bs) * global_bs;

            for (size_t bi = bi_start; bi + global_bs <= rl_samples.size() && !g_stop;
                 bi += global_bs) {
                if (g_max_steps > 0 && step >= g_max_steps) break;

                const size_t base = bi + (size_t)ddp_r * (size_t)g_batch_size;
                const int P = g_batch_size;      // prompts in this batch
                const int N = g_rl_n_samples;    // samples per prompt
                const int G = P * N;             // total sequences

                // ── 1. Sample N completions per prompt ──────────────────────
                std::vector<std::vector<int>> seq_tokens(G);
                std::vector<int> plen(G), gen_len(G, 0);
                int max_prompt = 0;
                for (int p = 0; p < P; p++) {
                    auto& s = rl_samples[base + p];
                    int m = std::min((int)s.prompt_tokens.size(), g_max_seq - 1);
                    max_prompt = std::max(max_prompt, m);
                    for (int n = 0; n < N; n++) {
                        int g = p * N + n;
                        seq_tokens[g].assign(s.prompt_tokens.begin(),
                                             s.prompt_tokens.begin() + m);
                        plen[g] = (int)seq_tokens[g].size();
                    }
                }
                int Lmax = max_prompt + g_rl_max_new;
                std::vector<int> tok_buf((size_t)G * Lmax, TOK_PAD);
                for (int g = 0; g < G; g++)
                    for (int i = 0; i < plen[g]; i++)
                        tok_buf[(size_t)g * Lmax + i] = seq_tokens[g][i];

                std::vector<int> alive(G, 1);
                std::vector<int> next_tok(G);
                std::vector<int> row_idx(G);
                std::vector<__half> gathered_logits((size_t)G * MODEL_VOCAB);
                std::mt19937 sampler_rng(1234 + step + ddp_r);

                for (int t = 0; t < g_rl_max_new; t++) {
                    int L = max_prompt + t;
                    std::vector<int> tok_slice((size_t)G * L);
                    for (int g = 0; g < G; g++)
                        for (int i = 0; i < L; i++)
                            tok_slice[(size_t)g * L + i] = tok_buf[(size_t)g * Lmax + i];

                    auto d_tok_rs = cuda::CudaTensor::from_host(tok_slice.data(), {G, L});
                    auto logits = forward_logits(params, d_tok_rs, G, L);

                    for (int g = 0; g < G; g++)
                        row_idx[g] = (g * L) + plen[g] + t - 1;

                    auto d_row_idx = cuda::CudaTensor::from_host(row_idx.data(), {G});
                    cuda::CudaTensor d_rows({(int64_t)G, (int64_t)MODEL_VOCAB}, cuda::DType::FP16);
                    launch_gather_logits_rows((const __half*)logits.data,
                                              (const int*)d_row_idx.data,
                                              (__half*)d_rows.data, G, MODEL_VOCAB,
                                              cuda::ctx.compute_stream);
                    cuda::sync_compute();
                    d_rows.copy_to_host(gathered_logits);

                    sample_top_k_host(gathered_logits, alive, next_tok,
                                      G, MODEL_VOCAB, g_rl_temp, g_rl_top_k,
                                      sampler_rng);

                    bool any_alive = false;
                    for (int g = 0; g < G; g++) {
                        if (!alive[g]) continue;
                        int gidx = plen[g] + t;
                        if (gidx < Lmax) tok_buf[(size_t)g * Lmax + gidx] = next_tok[g];
                        gen_len[g]++;
                        if (next_tok[g] == TOK_EOS) alive[g] = 0;
                        else any_alive = true;
                    }
                    if (!any_alive) break;
                }
                // Finalise sequences (prompt + completion).
                for (int g = 0; g < G; g++) {
                    seq_tokens[g].resize(plen[g] + gen_len[g]);
                    for (int i = 0; i < gen_len[g]; i++)
                        seq_tokens[g][plen[g] + i] = tok_buf[(size_t)g * Lmax + plen[g] + i];
                }

                // ── 2. Decode + run rule-based verifier ─────────────────────
                std::string req_path = g_rl_workdir + "/req.jsonl";
                std::string rew_path = g_rl_workdir + "/reward.jsonl";
                {
                    std::ofstream rf(req_path);
                    auto esc = [](const std::string& x) {
                        std::string o;
                        for (char c : x) {
                            if (c == '"' || c == '\\') o += '\\';
                            else if (c == '\n') o += "\\n";
                            else if (c == '\r') o += "\\r";
                            else if (c == '\t') o += "\\t";
                            else o += c;
                        }
                        return o;
                    };
                    for (int g = 0; g < G; g++) {
                        int p = g / N;
                        auto& s = rl_samples[base + p];
                        std::vector<TokenID> comp(seq_tokens[g].begin() + plen[g],
                                                  seq_tokens[g].end());
                        std::string code = tok.decode(comp, true);
                        // pass lang through as-is (en/ja/id) — the verifier
                        // only logs it, but coercion hid Indonesian rows
                        std::string lang_out = s.lang.empty() ? "en" : s.lang;
                        rf << "{\"id\":" << g
                           << ",\"lang\":\"" << lang_out << "\""
                           << ",\"code\":\"" << esc(code) << "\""
                           << ",\"input\":\"" << esc(s.mock_input) << "\""
                           << ",\"expected\":\"" << esc(s.expected_output) << "\"}\n";
                    }
                    rf.close();
                }
                std::string cmd = g_verifier_cmd + " " + req_path + " " + rew_path;
                int rc = system(cmd.c_str());
                if (rc != 0)
                    fprintf(stderr, "[RL][WARN] verifier failed (rc=%d), zero rewards\n", rc);

                std::vector<float> rewards(G, 0.0f);
                {
                    std::ifstream wf(rew_path);
                    std::string line;
                    while (std::getline(wf, line)) {
                        auto idp = line.find("\"id\":");
                        auto rp = line.find("\"reward\":");
                        if (idp == std::string::npos || rp == std::string::npos) continue;
                        long id = std::strtol(line.c_str() + idp + 5, nullptr, 10);
                        float r = (float)std::strtod(line.c_str() + rp + 9, nullptr);
                        if (id >= 0 && id < G) rewards[id] = r;
                    }
                }

                // ── 3. Group-normalised advantage ───────────────────────────
                std::vector<float> adv(G, 0.0f);
                for (int p = 0; p < P; p++) {
                    float mean = 0.0f;
                    for (int n = 0; n < N; n++) mean += rewards[p * N + n];
                    mean /= (float)N;
                    float var = 0.0f;
                    for (int n = 0; n < N; n++) {
                        float d = rewards[p * N + n] - mean;
                        var += d * d;
                    }
                    var /= (float)N;
                    float stdv = sqrtf(var) + 1e-8f;
                    for (int n = 0; n < N; n++)
                        adv[p * N + n] = (rewards[p * N + n] - mean) / stdv;
                }

                // ── 4+5. Token matrix, per-token KL, scaled backprop ────────
                int S = 0;
                for (int g = 0; g < G; g++)
                    S = std::max(S, plen[g] + gen_len[g]);
                if (S < 1) S = 1;
                if (S > g_max_seq) S = g_max_seq;

                std::vector<int> tok_mat((size_t)G * S, TOK_PAD);
                std::vector<int> tgt_mat((size_t)G * S, TOK_PAD);
                std::vector<float> mask((size_t)G * S, 0.0f);
                std::vector<int> Nt(G, 0);
                for (int g = 0; g < G; g++) {
                    int real_end = std::min(plen[g] + gen_len[g], S);
                    for (int i = 0; i < real_end; i++)
                        tok_mat[(size_t)g * S + i] = seq_tokens[g][i];
                    // Loss at positions [plen-1, real_end-2] predicts the
                    // completion tokens (the last one is EOS if the model
                    // stopped). This is exactly the sampled-token log-prob.
                    int start_pred = plen[g] - 1;
                    int end_pred = real_end - 2;
                    for (int i = 0; i < real_end; i++) {
                        tgt_mat[(size_t)g * S + i] = (i + 1 < real_end)
                            ? seq_tokens[g][i + 1] : TOK_EOS;
                        if (i >= start_pred && i <= end_pred) {
                            mask[(size_t)g * S + i] = 1.0f;
                            Nt[g]++;
                        }
                    }
                }

                auto d_tok_g  = cuda::CudaTensor::from_host(tok_mat.data(), {G, S});
                auto d_tgt_g  = cuda::CudaTensor::from_host(tgt_mat.data(), {G, S});
                auto d_mask_g = cuda::CudaTensor::from_host(mask.data(), {G, S});

                // Policy forward + fused CE (loss + gradient).
                ts::FwdCache cache;
                st.dims.B = G; st.dims.S = S;
                cuda::CudaTensor x_final = ts::forward(params, d_tok_g, st.dims,
                                                       rope_cos, rope_sin, cache);
                auto pol_logits = cuda::matmul_transb(x_final, params[0]);
                int BS = G * S;
                cuda::CudaTensor pol_loss({(int64_t)BS}, cuda::DType::F32);
                cuda::CudaTensor d_logits({(int64_t)BS, (int64_t)MODEL_VOCAB}, cuda::DType::FP16);
                launch_ce_fwd_bwd((const __half*)pol_logits.data, (const int*)d_tgt_g.data,
                                  (const float*)d_mask_g.data,
                                  (float*)pol_loss.data, (__half*)d_logits.data,
                                  BS, MODEL_VOCAB, cuda::ctx.compute_stream);
                cuda::sync_compute();

                // Reference per-token loss for the KL term.
                auto ref_logits = forward_logits(ref_params, d_tok_g, G, S);
                cuda::CudaTensor ref_loss({(int64_t)BS}, cuda::DType::F32);
                cuda::CudaTensor dummy_dl({(int64_t)BS, (int64_t)MODEL_VOCAB}, cuda::DType::FP16);
                launch_ce_fwd_bwd((const __half*)ref_logits.data, (const int*)d_tgt_g.data,
                                  (const float*)d_mask_g.data,
                                  (float*)ref_loss.data, (__half*)dummy_dl.data,
                                  BS, MODEL_VOCAB, cuda::ctx.compute_stream);
                cuda::sync_compute();

                std::vector<float> h_pol(BS), h_ref(BS);
                cudaMemcpy(h_pol.data(), pol_loss.data, BS * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_ref.data(), ref_loss.data, BS * sizeof(float), cudaMemcpyDeviceToHost);
                cuda::sync_compute();

                // Per-token scale for the CE gradient:
                //   r_t  = logp_ref - logp_pol = loss_pol - loss_ref
                //   scale_t = (A[g] - beta*(1 - exp(r_t))) / Nt[g]
                // The KL term uses the kl-1 estimator exp(r)-r-1, whose
                // gradient contributes -beta*(1-exp(r)) per token.
                std::vector<float> scale(BS, 0.0f);
                double loss_sum = 0.0; int mask_count = 0;
                for (int g = 0; g < G; g++) {
                    int nt = Nt[g];
                    if (nt <= 0) continue;
                    for (int i = 0; i < S; i++) {
                        int idx = g * S + i;
                        if (mask[idx] <= 0.0f) continue;
                        float r_t = h_pol[idx] - h_ref[idx];
                        scale[idx] = (adv[g] - g_rl_beta * (1.0f - expf(r_t))) / (float)nt;
                        loss_sum += h_pol[idx];
                        mask_count++;
                    }
                }
                float loss_val = mask_count > 0 ? (float)(loss_sum / mask_count) : 0.0f;

                auto d_scale = cuda::CudaTensor::from_host(scale.data(), {BS});
                launch_scale_rows_fp16((__half*)d_logits.data,
                                       (const float*)d_scale.data,
                                       BS, MODEL_VOCAB, cuda::ctx.compute_stream);
                cuda::sync_compute();

                // Backward (weight-tying handled like SFT/DPO).
                auto dx_final = cuda::matmul(d_logits, params[0]);
                auto d_embed_out = cuda::matmul(d_logits, x_final, true, false);
                launch_f16_to_f32_seed((const __half*)d_embed_out.data,
                                       (float*)st.grads.embed_accum.data,
                                       (size_t)MODEL_VOCAB * D_MODEL, cuda::ctx.compute_stream);
                ts::backward(params, d_tok_g, x_final, dx_final, st.dims,
                             rope_cos, rope_sin, cache, st.grads);

#ifdef USE_DDP
                ddp_allreduce(st.grads);
#endif
                clip_grads(st.grads, g_grad_clip);
                float lr = sched_lr(step, total_steps);
                adamw_all(params, master_params, opt_m, opt_v, st.grads, lr, 0.005f, step);

                step++;
                float mean_rew = 0.0f;
                for (float r : rewards) mean_rew += r;
                mean_rew /= (float)G;
                running_loss += loss_val;
                running_rew += mean_rew;

                if (step % g_log_every == 0) {
#ifdef USE_DDP
                    if (!g_use_ddp || cuda::ddp::is_main())
#endif
                    {
                        double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
                        printf("[RL] ep=%d step=%d/%d loss=%.4f reward=%.3f lr=%.2e %.1f tok/s\n",
                               epoch + 1, step, total_steps,
                               running_loss / g_log_every, running_rew / g_log_every,
                               lr, (double)(step - start_step) * (double)G * S / elapsed);
                    }
                    running_loss = 0.0f; running_rew = 0.0f;
                }
                if (step % g_save_every == 0)
                    save_stage_checkpoint("_rl", step, epoch, bi + global_bs);
            }
        }
        save_stage_checkpoint("_rl_final", step, g_epochs, 0);
        printf("\n[DONE] RL complete. %d steps.\n", step);
        trigger_upload(step, "rl");
    }

    cudaFree(rope_cos);
    cudaFree(rope_sin);
#ifdef USE_DDP
    if (g_use_ddp) cuda::ddp::shutdown();
#endif
    cuda::nvml_shutdown();
    return 0;
}
