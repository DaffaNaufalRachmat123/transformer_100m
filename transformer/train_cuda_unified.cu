// =============================================================================
// train_cuda_unified.cu — SFT / DPO / Phone Training (CUDA, 204M Model)
//
// Single binary, stage selected via --stage flag:
//   --stage 2  = SFT (Supervised Fine-Tuning)
//   --stage 3  = DPO (Direct Preference Optimization)
//   --stage 4  = Phone Adapter
//
// Architecture: 204M (D=768, L=18, QH=12, KVH=3, DH=64, FFN=3072)
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
#include "transformer/safetensors.h"
#include "transformer/train_step.cuh"
#include "cuda/ce_embed_kernels.cuh"

#ifdef USE_DDP
#include "cuda/cuda_ddp.h"
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

// ─── Architecture constants (204M model — matches pretrain) ────────────────
static constexpr int D_MODEL    = 768;
static constexpr int N_LAYERS   = 18;
static constexpr int N_Q_HEADS  = 12;
static constexpr int N_KV_HEADS = 3;
static constexpr int D_HEAD     = 64;
static constexpr int FFN_DIM    = 3072;
static constexpr int MODEL_VOCAB = 65536;
static constexpr float ROPE_THETA = 500000.0f;

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
static int    g_log_every             = 10;
static int    g_save_every            = 200;
static int    g_warmup                = 100;
static float  g_grad_clip             = 1.0f;
static bool   g_use_ddp               = false;
static int    g_num_gpus              = 2;
static bool   g_resume                = false;

static volatile bool g_stop = false;
void sig_handler(int) { g_stop = true; }

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
            // Find the closing quote by scanning forward from text_start
            auto text_end = text_start;
            while (text_end < line.size() && line[text_end] != '"') {
                if (line[text_end] == '\\' && text_end + 1 < line.size()) text_end++;
                text_end++;
            }
            if (text_end >= line.size()) continue;
            std::string text = line.substr(text_start, text_end - text_start);

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
static float sched_lr(int step, int total_steps) {
    float lr = g_lr * std::min(1.0f, (float)(step + 1) / (float)std::max(1, g_warmup));
    if (step >= g_warmup && total_steps > g_warmup) {
        float progress = (float)(step - g_warmup) / (float)(total_steps - g_warmup);
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
        else if (arg == "--grad_clip" && i+1 < argc) g_grad_clip = std::stof(argv[++i]);
        else if (arg == "--log_every" && i+1 < argc) g_log_every = std::stoi(argv[++i]);
        else if (arg == "--save_every" && i+1 < argc) g_save_every = std::stoi(argv[++i]);
        else if (arg == "--warmup" && i+1 < argc)    g_warmup = std::stoi(argv[++i]);
        else if (arg == "--ddp")                     g_use_ddp = true;
        else if (arg == "--num_gpus" && i+1 < argc)  g_num_gpus = std::stoi(argv[++i]);
        else if (arg == "--resume")                  g_resume = true;
        else if (arg == "--help") {
            printf("Usage: train_sft_cuda --stage <2|3|4> [options]\n");
            printf("  --stage 2          SFT (Supervised Fine-Tuning)\n");
            printf("  --stage 3          DPO (Direct Preference Optimization)\n");
            printf("  --stage 4          Phone Adapter\n");
            printf("  --base_model       Pretrained model checkpoint\n");
            printf("  --sft_dir          SFT data directory\n");
            printf("  --dpo_dir          DPO data directory\n");
            printf("  --batch_size       Batch size [4]\n");
            printf("  --epochs           Number of epochs [3]\n");
            printf("  --lr               Learning rate [3e-5]\n");
            printf("  --max_seq          Max sequence length [512]\n");
            printf("  --grad_clip        Global grad-norm clip [1.0]\n");
            printf("  --ddp              Enable multi-GPU DDP (NCCL)\n");
            printf("  --resume           Resume from checkpoint (weights + optim + step)\n");
            return 0;
        }
    }

    // ─── Init CUDA (DDP: fork BEFORE any CUDA call) ─────────────────────────
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

    const char* stage_names[] = {"", "", "SFT", "DPO", "Phone"};
    printf("\n[Stage %d: %s] 204M model | FP16 | batch=%d | seq=%d | lr=%.1e%s\n",
           g_stage, stage_names[g_stage], g_batch_size, g_max_seq, g_lr,
           g_use_ddp ? " | DDP x2" : "");

    // Load tokenizer
    BPEEncoder tok;
    if (!g_bpe_path.empty()) tok.load(g_bpe_path);
    else tok.load_vocab(g_vocab_dir);
    TOK_BOS = tok.bos_id(); TOK_EOS = tok.eos_id(); TOK_PAD = tok.pad_id();
    if (TOK_PAD < 0) TOK_PAD = TOK_BOS;
    TOK_SYSTEM = (int)tok.special_id(SpecialTokens::SYSTEM);
    TOK_USER = (int)tok.special_id(SpecialTokens::USER);
    TOK_ASSISTANT = (int)tok.special_id(SpecialTokens::ASSISTANT);
    TOK_END = (int)tok.special_id(SpecialTokens::END);

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
        auto copy_embed_row = [&](int dst_id, int src_id) {
            size_t row_bytes = (size_t)D_MODEL * sizeof(__half);
            cudaMemcpy((__half*)params[0].data + (size_t)dst_id * D_MODEL,
                       (const __half*)params[0].data + (size_t)src_id * D_MODEL,
                       row_bytes, cudaMemcpyDeviceToDevice);
            size_t f32_row_bytes = (size_t)D_MODEL * sizeof(float);
            cudaMemcpy((float*)master_params[0].data + (size_t)dst_id * D_MODEL,
                       (const float*)master_params[0].data + (size_t)src_id * D_MODEL,
                       f32_row_bytes, cudaMemcpyDeviceToDevice);
        };
        copy_embed_row(TOK_SYSTEM,    TOK_BOS);
        copy_embed_row(TOK_USER,      TOK_BOS);
        copy_embed_row(TOK_ASSISTANT, TOK_BOS);
        copy_embed_row(TOK_PAD,       TOK_EOS);
        printf("[INIT] Copied BOS(0) embedding → SYSTEM(%d) USER(%d) ASSISTANT(%d)\n",
               TOK_SYSTEM, TOK_USER, TOK_ASSISTANT);
        printf("[INIT] Copied EOS(4) embedding → PAD(%d)\n", TOK_PAD);
    }

    // RoPE tables
    float *rope_cos = nullptr, *rope_sin = nullptr;
    {
        int half_dim = D_HEAD / 2;
        std::vector<float> cos_h(2048 * half_dim), sin_h(2048 * half_dim);
        for (int pos = 0; pos < 2048; pos++)
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
    const char* stage_tag_early = (g_stage == 2) ? "sft" : (g_stage == 3) ? "dpo" : "phone";
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
    const char* stage_tag = (g_stage == 2) ? "sft" : (g_stage == 3) ? "dpo" : "phone";

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

        int total_steps = g_epochs * (int)samples.size() / g_batch_size;
        if (g_max_steps > 0) total_steps = std::min(total_steps, g_max_steps);

        auto t0 = Clock::now();
        int step = start_step;
        float running_loss = 0;

        for (int epoch = start_epoch; epoch < g_epochs && !g_stop; epoch++) {
            std::mt19937 rng(42 + epoch);
            std::shuffle(samples.begin(), samples.end(), rng);

            // On the first (resumed) epoch, skip already-processed samples.
            size_t bi_start = (epoch == start_epoch) ? start_sample_idx : 0;
            for (size_t bi = bi_start; bi + g_batch_size <= samples.size() && !g_stop; bi += g_batch_size) {
                if (g_max_steps > 0 && step >= g_max_steps) break;

                int S = g_max_seq, B = g_batch_size;
                std::vector<int> batch_tokens(B * S, TOK_PAD);
                std::vector<int> batch_targets(B * S, TOK_PAD);
                std::vector<float> batch_mask(B * S, 0.0f);

                for (int b = 0; b < B; b++) {
                    auto& sample = samples[bi + b];
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
                               (double)step * B * S / elapsed);
                    }
                    running_loss = 0;
                }
                if (step % g_save_every == 0)
                    save_stage_checkpoint(std::string("_") + (g_stage == 2 ? "sft" : "phone")
                                          + "_step" + std::to_string(step),
                                          step, epoch, bi + g_batch_size);
            }
        }
        save_stage_checkpoint(std::string("_") + (g_stage == 2 ? "sft" : "phone") + "_final",
                              step, g_epochs, 0);
        printf("\n[DONE] %s complete. %d steps.\n", tag, step);
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

        int total_steps = g_epochs * (int)dpo_samples.size() / g_batch_size;
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
            std::mt19937 rng(42 + epoch);
            std::shuffle(dpo_samples.begin(), dpo_samples.end(), rng);

            size_t bi_start = (epoch == start_epoch) ? start_sample_idx : 0;
            for (size_t bi = bi_start; bi + g_batch_size <= dpo_samples.size() && !g_stop; bi += g_batch_size) {
                if (g_max_steps > 0 && step >= g_max_steps) break;
                int S = g_max_seq, B = g_batch_size;

                std::vector<int> chosen_tok(B * S, TOK_PAD), rejected_tok(B * S, TOK_PAD);
                std::vector<int> chosen_tgt(B * S, TOK_PAD), rejected_tgt(B * S, TOK_PAD);
                std::vector<float> chosen_mask(B * S, 0.0f), rejected_mask(B * S, 0.0f);

                for (int b = 0; b < B; b++) {
                    auto& sample = dpo_samples[bi + b];
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

                float logit_val = g_dpo_beta * ((lp_c - rlp_c) - (lp_r - rlp_r));
                float dpo_loss  = logf(1.0f + expf(-logit_val));
                running_loss   += dpo_loss;
                running_margin += logit_val / g_dpo_beta;

                // DPO coefficient: d/dlogit [-log sigmoid(logit)] = -sigmoid(-logit)
                float coeff = -g_dpo_beta * (1.0f / (1.0f + expf(logit_val)));

                // ── Backward through the CHOSEN branch with a fresh forward ──
                // (increasing chosen log-prob lowers the loss when coeff<0).
                st.dims.B = B; st.dims.S = S;
                ts::FwdCache cache;
                cuda::CudaTensor x_final = ts::forward(params, d_chosen_tok, st.dims,
                                                       rope_cos, rope_sin, cache);
                auto logits = cuda::matmul_transb(x_final, params[0]);

                int BS = B * S;
                cuda::CudaTensor per_token_loss({(int64_t)BS}, cuda::DType::F32);
                cuda::CudaTensor d_logits({(int64_t)BS, (int64_t)MODEL_VOCAB}, cuda::DType::FP16);
                launch_ce_fwd_bwd((const __half*)logits.data, (const int*)d_chosen_tgt.data,
                                  (const float*)d_chosen_mask.data,
                                  (float*)per_token_loss.data, (__half*)d_logits.data,
                                  BS, MODEL_VOCAB, cuda::ctx.compute_stream);

                // Count masked tokens for mean normalization
                cuda::sync_compute();
                int tok_count = 0;
                for (int i = 0; i < BS; i++) if (chosen_mask[i] > 0.0f) tok_count++;
                float norm = (tok_count > 0) ? (coeff / tok_count) : 0.0f;
                launch_scale_fp16((__half*)d_logits.data, norm,
                                  d_logits.numel(), cuda::ctx.compute_stream);

                auto dx_final = cuda::matmul(d_logits, params[0]);
                auto d_embed_out = cuda::matmul(d_logits, x_final, true, false);
                launch_f16_to_f32_seed((const __half*)d_embed_out.data,
                                       (float*)st.grads.embed_accum.data,
                                       (size_t)MODEL_VOCAB * D_MODEL, cuda::ctx.compute_stream);
                ts::backward(params, d_chosen_tok, x_final, dx_final, st.dims,
                             rope_cos, rope_sin, cache, st.grads);

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
                               lr, (double)step * B * S * 2 / elapsed);
                    }
                    running_loss = 0; running_margin = 0;
                }
                if (step % g_save_every == 0)
                    save_stage_checkpoint("_dpo_step" + std::to_string(step),
                                          step, epoch, bi + g_batch_size);
            }
        }
        save_stage_checkpoint("_dpo_final", step, g_epochs, 0);
        printf("\n[DONE] DPO complete. %d steps.\n", step);
    }

    cudaFree(rope_cos);
    cudaFree(rope_sin);
#ifdef USE_DDP
    if (g_use_ddp) cuda::ddp::shutdown();
#endif
    cuda::nvml_shutdown();
    return 0;
}
