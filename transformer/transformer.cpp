#include "transformer.h"
#include "safetensors.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

// =============================================================================
// Transformer constructor
// =============================================================================

Transformer::Transformer(const ModelConfig& cfg) : cfg_(cfg) {}

// =============================================================================
// MatVec  y = x @ W   x:[D_in]  W:[D_in, D_out] row-major  y:[D_out]
// =============================================================================

void Transformer::matmul_vec(const float* x, const float* W,
                              float* out, int D_in, int D_out) {
    std::memset(out, 0, D_out * sizeof(float));
    for (int i = 0; i < D_in; ++i) {
        float xi = x[i];
        if (xi == 0.0f) continue;
        const float* row = W + (size_t)i * D_out;
        for (int j = 0; j < D_out; ++j)
            out[j] += xi * row[j];
    }
}

// =============================================================================
// RMSNorm  y = x * (1 / sqrt(mean(x²) + eps)) * gamma  (single row)
// =============================================================================

void Transformer::rms_norm(const float* x, const float* gamma,
                            float* out, int D, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < D; ++i) sum_sq += x[i] * x[i];
    float rscale = 1.0f / std::sqrt(sum_sq / (float)D + eps);
    for (int i = 0; i < D; ++i)
        out[i] = x[i] * rscale * gamma[i];
}

// =============================================================================
// RoPE precompute (matching train_cuda.cu RoPETables::init exactly)
// =============================================================================

void Transformer::precompute_rope() {
    int max_seq = (int)cfg_.max_seq_len;
    int d_head  = (int)cfg_.d_head;
    int half    = d_head / 2;
    size_t size = (size_t)max_seq * half;

    rope_cos_.resize(size);
    rope_sin_.resize(size);

    for (int pos = 0; pos < max_seq; ++pos) {
        for (int j = 0; j < half; ++j) {
            float freq = (float)pos /
                std::pow(cfg_.rope_theta, 2.0f * j / d_head);
            rope_cos_[pos * half + j] = std::cos(freq);
            rope_sin_[pos * half + j] = std::sin(freq);
        }
    }
}

// =============================================================================
// Apply RoPE in-place (interleaved pairs, matching rope_forward_kernel)
// =============================================================================

void Transformer::apply_rope(float* vec, int n_heads, int d_head, int pos) const {
    int half = d_head / 2;
    const float* cos_row = rope_cos_.data() + (size_t)pos * half;
    const float* sin_row = rope_sin_.data() + (size_t)pos * half;

    for (int h = 0; h < n_heads; ++h) {
        float* head = vec + (size_t)h * d_head;
        for (int j = 0; j < half; ++j) {
            float c = cos_row[j];
            float s = sin_row[j];
            float x0 = head[j * 2];
            float x1 = head[j * 2 + 1];
            head[j * 2]     = x0 * c - x1 * s;
            head[j * 2 + 1] = x0 * s + x1 * c;
        }
    }
}

// =============================================================================
// SwiGLU  out = gate * silu(up)   (single row, FFN dim)
// =============================================================================

void Transformer::swiglu(const float* gate, const float* up,
                          float* out, int N) {
    for (int i = 0; i < N; ++i) {
        float u = up[i];
        // silu(u) = u * sigmoid(u) = u / (1 + exp(-u))
        float silu_val = u / (1.0f + std::exp(-u));
        out[i] = gate[i] * silu_val;
    }
}

// =============================================================================
// Load safetensors checkpoint
// =============================================================================

void Transformer::load_safetensors(const std::string& path) {
    SafeTensorsLoader loader(path);

    std::cerr << "[Transformer] Loading " << path << "\n";
    loader.print_info();

    uint32_t V   = cfg_.vocab_size;
    uint32_t D   = cfg_.d_model;
    uint32_t L   = cfg_.n_layers;
    uint32_t KVD = cfg_.kv_dim();
    uint32_t FFN = cfg_.ffn_dim;

    // Embedding
    embed_.resize((size_t)V * D);
    loader.read_into_f32("embed.weight", embed_.data());

    // Layers
    layers_.resize(L);
    for (uint32_t l = 0; l < L; ++l) {
        std::string prefix = "layer." + std::to_string(l) + ".";
        auto& lw = layers_[l];

        // Q, K, V, O projections
        lw.wq.resize((size_t)D * D);         loader.read_into_f32((prefix + "wq.weight").c_str(), lw.wq.data());
        lw.wk.resize((size_t)D * KVD);       loader.read_into_f32((prefix + "wk.weight").c_str(), lw.wk.data());
        lw.wv.resize((size_t)D * KVD);       loader.read_into_f32((prefix + "wv.weight").c_str(), lw.wv.data());
        lw.wo.resize((size_t)D * D);         loader.read_into_f32((prefix + "wo.weight").c_str(), lw.wo.data());

        // SwiGLU: gate = w1, down = w2, up = w3
        lw.w1.resize((size_t)D * FFN);       loader.read_into_f32((prefix + "w1.weight").c_str(), lw.w1.data());
        lw.w2.resize((size_t)FFN * D);       loader.read_into_f32((prefix + "w2.weight").c_str(), lw.w2.data());
        lw.w3.resize((size_t)D * FFN);       loader.read_into_f32((prefix + "w3.weight").c_str(), lw.w3.data());

        // RMSNorm gains
        lw.rms_attn.resize(D);               loader.read_into_f32((prefix + "rms_attn.weight").c_str(), lw.rms_attn.data());
        lw.rms_ffn.resize(D);                loader.read_into_f32((prefix + "rms_ffn.weight").c_str(),  lw.rms_ffn.data());
    }

    // Final RMSNorm
    final_rms_.resize(D);
    loader.read_into_f32("final_rms.weight", final_rms_.data());

    std::cerr << "[Transformer] Loaded " << (1 + L * 9 + 1) << " tensors\n";

    // Precompute RoPE tables
    precompute_rope();

    // Allocate KV cache
    uint32_t max_seq = cfg_.max_seq_len;
    kv_cache_.resize(L);
    for (uint32_t l = 0; l < L; ++l) {
        auto& c = kv_cache_[l];
        c.k.resize((size_t)max_seq * KVD, 0.0f);
        c.v.resize((size_t)max_seq * KVD, 0.0f);
    }

    // Allocate scratch buffers
    scratch_.x.resize(D);
    scratch_.n1.resize(D);
    scratch_.q.resize(D);
    scratch_.k.resize(KVD);
    scratch_.v.resize(KVD);
    scratch_.scores.resize(max_seq, 0.0f);
    scratch_.attn.resize(D);
    scratch_.o.resize(D);
    scratch_.x_mid.resize(D);
    scratch_.n2.resize(D);
    scratch_.gate.resize(FFN);
    scratch_.up.resize(FFN);
    scratch_.ff.resize(FFN);
    scratch_.down.resize(D);
    x_final_.resize(D);

    std::cerr << "[Transformer] KV-cache: " << L << " layers x "
              << max_seq << " pos x " << KVD << " dim (K+V) = "
              << (L * max_seq * KVD * 2 * 4 / (1024*1024)) << " MB\n";
}

// =============================================================================
// Reset KV cache
// =============================================================================

void Transformer::reset_kv_cache() {
    for (auto& c : kv_cache_) {
        std::memset(c.k.data(), 0, c.k.size() * sizeof(float));
        std::memset(c.v.data(), 0, c.v.size() * sizeof(float));
    }
    std::memset(scratch_.scores.data(), 0, scratch_.scores.size() * sizeof(float));
}

// =============================================================================
// Forward one token  (incremental decode with causal GQA + KV-cache)
// =============================================================================

void Transformer::forward_one_token(std::vector<float>& logits, int token_id, int pos) {
    uint32_t V   = cfg_.vocab_size;
    uint32_t D   = cfg_.d_model;
    uint32_t L   = cfg_.n_layers;
    uint32_t HQ  = cfg_.n_q_heads;
    uint32_t HKV = cfg_.n_kv_heads;
    uint32_t DH  = cfg_.d_head;
    uint32_t KVD = cfg_.kv_dim();
    uint32_t FFN = cfg_.ffn_dim;

    const float eps = 1e-6f;
    const float attn_scale = 1.0f / std::sqrt((float)DH);

    // 1. Embedding lookup  x = embed[token_id]
    const float* embed_row = embed_.data() + (size_t)token_id * D;
    float* x = scratch_.x.data();
    std::memcpy(x, embed_row, D * sizeof(float));

    // 2. For each layer
    for (uint32_t l = 0; l < L; ++l) {
        auto& lw = layers_[l];
        auto& kcache = kv_cache_[l];

        // (a) RMSNorm → n1
        rms_norm(x, lw.rms_attn.data(), scratch_.n1.data(), D, eps);

        // (b) Q, K, V projections
        float* q = scratch_.q.data();
        float* k = scratch_.k.data();
        float* v = scratch_.v.data();
        matmul_vec(scratch_.n1.data(), lw.wq.data(), q, D, D);     // [D]  = HQ*DH
        matmul_vec(scratch_.n1.data(), lw.wk.data(), k, D, KVD);   // [KVD]
        matmul_vec(scratch_.n1.data(), lw.wv.data(), v, D, KVD);   // [KVD]

        // (c) RoPE in-place
        apply_rope(q, (int)HQ,  (int)DH, pos);
        apply_rope(k, (int)HKV, (int)DH, pos);

        // (d) Save K, V into cache at position pos
        size_t kvo_base = (size_t)pos * KVD;
        std::memcpy(kcache.k.data() + kvo_base, k, KVD * sizeof(float));
        std::memcpy(kcache.v.data() + kvo_base, v, KVD * sizeof(float));

        // (e) GQA causal attention  (query over all positions 0..pos)
        float* attn = scratch_.attn.data();
        for (uint32_t hq = 0; hq < HQ; ++hq) {
            uint32_t hkv = hq / (HQ / HKV);         // 0→0 1→0 2→0  3→1 ...
            float* q_head  = q  + (size_t)hq  * DH; // [DH]
            float* a_head  = attn + (size_t)hq * DH;

            // Compute scores
            float max_score = -1e30f;
            size_t n_tokens = (size_t)pos + 1;
            for (size_t j = 0; j < n_tokens; ++j) {
                const float* k_head = kcache.k.data() + j * KVD + (size_t)hkv * DH;
                float dot = 0.0f;
                for (uint32_t d = 0; d < DH; ++d)
                    dot += q_head[d] * k_head[d];
                float s = dot * attn_scale;
                scratch_.scores[j] = s;
                if (s > max_score) max_score = s;
            }

            // Softmax (numerically stable)
            float sum_exp = 0.0f;
            for (size_t j = 0; j < n_tokens; ++j) {
                float e = std::exp(scratch_.scores[j] - max_score);
                scratch_.scores[j] = e;
                sum_exp += e;
            }
            float inv_sum = 1.0f / sum_exp;

            // Weighted sum of values
            std::memset(a_head, 0, DH * sizeof(float));
            for (size_t j = 0; j < n_tokens; ++j) {
                const float* v_head = kcache.v.data() + j * KVD + (size_t)hkv * DH;
                float w = scratch_.scores[j] * inv_sum;
                for (uint32_t d = 0; d < DH; ++d)
                    a_head[d] += w * v_head[d];
            }
        }

        // (f) Output projection  o = attn @ Wo
        matmul_vec(attn, lw.wo.data(), scratch_.o.data(), D, D);

        // x_mid = x + o  (residual)
        float* x_mid = scratch_.x_mid.data();
        for (uint32_t d = 0; d < D; ++d)
            x_mid[d] = x[d] + scratch_.o.data()[d];

        // (g) RMSNorm → n2
        rms_norm(x_mid, lw.rms_ffn.data(), scratch_.n2.data(), D, eps);

        // (h) SwiGLU FFN
        float* gate = scratch_.gate.data();
        float* up   = scratch_.up.data();
        float* ff   = scratch_.ff.data();
        float* down = scratch_.down.data();

        matmul_vec(scratch_.n2.data(), lw.w1.data(), gate, D, FFN);  // gate
        matmul_vec(scratch_.n2.data(), lw.w3.data(), up,   D, FFN);  // up
        swiglu(gate, up, ff, (int)FFN);                                // ff = gate * silu(up)
        matmul_vec(ff, lw.w2.data(), down, (int)FFN, D);             // down

        // (i) x = x_mid + down  (residual)
        for (uint32_t d = 0; d < D; ++d)
            x[d] = x_mid[d] + down[d];
    }

    // 3. Final RMSNorm
    rms_norm(x, final_rms_.data(), x_final_.data(), D, eps);

    // 4. Logits via weight-tied output projection  logits = x_final @ embedᵗ
    logits.resize(V);
    for (uint32_t v = 0; v < V; ++v) {
        const float* row = embed_.data() + (size_t)v * D;
        float dot = 0.0f;
        for (uint32_t d = 0; d < D; ++d)
            dot += x_final_[d] * row[d];
        logits[v] = dot;
    }
}
