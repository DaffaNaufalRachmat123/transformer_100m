#pragma once

#include "model_config.h"

#include <cstdint>
#include <string>
#include <vector>

class Transformer {
public:
    struct GenerateConfig {
        int      max_new_tokens = 200;
        float    temperature    = 0.8f;
        float    top_p          = 0.9f;
        int      top_k          = 50;
        uint32_t eos_id         = 4;
    };

    explicit Transformer(const ModelConfig& cfg);

    void load_safetensors(const std::string& path);
    void forward_one_token(std::vector<float>& logits, int token_id, int pos);
    void reset_kv_cache();

private:
    ModelConfig cfg_;

    // ── Weights (host F32, auto-converted from F16/BF16 at load time) ──

    std::vector<float> embed_; // [V, D] row-major

    struct LayerWeights {
        std::vector<float> wq, wk, wv, wo; // [D,D] [D,KVD] [D,KVD] [D,D]
        std::vector<float> w1, w2, w3;     // [D,FFN] [FFN,D] [D,FFN]
        std::vector<float> rms_attn;       // [D]
        std::vector<float> rms_ffn;        // [D]
    };
    std::vector<LayerWeights> layers_;

    std::vector<float> final_rms_; // [D]

    // ── RoPE precomputed tables ──

    std::vector<float> rope_cos_; // [max_seq_len, half_dim]
    std::vector<float> rope_sin_;
    void precompute_rope();

    // ── KV-cache ──

    struct LayerKVCache {
        std::vector<float> k; // [max_seq_len, n_kv_heads, d_head]
        std::vector<float> v;
    };
    std::vector<LayerKVCache> kv_cache_;

    // ── CPU helpers ──

    static void rms_norm(const float* x, const float* gamma,
                         float* out, int D, float eps);

    static void matmul_vec(const float* x, const float* W,
                           float* out, int D_in, int D_out);

    void apply_rope(float* vec, int n_heads, int d_head, int pos) const;

    static void swiglu(const float* gate, const float* up,
                       float* out, int N);

    // ── Temporary scratch buffers (allocated once, reused per call) ──

    // per-layer:  x[576], n1[576], q[576], k[192], v[192],
    //              attn[576], o[576], x_mid[576], n2[576],
    //              gate[2304], up[2304], ff[2304], down[576]
    // plus global: x_final[576]
    // biggest single: gate/up/ff @ 2304 floats each → ~9 KB per layer
    struct Scratch {
        std::vector<float> x;       // [D]
        std::vector<float> n1;      // [D]
        std::vector<float> q;       // [D]  = n_q_heads * d_head
        std::vector<float> k;       // [KVD]= n_kv_heads * d_head
        std::vector<float> v;       // [KVD]
        std::vector<float> scores;   // allocated to max_seq_len
        std::vector<float> attn;    // [D]
        std::vector<float> o;       // [D]
        std::vector<float> x_mid;   // [D]
        std::vector<float> n2;      // [D]
        std::vector<float> gate;    // [FFN]
        std::vector<float> up;      // [FFN]
        std::vector<float> ff;      // [FFN]
        std::vector<float> down;    // [D]
    };
    Scratch scratch_;

    std::vector<float> x_final_; // [D]
};
