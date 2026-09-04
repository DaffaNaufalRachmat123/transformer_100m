#pragma once
// =============================================================================
// Transformer 100M — Model Configuration
// Architecture: GQA + RoPE + RMSNorm + SwiGLU
// Target: NVIDIA T4 (CUDA FP16) training, INT4 mobile deployment
// =============================================================================

#include <cstdint>
#include <string>

struct ModelConfig {
    // -------------------------------------------------------------------------
    // Architecture — 492M EN/JA ("S3"). Override at runtime with --arch_config.
    // Compute-optimal for a ~4.4e19 FLOP budget (~300h on a rented RTX 4090),
    // which buys ~11B tokens at ~23 tokens/param. Sizing larger without also
    // raising the token budget makes the model worse, not better.
    // -------------------------------------------------------------------------
    uint32_t vocab_size    = 49152;  // BPE (EN/JA/code; ID+KO merges dropped)
    uint32_t d_model       = 1280;   // embedding dimension
    uint32_t n_layers      = 24;     // transformer layers
    uint32_t n_q_heads     = 20;     // query heads
    uint32_t n_kv_heads    = 5;      // GQA: key/value heads (4x KV-cache saving)
    uint32_t d_head        = 64;     // d_model / n_q_heads = 1280/20
    uint32_t ffn_dim       = 3584;   // SwiGLU hidden dim = 2.8 * d_model
    uint32_t max_seq_len   = 2048;   // max context window

    // -------------------------------------------------------------------------
    // Derived dimensions (auto-computed)
    // -------------------------------------------------------------------------
    uint32_t kv_dim() const { return n_kv_heads * d_head; }   // 5*64 = 320
    uint32_t q_dim()  const { return n_q_heads  * d_head; }   // 20*64 = 1280

    // -------------------------------------------------------------------------
    // RoPE configuration
    // -------------------------------------------------------------------------
    float rope_theta       = 500000.0f;  // high theta for long context
    bool  rope_scale       = false;

    // -------------------------------------------------------------------------
    // Training hyperparameters
    // -------------------------------------------------------------------------
    float weight_decay     = 0.1f;
    float grad_clip        = 1.0f;
    bool  weight_tying     = true;  // share input/output embedding

    // -------------------------------------------------------------------------
    // Special token IDs (must match BPE vocab)
    // -------------------------------------------------------------------------
    uint32_t pad_token_id  = 15;    // <|pad|>
    uint32_t eos_token_id  = 4;     // <|end|>
    uint32_t bos_token_id  = 0;     // <|endoftext|>

    // -------------------------------------------------------------------------
    // Serialization
    // -------------------------------------------------------------------------
    void print() const;
    void save(const std::string& path) const;
    static ModelConfig load(const std::string& path);
    static ModelConfig default_100m();

    // -------------------------------------------------------------------------
    // Parameter count estimation
    // -------------------------------------------------------------------------
    uint64_t param_count() const {
        uint64_t embed    = (uint64_t)vocab_size * d_model;
        uint64_t wq       = (uint64_t)d_model * q_dim();
        uint64_t wk       = (uint64_t)d_model * kv_dim();
        uint64_t wv       = (uint64_t)d_model * kv_dim();
        uint64_t wo       = (uint64_t)q_dim()  * d_model;
        uint64_t w1       = (uint64_t)d_model  * ffn_dim;  // gate
        uint64_t w2       = (uint64_t)ffn_dim  * d_model;  // down
        uint64_t w3       = (uint64_t)d_model  * ffn_dim;  // up
        uint64_t rms      = d_model * 2;
        uint64_t per_layer = wq + wk + wv + wo + w1 + w2 + w3 + rms;
        uint64_t total    = embed + per_layer * n_layers + d_model;
        return total; // weight tying: embed counts once
    }
};