#pragma once
// =============================================================================
// LoRA (Low-Rank Adaptation) — Mobile fine-tuning
// r=16, alpha=32, targets: Wq, Wv only
// Per-persona adapters: ~8 MB each
// =============================================================================

#include "model_config.h"
#include "../bpe/bpe_tokenizer.h"

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

// Forward declaration — full Transformer class lives in train_cuda.cu
class Transformer {
public:
    void reset_kv_cache();
    void forward_one_token(std::vector<float>& logits, int token_id, int pos);
};

// ---------------------------------------------------------------------------
// LoRA hyperparameters
// ---------------------------------------------------------------------------
struct LoRAConfig {
    int rank    = 16;     // LoRA rank r
    float alpha = 32.0f;  // LoRA alpha (scaling = alpha/rank = 2.0)
    float dropout = 0.05f;

    // Which modules to adapt
    bool adapt_wq = true;
    bool adapt_wv = true;
    bool adapt_wk = false;
    bool adapt_wo = false;

    float scaling() const { return alpha / (float)rank; }
};

// ---------------------------------------------------------------------------
// LoRA weight pair: A [in, r] and B [r, out]
// ΔW = B @ A * scaling
// A initialized to N(0, 0.02), B initialized to 0
// ---------------------------------------------------------------------------
struct LoRAWeight {
    std::vector<float> A;  // [in_features, rank]
    std::vector<float> B;  // [rank, out_features]
    int in_features;
    int out_features;
    int rank;
    float scaling;
    std::string name;

    LoRAWeight(int in, int out, int r, float scale, const std::string& n);

    // Compute delta contribution: out_delta += (x @ A @ B) * scaling
    // x: [in_features] → out_delta: [out_features]
    void apply(float* out_delta, const float* x) const;

    // Zero B matrix (standard init)
    void zero_B();

    // Initialize A with normal(0, 0.02)
    void normal_A(float std = 0.02f);

    int size_bytes() const {
        return (int)((A.size() + B.size()) * sizeof(float));
    }
};

// ---------------------------------------------------------------------------
// LoRA adapter: collection of LoRA weights for all layers
// ---------------------------------------------------------------------------
class LoRAAdapter {
public:
    std::string persona_name;
    LoRAConfig  config;

    // Per-layer LoRA weights: layer_idx → {wq_lora, wv_lora}
    struct LayerLoRA {
        std::unique_ptr<LoRAWeight> wq;  // nullptr if not adapted
        std::unique_ptr<LoRAWeight> wv;
        std::unique_ptr<LoRAWeight> wk;
        std::unique_ptr<LoRAWeight> wo;
    };

    std::vector<LayerLoRA> layers;

    // -------------------------------------------------------------------------
    // Initialize adapter for a given model config
    // -------------------------------------------------------------------------
    explicit LoRAAdapter(const ModelConfig& model_cfg,
                          const LoRAConfig&  lora_cfg,
                          const std::string& name);

    // -------------------------------------------------------------------------
    // Apply LoRA to Q/V projections of a specific layer
    // This modifies the output of linear projections in-place
    // -------------------------------------------------------------------------
    void apply_to_layer(int layer_idx,
                         float* q_proj,   // [q_dim]
                         float* v_proj,   // [kv_dim]
                         const float* x,  // input [d_model]
                         const ModelConfig& cfg) const;

    // -------------------------------------------------------------------------
    // Save/Load adapter
    // -------------------------------------------------------------------------
    void save(const std::string& path) const;
    void load(const std::string& path);

    // -------------------------------------------------------------------------
    // Merge LoRA weights into base model permanently (for export)
    // -------------------------------------------------------------------------
    void merge_into(Transformer& model) const;

    // -------------------------------------------------------------------------
    // Stats
    // -------------------------------------------------------------------------
    size_t total_params() const;
    float  size_mb() const;
    void   print_info() const;

    // -------------------------------------------------------------------------
    // Get all trainable params (for optimizer)
    // -------------------------------------------------------------------------
    std::vector<float*> trainable_params();
    std::vector<int>    trainable_param_sizes();
};

// ---------------------------------------------------------------------------
// LoRA fine-tuning trainer
// ---------------------------------------------------------------------------
struct LoRATrainConfig {
    float lr       = 1e-4f;
    float lr_min   = 1e-5f;
    int   epochs   = 1;
    int   batch    = 4;
    float clip     = 1.0f;
    int   save_every = 500;
    std::string output_path;
};

class LoRATrainer {
public:
    LoRATrainer(Transformer& base_model,
                 LoRAAdapter&  adapter,
                 BPEEncoder&   enc,
                 const LoRATrainConfig& cfg);

    void train(const std::string& data_dir);

    // On-device training (mobile): smaller batches, limited steps
    void train_on_device(const std::vector<std::string>& samples,
                          int max_steps = 100);

private:
    Transformer&      base_model_;
    LoRAAdapter&      adapter_;
    BPEEncoder&       enc_;
    LoRATrainConfig   cfg_;
};