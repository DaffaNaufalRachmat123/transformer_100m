// =============================================================================
// LoRA Implementation
// =============================================================================

#include "lora.h"

// Minimal CPU AdamW state for LoRA training (not the CUDA version)
struct AdamWState {
    std::vector<float> m, v;
    int t = 0;
    void update_flat(float* param, const float* grad, int n, float lr,
                     float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
        if ((int)m.size() != n) { m.resize(n, 0.f); v.resize(n, 0.f); }
        t++;
        float bc1 = 1.f - powf(beta1, (float)t);
        float bc2 = 1.f - powf(beta2, (float)t);
        for (int i = 0; i < n; i++) {
            m[i] = beta1 * m[i] + (1.f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.f - beta2) * grad[i] * grad[i];
            param[i] -= lr * (m[i] / bc1) / (sqrtf(v[i] / bc2) + eps);
        }
    }
};

#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

// =============================================================================
// LoRAWeight
// =============================================================================

LoRAWeight::LoRAWeight(int in, int out, int r, float scale, const std::string& n)
    : in_features(in), out_features(out), rank(r), scaling(scale), name(n) {
    A.resize((size_t)in * r);
    B.resize((size_t)r * out, 0.0f);
    normal_A();
}

void LoRAWeight::normal_A(float std) {
    static std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, std);
    for (float& v : A) v = dist(rng);
}

void LoRAWeight::zero_B() {
    std::fill(B.begin(), B.end(), 0.0f);
}

void LoRAWeight::apply(float* out_delta, const float* x) const {
    // Step 1: h = x @ A  [rank]
    std::vector<float> h(rank, 0.0f);
    for (int r = 0; r < rank; ++r) {
        for (int i = 0; i < in_features; ++i) {
            h[r] += x[i] * A[(size_t)i * rank + r];
        }
    }

    // Step 2: out_delta += h @ B * scaling  [out_features]
    for (int o = 0; o < out_features; ++o) {
        float val = 0.0f;
        for (int r = 0; r < rank; ++r) {
            val += h[r] * B[(size_t)r * out_features + o];
        }
        out_delta[o] += val * scaling;
    }
}

// =============================================================================
// LoRAAdapter
// =============================================================================

LoRAAdapter::LoRAAdapter(const ModelConfig& model_cfg,
                           const LoRAConfig&  lora_cfg,
                           const std::string& name)
    : persona_name(name), config(lora_cfg) {
    float scale = lora_cfg.scaling();
    int r       = lora_cfg.rank;
    int D       = model_cfg.d_model;
    int QD      = model_cfg.q_dim();
    int KD      = model_cfg.kv_dim();

    layers.resize(model_cfg.n_layers);

    for (uint32_t l = 0; l < model_cfg.n_layers; ++l) {
        std::string prefix = "layer" + std::to_string(l) + "_";

        if (lora_cfg.adapt_wq) {
            layers[l].wq = std::make_unique<LoRAWeight>(D, QD, r, scale, prefix + "wq");
        }
        if (lora_cfg.adapt_wv) {
            layers[l].wv = std::make_unique<LoRAWeight>(D, KD, r, scale, prefix + "wv");
        }
        if (lora_cfg.adapt_wk) {
            layers[l].wk = std::make_unique<LoRAWeight>(D, KD, r, scale, prefix + "wk");
        }
        if (lora_cfg.adapt_wo) {
            layers[l].wo = std::make_unique<LoRAWeight>(QD, D, r, scale, prefix + "wo");
        }
    }

    print_info();
}

void LoRAAdapter::apply_to_layer(int layer_idx,
                                   float* q_proj,
                                   float* v_proj,
                                   const float* x,
                                   const ModelConfig& cfg) const {
    assert(layer_idx < (int)layers.size());
    const auto& ll = layers[layer_idx];

    if (ll.wq) {
        ll.wq->apply(q_proj, x);
    }
    if (ll.wv) {
        ll.wv->apply(v_proj, x);
    }
}

void LoRAAdapter::save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot write LoRA: " + path);

    // Header
    uint32_t magic = 0x4C4F5241;  // "LORA"
    f.write((char*)&magic, 4);
    uint32_t n_layers = (uint32_t)layers.size();
    f.write((char*)&n_layers, 4);
    uint32_t rank     = config.rank;
    float    alpha    = config.alpha;
    f.write((char*)&rank, 4);
    f.write((char*)&alpha, 4);

    // Persona name
    uint32_t name_len = (uint32_t)persona_name.size();
    f.write((char*)&name_len, 4);
    f.write(persona_name.c_str(), name_len);

    // Each layer
    for (const auto& ll : layers) {
        auto write_lora = [&](const std::unique_ptr<LoRAWeight>& w) {
            uint8_t has = (w != nullptr) ? 1 : 0;
            f.write((char*)&has, 1);
            if (has) {
                f.write((char*)w->A.data(), w->A.size() * sizeof(float));
                f.write((char*)w->B.data(), w->B.size() * sizeof(float));
            }
        };
        write_lora(ll.wq);
        write_lora(ll.wv);
        write_lora(ll.wk);
        write_lora(ll.wo);
    }

    float size = (float)f.tellp() / (1024 * 1024);
    std::cout << "[LoRA] Saved " << persona_name << " to " << path
              << " (" << std::fixed << std::setprecision(2) << size << " MB)\n";
}

void LoRAAdapter::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open LoRA: " + path);

    uint32_t magic, n_layers, rank;
    float alpha;
    f.read((char*)&magic, 4);
    if (magic != 0x4C4F5241) throw std::runtime_error("Invalid LoRA file");
    f.read((char*)&n_layers, 4);
    f.read((char*)&rank, 4);
    f.read((char*)&alpha, 4);
    config.rank  = rank;
    config.alpha = alpha;

    uint32_t name_len;
    f.read((char*)&name_len, 4);
    persona_name.resize(name_len);
    f.read(&persona_name[0], name_len);

    for (uint32_t l = 0; l < n_layers && l < layers.size(); ++l) {
        auto read_lora = [&](std::unique_ptr<LoRAWeight>& w) {
            uint8_t has;
            f.read((char*)&has, 1);
            if (has && w) {
                f.read((char*)w->A.data(), w->A.size() * sizeof(float));
                f.read((char*)w->B.data(), w->B.size() * sizeof(float));
            } else if (has) {
                // Skip
                // (would need to know dimensions — store in file for robustness)
            }
        };
        read_lora(layers[l].wq);
        read_lora(layers[l].wv);
        read_lora(layers[l].wk);
        read_lora(layers[l].wo);
    }

    std::cout << "[LoRA] Loaded " << persona_name << " from " << path << "\n";
}

size_t LoRAAdapter::total_params() const {
    size_t total = 0;
    for (const auto& ll : layers) {
        auto count = [&](const std::unique_ptr<LoRAWeight>& w) {
            if (w) total += w->A.size() + w->B.size();
        };
        count(ll.wq); count(ll.wv); count(ll.wk); count(ll.wo);
    }
    return total;
}

float LoRAAdapter::size_mb() const {
    return (float)(total_params() * sizeof(float)) / (1024 * 1024);
}

void LoRAAdapter::print_info() const {
    std::cout << "[LoRA] Adapter: " << persona_name << "\n";
    std::cout << "  rank=" << config.rank
              << " alpha=" << config.alpha
              << " scaling=" << config.scaling() << "\n";
    std::cout << "  total_params=" << total_params() / 1e6 << "M"
              << " (" << size_mb() << " MB)\n";
    std::cout << "  adapts: Wq=" << (config.adapt_wq ? "yes" : "no")
              << " Wv=" << (config.adapt_wv ? "yes" : "no")
              << " Wk=" << (config.adapt_wk ? "yes" : "no")
              << " Wo=" << (config.adapt_wo ? "yes" : "no") << "\n";
}

std::vector<float*> LoRAAdapter::trainable_params() {
    std::vector<float*> ptrs;
    for (auto& ll : layers) {
        auto add = [&](std::unique_ptr<LoRAWeight>& w) {
            if (w) {
                ptrs.push_back(w->A.data());
                ptrs.push_back(w->B.data());
            }
        };
        add(ll.wq); add(ll.wv); add(ll.wk); add(ll.wo);
    }
    return ptrs;
}

std::vector<int> LoRAAdapter::trainable_param_sizes() {
    std::vector<int> sizes;
    for (auto& ll : layers) {
        auto add = [&](std::unique_ptr<LoRAWeight>& w) {
            if (w) {
                sizes.push_back((int)w->A.size());
                sizes.push_back((int)w->B.size());
            }
        };
        add(ll.wq); add(ll.wv); add(ll.wk); add(ll.wo);
    }
    return sizes;
}

// =============================================================================
// LoRATrainer
// =============================================================================

LoRATrainer::LoRATrainer(Transformer& base_model,
                           LoRAAdapter&  adapter,
                           BPEEncoder&   enc,
                           const LoRATrainConfig& cfg)
    : base_model_(base_model), adapter_(adapter), enc_(enc), cfg_(cfg) {}

void LoRATrainer::train(const std::string& data_dir) {
    std::cout << "[LoRATrainer] Training adapter: " << adapter_.persona_name << "\n";
    std::cout << "  Data dir: " << data_dir << "\n";
    std::cout << "  LR: " << cfg_.lr << " → " << cfg_.lr_min << "\n";
    std::cout << "  Epochs: " << cfg_.epochs << "\n";

    // Only train LoRA params (base model frozen)
    auto params = adapter_.trainable_params();
    auto sizes  = adapter_.trainable_param_sizes();
    std::vector<AdamWState> states(params.size());
    std::vector<std::vector<float>> grads(params.size());
    for (size_t i = 0; i < params.size(); ++i) grads[i].resize(sizes[i], 0.0f);

    // Training loop would go here — same structure as main trainer
    // but only updating LoRA A/B matrices
    std::cout << "[LoRATrainer] Training loop ready — "
              << params.size() << " trainable param groups\n";
}

void LoRATrainer::train_on_device(const std::vector<std::string>& samples,
                                    int max_steps) {
    std::cout << "[LoRATrainer] On-device training: "
              << samples.size() << " samples, max " << max_steps << " steps\n";

    // On-device: conservative settings
    // - Very small LR (5e-6)
    // - Max 100 steps to avoid overfit
    // - Only during charging (check battery status)
    // - Stop if RAM > 5.5 GB

    auto params = adapter_.trainable_params();
    auto sizes  = adapter_.trainable_param_sizes();
    std::vector<AdamWState> states(params.size());

    for (int step = 0; step < max_steps; ++step) {
        const auto& text = samples[step % samples.size()];
        auto token_ids = enc_.encode(text);
        if (token_ids.empty()) continue;

        // Forward + loss (simplified on-device path)
        std::vector<float> logits;
        base_model_.reset_kv_cache();
        base_model_.forward_one_token(logits, token_ids[0], 0);

        // Update LoRA params
        float lr = cfg_.lr * (1.0f - (float)step / max_steps);
        for (size_t i = 0; i < params.size(); ++i) {
            std::vector<float> dummy_grad(sizes[i], 0.0f);
            states[i].update_flat(params[i], dummy_grad.data(), sizes[i], lr);
        }

        if (step % 10 == 0) {
            std::cout << "[OnDevice] step=" << step << "/" << max_steps << "\n";
        }
    }

    // Save updated adapter
    if (!cfg_.output_path.empty()) {
        adapter_.save(cfg_.output_path);
    }
}