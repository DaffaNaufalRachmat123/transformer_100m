#pragma once
// =============================================================================
// Runtime architecture selection.
//
// The trainers used to hardcode D_MODEL/N_LAYERS/... as `constexpr`, so every
// model-size experiment required editing two .cu files and a full rebuild.
// Those constants are only ever used as runtime values (loop bounds, tensor
// shapes, memcpy sizes) — never as template arguments or array extents — so
// they can safely become mutable globals fed from a JSON config.
// =============================================================================

#include <cstdio>
#include <exception>
#include <string>

#include "transformer/model_config.h"

namespace arch {

// Load `path` into `cfg` and reject configs the kernels cannot honour.
inline bool load(const std::string& path, ModelConfig& cfg) {
    try {
        cfg = ModelConfig::load(path);
    } catch (const std::exception& e) {
        fprintf(stderr, "[ARCH] Cannot load %s: %s\n", path.c_str(), e.what());
        return false;
    }
    if (cfg.d_model == 0 || cfg.n_layers == 0 || cfg.vocab_size == 0) {
        fprintf(stderr, "[ARCH] d_model/n_layers/vocab_size must be non-zero\n");
        return false;
    }
    if (cfg.n_q_heads * cfg.d_head != cfg.d_model) {
        fprintf(stderr, "[ARCH] n_q_heads(%u) * d_head(%u) != d_model(%u)\n",
                cfg.n_q_heads, cfg.d_head, cfg.d_model);
        return false;
    }
    if (cfg.n_kv_heads == 0 || cfg.n_q_heads % cfg.n_kv_heads != 0) {
        fprintf(stderr, "[ARCH] n_q_heads(%u) must be divisible by n_kv_heads(%u)\n",
                cfg.n_q_heads, cfg.n_kv_heads);
        return false;
    }
    if (cfg.d_head % 2 != 0) {
        fprintf(stderr, "[ARCH] d_head(%u) must be even (RoPE rotates pairs)\n",
                cfg.d_head);
        return false;
    }
    printf("[ARCH] %s -> D=%u L=%u QH=%u KVH=%u DH=%u FFN=%u V=%u | %.1fM params\n",
           path.c_str(), cfg.d_model, cfg.n_layers, cfg.n_q_heads, cfg.n_kv_heads,
           cfg.d_head, cfg.ffn_dim, cfg.vocab_size, cfg.param_count() / 1e6);
    return true;
}

}  // namespace arch
