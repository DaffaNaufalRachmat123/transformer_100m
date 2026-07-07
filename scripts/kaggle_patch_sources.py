#!/usr/bin/env python3
"""
Kaggle patch script -- targeted search & replace to apply 4 fixes.
Copy-paste this entire cell into a Kaggle notebook.
"""
import os

BASE = os.environ.get("TARGET_DIR", "/kaggle/working/transformer_100m_cuda")

# ========================================================================
# PATCH 1/3: transformer/train_cuda.cu
# ========================================================================
path = os.path.join(BASE, "transformer/train_cuda.cu")
with open(path, "r") as f:
    c = f.read()

# 1a: Add grad norm globals
c = c.replace(
    "static int g_current_phase = 0;",
    "static int g_current_phase = 0;\nstatic float g_last_grad_norm = 0.0f;\nstatic bool  g_last_clipped   = false;"
)

# 1b: Update save_checkpoint signature
c = c.replace(
    "void save_checkpoint(const std::string& dir,\n"
    "                     const std::vector<cuda::CudaTensor>& params,\n"
    "                     const std::vector<cuda::CudaTensor>& optimizer_m,",
    "void save_checkpoint(const std::string& dir,\n"
    "                     const std::vector<cuda::CudaTensor>& params,\n"
    "                     const std::vector<cuda::CudaTensor>& master_params,\n"
    "                     const std::vector<cuda::CudaTensor>& optimizer_m,"
)

# 1c: Add temp model path + metadata
c = c.replace(
    '    std::string model_path = dir + "/ckpt_pretrain" + suffix + ".safetensors";\n'
    '    SafeTensorsWriter w;',
    '    std::string model_path = dir + "/ckpt_pretrain" + suffix + ".safetensors";\n'
    '    std::string tmp_model  = dir + "/ckpt_pretrain" + suffix + ".tmp";\n'
    '    SafeTensorsWriter w;\n'
    '    w.set_metadata("step", std::to_string(step));\n'
    '    w.set_metadata("tokens", std::to_string(tokens_done));'
)

# 1d: Atomic write model
c = c.replace(
    "    w.finalize(model_path);",
    "    w.finalize(tmp_model);\n    fs::rename(tmp_model, model_path);"
)

# 1e: Atomic write optimizer + add master params
c = c.replace(
    '        std::string optim_path = dir + "/ckpt_optim" + suffix + ".safetensors";\n'
    '        SafeTensorsWriter ow;',
    '        std::string optim_path = dir + "/ckpt_optim" + suffix + ".safetensors";\n'
    '        std::string tmp_optim  = dir + "/ckpt_optim" + suffix + ".tmp";\n'
    '        SafeTensorsWriter ow;'
)

c = c.replace(
    '        save_f32_tensor(optimizer_m[0], "m.embed");\n'
    '        save_f32_tensor(optimizer_v[0], "v.embed");',
    '        save_f32_tensor(optimizer_m[0], "m.embed");\n'
    '        save_f32_tensor(optimizer_v[0], "v.embed");\n'
    '        save_f32_tensor(master_params[0], "master.embed");'
)

c = c.replace(
    '        // Layer optimizer states',
    '        // Layer optimizer states + master weights'
)

# Add master. prefix to layer save
c = c.replace(
    '                save_f32_tensor(optimizer_m[base + i], "m." + prefix);\n'
    '                save_f32_tensor(optimizer_v[base + i], "v." + prefix);\n'
    '            }\n'
    '        }\n'
    '\n'
    '        // Final RMS optimizer states',
    '                save_f32_tensor(optimizer_m[base + i], "m." + prefix);\n'
    '                save_f32_tensor(optimizer_v[base + i], "v." + prefix);\n'
    '                save_f32_tensor(master_params[base + i], "master." + prefix);\n'
    '            }\n'
    '        }\n'
    '\n'
    '        // Final RMS optimizer states + master'
)

# Add master final_rms + atomic write
c = c.replace(
    '        save_f32_tensor(optimizer_m[frms_idx], "m.final_rms");\n'
    '        save_f32_tensor(optimizer_v[frms_idx], "v.final_rms");\n'
    '\n'
    '        ow.finalize(optim_path);',
    '        save_f32_tensor(optimizer_m[frms_idx], "m.final_rms");\n'
    '        save_f32_tensor(optimizer_v[frms_idx], "v.final_rms");\n'
    '        save_f32_tensor(master_params[frms_idx], "master.final_rms");\n'
    '\n'
    '        ow.finalize(tmp_optim);\n'
    '        fs::rename(tmp_optim, optim_path);'
)

# 1f: Update resume comment
c = c.replace(
    '            // Restore FP32 master weights\n'
    '            for (size_t i = 0; i < params.size(); i++) {\n'
    '                master_params[i] = cuda::astype(params[i], cuda::DType::F32);\n'
    '            }',
    '            // Load master weights (FP32 from checkpoint, fallback to FP16 cast)\n'
    '            // Master params will be loaded below together with optimizer states;\n'
    '            // fallback: cast FP16 params to FP32 if no master tensors found.\n'
    '            for (size_t i = 0; i < params.size(); i++) {\n'
    '                master_params[i] = cuda::astype(params[i], cuda::DType::F32);\n'
    '            }'
)

# 1g: Resume - load master params (line-based to avoid Unicode arrow issue)
lines = c.split('\n')
for i, line in enumerate(lines):
    if 'Also load master weights from model (FP32 precision preserved)' in line:
        lines[i] = '                // Load master weights (FP32 precision preserved from training)'
        lines[i+1] = '                bool has_master = opt_loader.has_tensor("master.embed");'
        new_block = [
            '                if (has_master) {',
            '                    load_f32_tensor(master_params[0], "master.embed");',
            '                    for (int l = 0; l < N_LAYERS; l++) {',
            '                        int base = 1 + l * 9;',
            '                        for (int i = 0; i < 9; i++) {',
            '                            std::string prefix = "layer." + std::to_string(l) + "." + layer_names[i];',
            '                            load_f32_tensor(master_params[base + i], "master." + prefix);',
            '                        }',
            '                    }',
            '                    load_f32_tensor(master_params[frms_idx], "master.final_rms");',
            '                    printf("[RESUME] Master params restored (FP32 precision preserved)\\n");',
            '                } else {',
            '                    printf("[RESUME] Master params from FP16 cast (legacy checkpoint)\\n");',
            '                }',
        ]
        del lines[i+2]  # Remove old "but if optimizer file exists..." line
        for j, nl in enumerate(new_block):
            lines.insert(i+2+j, nl)
        break
c = '\n'.join(lines)

# 1h: Grad clip logging
c = c.replace(
    '            float global_norm = sqrtf(h_norm_sq);\n'
    '            if (global_norm > g_grad_clip) {\n'
    '                float clip_coeff = g_grad_clip / (global_norm + 1e-6f);',
    '            float global_norm = sqrtf(h_norm_sq);\n'
    '            g_last_grad_norm = global_norm;\n'
    '            g_last_clipped   = false;\n'
    '            if (global_norm > g_grad_clip) {\n'
    '                g_last_clipped = true;\n'
    '                float clip_coeff = g_grad_clip / (global_norm + 1e-6f);'
)

# 1i: Log line - add grad norm
c = c.replace(
    '                   "loss=%.4f lr=%.2e %s tok/s %.1fs/step elapsed=%s ETA %s | "',
    '                   "loss=%.4f lr=%.2e grad=%.2f%s %s tok/s %.1fs/step elapsed=%s ETA %s | "'
)
c = c.replace(
    '                   avg_loss, lr, tps, s_per_step,',
    '                   avg_loss, lr, g_last_grad_norm,\n'
    '                   g_last_clipped ? " CLIP" : "",\n'
    '                   tps, s_per_step,'
)

# 1j: Update save_checkpoint call sites
c = c.replace(
    '            save_checkpoint(g_checkpoint_dir, params, adamw_m, adamw_v,\n'
    '                           g_tokens_processed, step);',
    '            save_checkpoint(g_checkpoint_dir, params, master_params, adamw_m, adamw_v,\n'
    '                           g_tokens_processed, step);'
)
c = c.replace(
    '    save_checkpoint(g_checkpoint_dir, params, adamw_m, adamw_v,\n'
    '                   g_tokens_processed, step, "_final");',
    '    save_checkpoint(g_checkpoint_dir, params, master_params, adamw_m, adamw_v,\n'
    '                   g_tokens_processed, step, "_final");'
)

with open(path, "w") as f:
    f.write(c)
print("Patched: " + path)

# ========================================================================
# PATCH 2/3: transformer/pretrain_dataset.cpp
# ========================================================================
path = os.path.join(BASE, "transformer/pretrain_dataset.cpp")
with open(path, "r") as f:
    c = f.read()

old_refill = """    auto refill_active = [&]() {
        // Try to top up active_shards_ to kMixShards from the ready queue.
        while (active_shards_.size() < kMixShards) {
            std::shared_ptr<Shard> shard;
            {
                std::unique_lock<std::mutex> lock(ring_mtx_);
                if (ready_shards_.empty()) {
                    // If we already have at least one active shard, don't block.
                    if (!active_shards_.empty()) break;
                    // Otherwise wait for at least one shard.
                    ring_cv_not_empty_.wait(lock, [this]{
                        return !ready_shards_.empty() || stop_flag_.load();
                    });
                    if (ready_shards_.empty()) return;  // stopping
                }
                shard = ready_shards_.front();
                ready_shards_.pop_front();
                ring_cv_not_full_.notify_one();
            }
            if (shard && shard->n_samples > 0 && shard->seq_len == cfg_.seq_len)
                active_shards_.push_back({shard, 0});
            // else: stale shard (old seq_len) - discard silently
        }
    };"""

new_refill = """    auto refill_active = [&]() {
        // Try to top up active_shards_ to kMixShards from the ready queue.
        // Prefer shards with a language not already represented in active
        // shards, so each batch contains maximally diverse languages.
        while (active_shards_.size() < kMixShards) {
            std::shared_ptr<Shard> shard;
            {
                std::unique_lock<std::mutex> lock(ring_mtx_);
                if (ready_shards_.empty()) {
                    // If we already have at least one active shard, don't block.
                    if (!active_shards_.empty()) break;
                    // Otherwise wait for at least one shard.
                    ring_cv_not_empty_.wait(lock, [this]{
                        return !ready_shards_.empty() || stop_flag_.load();
                    });
                    if (ready_shards_.empty()) return;  // stopping
                }

                // Prefer a shard whose language is NEW to active_shards_
                auto it = std::find_if(ready_shards_.begin(), ready_shards_.end(),
                    [this](const std::shared_ptr<Shard>& s) {
                        return std::none_of(active_shards_.begin(), active_shards_.end(),
                            [&s](const ActiveShard& as) { return as.shard->lang == s->lang; });
                    });

                if (it == ready_shards_.end()) {
                    // No diverse shard available - fall back to first available
                    it = ready_shards_.begin();
                }

                shard = *it;
                ready_shards_.erase(it);
                ring_cv_not_full_.notify_one();
            }
            if (shard && shard->n_samples > 0 && shard->seq_len == cfg_.seq_len)
                active_shards_.push_back({shard, 0});
            // else: stale shard (old seq_len) - discard silently
        }
    };"""

c = c.replace(old_refill, new_refill)
with open(path, "w") as f:
    f.write(c)
print("Patched: " + path)

# ========================================================================
# PATCH 3/3: transformer/safetensors.h
# ========================================================================
path = os.path.join(BASE, "transformer/safetensors.h")
with open(path, "r") as f:
    c = f.read()

# 3a: Add set_metadata method
c = c.replace(
    '    void add_tensor(const std::string& name,\n'
    '                    STDtype dtype,\n'
    '                    std::vector<uint32_t> shape,\n'
    '                    const void* data) {\n'
    '        pending_.push_back({name, dtype, std::move(shape), data});\n'
    '    }',
    '    void add_tensor(const std::string& name,\n'
    '                    STDtype dtype,\n'
    '                    std::vector<uint32_t> shape,\n'
    '                    const void* data) {\n'
    '        pending_.push_back({name, dtype, std::move(shape), data});\n'
    '    }\n'
    '\n'
    '    void set_metadata(const std::string& key, const std::string& value) {\n'
    '        metadata_[key] = value;\n'
    '    }'
)

# 3b: Modify finalize to prepend metadata
c = c.replace(
    '        // Build header JSON\n'
    '        uint64_t offset = 0;\n'
    '        std::string header = "{";\n'
    '        bool first = true;',
    '        // Build header JSON\n'
    '        uint64_t offset = 0;\n'
    '        std::string header = "{";\n'
    '\n'
    '        // Prepend metadata section if present\n'
    '        if (!metadata_.empty()) {\n'
    '            header += "\\"__metadata__\\":{";\n'
    '            bool first_meta = true;\n'
    '            for (const auto& kv : metadata_) {\n'
    '                if (!first_meta) header += ",";\n'
    '                first_meta = false;\n'
    '                header += detail::json_string(kv.first) + ":" + detail::json_string(kv.second);\n'
    '            }\n'
    '            header += "},";\n'
    '        }\n'
    '\n'
    '        bool first = true;'
)

# 3c: Add metadata_ member
c = c.replace(
    'private:\n'
    '    std::vector<PendingTensor> pending_;\n'
    '};',
    'private:\n'
    '    std::vector<PendingTensor> pending_;\n'
    '    std::map<std::string, std::string> metadata_;\n'
    '};'
)

with open(path, "w") as f:
    f.write(c)
print("Patched: " + path)

print("\nAll 3 source files patched successfully.")
print("Recompile:  !cd " + BASE + "/build && make -j$(nproc)")
