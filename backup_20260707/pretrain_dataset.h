#pragma once
// =============================================================================
// Pretraining Dataset — Streaming Corpus Loader
// Handles multi-language corpus loading for Stage 1 pretraining.
//
// Design goals:
//   - Stream from disk; do NOT load entire corpus into RAM
//   - Shuffle at file level + within-shard level (bounded memory)
//   - Support curriculum: sequence lengths 256 → 512 → 2048
//   - Thread-safe: one prefetch thread fills a ring buffer
// =============================================================================

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <deque>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Language tag — used to sample proportions correctly
// ---------------------------------------------------------------------------
enum class Language : uint8_t {
    Indonesian = 0,
    English,
    Japanese,
    Korean,
    Mixed,
    Unknown
};

inline Language lang_from_dirname(const std::string& name) {
    if (name == "indonesian") return Language::Indonesian;
    if (name == "english")    return Language::English;
    if (name == "japanese")   return Language::Japanese;
    if (name == "korean")     return Language::Korean;
    if (name == "mixed")      return Language::Mixed;
    return Language::Unknown;
}

// Per-phase language sampling weights
// 4-language focus (ID/EN/JA/KO) + Mixed. Code/Math removed.
// JA/KO weighted higher early for character coverage; EN dominates later.
inline float lang_sample_weight(Language l, int phase = 0) {
    if (phase == 0) {
        switch (l) {
            case Language::Indonesian: return 0.20f;
            case Language::English:    return 0.22f;
            case Language::Japanese:   return 0.26f;
            case Language::Korean:     return 0.22f;
            case Language::Mixed:      return 0.10f;
            default:                   return 0.00f;
        }
    }
    if (phase == 1) {
        switch (l) {
            case Language::Indonesian: return 0.23f;
            case Language::English:    return 0.26f;
            case Language::Japanese:   return 0.24f;
            case Language::Korean:     return 0.19f;
            case Language::Mixed:      return 0.08f;
            default:                   return 0.00f;
        }
    }
    if (phase == 2) {
        switch (l) {
            case Language::Indonesian: return 0.26f;
            case Language::English:    return 0.29f;
            case Language::Japanese:   return 0.23f;
            case Language::Korean:     return 0.17f;
            case Language::Mixed:      return 0.05f;
            default:                   return 0.00f;
        }
    }
    switch (l) {
        case Language::Indonesian: return 0.28f;
        case Language::English:    return 0.31f;
        case Language::Japanese:   return 0.20f;
        case Language::Korean:     return 0.16f;
        case Language::Mixed:      return 0.05f;
        default:                   return 0.00f;
    }
}

// ---------------------------------------------------------------------------
// CorpusFile descriptor
// ---------------------------------------------------------------------------
struct CorpusFile {
    std::string path;
    Language    lang;
    float       weight;
    uint64_t    byte_size;
};

// ---------------------------------------------------------------------------
// Curriculum stage
// ---------------------------------------------------------------------------
struct CurriculumStage {
    uint32_t seq_len;          // tokens per sample
    uint64_t target_tokens;    // tokens to train on at this stage
    std::string name;
};

// ---------------------------------------------------------------------------
// PretrainDatasetConfig
// ---------------------------------------------------------------------------
struct PretrainDatasetConfig {
    std::string corpus_root;                  // e.g. "./datasets"
    uint32_t    seq_len         = 256;        // current stage seq length
    uint32_t    batch_size      = 1;
    uint32_t    prefetch_shards = 4;          // shards in ring buffer
    uint32_t    shard_tokens    = 1 << 20;    // 1M tokens per shard
    uint64_t    seed            = 42;
    bool        weighted_sampling = true;     // follow lang proportions

    // Curriculum schedule — 2B plan (800M + 600M + 300M + 300M)
    // Balanced for 100M model on T4 x2 Kaggle budget
    std::vector<CurriculumStage> curriculum = {
        {256,   800'000'000ULL, "Phase1_basic"},
        {512,   600'000'000ULL, "Phase2_medium"},
        {1024,  300'000'000ULL, "Phase25_transition"},
        {2048,  300'000'000ULL, "Phase3_long"},
    };
};

// ---------------------------------------------------------------------------
// Shard: a fixed-size vector of token sequences ready for training
// ---------------------------------------------------------------------------
struct Shard {
    // tokens[i * seq_len .. (i+1)*seq_len - 1] is one training sequence
    std::vector<uint32_t> tokens;
    uint32_t seq_len   = 0;
    uint32_t n_samples = 0;
    Language lang = Language::Unknown;
};

// ---------------------------------------------------------------------------
// PretrainDataset
// ---------------------------------------------------------------------------
class PretrainDataset {
public:
    explicit PretrainDataset(const PretrainDatasetConfig& cfg);
    ~PretrainDataset();

    // Must be called after construction. Discovers corpus files.
    bool init();

    // Advance curriculum to next stage (call when stage token budget exhausted)
    bool advance_curriculum();

    // Get current curriculum stage
    const CurriculumStage& current_stage() const { return cfg_.curriculum[stage_idx_]; }
    uint32_t current_seq_len() const { return cfg_.seq_len; }

    // Statistics
    uint64_t tokens_consumed() const { return tokens_consumed_.load(); }
    size_t   num_corpus_files() const { return files_.size(); }

    // Language of the last sampled batch
    Language current_lang() const { return last_batch_lang_; }

    // Blocking: fill batch with the next n_samples sequences.
    // tokens_out must have space for n_samples * seq_len tokens.
    // Returns false at end-of-corpus OR if interrupt() was called.
    bool next_batch(uint32_t* tokens_out, uint32_t n_samples);

    // Unblock any pending next_batch() call immediately (call from signal handler).
    void interrupt() {
        stop_flag_.store(true);
        ring_cv_not_empty_.notify_all();
        ring_cv_not_full_.notify_all();
    }

    // Set stage index directly (for resume from checkpoint)
    void set_stage_idx(uint32_t stage) {
        stage_idx_ = stage;
        cfg_.seq_len = cfg_.curriculum[stage].seq_len;
        update_lang_weights();
        rebuild_distribution();
    }

    // Print corpus summary
    void print_info(std::ostream& os = std::cout) const;

    // Set external tokenizer callback (call this before init)
    // signature: void(const std::string& text, std::vector<uint32_t>& out_ids)
    using TokenizeFn = std::function<void(const std::string&, std::vector<uint32_t>&)>;
    void set_tokenize_fn(TokenizeFn fn) { tokenize_fn_ = std::move(fn); }

private:
    PretrainDatasetConfig cfg_;
    std::vector<CorpusFile> files_;
    uint32_t stage_idx_ = 0;

    // Weighted random file sampler
    std::mt19937_64 rng_;
    std::vector<float> file_weights_;
    std::discrete_distribution<size_t> file_dist_;

    // Prefetch ring buffer
    std::deque<std::shared_ptr<Shard>> ready_shards_;
    std::mutex ring_mtx_;
    std::condition_variable ring_cv_not_empty_;
    std::condition_variable ring_cv_not_full_;
    std::atomic<bool> stop_flag_{false};
    std::thread prefetch_thread_;

    // Leftover shard from previous next_batch() call to avoid waste
    std::shared_ptr<Shard> leftover_shard_;
    uint32_t leftover_pos_ = 0;

    // Active shards for cross-language batch mixing. next_batch() round-robins
    // one sample from each active shard so every batch contains a MIX of
    // languages/domains (prevents single-language batches that cause loss to
    // oscillate on shard boundaries).
    struct ActiveShard {
        std::shared_ptr<Shard> shard;
        uint32_t pos = 0;
    };
    std::vector<ActiveShard> active_shards_;
    static constexpr uint32_t kMixShards = 4;  // languages mixed per batch

    std::atomic<uint64_t> tokens_consumed_{0};

    Language last_batch_lang_ = Language::Unknown;

    TokenizeFn tokenize_fn_;

    // Internal: discover and register corpus files
    void discover_files();

    // Internal: prefetch loop (runs on background thread)
    void prefetch_loop();

    // Internal: load one shard from a randomly selected file
    std::shared_ptr<Shard> load_shard();

    // Internal: tokenize a line of text
    void tokenize_line(const std::string& text, std::vector<uint32_t>& ids);

    // Mutex protecting distribution and weight state (read by prefetch thread)
    std::mutex dist_mtx_;

    // Rebuild discrete_distribution from file weights
    void rebuild_distribution();

    // Refresh per-file language weights for current stage_idx_
    void update_lang_weights();
};

// ---------------------------------------------------------------------------
// Token stream packer: packs variable-length sequences into fixed-length
// training blocks with BOS/EOS boundaries respected.
// ---------------------------------------------------------------------------
class TokenPacker {
public:
    explicit TokenPacker(uint32_t seq_len, uint32_t eos_id = 0)
        : seq_len_(seq_len), eos_id_(eos_id) {}

    // Feed token ids from one document.
    void feed(const std::vector<uint32_t>& ids);

    // Returns true if a complete packed sequence is available.
    bool has_sequence() const { return buffer_.size() >= seq_len_; }

    // Pop one packed sequence into dst[0..seq_len-1]. Returns true on success.
    bool pop_sequence(uint32_t* dst);

    void reset() { buffer_.clear(); }

    uint64_t sequences_produced() const { return n_seqs_; }

private:
    uint32_t seq_len_;
    uint32_t eos_id_;
    std::deque<uint32_t> buffer_;
    uint64_t n_seqs_ = 0;
};