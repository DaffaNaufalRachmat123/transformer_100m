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
    Mixed,
    Unknown
};

inline Language lang_from_dirname(const std::string& name) {
    if (name == "indonesian") return Language::Indonesian;
    if (name == "english")    return Language::English;
    if (name == "japanese")   return Language::Japanese;
    if (name == "mixed")      return Language::Mixed;
    return Language::Unknown;
}

// Per-phase language sampling weights.
// Two supported corpora layouts:
//   - Bilingual (EN+JA, no indonesian/ dir): EN 0.55 / JA 0.45 (original).
//   - Trilingual (indonesian/ dir present):   EN 0.30 / JA 0.25 / ID 0.45.
//     ID-heavy for the language-acquisition continuation phase; renormalize
//     back toward EN/JA once ID competence is reached.
// The caller passes whether Indonesian data exists so the same binary can
// serve both corpus layouts without recompiling.
inline float lang_sample_weight(Language l, int phase = 0, bool has_indonesian = false) {
    (void)phase;  // single-phase, no curriculum
    if (has_indonesian) {
        switch (l) {
            case Language::English:    return 0.30f;
            case Language::Japanese:   return 0.25f;
            case Language::Indonesian: return 0.45f;
            default:                   return 0.00f;
        }
    }
    switch (l) {
        case Language::English:    return 0.55f;
        case Language::Japanese:   return 0.45f;
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
    // Ring buffer depth. Must comfortably exceed kMixShards (12) or
    // next_batch() can never reach a full mix: refill_active() drains
    // whatever is ready and returns, so a shallow ring keeps the active set
    // permanently under-filled and re-introduces the low-diversity batches
    // this design is meant to avoid.
    // Cost: 16 x 1M tokens x 4B = 64 MB host RAM.
    uint32_t    prefetch_shards = 16;         // shards in ring buffer
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

    // Set seq_len directly (for phase transitions driven by training loop).
    // Clears stale shards that were packed with the old seq_len.
    void set_seq_len(uint32_t new_seq) {
        if (new_seq == cfg_.seq_len) return;
        cfg_.seq_len = new_seq;
        active_shards_.clear();
        // Restart the staggered-admission fan-out (see next_batch()).
        initial_fill_done_ = false;
        admitted_count_    = 0;
        leftover_shard_.reset();
        leftover_pos_ = 0;
        // Advance stage_idx_ to match, if possible
        for (uint32_t i = 0; i < cfg_.curriculum.size(); i++) {
            if (cfg_.curriculum[i].seq_len == new_seq) { stage_idx_ = i; break; }
        }
        update_lang_weights();
        rebuild_distribution();
        printf("[Dataset] seq_len updated to %u (stage %u)\n", new_seq, stage_idx_);
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
    //
    // kMixShards=4 was too small. With shard_tokens=1M and seq_len=256 each
    // shard holds 4096 samples; at batch=24 round-robined over 4 shards each
    // shard is drained in 4096/(24/4) = 682.67 steps. All four then expire in
    // the same window and get replaced by four fresh (file, offset) draws, so
    // the entire batch composition flips at once. Measured effect: the EMA
    // loss stepped between ~3.25 and ~3.75 (1.65x perplexity) at gaps of
    // exactly 680 / 2050 / 2730 steps = 1 / 3 / 4 shard-cycles, with gnorm
    // completely flat (1.9-2.4) — i.e. a data-composition artefact, not
    // optimizer divergence.
    //
    // 12 shards gives each batch a much broader sample of the corpus, so one
    // noisy file (mc4_ja) can no longer dominate a whole batch. Combined with
    // the staggered-retirement logic in next_batch(), shards now expire a few
    // at a time instead of all together, removing the periodic cliff.
    //
    // Memory cost is bounded and small: 12 shards x 1M tokens x 4B = 48 MB
    // host RAM (was 16 MB). Observed process RSS is 0.4 GB, so this is noise.
    struct ActiveShard {
        std::shared_ptr<Shard> shard;
        uint32_t pos = 0;
    };
    std::vector<ActiveShard> active_shards_;
    static constexpr uint32_t kMixShards = 12;  // files mixed per batch

    // Staggered-retirement state (see next_batch() in the .cpp).
    //
    // Every active shard is drained at the same rate, so if they all enter at
    // pos=0 they also all expire together and the whole batch composition
    // flips at once. We therefore admit shards with a non-zero starting
    // offset: an even spread on the initial fill, then a small random jitter
    // on each later admission so the retirement times never re-synchronise.
    //
    // mix_rng_ is deliberately SEPARATE from rng_: rng_ is owned by the
    // prefetch thread (load_shard picks file + byte offset with it, partly
    // outside dist_mtx_), and next_batch() runs on the training thread.
    // Sharing one generator across the two would be a data race.
    std::mt19937_64 mix_rng_;
    bool     initial_fill_done_ = false;
    uint32_t admitted_count_    = 0;

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