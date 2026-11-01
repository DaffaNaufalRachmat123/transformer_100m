#pragma once
// =============================================================================
// BPE Tokenizer — Quadrilingual (ID + EN + JA + KO)
// CUDA-only build: MLX dependencies removed.
// =============================================================================

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
// Training target only; BPEEncoder takes its vocab size from the loaded file.
static constexpr uint32_t VOCAB_SIZE          = 49152;  // EN/JA/code
static constexpr uint32_t MIN_FREQUENCY       = 15;
static constexpr uint32_t MIN_FREQUENCY_CJK   = 10;
static constexpr uint32_t MIN_FREQUENCY_LATIN = 25;
static constexpr uint32_t MIN_FREQUENCY_CODE  = 20;
static constexpr uint32_t NUM_THREADS_DEFAULT = 8;
static constexpr uint32_t CHECKPOINT_INTERVAL = 500;
static constexpr uint32_t PROGRESS_INTERVAL   = 50;

// ---------------------------------------------------------------------------
// Special tokens
// ---------------------------------------------------------------------------
struct SpecialTokens {
    static constexpr const char* ENDOFTEXT  = "<|endoftext|>";
    static constexpr const char* SYSTEM     = "<|system|>";
    static constexpr const char* USER       = "<|user|>";
    static constexpr const char* ASSISTANT  = "<|assistant|>";
    static constexpr const char* END        = "<|end|>";
    static constexpr const char* CODE       = "<|code|>";
    static constexpr const char* CODE_END   = "<|/code|>";
    static constexpr const char* TRANSLATE  = "<|translate|>";
    static constexpr const char* SUMMARY    = "<|summary|>";
    // Slots 9 and 10 held <|phone_event|> and <|app_usage|>, which an EN/JA
    // model never emits. Reused as reasoning delimiters so "thinking" costs one
    // token per side instead of the ~6 a literal "<think>" string would take.
    static constexpr const char* THINK      = "<|think|>";
    static constexpr const char* THINK_END  = "<|/think|>";
    static constexpr const char* LANG_ID    = "<|lang_id|>";
    static constexpr const char* LANG_EN    = "<|lang_en|>";
    static constexpr const char* LANG_JA    = "<|lang_ja|>";
    static constexpr const char* LANG_KO    = "<|lang_ko|>";
    static constexpr const char* PAD        = "<|pad|>";

    static std::vector<std::string> all() {
        return {
            ENDOFTEXT, SYSTEM, USER, ASSISTANT, END,
            CODE, CODE_END, TRANSLATE, SUMMARY,
            THINK, THINK_END,
            LANG_ID, LANG_EN, LANG_JA, LANG_KO, PAD
        };
    }
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
using TokenID    = uint32_t;
using SymbolPair = std::pair<TokenID, TokenID>;

struct PairHash {
    size_t operator()(const SymbolPair& p) const noexcept {
        return std::hash<uint64_t>{}(((uint64_t)p.first << 32) | p.second);
    }
};

struct MergeRule {
    SymbolPair pair;
    TokenID    result;
    uint64_t   frequency;
    bool operator<(const MergeRule& o) const { return frequency > o.frequency; }
};

// ---------------------------------------------------------------------------
// PairTracker — O(1) best-pair lookup with lazy-invalidation priority queue
// ---------------------------------------------------------------------------
struct PairEntry {
    uint64_t   freq;
    SymbolPair pair;
    bool operator<(const PairEntry& o) const { return freq < o.freq; }
};

class PairTracker {
public:
    void insert_or_add(const SymbolPair& p, int64_t delta) {
        counts_[p] += delta;
        if (counts_[p] > 0)
            heap_.push({(uint64_t)counts_[p], p});
        else
            counts_.erase(p);
    }

    std::pair<SymbolPair, uint64_t> pop_best() {
        while (!heap_.empty()) {
            auto top = heap_.top();
            heap_.pop();
            auto it = counts_.find(top.pair);
            if (it == counts_.end() || it->second != (int64_t)top.freq)
                continue;
            return {top.pair, top.freq};
        }
        return {{0, 0}, 0};
    }

    std::pair<SymbolPair, uint64_t> peek_best() {
        while (!heap_.empty()) {
            auto top = heap_.top();
            auto it = counts_.find(top.pair);
            if (it == counts_.end() || it->second != (int64_t)top.freq) {
                heap_.pop();
                continue;
            }
            return {top.pair, top.freq};
        }
        return {{0, 0}, 0};
    }

    uint64_t count(const SymbolPair& p) const {
        auto it = counts_.find(p);
        return (it == counts_.end()) ? 0 : (uint64_t)it->second;
    }

    void pop_pair(const SymbolPair& p) { counts_.erase(p); }
    size_t size() const { return counts_.size(); }
    bool   empty() const { return counts_.empty(); }

    void save(std::ostream& os) const;
    void load(std::istream& is);

private:
    std::unordered_map<SymbolPair, int64_t, PairHash> counts_;
    std::priority_queue<PairEntry>                      heap_;
};

// ---------------------------------------------------------------------------
// ThreadPool
// ---------------------------------------------------------------------------
class BPEThreadPool {
public:
    explicit BPEThreadPool(uint32_t n);
    ~BPEThreadPool();
    void submit(std::function<void()> task);
    void wait_all();

private:
    std::vector<std::thread>          workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex                        mtx_;
    std::condition_variable           cv_;
    std::condition_variable           done_cv_;
    uint32_t                          active_ = 0;
    bool                              stop_   = false;
};

// ---------------------------------------------------------------------------
// BPETrainer
// ---------------------------------------------------------------------------
class BPETrainer {
public:
    explicit BPETrainer(uint32_t target_vocab   = VOCAB_SIZE,
                        uint32_t min_freq       = MIN_FREQUENCY,
                        uint32_t num_threads    = NUM_THREADS_DEFAULT);

    void init_base_vocab();
    void load_corpus(const std::vector<std::string>& file_paths,
                     uint64_t max_bytes_per_file = 0);
    void train();
    bool load_seed_merges(const std::string& path);
    void pause();
    void resume();
    bool is_paused() const { return paused_.load(); }
    void save_checkpoint(const std::string& path) const;
    bool load_checkpoint(const std::string& path);
    // Where train() writes its every-500-merge auto-checkpoint.
    void set_checkpoint_path(const std::string& p) { ckpt_path_ = p; }
    void save(const std::string& vocab_path, const std::string& merges_path) const;
    void load(const std::string& vocab_path, const std::string& merges_path);
    void save_binary(const std::string& path) const;
    void load_binary(const std::string& path);
    std::vector<TokenID> encode(const std::string& text, bool add_special = false) const;
    std::string          decode(const std::vector<TokenID>& ids) const;
    void validate(const std::unordered_map<std::string, std::string>& test_sentences) const;
    size_t   vocab_size()  const { return id_to_token_.size(); }
    size_t   num_merges()  const { return merges_.size(); }
    size_t   merge_step()  const { return merge_step_; }
    uint32_t token_to_id(const std::string& s) const;

private:
    static std::vector<uint32_t> utf8_to_codepoints(const std::string& s);
    std::vector<std::string> pretokenize(const std::string& text) const;
    uint32_t pair_min_freq(TokenID a, TokenID b) const;
    void build_pair_counts_parallel();
    void update_pair_counts(const SymbolPair& merged_pair, TokenID new_id,
                            const std::vector<size_t>& affected_word_ids);
    std::vector<size_t> apply_merge(const SymbolPair& pair, TokenID new_id);
    void render_progress(size_t step, size_t total, uint64_t freq,
                         const std::string& left, const std::string& right,
                         double merges_per_sec) const;

    uint32_t target_vocab_;
    uint32_t min_frequency_;
    uint32_t num_threads_;
    std::unordered_map<std::string_view, TokenID> token_to_id_;
    std::vector<std::string>                     id_to_token_;
    std::vector<MergeRule>                       merges_;
    std::unordered_map<SymbolPair, TokenID, PairHash> merge_lookup_;
    std::unordered_map<std::string, uint64_t> word_freq_;
    std::vector<std::string>                  words_;
    std::vector<uint64_t>                     word_freqs_;
    std::vector<std::vector<TokenID>>         word_tokens_;
    PairTracker                               tracker_;
    size_t                                    merge_step_ = 0;
    std::unique_ptr<BPEThreadPool> pool_;
    std::vector<std::pair<std::string, std::string>> seed_merges_;
    std::atomic<bool>       paused_  {false};
    std::mutex              pause_mtx_;
    std::condition_variable pause_cv_;
    mutable std::mutex mtx_;
    uint64_t total_words_ = 0;
    std::string ckpt_path_ = "bpe_checkpoints/bpe_ckpt.bin";
};

// ---------------------------------------------------------------------------
// BPEEncoder — inference-only, minimal overhead
// ---------------------------------------------------------------------------
class BPEEncoder {
public:
    BPEEncoder() = default;
    explicit BPEEncoder(const std::string& vocab_path, const std::string& merges_path);

    void load_text(const std::string& vocab_path, const std::string& merges_path);
    void save_binary(const std::string& path) const;
    bool load_binary(const std::string& path);

    // Convenience wrappers used by training code
    bool load(const std::string& path) { return load_binary(path); }
    void load_vocab(const std::string& vocab_dir) {
        load_text(vocab_dir + "/vocab.txt", vocab_dir + "/merges.txt");
    }

    std::vector<TokenID> encode(const std::string& text,
                                bool add_bos = false,
                                bool add_eos = false) const;
    std::string decode(const std::vector<TokenID>& ids, bool skip_special = false) const;

    TokenID special_id(const std::string& token) const;
    size_t  vocab_size() const { return id_to_token_.size(); }

    // Token ID accessors
    int bos_id() const { return (int)special_id(SpecialTokens::ENDOFTEXT); }
    int eos_id() const { return (int)special_id(SpecialTokens::END); }
    int pad_id() const { return (int)special_id(SpecialTokens::PAD); }

private:
    std::vector<std::string>                            id_to_token_;
    std::unordered_map<std::string_view, TokenID>       token_to_id_;
    std::unordered_map<SymbolPair, TokenID, PairHash>   merge_lookup_;
    std::vector<std::pair<SymbolPair, TokenID>>         merges_ordered_;

    std::vector<std::string> pretokenize(const std::string& text) const;
    std::vector<TokenID>     encode_word(const std::string& word) const;
    std::vector<TokenID>     encode_chunk(std::string_view word) const;
};
