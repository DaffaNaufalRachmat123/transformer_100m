// =============================================================================
// pretrain_dataset.cpp
// =============================================================================

#include "pretrain_dataset.h"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>


namespace fs = std::filesystem;

// =============================================================================
// PretrainDataset
// =============================================================================

PretrainDataset::PretrainDataset(const PretrainDatasetConfig& cfg)
    : cfg_(cfg), rng_(cfg.seed)
{
    if (cfg_.curriculum.empty())
        throw std::invalid_argument("PretrainDataset: curriculum is empty");
    cfg_.seq_len = cfg_.curriculum[0].seq_len;
}

PretrainDataset::~PretrainDataset() {
    stop_flag_.store(true);
    ring_cv_not_full_.notify_all();
    ring_cv_not_empty_.notify_all();
    if (prefetch_thread_.joinable())
        prefetch_thread_.join();
}

bool PretrainDataset::init() {
    discover_files();
    if (files_.empty()) {
        std::cerr << "[Dataset] No corpus files found in: " << cfg_.corpus_root << "\n";
        return false;
    }
    rebuild_distribution();

    // Start prefetch background thread
    prefetch_thread_ = std::thread([this]{ prefetch_loop(); });
    return true;
}

bool PretrainDataset::advance_curriculum() {
    if (stage_idx_ + 1 >= cfg_.curriculum.size()) return false;
    ++stage_idx_;
    cfg_.seq_len = cfg_.curriculum[stage_idx_].seq_len;
    std::cout << "[Dataset] Curriculum advanced to stage " << stage_idx_
              << " (" << cfg_.curriculum[stage_idx_].name << ")"
              << " seq_len=" << cfg_.seq_len << "\n";
    // Drop shards produced with the previous seq_len (they have a different
    // sample stride and must not be mixed into the new stage).
    active_shards_.clear();
    leftover_shard_.reset();
    leftover_pos_ = 0;
    // Refresh language sampling weights for the new phase
    update_lang_weights();
    rebuild_distribution();
    return true;
}

bool PretrainDataset::next_batch(uint32_t* tokens_out, uint32_t n_samples) {
    // Cross-language mixing: keep up to kMixShards shards active at once and
    // round-robin one sample from each, so a single batch contains a MIX of
    // languages/domains. This removes the single-language-per-batch behaviour
    // that made the loss oscillate on shard boundaries.
    uint32_t filled = 0;

    auto refill_active = [&]() {
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
            // else: stale shard (old seq_len) — discard silently
        }
    };

    refill_active();
    if (active_shards_.empty()) return false;

    uint32_t rr = 0;  // round-robin index across active shards
    while (filled < n_samples) {
        if (active_shards_.empty()) {
            refill_active();
            if (active_shards_.empty()) break;  // end of corpus / stopping
        }
        if (rr >= active_shards_.size()) rr = 0;

        ActiveShard& as = active_shards_[rr];
        const uint32_t* src = as.shard->tokens.data() + (size_t)as.pos * cfg_.seq_len;
        uint32_t* dst = tokens_out + (size_t)filled * cfg_.seq_len;
        memcpy(dst, src, cfg_.seq_len * sizeof(uint32_t));
        ++as.pos;
        ++filled;
        last_batch_lang_ = as.shard->lang;
        tokens_consumed_.fetch_add(cfg_.seq_len, std::memory_order_relaxed);

        if (as.pos >= as.shard->n_samples) {
            // Shard exhausted: drop it. Do not advance rr (next shard shifts
            // into this index). Try to pull a replacement.
            active_shards_.erase(active_shards_.begin() + rr);
            refill_active();
            if (!active_shards_.empty() && rr >= active_shards_.size()) rr = 0;
        } else {
            ++rr;
        }
    }
    return filled == n_samples;
}

// ---------------------------------------------------------------------------
void PretrainDataset::discover_files() {
    files_.clear();
    std::unordered_set<std::string> seen_targets;
    size_t skipped_aliases = 0;

    std::vector<std::string> lang_dirs = {
        "indonesian", "english", "japanese", "korean", "mixed"
    };

    for (const auto& dir_name : lang_dirs) {
        fs::path dir = fs::path(cfg_.corpus_root) / dir_name;
        if (!fs::exists(dir)) {
            std::cout << "[Dataset] Skipping missing dir: " << dir << "\n";
            continue;
        }
        Language lang = lang_from_dirname(dir_name);
        float weight  = lang_sample_weight(lang, stage_idx_);

        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            if (ext != ".txt" && ext != ".jsonl" && ext != ".tsv") continue;

            std::error_code ec;
            fs::path resolved = fs::weakly_canonical(entry.path(), ec);
            if (ec || resolved.empty()) {
                ec.clear();
                resolved = fs::absolute(entry.path(), ec);
            }
            if (ec || resolved.empty()) {
                resolved = entry.path();
            }

            std::string resolved_str = resolved.string();
            if (!seen_targets.insert(resolved_str).second) {
                ++skipped_aliases;
                continue;
            }

            uint64_t byte_size = entry.file_size();
            ec.clear();
            uint64_t resolved_size = fs::file_size(resolved, ec);
            if (!ec) byte_size = resolved_size;

            CorpusFile cf;
            cf.path      = resolved_str;
            cf.lang      = lang;
            cf.weight    = weight;
            cf.byte_size = byte_size;
            files_.push_back(std::move(cf));
        }
    }

    // Sort by path for determinism
    std::sort(files_.begin(), files_.end(),
              [](const CorpusFile& a, const CorpusFile& b){ return a.path < b.path; });

    std::cout << "[Dataset] Found " << files_.size() << " unique corpus files";
    if (skipped_aliases > 0) {
        std::cout << " (skipped " << skipped_aliases << " symlink/alias duplicates)";
    }
    std::cout << "\n";
    for (const auto& f : files_) {
        std::cout << "  " << f.path << "  (" << f.byte_size/1024/1024 << " MB)\n";
    }
}

void PretrainDataset::rebuild_distribution() {
    std::vector<float> weights;
    weights.reserve(files_.size());
    for (const auto& f : files_) {
        float w = f.weight * std::max(1.0f, (float)f.byte_size / (1024.f * 1024.f));
        weights.push_back(w);
    }
    {
        std::lock_guard<std::mutex> lock(dist_mtx_);
        file_weights_ = std::move(weights);
        file_dist_ = std::discrete_distribution<size_t>(
            file_weights_.begin(), file_weights_.end()
        );
    }
}

void PretrainDataset::update_lang_weights() {
    for (auto& f : files_) {
        f.weight = lang_sample_weight(f.lang, (int)stage_idx_);
    }
}

void PretrainDataset::prefetch_loop() {
    uint64_t empty_count = 0;
    while (!stop_flag_.load()) {
        auto shard = load_shard();
        if (!shard || shard->n_samples == 0) {
            if (++empty_count % 100 == 0)
                std::cout << "[prefetch] " << empty_count << " empty shards\n" << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        empty_count = 0;

        std::unique_lock<std::mutex> lock(ring_mtx_);
        ring_cv_not_full_.wait(lock, [this]{
            return ready_shards_.size() < cfg_.prefetch_shards || stop_flag_.load();
        });
        if (stop_flag_.load()) break;
        ready_shards_.push_back(shard);
        ring_cv_not_empty_.notify_one();
    }
}

std::shared_ptr<Shard> PretrainDataset::load_shard() {
    if (files_.empty()) return nullptr;

    // Copy seq_len once to avoid race with advance_curriculum()
    uint32_t seq_len = cfg_.seq_len;

    // Pick a file with weighted sampling (protected against rebuild race)
    size_t file_idx;
    {
        std::lock_guard<std::mutex> lock(dist_mtx_);
        file_idx = file_dist_(rng_);
    }
    const CorpusFile& cf = files_[file_idx];

    auto shard = std::make_shared<Shard>();
    shard->seq_len = seq_len;
    shard->lang = cf.lang;

    std::ifstream f(cf.path);
    if (!f) {
        std::cerr << "[Dataset] Cannot open: " << cf.path << "\n";
        return shard;
    }

    // Seek to random offset to avoid always reading from the top
    uint64_t file_size = cf.byte_size;
    if (file_size > 1024) {
        uint64_t max_offset = file_size - 512;
        uint64_t offset = rng_() % max_offset;
        f.seekg(offset);
        std::string discard;
        std::getline(f, discard); // skip partial line
    }

    TokenPacker packer(seq_len, 0 /* EOS=0=<|endoftext|> */);
    std::vector<uint32_t> ids;
    std::string line;
    uint32_t target_samples = cfg_.shard_tokens / seq_len;

    // Pre-allocate shard token storage
    shard->tokens.reserve((size_t)target_samples * seq_len);
    std::vector<uint32_t> seq(seq_len);

    uint64_t n_lines = 0;
    while (shard->n_samples < target_samples && std::getline(f, line)) {
        if (line.empty()) continue;
        // Limit to 1024 bytes to prevent BPE O(n²) blow-up on pathological
        // lines with no whitespace (e.g. base64 data, minified code, math).
        // corpus_final has lines up to 100K+ chars; 1024 is safe and still
        // captures complete sentences for most natural-language content.
        //
        // IMPORTANT: this is a byte-level cut, so for multi-byte UTF-8
        // languages (JA/KO especially — avg line length 5-8KB, so the
        // vast majority of their lines get truncated here) a naive
        // resize() can slice a multi-byte codepoint in half, leaving a
        // dangling continuation byte (0x80-0xBF) at the tail. The BPE
        // encoder then falls back to a rare/unnatural byte-level token
        // for that garbage tail right before the next document's BOS —
        // and this corruption rate roughly doubles per training sample
        // whenever the packer concatenates more documents per sequence
        // (i.e. every time seq_len doubles in the curriculum). Trim back
        // to the last valid UTF-8 codepoint boundary instead.
        if (line.size() > 1024) {
            size_t cut = 1024;
            // Walk backwards while we're sitting on a UTF-8 continuation
            // byte (10xxxxxx) — back up until we reach a leading byte
            // (0xxxxxxx ASCII, or a 11xxxxxx multi-byte lead byte).
            while (cut > 0 && (static_cast<unsigned char>(line[cut]) & 0xC0) == 0x80) {
                --cut;
            }
            line.resize(cut);
        }
        tokenize_line(line, ids);
        if (ids.empty()) continue;
        packer.feed(ids);
        ++n_lines;

        // Pop completed sequences immediately so shard->n_samples advances
        while (packer.has_sequence() && shard->n_samples < target_samples) {
            packer.pop_sequence(seq.data());
            shard->tokens.insert(shard->tokens.end(), seq.begin(), seq.end());
            ++shard->n_samples;
        }
    }

    // Debug: one-time print showing prefetch is working
    static bool showed_nlines = false;
    if (n_lines > 0 && !showed_nlines) {
        showed_nlines = true;
        std::cout << "[prefetch] file=" << cf.path.substr(cf.path.find_last_of('/')+1)
                  << " lines=" << n_lines
                  << " samples=" << shard->n_samples
                  << "\n" << std::flush;
    }

    // Shuffle sequences within shard
    if (shard->n_samples > 1) {
        std::vector<uint32_t> indices(shard->n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng_);
        std::vector<uint32_t> shuffled;
        shuffled.reserve(shard->tokens.size());
        for (uint32_t idx : indices) {
            auto it = shard->tokens.begin() + (size_t)idx * seq_len;
            shuffled.insert(shuffled.end(), it, it + seq_len);
        }
        shard->tokens = std::move(shuffled);
    }

    return shard;
}

void PretrainDataset::tokenize_line(const std::string& text, std::vector<uint32_t>& ids) {
    ids.clear();
    if (tokenize_fn_) {
        tokenize_fn_(text, ids);
    } else {
        // Fallback: character-level UTF-8 byte tokenization (placeholder)
        // In production, always set tokenize_fn_ before calling init().
        ids.reserve(text.size());
        for (unsigned char c : text) ids.push_back((uint32_t)c);
    }
}

void PretrainDataset::print_info(std::ostream& os) const {
    os << "=== PretrainDataset ===\n";
    os << "  Corpus root  : " << cfg_.corpus_root << "\n";
    os << "  Files        : " << files_.size() << "\n";
    os << "  Seq len      : " << cfg_.seq_len << "\n";
    os << "  Stage        : " << stage_idx_ << " / " << cfg_.curriculum.size() << "\n";
    os << "  Tokens used  : " << tokens_consumed_.load() << "\n";

    uint64_t total_bytes = 0;
    for (const auto& f : files_) total_bytes += f.byte_size;
    os << "  Total data   : " << total_bytes/1024/1024/1024 << " GB\n";
}

// =============================================================================
// TokenPacker
// =============================================================================

void TokenPacker::feed(const std::vector<uint32_t>& ids) {
    // NOTE: `ids` already carries its own BOS/END markers — it comes from
    // tok.encode(text, /*add_bos=*/true, /*add_eos=*/true) in train_cuda.cu,
    // so every document is already wrapped as [BOS(0) ... END(4)]. This
    // function used to additionally push_back(eos_id_) here, where
    // eos_id_ was passed in as 0 (actually the model's BOS/<|endoftext|>
    // id, not the true EOS id 4). That duplicated a BOS-like token at
    // every document boundary, producing "...END(4), 0, 0, NEXT_DOC..."
    // instead of the intended "...END(4), 0, NEXT_DOC...". The natural
    // END->BOS transition already IS the document boundary marker, so no
    // extra token is needed — just append the document's own ids.
    for (uint32_t id : ids) buffer_.push_back(id);
}

bool TokenPacker::pop_sequence(uint32_t* dst) {
    if (buffer_.size() < seq_len_) return false;
    for (uint32_t i = 0; i < seq_len_; ++i) {
        dst[i] = buffer_.front();
        buffer_.pop_front();
    }
    ++n_seqs_;
    return true;
}