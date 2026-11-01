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
    : cfg_(cfg), rng_(cfg.seed), mix_rng_(cfg.seed ^ 0x9E3779B97F4A7C15ULL)
{
    if (cfg_.curriculum.empty())
        throw std::invalid_argument("PretrainDataset: curriculum is empty");
    // cfg.seq_len is authoritative. Overwriting it with curriculum[0].seq_len
    // (256) silently broke every run that passed --max_seq != 256: the training
    // loop sizes its batch buffer as B*max_seq while next_batch() strides by
    // cfg_.seq_len, so the last (max_seq - 256) tokens of every sequence stayed
    // at their zero-initialised value and the model trained on runs of token 0.
    if (cfg_.seq_len == 0) cfg_.seq_len = cfg_.curriculum[0].seq_len;
    for (uint32_t i = 0; i < cfg_.curriculum.size(); ++i) {
        if (cfg_.curriculum[i].seq_len == cfg_.seq_len) { stage_idx_ = i; break; }
    }
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
    // Restart the staggered-admission fan-out for the new seq_len, otherwise
    // the next fill admits every shard with steady-state jitter and can land
    // them back in lockstep.
    initial_fill_done_ = false;
    admitted_count_    = 0;
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
    //
    // STAGGERED RETIREMENT: admitting every shard at pos=0 means they all
    // drain at the same rate and therefore all expire in the same window, so
    // the batch composition flips wholesale every shard-lifetime (measured:
    // a periodic 0.5-nat EMA step every 683 steps at batch=24/seq=256). To
    // break that synchronisation each shard is admitted at a non-zero start
    // offset — evenly spread across the first fill, then randomly jittered
    // afterwards — so shards retire a few at a time instead of together.
    //
    // Skipping those leading samples is harmless: load_shard() already seeks
    // to a random byte offset in a random file and shuffles within the shard,
    // so no sample is privileged. The tokens_consumed_ counter only advances
    // for samples actually emitted, so the token budget stays exact.
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
            if (shard && shard->n_samples > 0 && shard->seq_len == cfg_.seq_len) {
                uint32_t start_pos = 0;
                if (!initial_fill_done_) {
                    // First fill: fan the shards out evenly over their lifetime
                    // so the very first retirement wave is already staggered.
                    // Shard i starts (i / kMixShards) of the way through.
                    start_pos = (uint32_t)(((uint64_t)shard->n_samples *
                                            admitted_count_) / kMixShards);
                } else {
                    // Steady state: jitter within the first 25% so repeated
                    // replacements never drift back into lockstep.
                    uint32_t span = shard->n_samples / 4;
                    if (span > 0) start_pos = (uint32_t)(mix_rng_() % span);
                }
                // Never consume the whole shard as offset.
                if (start_pos >= shard->n_samples) start_pos = shard->n_samples - 1;

                active_shards_.push_back({shard, start_pos});
                if (++admitted_count_ >= kMixShards) initial_fill_done_ = true;
            }
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
// Does this corpus file use a document-separator line? Probing the head is
// enough: the writer emits one separator per document, so a corpus that uses
// them always has one within the first few documents.
static bool file_uses_separator(const std::string& path,
                                const std::string& sep,
                                size_t probe_bytes = 1u << 20) {
    if (sep.empty()) return false;
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    size_t read = 0;
    while (read < probe_bytes && std::getline(f, line)) {
        read += line.size() + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == sep) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
void PretrainDataset::discover_files() {
    files_.clear();
    std::unordered_set<std::string> seen_targets;
    std::vector<std::string> broken_links;
    size_t skipped_aliases = 0;

    std::vector<std::string> lang_dirs = {
        "indonesian", "english", "japanese", "mixed", "code", "math"
    };

    // Mode detection:
    //  - code/ AND math/ present, no indonesian/ (EN/JA/Code/Math build):
    //    EN 0.40 / JA 0.30 / Code 0.20 / Math 0.10
    //  - code/ AND math/ AND indonesian/ (legacy six-bucket 1B build):
    //    EN 0.35 / JA 0.25 / ID 0.20 / Code 0.09 / Math 0.03 / Mixed 0.08
    //  - trilingual (indonesian/ with a usable file): EN 0.30 / JA 0.25 / ID 0.45
    //  - bilingual default: EN 0.55 / JA 0.45
    bool has_indonesian = false;
    {
        fs::path id_dir = fs::path(cfg_.corpus_root) / "indonesian";
        if (fs::exists(id_dir)) {
            for (const auto& entry : fs::directory_iterator(id_dir)) {
                if (!entry.is_regular_file()) continue;
                std::string ext = entry.path().extension().string();
                if (ext == ".txt" || ext == ".jsonl" || ext == ".tsv") {
                    has_indonesian = true;
                    break;
                }
            }
        }
    }
    bool has_code = false, has_math = false;
    for (const char* bucket : {"code", "math"}) {
        fs::path dir = fs::path(cfg_.corpus_root) / bucket;
        if (!fs::exists(dir)) continue;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            if (ext == ".txt" || ext == ".jsonl" || ext == ".tsv") {
                if (bucket[0] == 'c') has_code = true; else has_math = true;
                break;
            }
        }
    }
    const bool has_code_math = has_code && has_math;
    if (has_code_math && !has_indonesian) {
        std::cout << "[Dataset] EN/JA/Code/Math corpus detected — quad weights\n";
    } else if (has_code_math) {
        std::cout << "[Dataset] Code+Math corpora detected — six-bucket 1B weights\n";
    } else if (has_indonesian) {
        std::cout << "[Dataset] Indonesian corpus detected — trilingual weights "
                     "(EN 0.30 / JA 0.25 / ID 0.45)\n";
    }
    // The TW_W_* overrides are only consulted in the code+math branch of
    // lang_sample_weight. One empty bucket therefore turns a five-bucket 10B
    // run into a bilingual EN/JA one and ignores every weight the operator set,
    // with no output beyond a "Skipping missing dir" line. Say so loudly.
    if (!has_code_math) {
        for (const char* v : {"TW_W_EN", "TW_W_JA", "TW_W_CODE", "TW_W_MATH",
                              "TW_W_MIXED"}) {
            if (std::getenv(v)) {
                std::cout << "[Dataset] !! " << v << " is set but code/ or math/ "
                             "is empty — ALL TW_W_* overrides are being IGNORED "
                             "and this run is falling back to EN 0.55 / JA 0.45\n";
                break;
            }
        }
    }
    if (has_code_math) {
        std::cout << "[Dataset] weights:";
        for (auto [name, l] : {std::pair{"en", Language::English},
                               {"ja", Language::Japanese},
                               {"id", Language::Indonesian},
                               {"code", Language::Code},
                               {"math", Language::Math},
                               {"mixed", Language::Mixed}}) {
            float w = lang_sample_weight(l, stage_idx_, has_indonesian, true);
            if (w > 0.0f) std::cout << " " << name << "=" << w;
        }
        std::cout << "\n";
    }

    for (const auto& dir_name : lang_dirs) {
        fs::path dir = fs::path(cfg_.corpus_root) / dir_name;
        if (!fs::exists(dir)) {
            std::cout << "[Dataset] Skipping missing dir: " << dir << "\n";
            continue;
        }
        Language lang = lang_from_dirname(dir_name);
        float weight  = lang_sample_weight(lang, stage_idx_, has_indonesian,
                                           has_code_math);

        for (const auto& entry : fs::directory_iterator(dir)) {
            std::string ext = entry.path().extension().string();
            if (ext != ".txt" && ext != ".jsonl" && ext != ".tsv") continue;
            if (!entry.is_regular_file()) {
                // is_regular_file() is false for a dangling symlink, which is
                // how a whole bucket can vanish without a single message.
                broken_links.push_back(entry.path().string());
                continue;
            }

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
            cf.doc_mode  = file_uses_separator(resolved_str, cfg_.doc_separator);
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
        std::cout << "  " << f.path << "  (" << f.byte_size/1024/1024 << " MB, "
                  << (f.doc_mode ? "doc-mode" : "line-mode") << ")\n";
    }
    if (!broken_links.empty()) {
        std::cerr << "[Dataset] WARNING: " << broken_links.size()
                  << " unreadable entries (dangling symlink / permission):\n";
        for (const auto& b : broken_links) std::cerr << "    " << b << "\n";
    }
    // A bucket directory that exists but contributes no bytes silently changes
    // the sampling mode (this is how the math bucket disappeared: its only file
    // was a dangling symlink, so has_math stayed false and the six-bucket
    // weights never activated).
    for (const auto& dir_name : lang_dirs) {
        fs::path dir = fs::path(cfg_.corpus_root) / dir_name;
        if (!fs::exists(dir)) continue;
        Language l = lang_from_dirname(dir_name);
        uint64_t bytes = 0;
        for (const auto& f : files_) if (f.lang == l) bytes += f.byte_size;
        if (bytes == 0)
            std::cerr << "[Dataset] WARNING: bucket '" << dir_name
                      << "' exists but has 0 usable bytes — it will never be sampled\n";
    }
}

void PretrainDataset::rebuild_distribution() {
    // Sampling weight per file.
    //
    // The language weights from lang_sample_weight() (EN 0.55 / JA 0.45) must
    // control the LANGUAGE ratio. Multiplying them by the raw file size — as
    // this did previously — lets corpus size override that intent: with
    // EN=4071MB vs JA=2043MB the actual token exposure came out 70.9/29.1
    // instead of 55/45, starving JA by 35% of its budget.
    //
    // Every shard yields the same token count (target_samples =
    // shard_tokens / seq_len in load_shard), so the file-selection ratio IS
    // the trained-token ratio.
    //
    // Fix: normalise each file's size by the TOTAL size of its own language.
    // The language weight then becomes absolute, while files within a language
    // still stay proportional to their size.
    //     w = lang_weight * (file_size / total_size_of_that_language)
    constexpr size_t kNumLangs = 7;   // see enum class Language
    double lang_total[kNumLangs] = {0};
    for (const auto& f : files_) {
        size_t li = (size_t)f.lang;
        if (li < kNumLangs) lang_total[li] += (double)f.byte_size;
    }

    std::vector<float> weights;
    weights.reserve(files_.size());
    for (const auto& f : files_) {
        size_t li = (size_t)f.lang;
        double total = (li < kNumLangs) ? lang_total[li] : 0.0;
        // A zero-byte file yields no samples; giving it weight would just
        // burn prefetch cycles on empty shards. total==0 means every file of
        // this language is empty, which lands in the same branch.
        float w = (f.byte_size > 0 && total > 0.0)
                ? f.weight * (float)((double)f.byte_size / total)
                : 0.0f;
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
    bool has_indonesian = false;
    bool has_code = false, has_math = false;
    for (const auto& f : files_) {
        if (f.lang == Language::Indonesian) has_indonesian = true;
        if (f.lang == Language::Code)       has_code = true;
        if (f.lang == Language::Math)       has_math = true;
    }
    const bool has_code_math = has_code && has_math;
    for (auto& f : files_) {
        f.weight = lang_sample_weight(f.lang, (int)stage_idx_, has_indonesian,
                                      has_code_math);
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
    uint64_t n_docs  = 0;

    // Tokenize one document. Documents longer than max_doc_bytes are SPLIT at a
    // line boundary (falling back to a UTF-8 codepoint boundary), never
    // truncated — the previous 1024-byte hard cut discarded the tail of ~88% of
    // fineweb_edu_extra lines and ~81% of culturax_ja lines.
    const size_t max_doc = std::max<size_t>(256, cfg_.max_doc_bytes);
    auto emit_doc = [&](std::string& doc) {
        if (doc.empty()) return;
        size_t pos = 0;
        while (pos < doc.size() && shard->n_samples < target_samples) {
            size_t take = std::min(max_doc, doc.size() - pos);
            if (pos + take < doc.size()) {
                // Prefer to break on the last newline inside the window so a
                // source file is never split mid-statement.
                size_t nl = doc.rfind('\n', pos + take);
                if (nl != std::string::npos && nl > pos + max_doc / 2) {
                    take = nl - pos;
                } else {
                    // Otherwise back up to a UTF-8 lead byte so no codepoint is
                    // cut in half (a dangling continuation byte tokenizes to a
                    // rare byte-level token right before the next document).
                    while (take > 1 &&
                           (static_cast<unsigned char>(doc[pos + take]) & 0xC0) == 0x80)
                        --take;
                }
            }
            std::string chunk = doc.substr(pos, take);
            pos += take;
            while (pos < doc.size() && doc[pos] == '\n') ++pos;  // drop the break itself
            tokenize_line(chunk, ids);
            if (ids.empty()) continue;
            packer.feed(ids);
            ++n_docs;
            while (packer.has_sequence() && shard->n_samples < target_samples) {
                packer.pop_sequence(seq.data());
                shard->tokens.insert(shard->tokens.end(), seq.begin(), seq.end());
                ++shard->n_samples;
            }
        }
        doc.clear();
    };

    if (cf.doc_mode) {
        // Documents span multiple lines and end at a separator line. Blank
        // lines are content here (indentation/structure in code and math), so
        // they are kept verbatim.
        std::string doc;
        doc.reserve(max_doc + 1024);
        // The random seek above lands mid-document; skip to the next separator
        // so we never train on a half document.
        bool synced = (f.tellg() == std::streampos(0));
        while (shard->n_samples < target_samples && std::getline(f, line)) {
            ++n_lines;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line == cfg_.doc_separator) {
                if (!synced) { synced = true; doc.clear(); continue; }
                emit_doc(doc);
                continue;
            }
            if (!synced) continue;
            if (doc.size() + line.size() + 1 > (size_t)max_doc * 8) emit_doc(doc);
            if (!doc.empty()) doc += '\n';
            doc += line;
        }
        emit_doc(doc);
    } else {
        // Legacy layout: one document per line.
        while (shard->n_samples < target_samples && std::getline(f, line)) {
            if (line.empty()) continue;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            ++n_lines;
            emit_doc(line);
        }
    }

    // Debug: one-time print showing prefetch is working
    static bool showed_nlines = false;
    if (n_lines > 0 && !showed_nlines) {
        showed_nlines = true;
        std::cout << "[prefetch] file=" << cf.path.substr(cf.path.find_last_of('/')+1)
                  << (cf.doc_mode ? " (doc-mode)" : " (line-mode)")
                  << " lines=" << n_lines
                  << " docs=" << n_docs
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