// =============================================================================
// train_bpe — CLI for BPETrainer.
//
// The trainer class existed but had no entry point, so vocab_6 could not be
// regenerated. Needed for the 492M spec, which moves to a 49152-token EN/JA/code
// vocab (vocab_6 is 65536 and carries Indonesian/Korean merges we no longer use).
//
// Per-source byte budgets matter: BPE allocates merges in proportion to what it
// sees, so the sample mix should track the pretraining mix. Japanese needs fewer
// bytes for the same token share because CJK is ~2 bytes/token against ~4 for
// English prose.
//
//   train_bpe --out_dir vocab_7 --vocab_size 49152 \
//             --corpus data/corpus_final/english:2200 \
//             --corpus data/corpus_final/japanese:900 \
//             --corpus data/corpus_1b/code:500
// =============================================================================
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "bpe/bpe_tokenizer.h"

namespace fs = std::filesystem;

struct Source {
    std::string path;
    uint64_t    budget_mb = 0;  // 0 = unlimited
};

static void usage() {
    printf("Usage: train_bpe [options]\n");
    printf("  --corpus <path[:MB]>   File or directory; repeatable. MB caps each file.\n");
    printf("  --out_dir <dir>        Output dir for vocab.txt + merges.txt [vocab_7]\n");
    printf("  --binary <path>        Also write the packed binary [<out_dir>/bpe.bin]\n");
    printf("  --vocab_size <N>       Target vocab [49152]\n");
    printf("  --min_freq <N>         Minimum pair frequency [15]\n");
    printf("  --threads <N>          Worker threads [hardware concurrency]\n");
    printf("  --checkpoint <path>    Checkpoint file for pause/resume\n");
    printf("  --resume               Resume from --checkpoint if present\n");
}

// Collect .txt/.jsonl files, sorted so a run is reproducible across machines.
static void collect(const std::string& root, std::vector<std::string>& out) {
    std::error_code ec;
    if (fs::is_regular_file(root, ec)) {
        out.push_back(root);
        return;
    }
    if (!fs::is_directory(root, ec)) {
        fprintf(stderr, "[WARN] skipping missing path: %s\n", root.c_str());
        return;
    }
    std::vector<std::string> found;
    for (auto& e : fs::recursive_directory_iterator(root, ec)) {
        if (!e.is_regular_file(ec)) continue;
        auto ext = e.path().extension().string();
        if (ext == ".txt" || ext == ".jsonl") found.push_back(e.path().string());
    }
    std::sort(found.begin(), found.end());
    out.insert(out.end(), found.begin(), found.end());
}

int main(int argc, char** argv) {
    std::vector<Source> sources;
    std::string out_dir = "vocab_7";
    std::string binary_path, ckpt_path;
    uint32_t vocab_size = 49152, min_freq = 15;
    uint32_t threads = std::max(1u, std::thread::hardware_concurrency());
    bool resume = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--corpus" && i + 1 < argc) {
            std::string spec = argv[++i];
            Source s;
            auto colon = spec.rfind(':');
            // rfind, but a Windows drive letter or a bare path must not split.
            if (colon != std::string::npos && colon + 1 < spec.size() &&
                spec.find_first_not_of("0123456789", colon + 1) == std::string::npos) {
                s.path = spec.substr(0, colon);
                s.budget_mb = std::stoull(spec.substr(colon + 1));
            } else {
                s.path = spec;
            }
            sources.push_back(s);
        }
        else if (a == "--out_dir" && i + 1 < argc)    out_dir = argv[++i];
        else if (a == "--binary" && i + 1 < argc)     binary_path = argv[++i];
        else if (a == "--vocab_size" && i + 1 < argc) vocab_size = (uint32_t)std::stoul(argv[++i]);
        else if (a == "--min_freq" && i + 1 < argc)   min_freq = (uint32_t)std::stoul(argv[++i]);
        else if (a == "--threads" && i + 1 < argc)    threads = (uint32_t)std::stoul(argv[++i]);
        else if (a == "--checkpoint" && i + 1 < argc) ckpt_path = argv[++i];
        else if (a == "--resume")                     resume = true;
        else if (a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "[ERR] unknown arg: %s\n", a.c_str()); usage(); return 1; }
    }
    if (sources.empty()) { fprintf(stderr, "[ERR] need at least one --corpus\n"); return 1; }
    if (binary_path.empty()) binary_path = out_dir + "/bpe.bin";

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  BPE Trainer — vocab %-6u  threads %-3u                 ║\n", vocab_size, threads);
    printf("╚══════════════════════════════════════════════════════════╝\n");

    BPETrainer trainer(vocab_size, min_freq, threads);
    trainer.init_base_vocab();
    if (!ckpt_path.empty()) trainer.set_checkpoint_path(ckpt_path);

    if (resume && !ckpt_path.empty() && trainer.load_checkpoint(ckpt_path))
        printf("[RESUME] merge_step=%zu vocab=%zu\n",
               trainer.merge_step(), trainer.vocab_size());

    // Each source gets its own load_corpus call so budgets stay independent.
    auto t0 = std::chrono::steady_clock::now();
    for (auto& s : sources) {
        std::vector<std::string> files;
        collect(s.path, files);
        if (files.empty()) continue;
        uint64_t cap = s.budget_mb ? (s.budget_mb * 1024ULL * 1024ULL) / files.size() : 0;
        printf("\n[SRC] %s — %zu files, %s\n", s.path.c_str(), files.size(),
               s.budget_mb ? (std::to_string(s.budget_mb) + " MB total, " +
                              std::to_string(cap / (1024 * 1024)) + " MB/file").c_str()
                           : "unlimited");
        trainer.load_corpus(files, cap);
    }
    auto load_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    printf("\n[LOAD] done in %.1f min\n", load_s / 60.0);

    trainer.train();

    fs::create_directories(out_dir);
    trainer.save(out_dir + "/vocab.txt", out_dir + "/merges.txt");
    trainer.save_binary(binary_path);
    if (!ckpt_path.empty()) trainer.save_checkpoint(ckpt_path);

    printf("\n[DONE] vocab=%zu merges=%zu -> %s (+ %s)\n",
           trainer.vocab_size(), trainer.num_merges(), out_dir.c_str(),
           binary_path.c_str());
    return 0;
}
