// =============================================================================
// llama_runner — greedy generation from PRE-TOKENIZED input.
//
// Why this exists: the model's BPE (bpe/bpe_tokenizer.cpp) is not GPT-2
// compatible, so llama.cpp cannot encode prompts for it. Prompts are encoded
// outside (scripts/bpe_reference.py, or the Kotlin/Swift port) and fed here as
// token IDs. Detokenization IS compatible, so llama_token_to_piece is used.
//
// This is also the reference flow for the mobile apps.
//
// stdin : one prompt per line, space-separated token ids
// stdout: generated text per prompt, terminated by <<<EOT>>>
//
// Build:
//   cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=../llama.cpp
//   cmake --build build -j
// =============================================================================

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <chrono>

static void usage(const char* p) {
    fprintf(stderr,
        "usage: %s --model M.gguf [--n-predict 48] [--ctx 512] [--threads 4]\n"
        "          [--eot-marker] [--print-ids]\n"
        "stdin: one prompt per line as space-separated token ids\n", p);
}

int main(int argc, char** argv) {
    std::string model_path;
    int n_predict = 48, n_ctx = 512, n_threads = 4, debug_topk = 0, n_gpu_layers = 0;
    bool eot_marker = false, print_ids = false, ignore_eog = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--model")      model_path = next();
        else if (a == "--n-predict")  n_predict  = atoi(next());
        else if (a == "--ctx")        n_ctx      = atoi(next());
        else if (a == "--threads")    n_threads  = atoi(next());
        else if (a == "--eot-marker") eot_marker = true;
        else if (a == "--print-ids")  print_ids  = true;
        else if (a == "--ignore-eog") ignore_eog = true;
        else if (a == "--debug-topk") debug_topk = atoi(next());
        else if (a == "--n-gpu-layers" || a == "-ngl") n_gpu_layers = atoi(next());
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty()) { usage(argv[0]); return 1; }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;   // 0 = CPU, matches the reference engine
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path.c_str()); return 1; }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx         = n_ctx;
    cparams.n_batch       = n_ctx;
    cparams.n_threads     = n_threads;
    cparams.n_threads_batch = n_threads;
    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    fprintf(stderr, "[runner] ready. vocab=%d ctx=%d threads=%d\n",
            n_vocab, n_ctx, n_threads);

    auto piece = [&](llama_token id) {
        char buf[256];
        // special=false: control tokens (<|end|> etc.) render as empty
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
        return (n > 0) ? std::string(buf, n) : std::string();
    };

    std::string line;
    long long total_tok = 0;
    double total_s = 0.0;

    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        std::vector<llama_token> prompt;
        std::istringstream iss(line);
        int id;
        while (iss >> id) prompt.push_back((llama_token) id);
        if (prompt.empty()) continue;

        if ((int) prompt.size() >= n_ctx) {
            fprintf(stderr, "[runner] prompt %zu tokens >= ctx %d, truncating\n",
                    prompt.size(), n_ctx);
            prompt.resize(n_ctx - 1);
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        auto t0 = std::chrono::high_resolution_clock::now();

        llama_batch batch = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[runner] decode failed on prompt\n");
            fputs(eot_marker ? "<<<EOT>>>\n" : "\n", stdout);
            continue;
        }

        std::string out;
        std::vector<int> gen_ids;
        int n_past = (int) prompt.size();

        for (int t = 0; t < n_predict; ++t) {
            const float* logits = llama_get_logits_ith(ctx, -1);
            llama_token best = 0;
            float best_v = logits[0];
            for (int v = 1; v < n_vocab; ++v)
                if (logits[v] > best_v) { best_v = logits[v]; best = v; }

            if (t == 0 && debug_topk > 0) {
                std::vector<int> idx(n_vocab);
                for (int v = 0; v < n_vocab; ++v) idx[v] = v;
                std::partial_sort(idx.begin(), idx.begin() + debug_topk, idx.end(),
                                  [&](int a, int b) { return logits[a] > logits[b]; });
                fprintf(stderr, "[top%d]", debug_topk);
                for (int k = 0; k < debug_topk; ++k)
                    fprintf(stderr, "  %d(%.3f)'%s'", idx[k], logits[idx[k]],
                            piece(idx[k]).c_str());
                fprintf(stderr, "\n");
            }

            if (llama_vocab_is_eog(vocab, best) && !ignore_eog) break;

            gen_ids.push_back(best);
            out += piece(best);

            if (++n_past >= n_ctx) break;
            llama_batch nb = llama_batch_get_one(&best, 1);
            if (llama_decode(ctx, nb) != 0) break;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        total_tok += (long long) gen_ids.size();
        total_s   += secs;

        if (print_ids) {
            std::string s;
            for (size_t k = 0; k < gen_ids.size(); ++k)
                s += (k ? " " : "") + std::to_string(gen_ids[k]);
            fprintf(stderr, "[ids] %s\n", s.c_str());
        }

        fputs(out.c_str(), stdout);
        fputs(eot_marker ? "\n<<<EOT>>>\n" : "\n", stdout);
        fflush(stdout);
    }

    if (total_s > 0)
        fprintf(stderr, "[runner] %lld tokens in %.2fs (%.1f tok/s)\n",
                total_tok, total_s, total_tok / total_s);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
