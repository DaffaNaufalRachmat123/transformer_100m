// =============================================================================
// infer — CLI inference tool for the Quadrilingual 476M LLM
//
// Supports three modes:
//   1. Chat mode (default): interactive REPL with chat format
//   2. --prompt "text"    : single-shot generation, prints output then exits
//   3. --stdin            : read prompts line-by-line from stdin
//
// Usage:
//   ./infer --model  checkpoints/smoke_sft/sft_final.safetensors \
//           --config checkpoints/smoke_pretrain/config_final.json \
//           --vocab  vocab_6 \
//           --prompt "<|system|>You are helpful.<|user|>Hello<|assistant|>"
//
// Generation flags:
//   --max_tokens N      max new tokens to generate        (default: 200)
//   --temperature F     sampling temperature               (default: 0.8)
//   --top_p F           nucleus sampling cutoff            (default: 0.9)
//   --top_k N           top-k sampling                     (default: 50)
//   --rep_penalty F     repetition penalty >1 reduces repeats (default: 1.3)
//   --rep_window N      look-back window for rep penalty   (default: 64)
//   --greedy            greedy decoding (temperature=0, no rep penalty)
//   --system "text"     system prompt for chat mode
//   --chat_format       auto-wrap prompt in <|system|>...<|user|>...<|assistant|>
//
// NOTE: Output "always 256 tokens" happens because the model hasn't learned
//   to emit <|end|> yet (requires more training). Use --rep_penalty 1.3-1.5
//   to at least break repetition loops in under-trained models.
// =============================================================================

#include "../transformer/transformer.h"
#include "../transformer/model_config.h"
#include "../bpe/bpe_tokenizer.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <iomanip>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " [options]\n"
        << "\n"
        << "Required:\n"
        << "  --model  PATH    path to .safetensors model (TRNS format)\n"
        << "  --config PATH    path to config.json\n"
        << "  --vocab  DIR     path to vocab directory (contains vocab.txt + merges.txt)\n"
        << "\n"
        << "Input:\n"
        << "  --prompt TEXT    single prompt, generate and exit\n"
        << "  --stdin          read prompts from stdin (one per line)\n"
        << "  (no flag)        interactive chat REPL\n"
        << "\n"
        << "Generation:\n"
        << "  --max_tokens N   max new tokens  (default: 200)\n"
        << "  --temperature F  temperature     (default: 0.8)\n"
        << "  --top_p F        top-p cutoff    (default: 0.9)\n"
        << "  --top_k N        top-k           (default: 50)\n"
        << "  --rep_penalty F  repetition penalty; >1 penalises repeated tokens (default: 1.3)\n"
        << "  --rep_window N   look-back window for rep penalty (default: 64)\n"
        << "  --greedy         greedy decoding (disables sampling + rep penalty)\n"
        << "\n"
        << "Chat:\n"
        << "  --chat_format    wrap prompt as <|user|>PROMPT<|assistant|>\n"
        << "  --system TEXT    system prompt text (implies --chat_format)\n"
        << "\n"
        << "Note on 'always N tokens' output:\n"
        << "  Under-trained models never emit <|end|> so generation runs to --max_tokens.\n"
        << "  Use --rep_penalty 1.3-1.5 to break repetition loops.\n"
        << "  After sufficient training the model will stop naturally.\n"
        << "\n"
        << "Examples:\n"
        << "  # Single shot with repetition penalty\n"
        << "  ./infer --model sft.safetensors --config config.json --vocab vocab_6 \\\n"
        << "          --prompt \"<|system|>You are helpful.<|user|>What is Python?<|assistant|>\" \\\n"
        << "          --max_tokens 150 --rep_penalty 1.3\n"
        << "\n"
        << "  # Interactive chat\n"
        << "  ./infer --model dpo.safetensors --config config.json --vocab vocab_6 \\\n"
        << "          --system \"Kamu adalah asisten yang membantu.\"\n";
}

static void stream_generate(
        Transformer&          model,
        const BPEEncoder&     enc,
        const std::vector<uint32_t>& prompt_ids,
        const Transformer::GenerateConfig& gcfg,
        float rep_penalty  = 1.3f,   // >1 penalises recently-seen tokens
        int   rep_window   = 64) {   // how many recent tokens to look back

    model.reset_kv_cache();

    // Process prompt tokens (fill KV cache, no output)
    std::vector<float> logits;
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        model.forward_one_token(logits, prompt_ids[i], (int)i);
    }

    // Generate new tokens
    int kv_pos = (int)prompt_ids.size();
    std::vector<uint32_t> generated;
    std::vector<uint32_t> recent;   // sliding window for rep penalty
    auto t_start = std::chrono::high_resolution_clock::now();

    // Hard-block threshold: if a non-special token appears >= this many times
    // in the window, zero out its probability entirely.
    // This handles degenerate distributions from under-trained models.
    const int HARD_BLOCK_N = 3;

    for (int step = 0; step < gcfg.max_new_tokens; ++step) {
        uint32_t next_tok;



        if (gcfg.temperature <= 0.0f) {
            // Greedy
            next_tok = (uint32_t)std::distance(
                logits.begin(), std::max_element(logits.begin(), logits.end()));
        } else {
            // Count frequencies in recent window
            std::unordered_map<uint32_t, int> freq;
            for (uint32_t t : recent) freq[t]++;

            // Apply repetition penalty to logits before softmax
            if (rep_penalty > 1.0f) {
                for (auto& [tok, cnt] : freq) {
                    if (tok < (uint32_t)logits.size()) {
                        float& l = logits[tok];
                        float p = rep_penalty + (cnt - 1) * 0.2f;  // stronger for higher freq
                        l = (l > 0) ? l / p : l * p;
                    }
                }
            }

            // Temperature + softmax
            std::vector<float> probs(logits.size());
            float max_l = *std::max_element(logits.begin(), logits.end());
            float sum = 0.0f;
            for (size_t i = 0; i < logits.size(); ++i) {
                probs[i] = expf((logits[i] - max_l) / gcfg.temperature);
                sum += probs[i];
            }
            for (float& p : probs) p /= sum;
            

    // Hard-block: zero out any non-special token seen >= HARD_BLOCK_N times
    bool any_blocked = false;
    for (auto& [tok, cnt] : freq) {
        if (cnt >= HARD_BLOCK_N && tok >= 16 && tok < (uint32_t)probs.size()) {
            probs[tok] = 0.0f;
            any_blocked = true;
        }
    }
    // ALWAYS block the last generated token to force at least some diversity
    if (!recent.empty() && recent.back() >= 16 && recent.back() < (uint32_t)probs.size()) {
        probs[recent.back()] = 0.0f;
        any_blocked = true;
    }
    if (any_blocked) {
        float s = 0; for (float p : probs) s += p;
        if (s > 1e-8f) for (float& p : probs) p /= s;
    }

            // Top-K
            if (gcfg.top_k > 0 && gcfg.top_k < (int)probs.size()) {
                std::vector<int> idx(probs.size());
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + gcfg.top_k, idx.end(),
                                  [&](int a, int b){ return probs[a] > probs[b]; });
                float s_topk = 0;
                for (int i = 0; i < gcfg.top_k; ++i) s_topk += probs[idx[i]];
                std::vector<float> topk_vals(gcfg.top_k);
                for (int i = 0; i < gcfg.top_k; ++i) topk_vals[i] = probs[idx[i]];
                std::fill(probs.begin(), probs.end(), 0.0f);
                if (s_topk > 1e-8f) for (int i = 0; i < gcfg.top_k; ++i) probs[idx[i]] = topk_vals[i] / s_topk;
            }

            // Top-P (nucleus)
            if (gcfg.top_p < 1.0f) {
                std::vector<std::pair<float,int>> sp;
                sp.reserve(probs.size());
                for (size_t i = 0; i < probs.size(); ++i)
                    if (probs[i] > 0) sp.push_back({probs[i], (int)i});
                std::sort(sp.begin(), sp.end(), [](auto& a, auto& b){ return a.first > b.first; });
                float cum = 0;
                for (auto& [p, idx] : sp) {
                    cum += p;
                    if (cum > gcfg.top_p) probs[idx] = 0.0f;
                }
                float s = 0; for (float p : probs) s += p;
                if (s > 1e-8f) for (float& p : probs) p /= s;
            }

            // Multinomial sample
            static std::mt19937 rng(std::random_device{}());
            std::uniform_real_distribution<float> u(0.0f, 1.0f);
            float r = u(rng), cum = 0.0f;
            next_tok = (uint32_t)(probs.size() - 1);
            for (size_t i = 0; i < probs.size(); ++i) {
                cum += probs[i];
                if (r <= cum) { next_tok = (uint32_t)i; break; }
            }
        }

        // Stop on EOS
        if (next_tok == gcfg.eos_id) break;

        generated.push_back(next_tok);

        // Update sliding repetition window
        recent.push_back(next_tok);
        if ((int)recent.size() > rep_window)
            recent.erase(recent.begin());

        // Decode and stream this token
        std::string piece = enc.decode({next_tok}, false);
        std::cout << piece << std::flush;

        // Next forward step
        model.forward_one_token(logits, next_tok, kv_pos++);
    }

    std::cout << "\n";

    // Timing
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    double tps = (elapsed > 0) ? generated.size() / elapsed : 0;
    std::cerr << "[infer] " << generated.size() << " tokens in "
              << std::fixed << std::setprecision(2) << elapsed << "s  ("
              << std::setprecision(1) << tps << " tok/s)\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    // ---- Parse args --------------------------------------------------------
    std::string model_path, config_path, vocab_dir;
    std::string prompt_text, system_text;
    bool use_stdin     = false;
    bool chat_format   = false;
    bool show_help     = false;
    float rep_penalty  = 1.3f;
    int   rep_window   = 64;

    Transformer::GenerateConfig gcfg;
    gcfg.max_new_tokens = 200;    // default 200; model stops early if it emits <|end|>
    gcfg.temperature    = 0.8f;
    gcfg.top_p          = 0.9f;
    gcfg.top_k          = 50;
    gcfg.eos_id         = 4;   // <|end|>

    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if      (key == "--help"  || key == "-h")          show_help = true;
        else if (key == "--model"       && i+1 < argc) model_path  = argv[++i];
        else if (key == "--config"      && i+1 < argc) config_path = argv[++i];
        else if (key == "--vocab"       && i+1 < argc) vocab_dir   = argv[++i];
        else if (key == "--prompt"      && i+1 < argc) prompt_text = argv[++i];
        else if (key == "--system"      && i+1 < argc) { system_text = argv[++i]; chat_format = true; }
        else if (key == "--max_tokens"  && i+1 < argc) gcfg.max_new_tokens = std::stoi(argv[++i]);
        else if (key == "--temperature" && i+1 < argc) gcfg.temperature    = std::stof(argv[++i]);
        else if (key == "--top_p"       && i+1 < argc) gcfg.top_p          = std::stof(argv[++i]);
        else if (key == "--top_k"       && i+1 < argc) gcfg.top_k          = std::stoi(argv[++i]);
        else if (key == "--rep_penalty" && i+1 < argc) rep_penalty         = std::stof(argv[++i]);
        else if (key == "--rep_window"  && i+1 < argc) rep_window          = std::stoi(argv[++i]);
        else if (key == "--greedy")   { gcfg.temperature = 0.0f; rep_penalty = 1.0f; }
        else if (key == "--chat_format") chat_format = true;
        else if (key == "--stdin")    use_stdin = true;
    }

    if (show_help) { print_usage(argv[0]); return 0; }

    // ---- Validate required args -------------------------------------------
    if (model_path.empty() || config_path.empty() || vocab_dir.empty()) {
        std::cerr << "[infer] ERROR: --model, --config, --vocab are required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // ---- Load model --------------------------------------------------------
    std::cerr << "[infer] Loading config: " << config_path << "\n";
    ModelConfig cfg = ModelConfig::load(config_path);

    std::cerr << "[infer] Loading vocab: " << vocab_dir << "\n";
    BPEEncoder enc((vocab_dir + "/vocab.txt").c_str(),
                   (vocab_dir + "/merges.txt").c_str());

    // Look up special token IDs
    uint32_t sys_id  = enc.special_id("<|system|>");
    uint32_t usr_id  = enc.special_id("<|user|>");
    uint32_t ast_id  = enc.special_id("<|assistant|>");
    uint32_t end_id  = enc.special_id("<|end|>");
    gcfg.eos_id = end_id;

    std::cerr << "[infer] Loading model: " << model_path << "\n";
    Transformer model(cfg);
    model.load_safetensors(model_path);
    std::cerr << "[infer] Model ready. (vocab=" << cfg.vocab_size
              << " d=" << cfg.d_model << " L=" << cfg.n_layers << ")\n\n";

    // ---- Helper: build prompt ids from text --------------------------------
    auto build_prompt_ids = [&](const std::string& user_input) -> std::vector<uint32_t> {
        if (!chat_format) {
            // Raw encode — user controls format manually
            return enc.encode(user_input, false, false);
        }
        // Chat format: <|system|>SYS<|user|>USER<|assistant|>
        std::string full;
        if (!system_text.empty() && sys_id != UINT32_MAX) {
            full += "<|system|>" + system_text;
        }
        if (usr_id != UINT32_MAX)
            full += "<|user|>";
        full += user_input;
        if (ast_id != UINT32_MAX)
            full += "<|assistant|>";
        return enc.encode(full, false, false);
    };

    // ---- Mode: single --prompt ---------------------------------------------
    if (!prompt_text.empty() && !use_stdin) {
        auto ids = build_prompt_ids(prompt_text);
        stream_generate(model, enc, ids, gcfg, rep_penalty, rep_window);
        return 0;
    }

    // ---- Mode: --stdin -----------------------------------------------------
    if (use_stdin) {
        std::string line;
        int idx = 0;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            std::cerr << "[infer] Prompt " << ++idx << ": " << line.substr(0, 60)
                      << (line.size() > 60 ? "..." : "") << "\n";
            auto ids = build_prompt_ids(line);
            stream_generate(model, enc, ids, gcfg, rep_penalty, rep_window);
        }
        return 0;
    }

    // ---- Mode: interactive chat REPL ---------------------------------------
    std::cerr << "============================================================\n";
    std::cerr << "  Quadrilingual 476M LLM — Interactive Chat\n";
    std::cerr << "  Model: " << model_path << "\n";
    if (!system_text.empty())
        std::cerr << "  System: " << system_text << "\n";
    std::cerr << "  Commands: /quit  /reset  /greedy  /sample  /temp F\n";
    std::cerr << "============================================================\n\n";

    std::string line;
    int turn = 0;
    while (true) {
        std::cerr << "User> " << std::flush;
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

        // In-chat commands
        if (line == "/quit" || line == "/exit") break;
        if (line == "/reset") {
            model.reset_kv_cache();
            std::cerr << "[reset conversation]\n";
            turn = 0;
            continue;
        }
        if (line == "/greedy")  { gcfg.temperature = 0.0f;  std::cerr << "[greedy mode]\n"; continue; }
        if (line == "/sample")  { gcfg.temperature = 0.8f;  std::cerr << "[sampling mode]\n"; continue; }
        if (line.substr(0, 6) == "/temp ") {
            gcfg.temperature = std::stof(line.substr(6));
            std::cerr << "[temperature = " << gcfg.temperature << "]\n";
            continue;
        }

        auto ids = build_prompt_ids(line);
        std::cerr << "Assistant> ";
        stream_generate(model, enc, ids, gcfg, rep_penalty, rep_window);
        ++turn;
    }

    std::cerr << "\n[bye]\n";
    return 0;
}
