// =============================================================================
// filter_audit_v9_posttrain.cpp — Exact C++ BPEEncoder (bpe_v9.bin) filter &
// auditor mirroring train_cuda_unified.cu's load_sft_data, load_dpo_data, and
// load_rl_data byte-for-byte. Filters JSONL files in-place to guarantee 0%
// skipped/truncated/empty-mask rows on GPU!
// =============================================================================
#include "../bpe/bpe_tokenizer.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static constexpr int TOK_ASSISTANT = 3;
static constexpr int TOK_END       = 4;

static std::string extract_json_string(const std::string& line, const std::string& key) {
    std::string search = "\"" + key + "\":";
    auto pos = line.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size();
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) pos++;
    if (pos >= line.size() || line[pos] != '"') return "";
    pos++;
    std::string result;
    for (; pos < line.size() && line[pos] != '"'; pos++) {
        if (line[pos] == '\\' && pos + 1 < line.size()) {
            pos++;
            if (line[pos] == 'n') result += '\n';
            else if (line[pos] == 't') result += '\t';
            else if (line[pos] == 'r') result += '\r';
            else if (line[pos] == '"') result += '"';
            else if (line[pos] == '\\') result += '\\';
            else { result += '\\'; result += line[pos]; }
        } else {
            result += line[pos];
        }
    }
    return result;
}

static size_t count_assistant_mask(const std::vector<TokenID>& tokens, bool& any_end) {
    bool in_assistant = false;
    size_t masked = 0;
    any_end = false;
    for (size_t i = 0; i < tokens.size(); i++) {
        if ((int)tokens[i] == TOK_ASSISTANT) {
            in_assistant = true;
            masked++;
        } else if ((int)tokens[i] == TOK_END) {
            in_assistant = false;
            any_end = true;
        } else if (in_assistant) {
            masked++;
        }
    }
    return masked;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <bpe_v9.bin> <sft|dpo|rl> <jsonl_path> [max_seq]\n", argv[0]);
        return 1;
    }
    std::string bpe_path = argv[1];
    std::string mode     = argv[2];
    std::string path     = argv[3];
    int max_seq          = (argc >= 5) ? std::stoi(argv[4]) : (mode == "rl" ? 360 : 1000);

    BPEEncoder tok;
    if (!tok.load(bpe_path)) {
        fprintf(stderr, "[ERR] Cannot load %s\n", bpe_path.c_str());
        return 1;
    }

    std::ifstream in(path);
    std::string tmp_path = path + ".filtered_tmp";
    std::ofstream out(tmp_path);

    std::string line;
    size_t kept = 0, dropped_len = 0, dropped_mask = 0;
    uint64_t total_toks = 0, total_masked = 0;
    int max_seen = 0;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (mode == "sft") {
            std::string text = extract_json_string(line, "text");
            if (text.empty()) { dropped_mask++; continue; }
            bool has_end = text.size() >= 7 && text.compare(text.size() - 7, 7, SpecialTokens::END) == 0;
            auto ids = tok.encode(text, true, !has_end);
            if ((int)ids.size() > max_seq) { dropped_len++; continue; }
            bool any_end = false;
            size_t m = count_assistant_mask(ids, any_end);
            if (m == 0 || !any_end) { dropped_mask++; continue; }
            out << line << "\n";
            kept++;
            total_toks += ids.size();
            total_masked += m;
            if ((int)ids.size() > max_seen) max_seen = (int)ids.size();
        } else if (mode == "dpo") {
            std::string sys_t = extract_json_string(line, "system");
            std::string prm_t = extract_json_string(line, "prompt");
            std::string cho_t = extract_json_string(line, "chosen");
            std::string rej_t = extract_json_string(line, "rejected");
            if (prm_t.empty() || cho_t.empty() || rej_t.empty()) { dropped_mask++; continue; }
            if (cho_t == rej_t || cho_t.rfind(rej_t, 0) == 0 || rej_t.rfind(cho_t, 0) == 0) {
                dropped_mask++;
                continue;
            }
            std::string pfx = "<|system|>" + sys_t + "<|user|>" + prm_t + "<|assistant|>";
            auto c_ids = tok.encode(pfx + cho_t + "<|end|>", true, false);
            auto r_ids = tok.encode(pfx + rej_t + "<|end|>", true, false);
            if ((int)c_ids.size() > max_seq || (int)r_ids.size() > max_seq) { dropped_len++; continue; }
            bool ce = false, re = false;
            size_t cm = count_assistant_mask(c_ids, ce);
            size_t rm = count_assistant_mask(r_ids, re);
            if (cm == 0 || rm == 0) { dropped_mask++; continue; }
            out << line << "\n";
            kept++;
            total_toks += c_ids.size() + r_ids.size();
            total_masked += cm + rm;
            int mx = std::max((int)c_ids.size(), (int)r_ids.size());
            if (mx > max_seen) max_seen = mx;
        } else if (mode == "rl") {
            std::string prm_t = extract_json_string(line, "prompt");
            std::string exp_t = extract_json_string(line, "expected_output");
            std::string tst_t = extract_json_string(line, "tests");
            std::string sys_t = extract_json_string(line, "system");
            if (prm_t.empty() || (exp_t.empty() && tst_t.empty())) { dropped_mask++; continue; }
            if (sys_t.empty()) sys_t = "You are a coding assistant.";
            std::string pfx = "<|system|>" + sys_t + "<|user|>" + prm_t + "<|assistant|>";
            auto ids = tok.encode(pfx, true, false);
            if ((int)ids.size() > max_seq) { dropped_len++; continue; }
            out << line << "\n";
            kept++;
            total_toks += ids.size();
            if ((int)ids.size() > max_seen) max_seen = (int)ids.size();
        }
    }
    in.close();
    out.close();
    std::rename(tmp_path.c_str(), path.c_str());

    printf("[CPP-BPE-AUDIT %s] %s: kept=%zu | dropped_len=%zu | dropped_bad=%zu | max_tok=%d | total_tok=%.2fM | masked_tok=%.2fM\n",
           mode.c_str(), path.c_str(), kept, dropped_len, dropped_mask, max_seen,
           total_toks / 1e6, total_masked / 1e6);
    return 0;
}
