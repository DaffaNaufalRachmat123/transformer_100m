#include "model_config.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

// =============================================================================
// Minimal JSON writer/parser — no external dependencies.
// Supports the flat key-value config format used by ModelConfig::save/load.
// Values are: uint32_t, float, bool, string, or string-array (ignored on load).
// =============================================================================

// ---------------------------------------------------------------------------
// Writer: builds a pretty-printed JSON object string
// ---------------------------------------------------------------------------
struct JsonWriter {
    std::ostringstream buf;
    bool first = true;

    void begin() { buf << "{\n"; }
    void end()   { buf << "\n}"; }

    void sep() {
        if (!first) buf << ",\n";
        first = false;
        buf << "  ";
    }

    void write(const std::string& key, uint32_t v) {
        sep(); buf << "\"" << key << "\": " << v;
    }
    void write(const std::string& key, float v) {
        sep();
        buf << "\"" << key << "\": ";
        // Write full precision so round-trip is lossless
        std::ostringstream tmp;
        tmp << std::setprecision(9) << v;
        buf << tmp.str();
    }
    void write(const std::string& key, bool v) {
        sep(); buf << "\"" << key << "\": " << (v ? "true" : "false");
    }
    void write(const std::string& key, const std::string& v) {
        sep(); buf << "\"" << key << "\": \"" << v << "\"";
    }
    // String array — written for readability, ignored on load
    void write_array(const std::string& key,
                     std::initializer_list<const char*> vals) {
        sep(); buf << "\"" << key << "\": [";
        bool f = true;
        for (const char* s : vals) {
            if (!f) buf << ", ";
            buf << "\"" << s << "\"";
            f = false;
        }
        buf << "]";
    }

    std::string str() const { return buf.str(); }
};

// ---------------------------------------------------------------------------
// Parser: reads a flat JSON object written by JsonWriter.
// Strips whitespace/quotes, splits on ':', handles bool/number/string values.
// ---------------------------------------------------------------------------
struct JsonReader {
    std::unordered_map<std::string, std::string> kv;

    // Trim leading/trailing whitespace and optional surrounding quotes
    static std::string trim(const std::string& s) {
        size_t a = 0, b = s.size();
        while (a < b && (s[a] == ' ' || s[a] == '\t' ||
                         s[a] == '\n' || s[a] == '\r')) ++a;
        while (b > a && (s[b-1] == ' ' || s[b-1] == '\t' ||
                         s[b-1] == '\n' || s[b-1] == '\r')) --b;
        std::string t = s.substr(a, b - a);
        if (t.size() >= 2 && t.front() == '"' && t.back() == '"')
            t = t.substr(1, t.size() - 2);
        return t;
    }

    void parse(std::istream& in) {
        std::string line;
        while (std::getline(in, line)) {
            // Skip lines that don't contain a colon (braces, array lines, etc.)
            auto colon = line.find(':');
            if (colon == std::string::npos) continue;

            std::string key = trim(line.substr(0, colon));
            if (key.empty()) continue;

            // Value: everything after ':', up to optional trailing comma
            std::string val = line.substr(colon + 1);
            // Drop trailing comma
            while (!val.empty() && (val.back() == ',' ||
                                     val.back() == ' ' ||
                                     val.back() == '\n' ||
                                     val.back() == '\r'))
                val.pop_back();
            // Skip array values (start with '[')
            std::string vtrim = trim(val);
            if (!vtrim.empty() && vtrim.front() == '[') continue;

            kv[key] = vtrim;
        }
    }

    // Typed accessors with defaults
    uint32_t get_u32(const std::string& k, uint32_t def) const {
        auto it = kv.find(k);
        if (it == kv.end()) return def;
        try { return (uint32_t)std::stoul(it->second); } catch (...) { return def; }
    }
    float get_float(const std::string& k, float def) const {
        auto it = kv.find(k);
        if (it == kv.end()) return def;
        try { return std::stof(it->second); } catch (...) { return def; }
    }
    bool get_bool(const std::string& k, bool def) const {
        auto it = kv.find(k);
        if (it == kv.end()) return def;
        return it->second == "true";
    }
};

// =============================================================================
// ModelConfig implementations
// =============================================================================

void ModelConfig::print() const {
    std::cout << "\n=== Model Config ===\n";
    std::cout << std::left << std::setw(20) << "vocab_size"   << vocab_size   << "\n";
    std::cout << std::left << std::setw(20) << "d_model"      << d_model      << "\n";
    std::cout << std::left << std::setw(20) << "n_layers"     << n_layers     << "\n";
    std::cout << std::left << std::setw(20) << "n_q_heads"    << n_q_heads    << "\n";
    std::cout << std::left << std::setw(20) << "n_kv_heads"   << n_kv_heads   << "\n";
    std::cout << std::left << std::setw(20) << "d_head"       << d_head       << "\n";
    std::cout << std::left << std::setw(20) << "ffn_dim"      << ffn_dim      << "\n";
    std::cout << std::left << std::setw(20) << "max_seq_len"  << max_seq_len  << "\n";
    std::cout << std::left << std::setw(20) << "rope_theta"   << rope_theta   << "\n";
    std::cout << std::left << std::setw(20) << "weight_tying" << (weight_tying ? "true" : "false") << "\n";

    uint64_t params = param_count();
    double params_m = params / 1e6;
    double int4_mb  = params * 0.5 / (1024.0 * 1024.0);

    std::cout << "\n--- Parameter Estimate ---\n";
    std::cout << std::left << std::setw(20) << "total_params"
              << std::fixed << std::setprecision(1) << params_m << "M\n";
    std::cout << std::left << std::setw(20) << "INT4_size_MB"
              << std::fixed << std::setprecision(1) << int4_mb << " MB\n";
    std::cout << "=========================================\n\n";
}

void ModelConfig::save(const std::string& path) const {
    JsonWriter j;
    j.begin();
    j.write("vocab_size",   vocab_size);
    j.write("d_model",      d_model);
    j.write("n_layers",     n_layers);
    j.write("n_q_heads",    n_q_heads);
    j.write("n_kv_heads",   n_kv_heads);
    j.write("d_head",       d_head);
    j.write("ffn_dim",      ffn_dim);
    j.write("max_seq_len",  max_seq_len);
    j.write("rope_theta",   rope_theta);
    j.write("weight_decay", weight_decay);
    j.write("grad_clip",    grad_clip);
    j.write("weight_tying", weight_tying);
    j.write("pad_token_id", pad_token_id);
    j.write("eos_token_id", eos_token_id);
    j.write("bos_token_id", bos_token_id);
    j.write("architecture",
            std::to_string(param_count() / 1000000) + "m_gqa_rope_swiglu");
    j.write_array("languages", {"en", "ja"});
    j.end();

    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot write config: " + path);
    f << j.str();
    std::cout << "[Config] Saved: " << path << "\n";
}

ModelConfig ModelConfig::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open config: " + path);

    JsonReader r;
    r.parse(f);

    // Missing keys fall back to the struct defaults in model_config.h rather
    // than to literals here, so there is a single source of truth for the arch.
    ModelConfig cfg;
    cfg.vocab_size   = r.get_u32  ("vocab_size",   cfg.vocab_size);
    cfg.d_model      = r.get_u32  ("d_model",      cfg.d_model);
    cfg.n_layers     = r.get_u32  ("n_layers",     cfg.n_layers);
    cfg.n_q_heads    = r.get_u32  ("n_q_heads",    cfg.n_q_heads);
    cfg.n_kv_heads   = r.get_u32  ("n_kv_heads",   cfg.n_kv_heads);
    cfg.d_head       = r.get_u32  ("d_head",       cfg.d_head);
    cfg.ffn_dim      = r.get_u32  ("ffn_dim",      cfg.ffn_dim);
    cfg.max_seq_len  = r.get_u32  ("max_seq_len",  cfg.max_seq_len);
    cfg.rope_theta   = r.get_float("rope_theta",   cfg.rope_theta);
    cfg.weight_decay = r.get_float("weight_decay", cfg.weight_decay);
    cfg.grad_clip    = r.get_float("grad_clip",    cfg.grad_clip);
    cfg.weight_tying = r.get_bool ("weight_tying", cfg.weight_tying);
    cfg.pad_token_id = r.get_u32  ("pad_token_id", cfg.pad_token_id);
    cfg.eos_token_id = r.get_u32  ("eos_token_id", cfg.eos_token_id);
    cfg.bos_token_id = r.get_u32  ("bos_token_id", cfg.bos_token_id);
    return cfg;
}

ModelConfig ModelConfig::default_100m() {
    return ModelConfig{}; // all defaults already set in struct
}
