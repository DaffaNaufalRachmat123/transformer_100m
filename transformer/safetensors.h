#pragma once
// =============================================================================
// SafeTensors I/O  (subset implementation)
// Supports BF16 and F32 tensors — sufficient for training + export.
//
// Format spec: https://huggingface.co/docs/safetensors/index
//   bytes 0-7   : uint64_le  N = length of JSON header
//   bytes 8..8+N: UTF-8 JSON  {"tensor_name": {"dtype":"BF16","shape":[…],"data_offsets":[s,e]}, …}
//   bytes 8+N.. : raw tensor data (tightly packed, no alignment gaps)
//
// Usage:
//   // Save
//   SafeTensorsWriter w("model.safetensors");
//   w.add_tensor("embed.weight", TensorDesc::BF16, {65536, 1152}, ptr);
//   w.finalize();
//
//   // Load
//   SafeTensorsLoader r("model.safetensors");
//   auto desc = r.tensor("embed.weight");
//   r.read_into_f32("embed.weight", out_ptr);
// =============================================================================

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Dtype enum
// ---------------------------------------------------------------------------
enum class STDtype : uint8_t {
    F32  = 0,
    BF16 = 1,
    F16  = 2,
    I32  = 3,
    I8   = 4,
};

inline size_t dtype_bytes(STDtype d) {
    switch (d) {
        case STDtype::F32:  return 4;
        case STDtype::BF16: return 2;
        case STDtype::F16:  return 2;
        case STDtype::I32:  return 4;
        case STDtype::I8:   return 1;
    }
    return 4;
}

inline std::string dtype_str(STDtype d) {
    switch (d) {
        case STDtype::F32:  return "F32";
        case STDtype::BF16: return "BF16";
        case STDtype::F16:  return "F16";
        case STDtype::I32:  return "I32";
        case STDtype::I8:   return "I8";
    }
    return "F32";
}

inline STDtype dtype_from_str(const std::string& s) {
    if (s == "BF16") return STDtype::BF16;
    if (s == "F16")  return STDtype::F16;
    if (s == "I32")  return STDtype::I32;
    if (s == "I8")   return STDtype::I8;
    return STDtype::F32;
}

// ---------------------------------------------------------------------------
// BF16 ↔ F32 conversion utilities
// ---------------------------------------------------------------------------
inline float bf16_to_f32(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    // Round to nearest even
    uint32_t rounding_bias = 0x7fff + ((u >> 16) & 1);
    u += rounding_bias;
    return (uint16_t)(u >> 16);
}

// Convert array BF16→F32 in-place into dst (dst must have numel floats)
inline void bf16_to_f32_array(const uint16_t* src, float* dst, size_t numel) {
    for (size_t i = 0; i < numel; ++i)
        dst[i] = bf16_to_f32(src[i]);
}

// F32 → FP16 conversion (IEEE 754 half-precision)
inline uint16_t f32_to_f16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t exp = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (u >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)sign;           // underflow → 0
    if (exp >= 31) return (uint16_t)(sign | 0x7C00); // overflow → inf
    return (uint16_t)(sign | (exp << 10) | mant);
}

// FP16 → F32 conversion (IEEE 754 half-precision)
inline float f16_to_f32(uint16_t v) {
    uint32_t sign     = (v >> 15) & 0x1;
    uint32_t exponent = (v >> 10) & 0x1F;
    uint32_t mantissa = v & 0x3FF;
    if (exponent == 0)
        return (sign ? -1.0f : 1.0f) * (mantissa / 1024.0f) * 6.103515625e-5f;
    if (exponent == 31) {
        uint32_t bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
    uint32_t bits = (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// Convert array F16→F32
inline void f16_to_f32_array(const uint16_t* src, float* dst, size_t numel) {
    for (size_t i = 0; i < numel; ++i)
        dst[i] = f16_to_f32(src[i]);
}

// Convert array F32→BF16 into dst (dst must have numel uint16_t)
inline void f32_to_bf16_array(const float* src, uint16_t* dst, size_t numel) {
    for (size_t i = 0; i < numel; ++i)
        dst[i] = f32_to_bf16(src[i]);
}

// ---------------------------------------------------------------------------
// Tensor descriptor (metadata only, no data ownership)
// ---------------------------------------------------------------------------
struct TensorDesc {
    std::string name;
    STDtype     dtype = STDtype::F32;
    std::vector<uint32_t> shape;
    uint64_t    data_offset_start = 0;
    uint64_t    data_offset_end   = 0;

    size_t numel() const {
        if (shape.empty()) return 0;
        size_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }

    size_t byte_size() const { return numel() * dtype_bytes(dtype); }
};

// ---------------------------------------------------------------------------
// Minimal JSON emitter (no dependencies)
// ---------------------------------------------------------------------------
namespace detail {

inline std::string json_string(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else out += c;
    }
    out += '"';
    return out;
}

inline std::string shape_to_json(const std::vector<uint32_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

// Minimal JSON parser — parses only the SafeTensors header structure.
// Returns map: name → TensorDesc (shape + dtype + offsets)
// Does NOT validate full JSON — only the subset SafeTensors uses.
struct MiniParser {
    const char* p;
    const char* end;

    MiniParser(const char* data, size_t len) : p(data), end(data + len) {}

    void skip_ws() { while (p < end && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r')) ++p; }

    char peek() { skip_ws(); return (p < end) ? *p : 0; }
    char consume() { skip_ws(); return (p < end) ? *p++ : 0; }

    void expect(char c) {
        char got = consume();
        if (got != c) throw std::runtime_error(std::string("JSON parse: expected '") + c + "' got '" + got + "'");
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (p < end && *p != '"') {
            if (*p == '\\') { ++p; out += *p; }
            else out += *p;
            ++p;
        }
        expect('"');
        return out;
    }

    int64_t parse_int() {
        skip_ws();
        int64_t v = 0;
        bool neg = false;
        if (*p == '-') { neg = true; ++p; }
        while (p < end && *p >= '0' && *p <= '9') { v = v*10 + (*p-'0'); ++p; }
        return neg ? -v : v;
    }

    std::vector<uint32_t> parse_array_int() {
        expect('[');
        std::vector<uint32_t> arr;
        if (peek() == ']') { ++p; return arr; }
        do {
            skip_ws();
            if (*p == ',') ++p;
            skip_ws();
            arr.push_back((uint32_t)parse_int());
        } while (peek() == ',');
        expect(']');
        return arr;
    }

    std::map<std::string, TensorDesc> parse_header() {
        std::map<std::string, TensorDesc> result;
        expect('{');
        while (peek() != '}') {
            std::string key = parse_string();
            expect(':');
            if (key == "__metadata__") {
                // skip metadata — may be null (MLX) or {} (HF)
                char c = peek();
                if (c == 'n') {
                    // null — consume 4 chars
                    p += 4;
                } else {
                    expect('{');
                    int depth = 1;
                    while (p < end && depth > 0) {
                        if (*p == '{') ++depth;
                        else if (*p == '}') --depth;
                        ++p;
                    }
                }
            } else {
                TensorDesc td;
                td.name = key;
                expect('{');
                while (peek() != '}') {
                    std::string field = parse_string();
                    expect(':');
                    if (field == "dtype") {
                        td.dtype = dtype_from_str(parse_string());
                    } else if (field == "shape") {
                        td.shape = parse_array_int();
                    } else if (field == "data_offsets") {
                        expect('[');
                        td.data_offset_start = (uint64_t)parse_int();
                        skip_ws(); if (peek()==',') ++p;
                        td.data_offset_end   = (uint64_t)parse_int();
                        expect(']');
                    } else {
                        // skip unknown value
                        char c = peek();
                        if (c == '"') parse_string();
                        else if (c == '[') { parse_array_int(); }
                        else { while(p<end && *p!=',' && *p!='}') ++p; }
                    }
                    skip_ws();
                    if (peek() == ',') { ++p; }
                }
                expect('}');
                result[key] = td;
            }
            skip_ws();
            if (peek() == ',') ++p;
        }
        expect('}');
        return result;
    }
};

} // namespace detail

// =============================================================================
// SafeTensorsWriter
// =============================================================================
class SafeTensorsWriter {
public:
    struct PendingTensor {
        std::string            name;
        STDtype                dtype;
        std::vector<uint32_t>  shape;
        const void*            data;   // BF16 or F32 data in the requested dtype
    };

    void add_tensor(const std::string& name,
                    STDtype dtype,
                    std::vector<uint32_t> shape,
                    const void* data) {
        pending_.push_back({name, dtype, std::move(shape), data});
    }

    // Add F32 tensor, auto-convert to BF16 before writing
    void add_tensor_f32_as_bf16(const std::string& name,
                                 std::vector<uint32_t> shape,
                                 const float* data,
                                 std::vector<uint16_t>& scratch) {
        size_t n = 1;
        for (auto d : shape) n *= d;
        scratch.resize(n);
        f32_to_bf16_array(data, scratch.data(), n);
        add_tensor(name, STDtype::BF16, shape, scratch.data());
    }

    void finalize(const std::string& path) {
        // Build header JSON
        uint64_t offset = 0;
        std::string header = "{";
        bool first = true;
        for (const auto& t : pending_) {
            size_t nb = 1;
            for (auto d : t.shape) nb *= d;
            nb *= dtype_bytes(t.dtype);

            if (!first) header += ",";
            first = false;
            header += detail::json_string(t.name) + ":{";
            header += "\"dtype\":\"" + dtype_str(t.dtype) + "\",";
            header += "\"shape\":" + detail::shape_to_json(t.shape) + ",";
            header += "\"data_offsets\":[" + std::to_string(offset) + ","
                      + std::to_string(offset + nb) + "]";
            header += "}";
            offset += nb;
        }
        header += "}";

        // Pad header to 64-byte alignment
        while ((8 + header.size()) % 64 != 0) header += ' ';

        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("Cannot open for writing: " + path);

        uint64_t hlen = header.size();
        f.write(reinterpret_cast<const char*>(&hlen), 8);
        f.write(header.data(), hlen);

        for (const auto& t : pending_) {
            size_t nb = 1;
            for (auto d : t.shape) nb *= d;
            nb *= dtype_bytes(t.dtype);
            f.write(reinterpret_cast<const char*>(t.data), nb);
        }
    }

private:
    std::vector<PendingTensor> pending_;
};

// =============================================================================
// SafeTensorsLoader
// =============================================================================
class SafeTensorsLoader {
public:
    explicit SafeTensorsLoader(const std::string& path) {
        f_.open(path, std::ios::binary);
        if (!f_) throw std::runtime_error("Cannot open: " + path);

        uint64_t hlen = 0;
        f_.read(reinterpret_cast<char*>(&hlen), 8);
        if (!f_ || hlen == 0 || hlen > 100 * 1024 * 1024)
            throw std::runtime_error("Invalid safetensors header length");

        std::string header_json(hlen, '\0');
        f_.read(header_json.data(), hlen);
        if (!f_) throw std::runtime_error("Truncated header");

        data_base_offset_ = 8 + hlen;

        detail::MiniParser parser(header_json.data(), hlen);
        tensors_ = parser.parse_header();
    }

    bool has_tensor(const std::string& name) const {
        return tensors_.count(name) > 0;
    }

    const TensorDesc& tensor(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end())
            throw std::runtime_error("Tensor not found: " + name);
        return it->second;
    }

    // Read tensor data into pre-allocated buffer.
    // If tensor dtype == BF16 and dst_dtype == F32, auto-converts.
    // If tensor dtype == F32 and dst_dtype == BF16, auto-converts.
    // Otherwise copies raw bytes.
    void read_into_f32(const std::string& name, float* dst) {
        const TensorDesc& td = tensor(name);
        uint64_t file_offset = data_base_offset_ + td.data_offset_start;
        f_.seekg(file_offset);

        if (td.dtype == STDtype::F32) {
            f_.read(reinterpret_cast<char*>(dst), td.byte_size());
        } else if (td.dtype == STDtype::BF16) {
            std::vector<uint16_t> tmp(td.numel());
            f_.read(reinterpret_cast<char*>(tmp.data()), td.byte_size());
            bf16_to_f32_array(tmp.data(), dst, td.numel());
        } else if (td.dtype == STDtype::F16) {
            std::vector<uint16_t> tmp(td.numel());
            f_.read(reinterpret_cast<char*>(tmp.data()), td.byte_size());
            f16_to_f32_array(tmp.data(), dst, td.numel());
        } else {
            throw std::runtime_error("read_into_f32: unsupported dtype for " + name);
        }
        if (!f_) throw std::runtime_error("Read error for tensor: " + name);
    }

    // Read tensor into FP16 (__half) buffer. Converts from BF16/F32 if needed.
    void read_into_fp16(const std::string& name, void* dst) {
        const TensorDesc& td = tensor(name);
        uint64_t file_offset = data_base_offset_ + td.data_offset_start;
        f_.seekg(file_offset);

        if (td.dtype == STDtype::F16) {
            f_.read(reinterpret_cast<char*>(dst), td.byte_size());
        } else if (td.dtype == STDtype::BF16) {
            std::vector<uint16_t> tmp(td.numel());
            f_.read(reinterpret_cast<char*>(tmp.data()), td.byte_size());
            // BF16 → F32 → F16
            uint16_t* out = reinterpret_cast<uint16_t*>(dst);
            for (size_t i = 0; i < td.numel(); i++) {
                float v = bf16_to_f32(tmp[i]);
                out[i] = f32_to_f16(v);
            }
        } else if (td.dtype == STDtype::F32) {
            std::vector<float> tmp(td.numel());
            f_.read(reinterpret_cast<char*>(tmp.data()), td.byte_size());
            uint16_t* out = reinterpret_cast<uint16_t*>(dst);
            for (size_t i = 0; i < td.numel(); i++) {
                out[i] = f32_to_f16(tmp[i]);
            }
        } else {
            throw std::runtime_error("read_into_fp16: unsupported dtype for " + name);
        }
        if (!f_) throw std::runtime_error("Read error for tensor: " + name);
    }

    // List all tensor names
    std::vector<std::string> tensor_names() const {
        std::vector<std::string> names;
        for (const auto& kv : tensors_) names.push_back(kv.first);
        return names;
    }

    // Get tensor shape as uint64_t vector
    std::vector<uint64_t> tensor_shape(const std::string& name) const {
        const TensorDesc& td = tensor(name);
        std::vector<uint64_t> s;
        for (auto d : td.shape) s.push_back(d);
        return s;
    }

    // Load tensor data as F32 vector
    std::vector<float> load_tensor_f32(const std::string& name) {
        const TensorDesc& td = tensor(name);
        std::vector<float> data(td.numel());
        read_into_f32(name, data.data());
        return data;
    }

    void print_info(std::ostream& os = std::cout) const {
        os << "SafeTensors: " << tensors_.size() << " tensors\n";
        size_t total = 0;
        for (const auto& kv : tensors_) {
            const auto& td = kv.second;
            os << "  " << td.name << "  dtype=" << dtype_str(td.dtype)
               << "  shape=[";
            for (size_t i = 0; i < td.shape.size(); ++i) {
                if (i) os << ",";
                os << td.shape[i];
            }
            os << "]  bytes=" << td.byte_size() << "\n";
            total += td.byte_size();
        }
        os << "  TOTAL: " << (total / 1024.0 / 1024.0) << " MB\n";
    }

private:
    std::ifstream f_;
    uint64_t      data_base_offset_ = 0;
    std::map<std::string, TensorDesc> tensors_;
};