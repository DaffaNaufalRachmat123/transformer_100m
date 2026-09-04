import Foundation

/// Port of bpe/bpe_tokenizer.cpp (BPEEncoder).
///
/// This model's BPE is NOT GPT-2 compatible, so llama.cpp cannot encode prompts
/// for it. Encode here, then hand the ids to `LlamaRunner`.
///
/// Deliberate deviations from classic BPE (they match the trained model):
///  1. Pre-tokenizer splits on runs of bytes <= 0x20 and COLLAPSES them; only
///     the first word of a segment lacks the "Ġ" prefix.
///  2. Word seeding is greedy longest-match over up to 8 BYTES.
///  3. Merges are repeated left-to-right passes, not lowest-rank-first.
///  4. "<|...|>" specials are split out first.
///  5. merges.txt lines starting with '#' are skipped, matching the C++ loader
///     (which silently drops 42 real merges).
public final class BpeTokenizer {

    private let tokens: [String]
    private var tokenToId: [String: Int] = [:]
    /// byte-view -> id, used by the longest-match seeding pass
    private var byteKeyToId: [[UInt8]: Int] = [:]
    private var merges: [UInt64: Int] = [:]

    public let bosId: Int
    public let eosId: Int
    public let padId: Int
    private let nSpecial = 16

    private static let maxMatchBytes = 8
    private static let maxWordBytes = 512

    /// GPT-2 bytes_to_unicode(): index = byte value, value = glyph.
    public static let byteEncoder: [String] = {
        var bs: [Int] = Array(0x21...0x7E) + Array(0xA1...0xAC) + Array(0xAE...0xFF)
        var cs = bs
        var n = 0
        let present = Set(bs)
        for b in 0...255 where !present.contains(b) {
            bs.append(b); cs.append(256 + n); n += 1
        }
        var out = [String](repeating: "", count: 256)
        for i in 0..<bs.count { out[bs[i]] = String(UnicodeScalar(cs[i])!) }
        return out
    }()

    public static let byteDecoder: [Character: UInt8] = {
        var m: [Character: UInt8] = [:]
        for (b, g) in byteEncoder.enumerated() { m[Character(g)] = UInt8(b) }
        return m
    }()

    @inline(__always)
    private static func key(_ a: Int, _ b: Int) -> UInt64 {
        (UInt64(UInt32(a)) << 32) | UInt64(UInt32(b))
    }

    public init(vocabURL: URL, mergesURL: URL) throws {
        let vocabText = try String(contentsOf: vocabURL, encoding: .utf8)
        var lines = vocabText.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }
        tokens = lines

        tokenToId.reserveCapacity(tokens.count * 2)
        byteKeyToId.reserveCapacity(tokens.count * 2)
        for (i, t) in tokens.enumerated() {
            tokenToId[t] = i
            byteKeyToId[Array(t.utf8)] = i
        }

        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        merges.reserveCapacity(70000)
        for line in mergesText.split(separator: "\n", omittingEmptySubsequences: true) {
            if line.first == "#" { continue }
            guard let sp = line.firstIndex(of: " ") else { continue }
            let l = String(line[line.startIndex..<sp])
            let r = String(line[line.index(after: sp)...])
            guard let li = tokenToId[l], let ri = tokenToId[r],
                  let mi = tokenToId[l + r] else { continue }
            merges[Self.key(li, ri)] = mi
        }

        bosId = tokenToId["<|endoftext|>"] ?? 0
        eosId = tokenToId["<|end|>"] ?? 4
        padId = tokenToId["<|pad|>"] ?? 15
    }

    /// Stage 1: whitespace pre-tokenizer over raw bytes.
    private func pretokenize(_ b: [UInt8], _ from: Int, _ to: Int) -> [[UInt8]] {
        var out: [[UInt8]] = []
        var p = from
        var first = true
        while p < to {
            while p < to && b[p] <= 0x20 { p += 1 }
            if p >= to { break }
            let start = p
            while p < to && b[p] > 0x20 { p += 1 }
            var word: [UInt8] = []
            if !first { word.append(contentsOf: Array(Self.byteEncoder[0x20].utf8)) }
            for i in start..<p { word.append(contentsOf: Array(Self.byteEncoder[Int(b[i])].utf8)) }
            first = false
            out.append(word)
        }
        return out
    }

    /// Stage 2: longest-match seeding, then left-to-right merge passes.
    private func encodeWord(_ wordIn: [UInt8], into out: inout [Int]) {
        let wb = wordIn.count > Self.maxWordBytes
            ? Array(wordIn[0..<Self.maxWordBytes]) : wordIn

        var seq: [Int] = []
        seq.reserveCapacity(wb.count)
        var pos = 0
        while pos < wb.count {
            var hit = false
            var len = min(wb.count - pos, Self.maxMatchBytes)
            while len >= 1 {
                if let id = byteKeyToId[Array(wb[pos..<(pos + len)])] {
                    seq.append(id); pos += len; hit = true; break
                }
                len -= 1
            }
            if !hit { pos += 1 }          // C++ skips the unmatched byte
        }

        var changed = true
        while changed && seq.count > 1 {
            changed = false
            var next: [Int] = []
            next.reserveCapacity(seq.count)
            var i = 0
            while i < seq.count {
                if i + 1 < seq.count, let m = merges[Self.key(seq[i], seq[i + 1])] {
                    next.append(m); i += 2; changed = true; continue
                }
                next.append(seq[i]); i += 1
            }
            seq = next
        }
        out.append(contentsOf: seq)
    }

    /// Stage 3: split "<|...|>" specials, encode the rest.
    public func encode(_ text: String, addBos: Bool = false, addEos: Bool = false) -> [Int32] {
        var res: [Int] = []
        if addBos { res.append(bosId) }

        let bb = Array(text.utf8)
        let end = bb.count
        var p = 0
        let lt: UInt8 = 0x3C, pipe: UInt8 = 0x7C, gt: UInt8 = 0x3E

        while p < end {
            if end - p >= 3 && bb[p] == lt && bb[p + 1] == pipe {
                let scanEnd = (end - p < 128) ? end : p + 128
                var spEnd = -1
                var q = p + 2
                while q + 1 < scanEnd {
                    if bb[q] == pipe && bb[q + 1] == gt { spEnd = q + 2; break }
                    q += 1
                }
                if spEnd > 0,
                   let s = String(bytes: bb[p..<spEnd], encoding: .utf8),
                   let id = tokenToId[s] {
                    res.append(id); p = spEnd; continue
                }
            }
            let segStart = p
            while p < end {
                if end - p >= 2 && bb[p] == lt && bb[p + 1] == pipe { break }
                p += 1
            }
            if p > segStart {
                for w in pretokenize(bb, segStart, p) { encodeWord(w, into: &res) }
            } else {
                p += 2                    // unmatched "<|", avoid infinite loop
            }
        }

        if addEos { res.append(eosId) }
        return res.map { Int32($0) }
    }

    /// Map ids back to text (byte-level decode).
    public func decode(_ ids: [Int32], skipSpecial: Bool = true) -> String {
        var glyphs = ""
        for id in ids {
            let i = Int(id)
            if i < 0 || i >= tokens.count { continue }
            if skipSpecial && i < nSpecial { continue }
            glyphs += tokens[i]
        }
        var bytes: [UInt8] = []
        bytes.reserveCapacity(glyphs.count)
        for c in glyphs { if let b = Self.byteDecoder[c] { bytes.append(b) } }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// Single token -> text, for streaming output.
    public func decodeToken(_ id: Int32) -> String { decode([id]) }
}
