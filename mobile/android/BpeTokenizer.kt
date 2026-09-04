package com.example.trilingual

import java.io.InputStream

/**
 * Port of bpe/bpe_tokenizer.cpp (BPEEncoder).
 *
 * This model's BPE is NOT GPT-2 compatible, so llama.cpp cannot encode prompts
 * for it. Encode here, then pass the token ids to LlamaBridge.
 *
 * Differences from classic BPE, all deliberate (they match the trained model):
 *  1. Pre-tokenizer splits on runs of bytes <= 0x20 and COLLAPSES them. Only the
 *     first word of a segment lacks the 'Ġ' prefix.
 *  2. Word seeding is greedy longest-match over up to 8 BYTES.
 *  3. Merges are repeated left-to-right passes, not lowest-rank-first.
 *  4. '<|...|>' specials are split out first.
 *  5. merges.txt lines starting with '#' are skipped (matches the C++ loader,
 *     which silently drops 42 real merges).
 *
 * Bytes are held as ISO-8859-1 strings so that 1 char == 1 byte, which makes the
 * byte-level longest-match cheap and hashable.
 */
class BpeTokenizer(vocabStream: InputStream, mergesStream: InputStream) {

    private val tokens: List<String>
    private val tokenToId: HashMap<String, Int>
    /** byte-view (ISO-8859-1) -> id, for the longest-match seeding pass */
    private val byteKeyToId: HashMap<String, Int>
    private val merges: HashMap<Long, Int>

    val bosId: Int
    val eosId: Int
    val padId: Int
    private val nSpecial = 16

    companion object {
        private const val MAX_MATCH_BYTES = 8
        private const val MAX_WORD_BYTES = 512

        /** GPT-2 bytes_to_unicode(); index = byte value, value = glyph. */
        val BYTE_ENCODER: Array<String> = buildByteEncoder()
        val BYTE_DECODER: HashMap<Char, Int> = HashMap<Char, Int>().apply {
            BYTE_ENCODER.forEachIndexed { b, g -> this[g[0]] = b }
        }

        private fun buildByteEncoder(): Array<String> {
            val bs = ArrayList<Int>()
            (0x21..0x7E).forEach { bs.add(it) }
            (0xA1..0xAC).forEach { bs.add(it) }
            (0xAE..0xFF).forEach { bs.add(it) }
            val cs = ArrayList(bs)
            var n = 0
            for (b in 0..255) if (!bs.contains(b)) { bs.add(b); cs.add(256 + n); n++ }
            val out = arrayOfNulls<String>(256)
            for (i in bs.indices) out[bs[i]] = cs[i].toChar().toString()
            @Suppress("UNCHECKED_CAST")
            return out as Array<String>
        }

        private fun key(a: Int, b: Int): Long = (a.toLong() shl 32) or (b.toLong() and 0xFFFFFFFFL)
    }

    init {
        tokens = vocabStream.bufferedReader(Charsets.UTF_8).readLines()
        tokenToId = HashMap(tokens.size * 2)
        byteKeyToId = HashMap(tokens.size * 2)
        tokens.forEachIndexed { i, t ->
            tokenToId[t] = i
            byteKeyToId[String(t.toByteArray(Charsets.UTF_8), Charsets.ISO_8859_1)] = i
        }

        merges = HashMap(70000)
        mergesStream.bufferedReader(Charsets.UTF_8).forEachLine { line ->
            if (line.isEmpty() || line[0] == '#') return@forEachLine
            val sp = line.indexOf(' ')
            if (sp < 0) return@forEachLine
            val l = tokenToId[line.substring(0, sp)] ?: return@forEachLine
            val r = tokenToId[line.substring(sp + 1)] ?: return@forEachLine
            val m = tokenToId[line.substring(0, sp) + line.substring(sp + 1)] ?: return@forEachLine
            merges[key(l, r)] = m
        }

        bosId = tokenToId["<|endoftext|>"] ?: 0
        eosId = tokenToId["<|end|>"] ?: 4
        padId = tokenToId["<|pad|>"] ?: 15
    }

    /** Stage 1: whitespace pre-tokenizer over raw bytes. */
    private fun pretokenize(seg: ByteArray, from: Int, to: Int): List<String> {
        val out = ArrayList<String>()
        var p = from
        var first = true
        while (p < to) {
            while (p < to && (seg[p].toInt() and 0xFF) <= 0x20) p++
            if (p >= to) break
            val start = p
            while (p < to && (seg[p].toInt() and 0xFF) > 0x20) p++
            val sb = StringBuilder()
            if (!first) sb.append(BYTE_ENCODER[0x20])
            for (i in start until p) sb.append(BYTE_ENCODER[seg[i].toInt() and 0xFF])
            first = false
            out.add(sb.toString())
        }
        return out
    }

    /** Stage 2: longest-match seeding, then left-to-right merge passes. */
    private fun encodeWord(word: String, out: MutableList<Int>) {
        var wb = String(word.toByteArray(Charsets.UTF_8), Charsets.ISO_8859_1)
        if (wb.length > MAX_WORD_BYTES) wb = wb.substring(0, MAX_WORD_BYTES)

        var seq = ArrayList<Int>(wb.length)
        var pos = 0
        while (pos < wb.length) {
            var hit = false
            var len = minOf(wb.length - pos, MAX_MATCH_BYTES)
            while (len >= 1) {
                val id = byteKeyToId[wb.substring(pos, pos + len)]
                if (id != null) { seq.add(id); pos += len; hit = true; break }
                len--
            }
            if (!hit) pos++              // C++ skips the unmatched byte
        }

        var changed = true
        while (changed && seq.size > 1) {
            changed = false
            val next = ArrayList<Int>(seq.size)
            var i = 0
            while (i < seq.size) {
                if (i + 1 < seq.size) {
                    val m = merges[key(seq[i], seq[i + 1])]
                    if (m != null) { next.add(m); i += 2; changed = true; continue }
                }
                next.add(seq[i]); i++
            }
            seq = next
        }
        out.addAll(seq)
    }

    /** Stage 3: split '<|...|>' specials, encode the rest. */
    fun encode(text: String, addBos: Boolean = false, addEos: Boolean = false): IntArray {
        val res = ArrayList<Int>(text.length / 2 + 8)
        if (addBos) res.add(bosId)

        val bb = text.toByteArray(Charsets.UTF_8)
        val end = bb.size
        var p = 0
        val lt = '<'.code.toByte(); val pipe = '|'.code.toByte(); val gt = '>'.code.toByte()

        while (p < end) {
            if (end - p >= 3 && bb[p] == lt && bb[p + 1] == pipe) {
                val scanEnd = if (end - p < 128) end else p + 128
                var spEnd = -1
                var q = p + 2
                while (q + 1 < scanEnd) {
                    if (bb[q] == pipe && bb[q + 1] == gt) { spEnd = q + 2; break }
                    q++
                }
                if (spEnd > 0) {
                    val id = tokenToId[String(bb, p, spEnd - p, Charsets.UTF_8)]
                    if (id != null) { res.add(id); p = spEnd; continue }
                }
            }
            val segStart = p
            while (p < end) {
                if (end - p >= 2 && bb[p] == lt && bb[p + 1] == pipe) break
                p++
            }
            if (p > segStart) {
                for (w in pretokenize(bb, segStart, p)) encodeWord(w, res)
            } else {
                p += 2                   // unmatched '<|', avoid infinite loop
            }
        }

        if (addEos) res.add(eosId)
        return res.toIntArray()
    }

    /** Map ids back to text (byte-level decode). */
    fun decode(ids: IntArray, skipSpecial: Boolean = true): String {
        val sb = StringBuilder()
        for (id in ids) {
            if (id < 0 || id >= tokens.size) continue
            if (skipSpecial && id < nSpecial) continue
            sb.append(tokens[id])
        }
        val bytes = ByteArray(sb.length)
        var n = 0
        for (c in sb) BYTE_DECODER[c]?.let { bytes[n++] = it.toByte() }
        return String(bytes, 0, n, Charsets.UTF_8)
    }

    /** Single token -> text, for streaming output. */
    fun decodeToken(id: Int): String = decode(intArrayOf(id))

    fun isSpecial(id: Int): Boolean = id in 0 until nSpecial
}
