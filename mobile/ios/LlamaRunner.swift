import Foundation
import llama   // llama.xcframework

/// Swift wrapper over llama.cpp for the trilingual 204M GGUF model.
///
/// Prompts arrive ALREADY TOKENIZED (see `BpeTokenizer`) because the model's BPE
/// is not GPT-2 compatible and llama.cpp cannot encode for it.
public final class LlamaRunner {

    private var model: OpaquePointer?
    private var ctx: OpaquePointer?
    private var vocab: OpaquePointer?
    private let nCtx: Int32
    private var nPast: Int32 = 0

    public enum LlamaError: Error {
        case modelLoadFailed(String)
        case contextCreationFailed
        case decodeFailed
    }

    /// - Parameter nThreads: performance cores only; more usually hurts.
    public init(modelURL: URL,
                nCtx: Int32 = 512,
                nThreads: Int32 = Int32(max(2, min(4, ProcessInfo.processInfo.activeProcessorCount / 2))))
    throws {
        self.nCtx = nCtx
        llama_backend_init()

        var mp = llama_model_default_params()
        // Metal is a win on A-series; set to 0 to force CPU.
        mp.n_gpu_layers = 99

        guard let m = llama_model_load_from_file(modelURL.path, mp) else {
            throw LlamaError.modelLoadFailed(modelURL.lastPathComponent)
        }
        model = m
        vocab = UnsafeMutableRawPointer(mutating: llama_model_get_vocab(m))
            .assumingMemoryBound(to: llama_vocab.self)

        var cp = llama_context_default_params()
        cp.n_ctx = UInt32(nCtx)
        cp.n_batch = UInt32(nCtx)
        cp.n_threads = nThreads
        cp.n_threads_batch = nThreads
        cp.type_k = GGML_TYPE_Q8_0     // int8 KV cache: ~4x smaller than f32
        cp.type_v = GGML_TYPE_Q8_0

        guard let c = llama_init_from_model(m, cp) else {
            llama_model_free(m)
            throw LlamaError.contextCreationFailed
        }
        ctx = c
    }

    deinit {
        if let c = ctx { llama_free(c) }
        if let m = model { llama_model_free(m) }
        llama_backend_free()
    }

    /// Clear KV cache and ingest the prompt.
    public func beginPrompt(_ ids: [Int32]) throws {
        guard let ctx else { throw LlamaError.decodeFailed }
        var toks = ids
        if toks.count >= Int(nCtx) { toks = Array(toks[0..<(Int(nCtx) - 1)]) }

        llama_memory_clear(llama_get_memory(ctx), true)
        nPast = 0

        let rc = toks.withUnsafeMutableBufferPointer { buf -> Int32 in
            let batch = llama_batch_get_one(buf.baseAddress, Int32(buf.count))
            return llama_decode(ctx, batch)
        }
        guard rc == 0 else { throw LlamaError.decodeFailed }
        nPast = Int32(toks.count)
    }

    /// Greedy next token, or nil when the context is full.
    /// Sampling is left to Swift so tuning needs no native rebuild.
    public func nextToken() -> Int32? {
        guard let ctx, let vocab, nPast < nCtx else { return nil }

        let nVocab = llama_vocab_n_tokens(vocab)
        guard let logits = llama_get_logits_ith(ctx, -1) else { return nil }

        var best: Int32 = 0
        var bestV = logits[0]
        for v in 1..<Int(nVocab) where logits[v] > bestV {
            bestV = logits[v]; best = Int32(v)
        }

        var one = best
        let rc = withUnsafeMutablePointer(to: &one) { p -> Int32 in
            llama_decode(ctx, llama_batch_get_one(p, 1))
        }
        guard rc == 0 else { return nil }
        nPast += 1
        return best
    }

    public func isEndOfGeneration(_ token: Int32) -> Bool {
        guard let vocab else { return true }
        return llama_vocab_is_eog(vocab, token)
    }
}

/// End-to-end assistant: our BPE for encoding, llama.cpp for the math.
public final class TrilingualAssistant {

    public static let sysFactEN = "You are a helpful, accurate multilingual assistant."
    public static let sysFactJA = "あなたは多言語AIアシスタントです。親切で正直で正確に回答します。"
    public static let sysTranslateToJA = "You are a helpful translation engine. Output only the translation."
    public static let sysTranslateToEN = "You are a multilingual translation assistant."

    private let tokenizer: BpeTokenizer
    private let runner: LlamaRunner

    public init(modelURL: URL, vocabURL: URL, mergesURL: URL, nCtx: Int32 = 512) throws {
        tokenizer = try BpeTokenizer(vocabURL: vocabURL, mergesURL: mergesURL)
        runner = try LlamaRunner(modelURL: modelURL, nCtx: nCtx)
    }

    /// Must match the SFT template exactly or quality collapses.
    public func buildPrompt(system: String, user: String) -> String {
        "<|system|>\(system)<|user|>\(user)<|assistant|>"
    }

    public func askFact(_ question: String, japanese: Bool = false) -> AsyncStream<String> {
        generate(buildPrompt(system: japanese ? Self.sysFactJA : Self.sysFactEN, user: question))
    }

    public func translateToJapanese(_ text: String) -> AsyncStream<String> {
        generate(buildPrompt(system: Self.sysTranslateToJA, user: "日本語に翻訳: \(text)"))
    }

    public func translateToEnglish(_ text: String) -> AsyncStream<String> {
        generate(buildPrompt(system: Self.sysTranslateToEN, user: "Translate to English: \(text)"))
    }

    /// Streams decoded text pieces as they are produced.
    public func generate(_ prompt: String, maxTokens: Int = 64) -> AsyncStream<String> {
        AsyncStream { continuation in
            Task.detached(priority: .userInitiated) { [tokenizer, runner] in
                do {
                    try runner.beginPrompt(tokenizer.encode(prompt))
                } catch {
                    continuation.finish(); return
                }
                for _ in 0..<maxTokens {
                    guard let id = runner.nextToken(), !runner.isEndOfGeneration(id) else { break }
                    let piece = tokenizer.decodeToken(id)
                    if !piece.isEmpty { continuation.yield(piece) }
                }
                continuation.finish()
            }
        }
    }
}
