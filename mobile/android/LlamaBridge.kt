package com.example.trilingual

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.io.File

/** Thin Kotlin wrapper over llama_jni.cpp. Not thread-safe: use one per thread. */
class LlamaBridge {

    private var handle: Long = 0L

    private external fun nativeInit()
    private external fun nativeLoad(path: String, nCtx: Int, nThreads: Int): Long
    private external fun nativeBeginPrompt(handle: Long, ids: IntArray): Int
    private external fun nativeNextToken(handle: Long): Int
    private external fun nativeIsEog(handle: Long, token: Int): Boolean
    private external fun nativeFree(handle: Long)

    companion object {
        init { System.loadLibrary("trilingual") }

        /** Big cores only; more threads than that usually hurts on phones. */
        fun recommendedThreads(): Int =
            (Runtime.getRuntime().availableProcessors() / 2).coerceIn(2, 4)
    }

    fun load(modelFile: File, nCtx: Int = 512, nThreads: Int = recommendedThreads()) {
        require(modelFile.exists()) { "model not found: ${modelFile.absolutePath}" }
        nativeInit()
        handle = nativeLoad(modelFile.absolutePath, nCtx, nThreads)
        check(handle != 0L) { "failed to load ${modelFile.name}" }
    }

    fun beginPrompt(ids: IntArray) {
        check(handle != 0L) { "model not loaded" }
        val rc = nativeBeginPrompt(handle, ids)
        check(rc == 0) { "prompt decode failed (rc=$rc)" }
    }

    fun nextToken(): Int = nativeNextToken(handle)

    fun isEog(token: Int): Boolean = nativeIsEog(handle, token)

    fun close() {
        if (handle != 0L) { nativeFree(handle); handle = 0L }
    }
}

/**
 * End-to-end assistant: our BPE for encoding, llama.cpp for the math.
 *
 * The chat template must match SFT exactly or quality collapses:
 *   <|system|>{system}<|user|>{user}<|assistant|>
 */
class TrilingualAssistant(
    private val tokenizer: BpeTokenizer,
    private val llama: LlamaBridge,
) {
    companion object {
        const val SYS_FACT_EN = "You are a helpful, accurate multilingual assistant."
        const val SYS_FACT_JA = "あなたは多言語AIアシスタントです。親切で正直で正確に回答します。"
        const val SYS_TRANSLATE_TO_JA = "You are a helpful translation engine. Output only the translation."
        const val SYS_TRANSLATE_TO_EN = "You are a multilingual translation assistant."

        fun create(context: Context, modelName: String): TrilingualAssistant {
            val tok = BpeTokenizer(
                context.assets.open("vocab.txt"),
                context.assets.open("merges.txt"),
            )
            // mmap needs a real file, so assets are copied into filesDir once.
            val model = File(context.filesDir, modelName)
            if (!model.exists()) {
                context.assets.open(modelName).use { input ->
                    model.outputStream().use { input.copyTo(it) }
                }
            }
            val bridge = LlamaBridge().apply { load(model) }
            return TrilingualAssistant(tok, bridge)
        }
    }

    fun buildPrompt(system: String, user: String) = "<|system|>$system<|user|>$user<|assistant|>"

    fun askFact(question: String, japanese: Boolean = false) =
        generate(buildPrompt(if (japanese) SYS_FACT_JA else SYS_FACT_EN, question))

    fun translateToJapanese(text: String) =
        generate(buildPrompt(SYS_TRANSLATE_TO_JA, "日本語に翻訳: $text"))

    fun translateToEnglish(text: String) =
        generate(buildPrompt(SYS_TRANSLATE_TO_EN, "Translate to English: $text"))

    /** Streams decoded text pieces as they are produced. */
    fun generate(prompt: String, maxTokens: Int = 64): Flow<String> = flow {
        llama.beginPrompt(tokenizer.encode(prompt))
        repeat(maxTokens) {
            val id = llama.nextToken()
            if (id < 0 || llama.isEog(id)) return@flow
            val piece = tokenizer.decodeToken(id)
            if (piece.isNotEmpty()) emit(piece)
        }
    }.flowOn(Dispatchers.Default)

    fun close() = llama.close()
}
