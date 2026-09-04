// =============================================================================
// llama_jni.cpp — JNI bridge for the trilingual 204M GGUF model.
//
// Mirrors tools/llama_runner.cpp: prompts arrive ALREADY TOKENIZED (the model's
// BPE is not GPT-2 compatible, so llama.cpp cannot encode them). Kotlin does the
// tokenization via BpeTokenizer.kt and passes an int[] of token ids.
//
// Build with the NDK against llama.cpp (see mobile/README.md).
// =============================================================================

#include <jni.h>
#include <android/log.h>

#include "llama.h"

#include <algorithm>
#include <string>
#include <vector>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "LlamaJNI", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "LlamaJNI", __VA_ARGS__)

namespace {

struct Session {
    llama_model*   model = nullptr;
    llama_context* ctx   = nullptr;
    const llama_vocab* vocab = nullptr;
    int n_ctx = 512;
    int n_past = 0;
};

inline Session* as_session(jlong h) { return reinterpret_cast<Session*>(h); }

} // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_com_example_trilingual_LlamaBridge_nativeInit(JNIEnv*, jobject) {
    llama_backend_init();
}

JNIEXPORT jlong JNICALL
Java_com_example_trilingual_LlamaBridge_nativeLoad(
        JNIEnv* env, jobject, jstring jpath, jint n_ctx, jint n_threads) {

    const char* path = env->GetStringUTFChars(jpath, nullptr);

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;                 // CPU; ARM NEON/dotprod kernels

    llama_model* model = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!model) { LOGE("model load failed"); return 0; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) n_ctx;
    cp.n_batch         = (uint32_t) n_ctx;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    // int8 KV cache: ~4x smaller than f32, negligible quality cost at this size
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;

    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { LOGE("context creation failed"); llama_model_free(model); return 0; }

    auto* s = new Session();
    s->model = model;
    s->ctx   = ctx;
    s->vocab = llama_model_get_vocab(model);
    s->n_ctx = n_ctx;
    LOGI("loaded: n_ctx=%d threads=%d vocab=%d",
         n_ctx, n_threads, llama_vocab_n_tokens(s->vocab));
    return reinterpret_cast<jlong>(s);
}

/** Clear KV cache and ingest the prompt. Returns 0 on success. */
JNIEXPORT jint JNICALL
Java_com_example_trilingual_LlamaBridge_nativeBeginPrompt(
        JNIEnv* env, jobject, jlong handle, jintArray jids) {

    Session* s = as_session(handle);
    if (!s) return -1;

    const jsize n = env->GetArrayLength(jids);
    if (n <= 0) return -1;

    std::vector<llama_token> toks(n);
    {
        jint* raw = env->GetIntArrayElements(jids, nullptr);
        for (jsize i = 0; i < n; ++i) toks[i] = (llama_token) raw[i];
        env->ReleaseIntArrayElements(jids, raw, JNI_ABORT);
    }
    if ((int) toks.size() >= s->n_ctx) toks.resize(s->n_ctx - 1);

    llama_memory_clear(llama_get_memory(s->ctx), true);
    s->n_past = 0;

    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    if (llama_decode(s->ctx, batch) != 0) { LOGE("prompt decode failed"); return -2; }
    s->n_past = (int) toks.size();
    return 0;
}

/**
 * Greedy next token. Returns the token id, or -1 when the context is full.
 * Sampling (temperature/top-p) is intentionally left to Kotlin so product
 * tuning does not require rebuilding native code.
 */
JNIEXPORT jint JNICALL
Java_com_example_trilingual_LlamaBridge_nativeNextToken(
        JNIEnv*, jobject, jlong handle) {

    Session* s = as_session(handle);
    if (!s || s->n_past >= s->n_ctx) return -1;

    const int n_vocab = llama_vocab_n_tokens(s->vocab);
    const float* logits = llama_get_logits_ith(s->ctx, -1);

    llama_token best = 0;
    float best_v = logits[0];
    for (int v = 1; v < n_vocab; ++v)
        if (logits[v] > best_v) { best_v = logits[v]; best = v; }

    llama_batch nb = llama_batch_get_one(&best, 1);
    if (llama_decode(s->ctx, nb) != 0) return -1;
    s->n_past++;
    return (jint) best;
}

JNIEXPORT jboolean JNICALL
Java_com_example_trilingual_LlamaBridge_nativeIsEog(
        JNIEnv*, jobject, jlong handle, jint token) {
    Session* s = as_session(handle);
    if (!s) return JNI_TRUE;
    return llama_vocab_is_eog(s->vocab, (llama_token) token) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_example_trilingual_LlamaBridge_nativeFree(JNIEnv*, jobject, jlong handle) {
    Session* s = as_session(handle);
    if (!s) return;
    if (s->ctx)   llama_free(s->ctx);
    if (s->model) llama_model_free(s->model);
    delete s;
}

} // extern "C"
