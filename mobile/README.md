# Menjalankan model di Android (Kotlin) & iOS (Swift)

Artefak: `gguf/trilingual-204m-facts-Q4_K_M.gguf` (130 MB) + `vocab_6/vocab.txt`
+ `vocab_6/merges.txt`.

## ⚠️ Hal terpenting: tokenizer TIDAK boleh pakai llama.cpp

BPE model ini bukan GPT-2/HuggingFace standar. Perbedaannya (diverifikasi dari
`bpe/bpe_tokenizer.cpp`):

1. Pre-tokenizer memecah pada deretan byte `<= 0x20` dan **menggabungkannya**;
   hanya kata pertama tiap segmen yang tanpa prefiks `Ġ`.
2. Seeding kata = **longest-match serakah hingga 8 BYTE**, bukan per karakter.
3. Merge = **pass kiri-ke-kanan berulang**, bukan lowest-rank-first.
4. `<|...|>` dipisah lebih dulu.
5. Baris `merges.txt` berawalan `#` dibuang loader C++ — ikut membuang 42 merge sah.

Bukti nyata perbedaannya:

| | `What is the capital of Japan?` |
|---|---|
| Tokenizer llama.cpp | `9031 3598 3376 29091 3434 15797 79` (7 token) |
| Tokenizer model ini | `9031 3598 3376 60285 3352 3434 15797 79` (8 token) |

Karena itu alurnya: **encode di Kotlin/Swift → kirim token id ke llama.cpp**.
Detokenisasi tetap kompatibel, tapi port di sini sudah menanganinya sendiri.

Validasi port dengan `mobile/tokenizer_test_vectors.json` — wajib 100% sama.

---

## Android (Kotlin + NDK)

### 1. Struktur

```
app/src/main/
├── assets/
│   ├── trilingual-204m-facts-Q4_K_M.gguf
│   ├── vocab.txt        # dari vocab_6/
│   └── merges.txt       # dari vocab_6/
├── java/com/example/trilingual/
│   ├── BpeTokenizer.kt  # mobile/android/
│   └── LlamaBridge.kt   # mobile/android/
└── cpp/
    ├── CMakeLists.txt
    └── llama_jni.cpp    # mobile/android/
```

> File `.gguf` >100 MB tidak boleh dikompres APK. Tambahkan di `build.gradle`:
> ```gradle
> androidResources { noCompress += listOf("gguf") }
> ```
> Atau (lebih baik) unduh model saat pertama kali dijalankan agar APK tetap kecil.

### 2. `app/src/main/cpp/CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.22)
project(trilingual)

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)   # tidak ada di NDK

add_subdirectory(${CMAKE_SOURCE_DIR}/llama.cpp ${CMAKE_BINARY_DIR}/llama.cpp)

add_library(trilingual SHARED llama_jni.cpp)
target_link_libraries(trilingual llama android log)
```

### 3. `build.gradle` (module)

```gradle
android {
    ndkVersion "27.0.12077973"
    defaultConfig {
        ndk { abiFilters += "arm64-v8a" }        // 32-bit tidak perlu
        externalNativeBuild {
            cmake {
                // dotprod+fp16 = percepatan besar untuk kernel Q4_K
                arguments += listOf(
                    "-DGGML_NATIVE=OFF",
                    "-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16"
                )
            }
        }
    }
    externalNativeBuild { cmake { path "src/main/cpp/CMakeLists.txt" } }
}
```

### 4. Pemakaian

```kotlin
val assistant = TrilingualAssistant.create(context, "trilingual-204m-facts-Q4_K_M.gguf")

lifecycleScope.launch {
    assistant.askFact("What is the capital of Japan?").collect { piece ->
        textView.append(piece)      // "In Japan, the capital of Japan is Tokyo."
    }
}

// Terjemahan (pakai checkpoint translasi, lihat catatan di bawah)
assistant.translateToJapanese("Good evening.").collect { print(it) }

assistant.close()
```

---

## iOS (Swift)

### 1. Build `llama.xcframework`

```bash
cd llama.cpp
./build-xcframework.sh          # menghasilkan build-apple/llama.xcframework
```

Seret `llama.xcframework` ke target Xcode (Frameworks, *Embed & Sign*).

### 2. Tambahkan file

- `mobile/ios/BpeTokenizer.swift`
- `mobile/ios/LlamaRunner.swift`
- Bundle: `trilingual-204m-facts-Q4_K_M.gguf`, `vocab.txt`, `merges.txt`

### 3. Pemakaian

```swift
guard let model  = Bundle.main.url(forResource: "trilingual-204m-facts-Q4_K_M", withExtension: "gguf"),
      let vocab  = Bundle.main.url(forResource: "vocab",  withExtension: "txt"),
      let merges = Bundle.main.url(forResource: "merges", withExtension: "txt")
else { return }

let assistant = try TrilingualAssistant(modelURL: model, vocabURL: vocab, mergesURL: merges)

Task {
    for await piece in assistant.askFact("What is the capital of Japan?") {
        await MainActor.run { self.output += piece }
    }
}
```

### 4. Catatan iOS

- `Info.plist`: aktifkan **Increased Memory Limit** kalau menaikkan `nCtx`.
- `n_gpu_layers = 99` memakai Metal. Set `0` bila ingin CPU murni.
- Model dalam bundle sudah berupa file nyata, jadi `mmap` langsung bekerja.

---

## Anggaran memori & performa

Konfigurasi: Q4_K_M, `nCtx = 512`, KV cache `Q8_0`.

| Komponen | Ukuran |
|---|---|
| Bobot (mmap) | 130 MB |
| KV cache (18 lapis × 512 × 192 × 2, q8_0) | ~3,5 MB |
| Logits (65.536 float) | 0,26 MB |
| Buffer komputasi | ~4 MB |
| **Total** | **< 150 MB** |

Terukur di M3 Pro, 4 thread, CPU (`-ngl 0`):

| Varian | Ukuran | Prompt (pp128) | Generasi (tg64) |
|---|---|---|---|
| Q4_K_M | 128 MiB | 1827 tok/s | **495 tok/s** |
| Q8_0 | 207 MiB | 2868 tok/s | 429 tok/s |

Sebagai pembanding, engine `infer` asli (skalar, 1 thread) hanya **69,5 tok/s**.
Perkiraan di HP: 60–150 tok/s pada flagship Android/iOS terkini — jauh di atas
kecepatan baca manusia.

---

## Memilih checkpoint

Yang dipaketkan sekarang adalah **checkpoint fakta**. Checkpoint ini bagus untuk
tanya-jawab fakta EN/JA tapi **tidak bisa menerjemahkan** (lihat
`docs/QUANTIZATION_PLAN.md` §4).

Kalau nanti ada checkpoint yang bisa fakta **dan** translasi, tidak ada kode yang
perlu diubah — cukup ulangi konversinya:

```bash
./.venv/bin/python scripts/convert_to_gguf.py \
    --model path/ke/checkpoint_baru.safetensors \
    --name  trilingual-baru
python3 scripts/check_parity.py \
    --safetensors path/ke/checkpoint_baru.safetensors \
    --gguf gguf/trilingual-baru-f16.gguf        # harus PASS
../llama.cpp/build/bin/llama-quantize \
    gguf/trilingual-baru-f16.gguf gguf/trilingual-baru-Q4_K_M.gguf Q4_K_M 8
```

Arsitektur dideteksi otomatis dari bentuk tensor, jadi ukuran model yang berbeda
pun tetap jalan tanpa mengedit script.
