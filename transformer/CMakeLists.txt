cmake_minimum_required(VERSION 3.20)

add_library(transformer_core STATIC
    model_config.cpp
    pretrain_dataset.cpp
    # lora.cpp excluded — LoRA is handled directly in CUDA training files
)

target_include_directories(transformer_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/bpe
)

find_package(Threads REQUIRED)
target_link_libraries(transformer_core PUBLIC
    Threads::Threads
    bpe_tokenizer
)
