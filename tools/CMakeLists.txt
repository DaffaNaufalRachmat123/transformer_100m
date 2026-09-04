cmake_minimum_required(VERSION 3.20)

# Inference tool (CPU-only, portable)
add_executable(infer infer.cpp)
target_include_directories(infer PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/transformer
    ${CMAKE_SOURCE_DIR}/bpe
)
target_link_libraries(infer PRIVATE
    transformer_core
    bpe_tokenizer
)

# BPE trainer (CPU-only, portable)
add_executable(train_bpe train_bpe.cpp)
target_include_directories(train_bpe PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/bpe
)
target_link_libraries(train_bpe PRIVATE
    bpe_tokenizer
)
