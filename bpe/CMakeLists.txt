cmake_minimum_required(VERSION 3.20)

add_library(bpe_tokenizer STATIC
    bpe_tokenizer.cpp
)

target_include_directories(bpe_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

find_package(Threads REQUIRED)
target_link_libraries(bpe_tokenizer PUBLIC Threads::Threads)
