cmake_minimum_required(VERSION 3.20)
project(infer_cpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(ROOT ${CMAKE_SOURCE_DIR}/..)

add_executable(infer
    ${ROOT}/tools/infer.cpp
    ${ROOT}/transformer/transformer.cpp
    ${ROOT}/transformer/model_config.cpp
    ${ROOT}/bpe/bpe_tokenizer.cpp
)

target_include_directories(infer PRIVATE
    ${ROOT}
    ${ROOT}/transformer
    ${ROOT}/bpe
)

find_package(Threads REQUIRED)
target_link_libraries(infer PRIVATE Threads::Threads)

# Try linking stdc++fs if needed (older GCC for std::filesystem)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(infer PRIVATE stdc++fs)
endif()
