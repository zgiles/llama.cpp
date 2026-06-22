llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_CORE_SOURCES
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-chat.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama-grammar.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-io.cpp
    llama-kv-cache.cpp
    llama-kv-cache-iswa.cpp
    llama-kv-cache-dsa.cpp
    llama-kv-cache-dsa-iswa.cpp
    llama-kv-cache-msa.cpp
    llama-kv-cache-dsv4.cpp
    llama-memory.cpp
    llama-memory-hybrid.cpp
    llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-idx.cpp
    llama-memory-recurrent.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model-saver.cpp
    llama-model.cpp
    llama-quant.cpp
    llama-sampler.cpp
    llama-vocab.cpp
    unicode-data.cpp
    unicode.cpp
)

# CPU tensor parallelism (C, and order-sensitive statics): kept out of the unity build below
set(LLAMA_TP_SOURCES
    llama-tp-shard.c
    llama-tp-net.c
)

add_library(llama
            ../include/llama.h
            ${LLAMA_CORE_SOURCES}
            ${LLAMA_TP_SOURCES}
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 16
)

# exclude non-model sources from unity build
set_source_files_properties(${LLAMA_CORE_SOURCES} ${LLAMA_TP_SOURCES} ../include/llama.h unicode.h
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

configure_file(llama-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/llama-version.h @ONLY)

target_include_directories(llama PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

# Optional inter-node tensor parallelism over UCX/IB. Enable by passing the UCX paths, e.g.:
#   -DTP_UCX_INC=<ucx-dev>/include -DTP_UCX_LIB=<ucx>/lib
if (DEFINED TP_UCX_LIB)
    message(STATUS "llama: CPU tensor parallelism UCX backend enabled (${TP_UCX_LIB})")
    target_include_directories(llama PRIVATE ${TP_UCX_INC})
    target_link_libraries(llama PRIVATE ${TP_UCX_LIB}/libucp.so ${TP_UCX_LIB}/libucs.so)
    target_link_options(llama PRIVATE -Wl,-rpath,${TP_UCX_LIB})
    set_property(SOURCE llama-tp-net.c PROPERTY COMPILE_DEFINITIONS LLAMA_TP_UCX=1)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
