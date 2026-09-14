cmake_minimum_required(VERSION 3.19)

project(ernie_image_ncnn LANGUAGES C CXX)

option(ERNIE_IMAGE_VULKAN "Build the ncnn Vulkan backend" ON)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/CMakeLists.txt")
    message(FATAL_ERROR "Missing vendored third_party/ncnn")
endif()
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp/CMakeLists.txt")
    message(FATAL_ERROR "Missing vendored third_party/tokenizers-cpp")
endif()

# Keep the bundled ncnn build aligned with the validated runtime. Developer
# tools, examples, benchmarks, tests and Python bindings are not release
# dependencies.
set(NCNN_VULKAN ${ERNIE_IMAGE_VULKAN} CACHE BOOL "" FORCE)
set(NCNN_VERSION 20260526 CACHE STRING "" FORCE)
set(NCNN_SYSTEM_GLSLANG OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_PYTHON OFF CACHE BOOL "" FORCE)
set(NCNN_SHARED_LIB OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/ncnn ncnn-build)

set(MSGPACK_USE_BOOST OFF CACHE BOOL "" FORCE)
set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER ON CACHE BOOL "" FORCE)
add_subdirectory(
    third_party/tokenizers-cpp
    tokenizers-cpp-build
    EXCLUDE_FROM_ALL
)

find_package(PNG REQUIRED)

add_library(ernie_image_runtime STATIC
    cpp/src/runtime.cpp
)
target_include_directories(ernie_image_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)
target_compile_features(ernie_image_runtime PUBLIC cxx_std_17)

add_library(ernie_image_ncnn_runner STATIC
    cpp/src/ncnn_runner.cpp
)
target_include_directories(ernie_image_ncnn_runner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)
target_link_libraries(ernie_image_ncnn_runner PUBLIC ncnn)
target_compile_features(ernie_image_ncnn_runner PUBLIC cxx_std_17)

add_library(ernie_image_tokenizer STATIC
    cpp/src/tokenizer.cpp
)
target_include_directories(ernie_image_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)
target_link_libraries(ernie_image_tokenizer PUBLIC tokenizers_cpp)
target_compile_features(ernie_image_tokenizer PUBLIC cxx_std_17)

add_library(ernie_image_text_encoder STATIC
    cpp/src/text_encoder.cpp
)
target_include_directories(ernie_image_text_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)
target_link_libraries(ernie_image_text_encoder PUBLIC
    ernie_image_tokenizer
    ernie_image_runtime
    ernie_image_ncnn_runner
)
target_compile_features(ernie_image_text_encoder PUBLIC cxx_std_17)

add_library(ernie_image_pipeline STATIC
    cpp/src/model_layout.cpp
    cpp/src/dit.cpp
    cpp/src/latent.cpp
    cpp/src/image_io.cpp
    cpp/src/vae.cpp
    cpp/src/random.cpp
    cpp/src/pipeline.cpp
)
target_include_directories(ernie_image_pipeline PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)
target_link_libraries(ernie_image_pipeline PUBLIC
    ernie_image_text_encoder
    ernie_image_ncnn_runner
    ernie_image_runtime
    PNG::PNG
)
target_compile_features(ernie_image_pipeline PUBLIC cxx_std_17)

add_executable(ernie_image_cli
    cpp/tools/ernie_image_cli.cpp
)
target_link_libraries(ernie_image_cli PRIVATE ernie_image_pipeline)
target_compile_features(ernie_image_cli PRIVATE cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(target
        ernie_image_runtime
        ernie_image_ncnn_runner
        ernie_image_tokenizer
        ernie_image_text_encoder
        ernie_image_pipeline
        ernie_image_cli
    )
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -ffp-contract=off
        )
    endforeach()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(ernie_image_foundation_tests
        cpp/tests/foundation_tests.cpp
    )
    target_include_directories(ernie_image_foundation_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src
    )
    target_link_libraries(ernie_image_foundation_tests PRIVATE
        ernie_image_pipeline
    )
    target_compile_features(ernie_image_foundation_tests PRIVATE cxx_std_17)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(ernie_image_foundation_tests PRIVATE
            -Wall -Wextra -Wpedantic -ffp-contract=off
        )
    endif()
    add_test(
        NAME ernie_image_foundation_tests
        COMMAND ernie_image_foundation_tests
    )

    # Manual model-backed acceptance driver. It is intentionally not a CTest:
    # the normal test suite remains lightweight and does not require weights.
    add_executable(ernie_image_dynamic_acceptance
        cpp/tests/dynamic_resolution_acceptance.cpp
    )
    target_link_libraries(ernie_image_dynamic_acceptance PRIVATE
        ernie_image_pipeline
    )
    target_compile_features(ernie_image_dynamic_acceptance PRIVATE cxx_std_17)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(ernie_image_dynamic_acceptance PRIVATE
            -Wall -Wextra -Wpedantic -ffp-contract=off
        )
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS ernie_image_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
