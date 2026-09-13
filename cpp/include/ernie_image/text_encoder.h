#pragma once

#include "ernie_image/ncnn_runner.h"
#include "ernie_image/runtime.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ernie_image {

struct TextEncoderBoundary {
    int first_layer = 0;
    int last_layer = 0;
    std::vector<float> hidden_states;
};

struct TextEncoderCaseResult {
    int text_length = 0;
    std::vector<std::int32_t> input_ids;
    std::vector<float> embedding_fp32;
    TextRuntimeTensors runtime_tensors;
    std::vector<TextEncoderBoundary> boundaries;
    std::vector<float> hidden_states;
};

struct TextEncoderStageTiming {
    std::string name;
    std::filesystem::path param_path;
    std::filesystem::path model_path;
    double load_seconds = 0.0;
    std::vector<double> inference_seconds;
};

struct TextEncoderBatchResult {
    std::vector<TextEncoderCaseResult> cases;
    std::vector<TextEncoderStageTiming> stages;
};

// Complete dynamic Text Encoder chain:
// UTF-8 prompt -> tokenizer -> ncnn Embed -> ncnn layers 0-24.
class DynamicTextEncoder {
public:
    DynamicTextEncoder(
        const std::filesystem::path& tokenizer_json,
        const std::filesystem::path& model_directory,
        NcnnRuntimeOptions options = {}
    );
    ~DynamicTextEncoder();

    DynamicTextEncoder(DynamicTextEncoder&&) noexcept;
    DynamicTextEncoder& operator=(DynamicTextEncoder&&) noexcept;

    DynamicTextEncoder(const DynamicTextEncoder&) = delete;
    DynamicTextEncoder& operator=(const DynamicTextEncoder&) = delete;

    TextEncoderBatchResult encode_batch(
        const std::vector<std::string>& utf8_prompts,
        bool capture_intermediates = false
    ) const;

    const std::filesystem::path& tokenizer_json_path() const noexcept;
    const std::filesystem::path& model_directory() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ernie_image
