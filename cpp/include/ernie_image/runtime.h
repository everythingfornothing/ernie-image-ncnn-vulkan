#pragma once

#include <cstdint>
#include <vector>

namespace ernie_image {

constexpr int kImageHeightTokens = 64;
constexpr int kImageWidthTokens = 64;
constexpr int kImageTokens = kImageHeightTokens * kImageWidthTokens;
constexpr int kTokenizerMaxLength = 2048;
constexpr int kTextHeadDim = 128;
constexpr int kDitHeadDim = 128;
constexpr int kNumInferenceSteps = 8;

struct TextRuntimeTensors {
    int text_length = 0;
    std::vector<float> rotary_cos;
    std::vector<float> rotary_sin;
    std::vector<float> causal_mask;
};

struct DitRuntimeTensors {
    int text_length = 0;
    int sequence_length = 0;
    std::vector<float> position_ids;
    std::vector<float> rotary_frequencies;
    std::vector<std::uint8_t> attention_mask;
    std::vector<float> additive_attention_mask;
};

struct FlowMatchSchedule {
    std::vector<float> requested_sigmas;
    std::vector<float> sigmas;
    std::vector<float> timesteps;
    std::vector<float> model_timesteps_bf16_as_fp32;
};

TextRuntimeTensors make_text_runtime_tensors(int text_length);

DitRuntimeTensors make_dit_runtime_tensors(
    int text_length,
    int image_height_tokens = kImageHeightTokens,
    int image_width_tokens = kImageWidthTokens
);

FlowMatchSchedule make_flow_match_schedule(
    int num_inference_steps = kNumInferenceSteps,
    float shift = 4.0f,
    float num_train_timesteps = 1000.0f
);

std::vector<float> flow_match_euler_step(
    const std::vector<float>& sample,
    const std::vector<float>& model_output,
    float sigma,
    float sigma_next
);

float round_to_bfloat16_as_float(float value);

}  // namespace ernie_image
