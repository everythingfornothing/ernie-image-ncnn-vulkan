#include "ernie_image/runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ernie_image {
namespace {

constexpr float kTextRopeTheta = 1'000'000.0f;
constexpr float kTextRopeFactor = 16.0f;
constexpr int kTextOriginalMaxPositionEmbeddings = 16'384;
constexpr float kTextBetaFast = 32.0f;
constexpr float kTextBetaSlow = 1.0f;

constexpr std::array<float, 16> kDitOmega32 = {
    1.0f,
    0.7071067690849304f,
    0.5f,
    0.3535533845424652f,
    0.25f,
    0.1767766922712326f,
    0.125f,
    0.0883883461356163f,
    0.0625f,
    0.04419417306780815f,
    0.03125f,
    0.022097086533904076f,
    0.015625f,
    0.011048543266952038f,
    0.0078125f,
    0.005524271633476019f,
};

constexpr std::array<float, 24> kDitOmega48 = {
    1.0f,
    0.7937005162239075f,
    0.6299604773521423f,
    0.5f,
    0.39685025811195374f,
    0.31498023867607117f,
    0.25f,
    0.19842511415481567f,
    0.15749011933803558f,
    0.125f,
    0.09921255707740784f,
    0.07874505966901779f,
    0.0625f,
    0.04960627853870392f,
    0.0393725261092186f,
    0.03125f,
    0.02480313926935196f,
    0.0196862630546093f,
    0.015625f,
    0.01240156963467598f,
    0.00984313152730465f,
    0.0078125f,
    0.00620078481733799f,
    0.004921565763652325f,
};

float correction_dimension(float rotations) {
    const double numerator = static_cast<double>(kTextHeadDim) *
        std::log(
            static_cast<double>(kTextOriginalMaxPositionEmbeddings) /
            (static_cast<double>(rotations) * 2.0 * std::acos(-1.0))
        );
    const double denominator = 2.0 * std::log(static_cast<double>(kTextRopeTheta));
    return static_cast<float>(numerator / denominator);
}

std::array<float, kTextHeadDim / 2> make_text_inv_freq() {
    const int low = std::max(
        static_cast<int>(std::floor(correction_dimension(kTextBetaFast))),
        0
    );
    const int high = std::min(
        static_cast<int>(std::ceil(correction_dimension(kTextBetaSlow))),
        kTextHeadDim - 1
    );
    std::array<float, kTextHeadDim / 2> inv_freq{};
    for (int index = 0; index < kTextHeadDim / 2; ++index) {
        float ramp = static_cast<float>(index - low) /
            static_cast<float>(high == low ? 0.001f : high - low);
        ramp = std::clamp(ramp, 0.0f, 1.0f);
        const float extrapolation_factor = 1.0f - ramp;
        const float exponent = static_cast<float>(index * 2) /
            static_cast<float>(kTextHeadDim);
        const float position_frequency = std::pow(kTextRopeTheta, exponent);
        const float inv_extrapolation = 1.0f / position_frequency;
        const float inv_interpolation = 1.0f /
            (kTextRopeFactor * position_frequency);
        inv_freq[index] =
            inv_interpolation * (1.0f - extrapolation_factor) +
            inv_extrapolation * extrapolation_factor;
    }
    return inv_freq;
}

template <std::size_t Size>
void append_axis_frequencies(
    std::array<float, kDitHeadDim / 2>& output,
    std::size_t& offset,
    float position,
    const std::array<float, Size>& omega
) {
    for (float value : omega) {
        output[offset++] = position * value;
    }
}

}  // namespace

float round_to_bfloat16_as_float(float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "Unexpected float size");
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding_bias = 0x7fffu + ((bits >> 16u) & 1u);
    bits += rounding_bias;
    bits &= 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

TextRuntimeTensors make_text_runtime_tensors(int text_length) {
    if (text_length <= 0) {
        throw std::invalid_argument("text_length must be positive");
    }
    if (text_length > kTokenizerMaxLength) {
        throw std::invalid_argument("text_length exceeds tokenizer maximum of 2048");
    }
    TextRuntimeTensors result;
    result.text_length = text_length;
    const std::size_t rotary_size =
        static_cast<std::size_t>(text_length) * kTextHeadDim;
    result.rotary_cos.resize(rotary_size);
    result.rotary_sin.resize(rotary_size);
    const auto inv_freq = make_text_inv_freq();
    for (int position = 0; position < text_length; ++position) {
        for (int index = 0; index < kTextHeadDim / 2; ++index) {
            const float frequency = static_cast<float>(position) * inv_freq[index];
            const float cosine = std::cos(frequency);
            const float sine = std::sin(frequency);
            const std::size_t first =
                static_cast<std::size_t>(position) * kTextHeadDim + index;
            const std::size_t second = first + kTextHeadDim / 2;
            result.rotary_cos[first] = cosine;
            result.rotary_cos[second] = cosine;
            result.rotary_sin[first] = sine;
            result.rotary_sin[second] = sine;
        }
    }

    const std::size_t mask_size =
        static_cast<std::size_t>(text_length) * text_length;
    result.causal_mask.resize(mask_size);
    for (int query = 0; query < text_length; ++query) {
        for (int key = 0; key < text_length; ++key) {
            result.causal_mask[
                static_cast<std::size_t>(query) * text_length + key
            ] = key > query ? std::numeric_limits<float>::lowest() : 0.0f;
        }
    }
    return result;
}

DitRuntimeTensors make_dit_runtime_tensors(
    int text_length,
    int image_height_tokens,
    int image_width_tokens
) {
    if (text_length <= 0) {
        throw std::invalid_argument("text_length must be positive");
    }
    if (text_length > kTokenizerMaxLength) {
        throw std::invalid_argument("text_length exceeds tokenizer maximum of 2048");
    }
    if (image_height_tokens <= 0 || image_width_tokens <= 0) {
        throw std::invalid_argument("image token dimensions must be positive");
    }
    const int image_tokens = image_height_tokens * image_width_tokens;
    const int sequence_length = image_tokens + text_length;
    DitRuntimeTensors result;
    result.text_length = text_length;
    result.sequence_length = sequence_length;
    result.position_ids.resize(static_cast<std::size_t>(sequence_length) * 3);

    int token = 0;
    for (int y = 0; y < image_height_tokens; ++y) {
        for (int x = 0; x < image_width_tokens; ++x, ++token) {
            const std::size_t base = static_cast<std::size_t>(token) * 3;
            result.position_ids[base] = static_cast<float>(text_length);
            result.position_ids[base + 1] = static_cast<float>(y);
            result.position_ids[base + 2] = static_cast<float>(x);
        }
    }
    for (int index = 0; index < text_length; ++index, ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * 3;
        result.position_ids[base] = static_cast<float>(index);
        result.position_ids[base + 1] = 0.0f;
        result.position_ids[base + 2] = 0.0f;
    }

    result.rotary_frequencies.resize(
        static_cast<std::size_t>(sequence_length) * kDitHeadDim
    );
    for (int index = 0; index < sequence_length; ++index) {
        const std::size_t position_base = static_cast<std::size_t>(index) * 3;
        std::array<float, kDitHeadDim / 2> frequencies{};
        std::size_t offset = 0;
        append_axis_frequencies(
            frequencies, offset, result.position_ids[position_base], kDitOmega32
        );
        append_axis_frequencies(
            frequencies, offset, result.position_ids[position_base + 1], kDitOmega48
        );
        append_axis_frequencies(
            frequencies, offset, result.position_ids[position_base + 2], kDitOmega48
        );
        const std::size_t rotary_base =
            static_cast<std::size_t>(index) * kDitHeadDim;
        for (int channel = 0; channel < kDitHeadDim / 2; ++channel) {
            result.rotary_frequencies[rotary_base + channel * 2] =
                frequencies[channel];
            result.rotary_frequencies[rotary_base + channel * 2 + 1] =
                frequencies[channel];
        }
    }
    result.attention_mask.assign(sequence_length, std::uint8_t{1});
    result.additive_attention_mask.assign(sequence_length, 0.0f);
    return result;
}

FlowMatchSchedule make_flow_match_schedule(
    int num_inference_steps,
    float shift,
    float num_train_timesteps
) {
    if (num_inference_steps <= 0) {
        throw std::invalid_argument("num_inference_steps must be positive");
    }
    if (!(shift > 0.0f) || !(num_train_timesteps > 0.0f)) {
        throw std::invalid_argument("shift and num_train_timesteps must be positive");
    }
    FlowMatchSchedule result;
    result.requested_sigmas.resize(num_inference_steps);
    result.sigmas.resize(num_inference_steps + 1);
    result.timesteps.resize(num_inference_steps);
    result.model_timesteps_bf16_as_fp32.resize(num_inference_steps);
    for (int index = 0; index < num_inference_steps; ++index) {
        const float requested = 1.0f -
            static_cast<float>(index) / static_cast<float>(num_inference_steps);
        const float shifted = (shift * requested) /
            (1.0f + (shift - 1.0f) * requested);
        result.requested_sigmas[index] = requested;
        result.sigmas[index] = shifted;
        result.timesteps[index] = shifted * num_train_timesteps;
        result.model_timesteps_bf16_as_fp32[index] =
            round_to_bfloat16_as_float(result.timesteps[index]);
    }
    result.sigmas[num_inference_steps] = 0.0f;
    return result;
}

std::vector<float> flow_match_euler_step(
    const std::vector<float>& sample,
    const std::vector<float>& model_output,
    float sigma,
    float sigma_next
) {
    if (sample.size() != model_output.size()) {
        throw std::invalid_argument("sample and model_output sizes differ");
    }
    const float delta = sigma_next - sigma;
    std::vector<float> output(sample.size());
    for (std::size_t index = 0; index < sample.size(); ++index) {
        const float scaled = delta * model_output[index];
        output[index] = sample[index] + scaled;
    }
    return output;
}

}  // namespace ernie_image
