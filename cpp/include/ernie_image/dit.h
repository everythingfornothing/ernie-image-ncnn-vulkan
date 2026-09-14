#pragma once

#include "ernie_image/latent.h"
#include "ernie_image/model_layout.h"
#include "ernie_image/ncnn_runner.h"
#include "ernie_image/runtime.h"

#include <array>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace ernie_image {

constexpr int kTextEncoderHiddenSize = 3072;
constexpr int kDitHiddenSize = 4096;
constexpr int kDitAdaLnInputCount = 6;

inline constexpr std::array<const char*, kDitAdaLnInputCount> kDitAdaLnNames = {{
    "shift_msa",
    "scale_msa",
    "gate_msa",
    "shift_mlp",
    "scale_mlp",
    "gate_mlp",
}};

struct DitFrontendInputs {
    HostTensor latent;
    HostTensor timestep;
    HostTensor text_embeddings;
    ImageGeometry geometry = make_image_geometry(kImageWidth, kImageHeight);
};

struct DitFrontendOutputs {
    ImageGeometry geometry = make_image_geometry(kImageWidth, kImageHeight);
    int text_length = 0;
    int sequence_length = 0;
    HostTensor image_tokens;
    HostTensor text_tokens;
    std::array<HostTensor, kDitAdaLnInputCount> adaln;
    HostTensor conditioning;
    double model_load_seconds = 0.0;
    double inference_seconds = 0.0;
};

struct DitStageTiming {
    std::string name;
    int first_layer = -1;
    int last_layer = -1;
    std::filesystem::path param_path;
    std::filesystem::path bin_path;
    double model_load_seconds = 0.0;
    double inference_seconds = 0.0;
};

using DitChunkCallback =
    std::function<void(const DitStageTiming&, const HostTensor&)>;

struct DitPredictResult {
    ImageGeometry geometry = make_image_geometry(kImageWidth, kImageHeight);
    int text_length = 0;
    int sequence_length = 0;
    HostTensor prediction;
    std::vector<DitStageTiming> stages;
    double total_seconds = 0.0;
};

struct DitDenoiseInputs {
    // At start_step=0 this is the seed-derived initial latent. For checkpoint
    // resume it is the previous step's output sample.
    HostTensor initial_sample;
    HostTensor text_embeddings;
    ImageGeometry geometry = make_image_geometry(kImageWidth, kImageHeight);
    int start_step = 0;
    int num_inference_steps = kNumInferenceSteps;
};

struct DitDenoiseStepResult {
    int step_index = 0;
    float scheduler_timestep = 0.0f;
    float model_timestep_bf16_as_fp32 = 0.0f;
    float sigma = 0.0f;
    float sigma_next = 0.0f;
    HostTensor prediction;
    HostTensor sample;
    std::vector<DitStageTiming> stages;
    double dit_seconds = 0.0;
    double scheduler_seconds = 0.0;
    double total_seconds = 0.0;
};

struct DitDenoiseResult {
    ImageGeometry geometry = make_image_geometry(kImageWidth, kImageHeight);
    int text_length = 0;
    int sequence_length = 0;
    int start_step = 0;
    int num_inference_steps = 0;
    HostTensor final_sample;
    std::vector<DitDenoiseStepResult> steps;
    double total_seconds = 0.0;
};

using DitDenoiseStepCallback =
    std::function<void(const DitDenoiseStepResult&)>;

// Owns the ERNIE DiT model-file contract. Heavy ncnn models are loaded lazily:
// predict() runs frontend, each block chunk, and output head in sequence so
// their weights do not all remain resident at once. denoise() repeats that
// path with the official eight-step FlowMatch Euler schedule.
class DynamicDitModel {
public:
    DynamicDitModel(
        DitModelFiles model_files,
        NcnnRuntimeOptions options = {}
    );

    DitFrontendOutputs run_frontend(DitFrontendInputs inputs) const;
    DitPredictResult predict(
        DitFrontendInputs inputs,
        const DitChunkCallback& on_chunk = {}
    ) const;

    DitDenoiseResult denoise(
        DitDenoiseInputs inputs,
        const DitDenoiseStepCallback& on_step = {}
    ) const;
    const DitModelFiles& model_files() const noexcept;
    const NcnnRuntimeOptions& runtime_options() const noexcept;

private:
    DitModelFiles model_files_;
    NcnnRuntimeOptions options_;
};

}  // namespace ernie_image
