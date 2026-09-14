#pragma once

#include "ernie_image/dit.h"
#include "ernie_image/model_layout.h"
#include "ernie_image/ncnn_runner.h"
#include "ernie_image/random.h"
#include "ernie_image/vae.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ernie_image {

struct GenerationConfig {
    int batch_size = 1;
    int width = 1024;
    int height = 1024;
    int num_inference_steps = 8;
    float guidance_scale = 1.0f;
    std::uint64_t seed = 42;
    InitialLatentMode initial_latent_mode =
        InitialLatentMode::OfficialReference;
    bool use_prompt_enhancer = false;
};

struct PreparedPrompt {
    int text_length = 0;
    int sequence_length = 0;
    std::vector<float> text_embeddings;
};

struct GenerationRequest {
    std::string prompt;
    std::filesystem::path output_path;
    GenerationConfig config;
};

struct GenerationResult {
    std::filesystem::path output_path;
    std::uint64_t seed = 42;
    InitialLatentMode initial_latent_mode =
        InitialLatentMode::OfficialReference;
    int width = 0;
    int height = 0;
    int num_inference_steps = 0;
    int text_length = 0;
    int image_tokens = 0;
    int sequence_length = 0;
    std::string initial_latent_policy;
    double prompt_seconds = 0.0;
    double initial_latent_seconds = 0.0;
    double denoise_seconds = 0.0;
    double vae_seconds = 0.0;
    double png_seconds = 0.0;
    double elapsed_seconds = 0.0;
};

void validate_generation_config(const GenerationConfig& config);

// Top-level orchestration boundary. load() validates the deploy package and
// prepares lightweight adapters. Large ncnn models remain stage-local so that
// later DiT/VAE execution can control peak memory explicitly.
class ErnieImagePipeline {
public:
    ErnieImagePipeline(
        PipelineModelPaths model_paths,
        NcnnRuntimeOptions options = {}
    );
    ~ErnieImagePipeline();

    ErnieImagePipeline(ErnieImagePipeline&&) noexcept;
    ErnieImagePipeline& operator=(ErnieImagePipeline&&) noexcept;

    ErnieImagePipeline(const ErnieImagePipeline&) = delete;
    ErnieImagePipeline& operator=(const ErnieImagePipeline&) = delete;

    void load();
    bool loaded() const noexcept;

    PreparedPrompt prepare_prompt(const std::string& utf8_prompt) const;
    DitFrontendOutputs run_dit_frontend(DitFrontendInputs inputs) const;
    DitPredictResult predict_dit(
        DitFrontendInputs inputs,
        const DitChunkCallback& on_chunk = {}
    ) const;
    DitDenoiseResult denoise_dit(
        DitDenoiseInputs inputs,
        const DitDenoiseStepCallback& on_step = {}
    ) const;
    VaeDecodeResult decode_vae(const HostTensor& packed_latent) const;
    VaeDecodeResult decode_vae(
        const HostTensor& packed_latent,
        const ImageGeometry& geometry
    ) const;

    // Stable full-generation API. Reference mode preserves the official
    // seed=42 boundary; portable mode supports explicit arbitrary seeds.
    // Both continue through 8-step DiT/Scheduler -> VAE -> atomic PNG.
    GenerationResult generate(
        const GenerationRequest& request,
        const DitDenoiseStepCallback& on_step = {}
    ) const;

    const PipelineModelPaths& model_paths() const;
    const NcnnRuntimeOptions& runtime_options() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ernie_image
