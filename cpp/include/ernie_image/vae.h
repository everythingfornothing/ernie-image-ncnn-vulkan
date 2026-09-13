#pragma once

#include "ernie_image/image_io.h"
#include "ernie_image/model_layout.h"
#include "ernie_image/ncnn_runner.h"

#include <filesystem>
#include <vector>

namespace ernie_image {

constexpr float kVaeBatchNormEpsilon = 1.0e-5f;

struct VaeDecodeResult {
    HostTensor decoder_input;
    HostTensor decoder_output;
    RgbImage image;
    double model_load_seconds = 0.0;
    double inference_seconds = 0.0;
    double preprocess_seconds = 0.0;
    double postprocess_seconds = 0.0;
    double total_seconds = 0.0;
};

// Converts the packed DiT output [1,128,64,64] into the VAE decoder input
// [1,32,128,128] using the official ERNIE latent BN inverse and 2x2
// unpatchify mapping.
HostTensor prepare_vae_decoder_input(
    const HostTensor& packed_latent,
    const std::vector<float>& running_mean,
    const std::vector<float>& running_var
);

// Fixed-resolution ERNIE-Image-Turbo VAE Decoder. The ncnn model remains
// stage-local and is loaded only for decode(), preserving the pipeline's
// low-peak-memory execution contract.
class DynamicVaeDecoder {
public:
    DynamicVaeDecoder(
        NcnnModelFiles decoder_files,
        std::filesystem::path running_mean_path,
        std::filesystem::path running_var_path,
        NcnnRuntimeOptions options = {}
    );

    VaeDecodeResult decode(const HostTensor& packed_latent) const;

    const NcnnModelFiles& model_files() const noexcept;
    const std::filesystem::path& running_mean_path() const noexcept;
    const std::filesystem::path& running_var_path() const noexcept;

private:
    NcnnModelFiles decoder_files_;
    std::filesystem::path running_mean_path_;
    std::filesystem::path running_var_path_;
    NcnnRuntimeOptions options_;
    std::vector<float> running_mean_;
    std::vector<float> running_var_;
};

}  // namespace ernie_image
