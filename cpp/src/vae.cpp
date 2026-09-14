#include "ernie_image/vae.h"

#include "ernie_image/latent.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ernie_image {

namespace {

constexpr std::size_t kPackedChannels = kLatentChannels;

double elapsed_seconds(const std::chrono::steady_clock::time_point& started) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
}

std::vector<float> read_f32_constants(
    const std::filesystem::path& path,
    std::size_t expected_count,
    const char* label
) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            std::string(label) + " file not found: " + path.string()
        );
    }
    const std::uintmax_t expected_bytes = expected_count * sizeof(float);
    if (std::filesystem::file_size(path) != expected_bytes) {
        throw std::runtime_error(
            std::string(label) + " byte count is invalid: " + path.string()
        );
    }
    std::vector<float> values(expected_count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(expected_bytes)
    );
    if (!stream || stream.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error(
            std::string("failed to read ") + label + ": " + path.string()
        );
    }
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                std::string(label) + " contains NaN or Inf"
            );
        }
    }
    return values;
}

void require_decoder_contract(const ModelIoContract& contract) {
    if (contract.input_names != std::vector<std::string>{"in0"}) {
        throw std::runtime_error(
            "VAE Decoder ncnn input contract must be exactly in0"
        );
    }
    if (contract.output_names != std::vector<std::string>{"out0"}) {
        throw std::runtime_error(
            "VAE Decoder ncnn output contract must be exactly out0"
        );
    }
}

}  // namespace

HostTensor prepare_vae_decoder_input(
    const HostTensor& packed_latent,
    const std::vector<float>& running_mean,
    const std::vector<float>& running_var
) {
    return prepare_vae_decoder_input(
        packed_latent,
        running_mean,
        running_var,
        make_image_geometry(kImageWidth, kImageHeight)
    );
}

HostTensor prepare_vae_decoder_input(
    const HostTensor& packed_latent,
    const std::vector<float>& running_mean,
    const std::vector<float>& running_var,
    const ImageGeometry& geometry
) {
    const ImageGeometry checked =
        make_image_geometry(geometry.width, geometry.height);
    validate_latent_tensor(packed_latent, checked);
    if (
        running_mean.size() != kPackedChannels ||
        running_var.size() != kPackedChannels
    ) {
        throw std::invalid_argument(
            "VAE BN constants must each contain 128 FP32 values"
        );
    }
    const auto& input = std::get<std::vector<float>>(packed_latent.storage);
    std::vector<float> standard_deviation(kPackedChannels);
    for (std::size_t channel = 0; channel < kPackedChannels; ++channel) {
        const float variance = running_var[channel];
        if (!std::isfinite(running_mean[channel]) ||
            !std::isfinite(variance) ||
            variance + kVaeBatchNormEpsilon < 0.0f) {
            throw std::invalid_argument("VAE BN constants are invalid");
        }
        standard_deviation[channel] =
            std::sqrt(variance + kVaeBatchNormEpsilon);
    }

    const std::size_t packed_plane =
        static_cast<std::size_t>(checked.image_tokens);
    const std::size_t vae_plane =
        static_cast<std::size_t>(checked.vae_latent_height) *
        checked.vae_latent_width;
    std::vector<float> output(
        static_cast<std::size_t>(kVaeLatentChannels) * vae_plane
    );

    // PyTorch mapping:
    // [1,128,64,64] -> reshape [1,32,2,2,64,64]
    // -> permute [1,32,64,2,64,2] -> [1,32,128,128].
    for (int channel = 0; channel < kVaeLatentChannels; ++channel) {
        for (int patch_y = 0; patch_y < 2; ++patch_y) {
            for (int patch_x = 0; patch_x < 2; ++patch_x) {
                const int packed_channel =
                    channel * 4 + patch_y * 2 + patch_x;
                const float mean =
                    running_mean[static_cast<std::size_t>(packed_channel)];
                const float standard = standard_deviation[
                    static_cast<std::size_t>(packed_channel)
                ];
                for (int y = 0; y < checked.packed_height; ++y) {
                    for (int x = 0; x < checked.packed_width; ++x) {
                        const std::size_t source =
                            static_cast<std::size_t>(packed_channel) *
                                packed_plane +
                            static_cast<std::size_t>(y) *
                                checked.packed_width + x;
                        const int output_y = y * 2 + patch_y;
                        const int output_x = x * 2 + patch_x;
                        const std::size_t destination =
                            static_cast<std::size_t>(channel) * vae_plane +
                            static_cast<std::size_t>(output_y) *
                                checked.vae_latent_width +
                            output_x;
                        output[destination] = input[source] * standard + mean;
                    }
                }
            }
        }
    }

    HostTensor result;
    result.shape = {
        kBatchSize,
        kVaeLatentChannels,
        checked.vae_latent_height,
        checked.vae_latent_width,
    };
    result.storage = std::move(output);
    validate_vae_latent_tensor(result, checked);
    return result;
}

DynamicVaeDecoder::DynamicVaeDecoder(
    NcnnModelFiles decoder_files,
    std::filesystem::path running_mean_path,
    std::filesystem::path running_var_path,
    NcnnRuntimeOptions options
) : decoder_files_(std::move(decoder_files)),
    running_mean_path_(std::move(running_mean_path)),
    running_var_path_(std::move(running_var_path)),
    options_(options),
    running_mean_(read_f32_constants(
        running_mean_path_, kPackedChannels, "VAE running mean"
    )),
    running_var_(read_f32_constants(
        running_var_path_, kPackedChannels, "VAE running variance"
    )) {
    if (options_.num_threads <= 0) {
        throw std::invalid_argument("VAE ncnn num_threads must be positive");
    }
}

VaeDecodeResult DynamicVaeDecoder::decode(
    const HostTensor& packed_latent
) const {
    return decode(
        packed_latent,
        make_image_geometry(kImageWidth, kImageHeight)
    );
}

VaeDecodeResult DynamicVaeDecoder::decode(
    const HostTensor& packed_latent,
    const ImageGeometry& geometry
) const {
    const auto total_started = std::chrono::steady_clock::now();
    const ImageGeometry checked =
        make_image_geometry(geometry.width, geometry.height);
    VaeDecodeResult result;

    auto started = std::chrono::steady_clock::now();
    result.decoder_input = prepare_vae_decoder_input(
        packed_latent, running_mean_, running_var_, checked
    );
    result.preprocess_seconds = elapsed_seconds(started);

    NcnnModel decoder(options_);
    started = std::chrono::steady_clock::now();
    decoder.load(decoder_files_.param, decoder_files_.bin);
    result.model_load_seconds = elapsed_seconds(started);
    require_decoder_contract(decoder.io_contract());

    started = std::chrono::steady_clock::now();
    result.decoder_output = decoder.run_positional({result.decoder_input});
    result.inference_seconds = elapsed_seconds(started);
    const std::vector<int> expected_output = {
        kBatchSize, 3, checked.height, checked.width,
    };
    if (result.decoder_output.shape != expected_output) {
        throw std::runtime_error(
            "VAE Decoder produced an unexpected output shape"
        );
    }

    started = std::chrono::steady_clock::now();
    result.image = postprocess_vae_output(result.decoder_output);
    result.postprocess_seconds = elapsed_seconds(started);
    result.total_seconds = elapsed_seconds(total_started);
    return result;
}

const NcnnModelFiles& DynamicVaeDecoder::model_files() const noexcept {
    return decoder_files_;
}

const std::filesystem::path&
DynamicVaeDecoder::running_mean_path() const noexcept {
    return running_mean_path_;
}

const std::filesystem::path&
DynamicVaeDecoder::running_var_path() const noexcept {
    return running_var_path_;
}

}  // namespace ernie_image
