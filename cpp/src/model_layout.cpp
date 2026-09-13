#include "ernie_image/model_layout.h"

#include <array>
#include <stdexcept>
#include <string>

namespace ernie_image {

namespace {

constexpr std::array<std::array<int, 2>, kDitChunkCount> kDitChunkRanges = {{
    {{0, 1}},
    {{2, 3}},
    {{4, 7}},
    {{8, 15}},
    {{16, 23}},
    {{24, 31}},
    {{32, 35}},
}};

constexpr std::array<const char*, 8> kTextEncoderStems = {{
    "embedding",
    "layers_00_00",
    "layers_01_04",
    "layers_05_08",
    "layers_09_12",
    "layers_13_16",
    "layers_17_20",
    "layers_21_24",
}};

std::string two_digits(int value) {
    if (value < 0 || value > 99) {
        throw std::invalid_argument("layer index is outside two-digit range");
    }
    std::string result = std::to_string(value);
    if (result.size() == 1) {
        result.insert(result.begin(), '0');
    }
    return result;
}

NcnnModelFiles model_files(
    const std::filesystem::path& directory,
    const std::string& stem
) {
    return {
        directory / (stem + ".ncnn.param"),
        directory / (stem + ".ncnn.bin"),
    };
}

void require_directory(
    const std::filesystem::path& path,
    const char* label
) {
    if (!std::filesystem::is_directory(path)) {
        throw std::runtime_error(
            std::string(label) + " directory not found: " + path.string()
        );
    }
}

void require_file(const std::filesystem::path& path, const char* label) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            std::string(label) + " file not found: " + path.string()
        );
    }
}

void validate_model(const NcnnModelFiles& files, const char* label) {
    require_file(files.param, label);
    require_file(files.bin, label);
}

NcnnModelFiles canonical_model(const NcnnModelFiles& files) {
    return {
        std::filesystem::canonical(files.param),
        std::filesystem::canonical(files.bin),
    };
}

}  // namespace

PipelineModelPaths PipelineModelPaths::from_model_directory(
    const std::filesystem::path& directory
) {
    PipelineModelPaths paths;
    paths.model_directory = directory;
    paths.tokenizer_json = directory / "tokenizer" / "tokenizer.json";
    paths.text_encoder_directory = directory / "text_encoder";

    const std::filesystem::path dit_directory = directory / "dit";
    paths.dit.frontend = model_files(dit_directory, "frontend");
    for (std::size_t index = 0; index < kDitChunkRanges.size(); ++index) {
        const int first = kDitChunkRanges[index][0];
        const int last = kDitChunkRanges[index][1];
        paths.dit.chunks[index] = {
            first,
            last,
            model_files(
                dit_directory,
                "layers_" + two_digits(first) + "_" + two_digits(last)
            ),
        };
    }
    paths.dit.output_head = model_files(dit_directory, "output_head");
    paths.vae_decoder = model_files(directory / "vae", "decoder");
    paths.vae_running_mean = directory / "vae" / "running_mean.f32";
    paths.vae_running_var = directory / "vae" / "running_var.f32";
    paths.initial_latent_seed42 =
        directory / "rng" / "official_cuda_bf16_seed42.f32";
    return paths;
}

void PipelineModelPaths::validate() const {
    require_directory(model_directory, "model");
    require_file(tokenizer_json, "tokenizer");
    require_directory(text_encoder_directory, "Text Encoder");
    for (const char* stem : kTextEncoderStems) {
        validate_model(
            model_files(text_encoder_directory, stem),
            "Text Encoder"
        );
    }

    validate_model(dit.frontend, "DiT frontend");
    for (std::size_t index = 0; index < dit.chunks.size(); ++index) {
        const LayerChunkModelFiles& chunk = dit.chunks[index];
        if (
            chunk.first_layer != kDitChunkRanges[index][0] ||
            chunk.last_layer != kDitChunkRanges[index][1]
        ) {
            throw std::runtime_error(
                "DiT chunk ranges do not match the 0-35 deployment contract"
            );
        }
        validate_model(chunk.model, "DiT chunk");
    }
    validate_model(dit.output_head, "DiT output head");
    validate_model(vae_decoder, "VAE decoder");
    require_file(vae_running_mean, "VAE running mean");
    require_file(vae_running_var, "VAE running variance");
    require_file(initial_latent_seed42, "official seed=42 latent");
}

PipelineModelPaths PipelineModelPaths::canonicalized() const {
    validate();
    PipelineModelPaths paths = *this;
    paths.model_directory = std::filesystem::canonical(model_directory);
    paths.tokenizer_json = std::filesystem::canonical(tokenizer_json);
    paths.text_encoder_directory =
        std::filesystem::canonical(text_encoder_directory);
    paths.dit.frontend = canonical_model(dit.frontend);
    for (std::size_t index = 0; index < paths.dit.chunks.size(); ++index) {
        paths.dit.chunks[index].model = canonical_model(dit.chunks[index].model);
    }
    paths.dit.output_head = canonical_model(dit.output_head);
    paths.vae_decoder = canonical_model(vae_decoder);
    paths.vae_running_mean = std::filesystem::canonical(vae_running_mean);
    paths.vae_running_var = std::filesystem::canonical(vae_running_var);
    paths.initial_latent_seed42 =
        std::filesystem::canonical(initial_latent_seed42);
    return paths;
}

}  // namespace ernie_image
