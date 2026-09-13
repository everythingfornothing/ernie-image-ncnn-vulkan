#pragma once

#include <array>
#include <cstddef>
#include <filesystem>

namespace ernie_image {

struct NcnnModelFiles {
    std::filesystem::path param;
    std::filesystem::path bin;
};

struct LayerChunkModelFiles {
    int first_layer = 0;
    int last_layer = 0;
    NcnnModelFiles model;
};

constexpr std::size_t kDitChunkCount = 7;

struct DitModelFiles {
    NcnnModelFiles frontend;
    std::array<LayerChunkModelFiles, kDitChunkCount> chunks;
    NcnnModelFiles output_head;
};

struct PipelineModelPaths {
    std::filesystem::path model_directory;
    std::filesystem::path tokenizer_json;
    std::filesystem::path text_encoder_directory;
    DitModelFiles dit;
    NcnnModelFiles vae_decoder;
    std::filesystem::path vae_running_mean;
    std::filesystem::path vae_running_var;
    std::filesystem::path initial_latent_seed42;

    static PipelineModelPaths from_model_directory(
        const std::filesystem::path& model_directory
    );

    void validate() const;
    PipelineModelPaths canonicalized() const;
};

}  // namespace ernie_image
