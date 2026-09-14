#pragma once

#include "ernie_image/ncnn_runner.h"

#include <vector>

namespace ernie_image {

constexpr int kBatchSize = 1;
constexpr int kImageHeight = 1024;
constexpr int kImageWidth = 1024;
constexpr int kLatentChannels = 128;
constexpr int kLatentHeight = 64;
constexpr int kLatentWidth = 64;
constexpr int kVaeLatentChannels = 32;
constexpr int kVaeLatentHeight = 128;
constexpr int kVaeLatentWidth = 128;
constexpr int kImageDimensionAlignment = 16;

// Validated image geometry shared by all dynamic-resolution host stages.
struct ImageGeometry {
    int width = 0;
    int height = 0;
    int packed_width = 0;
    int packed_height = 0;
    int image_tokens = 0;
    int vae_latent_width = 0;
    int vae_latent_height = 0;

    std::vector<int> dit_packed_latent_shape() const;
    std::vector<int> vae_latent_shape() const;
};

ImageGeometry make_image_geometry(int width, int height);

HostTensor make_latent_tensor(std::vector<float> values);
HostTensor make_latent_tensor(
    std::vector<float> values,
    const ImageGeometry& geometry
);
void validate_latent_tensor(const HostTensor& latent);
void validate_latent_tensor(
    const HostTensor& latent,
    const ImageGeometry& geometry
);
void validate_vae_latent_tensor(const HostTensor& latent);
void validate_vae_latent_tensor(
    const HostTensor& latent,
    const ImageGeometry& geometry
);

}  // namespace ernie_image
