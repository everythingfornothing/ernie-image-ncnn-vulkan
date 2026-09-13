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

HostTensor make_latent_tensor(std::vector<float> values);
void validate_latent_tensor(const HostTensor& latent);
void validate_vae_latent_tensor(const HostTensor& latent);

}  // namespace ernie_image
