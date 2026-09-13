#include "ernie_image/latent.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace ernie_image {

namespace {

void validate_fp32_tensor(
    const HostTensor& tensor,
    const std::vector<int>& expected_shape,
    const char* label
) {
    if (tensor.shape != expected_shape) {
        throw std::invalid_argument(
            std::string(label) + " has an unexpected logical shape"
        );
    }
    const auto* values = std::get_if<std::vector<float>>(&tensor.storage);
    if (values == nullptr) {
        throw std::invalid_argument(std::string(label) + " must be FP32");
    }
    std::size_t expected_count = 1;
    for (int dimension : expected_shape) {
        expected_count *= static_cast<std::size_t>(dimension);
    }
    if (values->size() != expected_count) {
        throw std::invalid_argument(
            std::string(label) + " element count does not match shape"
        );
    }
    for (float value : *values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                std::string(label) + " contains NaN or Inf"
            );
        }
    }
}

}  // namespace

HostTensor make_latent_tensor(std::vector<float> values) {
    HostTensor latent;
    latent.shape = {
        kBatchSize,
        kLatentChannels,
        kLatentHeight,
        kLatentWidth,
    };
    latent.storage = std::move(values);
    validate_latent_tensor(latent);
    return latent;
}

void validate_latent_tensor(const HostTensor& latent) {
    validate_fp32_tensor(
        latent,
        {kBatchSize, kLatentChannels, kLatentHeight, kLatentWidth},
        "DiT latent"
    );
}

void validate_vae_latent_tensor(const HostTensor& latent) {
    validate_fp32_tensor(
        latent,
        {
            kBatchSize,
            kVaeLatentChannels,
            kVaeLatentHeight,
            kVaeLatentWidth,
        },
        "VAE latent"
    );
}

}  // namespace ernie_image
