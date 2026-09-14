#include "ernie_image/latent.h"

#include <cmath>
#include <limits>
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

std::vector<int> ImageGeometry::dit_packed_latent_shape() const {
    return {kBatchSize, kLatentChannels, packed_height, packed_width};
}

std::vector<int> ImageGeometry::vae_latent_shape() const {
    return {
        kBatchSize,
        kVaeLatentChannels,
        vae_latent_height,
        vae_latent_width,
    };
}

ImageGeometry make_image_geometry(int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("image width and height must be positive");
    }
    if (
        width % kImageDimensionAlignment != 0 ||
        height % kImageDimensionAlignment != 0
    ) {
        throw std::invalid_argument(
            "image width and height must be multiples of 16"
        );
    }

    const int packed_width = width / kImageDimensionAlignment;
    const int packed_height = height / kImageDimensionAlignment;
    if (packed_width > std::numeric_limits<int>::max() / packed_height) {
        throw std::overflow_error("image token count exceeds supported range");
    }

    ImageGeometry geometry;
    geometry.width = width;
    geometry.height = height;
    geometry.packed_width = packed_width;
    geometry.packed_height = packed_height;
    geometry.image_tokens = packed_width * packed_height;
    geometry.vae_latent_width = width / 8;
    geometry.vae_latent_height = height / 8;
    return geometry;
}

HostTensor make_latent_tensor(std::vector<float> values) {
    return make_latent_tensor(
        std::move(values),
        make_image_geometry(kImageWidth, kImageHeight)
    );
}

HostTensor make_latent_tensor(
    std::vector<float> values,
    const ImageGeometry& geometry
) {
    const ImageGeometry checked =
        make_image_geometry(geometry.width, geometry.height);
    HostTensor latent;
    latent.shape = {
        kBatchSize,
        kLatentChannels,
        checked.packed_height,
        checked.packed_width,
    };
    latent.storage = std::move(values);
    validate_latent_tensor(latent, checked);
    return latent;
}

void validate_latent_tensor(const HostTensor& latent) {
    validate_latent_tensor(
        latent,
        make_image_geometry(kImageWidth, kImageHeight)
    );
}

void validate_latent_tensor(
    const HostTensor& latent,
    const ImageGeometry& geometry
) {
    const ImageGeometry checked =
        make_image_geometry(geometry.width, geometry.height);
    validate_fp32_tensor(
        latent,
        checked.dit_packed_latent_shape(),
        "DiT latent"
    );
}

void validate_vae_latent_tensor(const HostTensor& latent) {
    validate_vae_latent_tensor(
        latent,
        make_image_geometry(kImageWidth, kImageHeight)
    );
}

void validate_vae_latent_tensor(
    const HostTensor& latent,
    const ImageGeometry& geometry
) {
    const ImageGeometry checked =
        make_image_geometry(geometry.width, geometry.height);
    validate_fp32_tensor(
        latent,
        checked.vae_latent_shape(),
        "VAE latent"
    );
}

}  // namespace ernie_image
