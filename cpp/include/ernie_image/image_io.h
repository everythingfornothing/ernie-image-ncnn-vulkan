#pragma once

#include "ernie_image/ncnn_runner.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace ernie_image {

struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;
};

void validate_rgb_image(const RgbImage& image);
RgbImage postprocess_vae_output(const HostTensor& decoder_output);
void write_png_atomic(
    const std::filesystem::path& output_path,
    const RgbImage& image
);

}  // namespace ernie_image
