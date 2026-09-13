#include "ernie_image/image_io.h"

#include "ernie_image/latent.h"

#include <png.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ernie_image {

void validate_rgb_image(const RgbImage& image) {
    if (image.width <= 0 || image.height <= 0) {
        throw std::invalid_argument("RGB image dimensions must be positive");
    }
    const std::size_t expected =
        static_cast<std::size_t>(image.width) *
        static_cast<std::size_t>(image.height) * 3;
    if (image.pixels.size() != expected) {
        throw std::invalid_argument(
            "RGB image byte count does not match width and height"
        );
    }
}

RgbImage postprocess_vae_output(const HostTensor& decoder_output) {
    const std::vector<int> expected_shape = {
        1, 3, kImageHeight, kImageWidth,
    };
    if (decoder_output.shape != expected_shape) {
        throw std::invalid_argument(
            "VAE decoder output must have shape [1,3,1024,1024]"
        );
    }
    const auto* values =
        std::get_if<std::vector<float>>(&decoder_output.storage);
    if (values == nullptr) {
        throw std::invalid_argument("VAE decoder output must be FP32");
    }
    const std::size_t plane =
        static_cast<std::size_t>(kImageHeight) * kImageWidth;
    if (values->size() != plane * 3) {
        throw std::invalid_argument(
            "VAE decoder output element count is invalid"
        );
    }

    RgbImage image;
    image.width = kImageWidth;
    image.height = kImageHeight;
    image.pixels.resize(plane * 3);
    for (int y = 0; y < kImageHeight; ++y) {
        for (int x = 0; x < kImageWidth; ++x) {
            const std::size_t pixel =
                static_cast<std::size_t>(y) * kImageWidth + x;
            for (int channel = 0; channel < 3; ++channel) {
                const float value =
                    (*values)[static_cast<std::size_t>(channel) * plane + pixel];
                if (!std::isfinite(value)) {
                    throw std::invalid_argument(
                        "VAE decoder output contains NaN or Inf"
                    );
                }
                // Exact Diffusers VaeImageProcessor path:
                // (x * 0.5 + 0.5).clamp(0, 1), then round(x * 255).
                const float normalized = std::clamp(
                    value * 0.5f + 0.5f, 0.0f, 1.0f
                );
                const float rounded = std::nearbyint(normalized * 255.0f);
                image.pixels[pixel * 3 + channel] =
                    static_cast<std::uint8_t>(rounded);
            }
        }
    }
    validate_rgb_image(image);
    return image;
}

void write_png_atomic(
    const std::filesystem::path& output_path,
    const RgbImage& image
) {
    validate_rgb_image(image);
    if (output_path.empty()) {
        throw std::invalid_argument("PNG output path must not be empty");
    }
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    const std::filesystem::path temporary =
        output_path.parent_path() /
        ("." + output_path.filename().string() + ".partial");
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);

    png_image png;
    std::memset(&png, 0, sizeof(png));
    png.version = PNG_IMAGE_VERSION;
    png.width = static_cast<png_uint_32>(image.width);
    png.height = static_cast<png_uint_32>(image.height);
    png.format = PNG_FORMAT_RGB;
    const int success = png_image_write_to_file(
        &png,
        temporary.c_str(),
        0,
        image.pixels.data(),
        image.width * 3,
        nullptr
    );
    if (success == 0) {
        const std::string message = png.message;
        png_image_free(&png);
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error("libpng write failed: " + message);
    }
    png_image_free(&png);
    std::filesystem::rename(temporary, output_path);
}

}  // namespace ernie_image
