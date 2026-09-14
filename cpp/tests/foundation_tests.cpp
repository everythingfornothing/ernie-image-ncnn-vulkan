#include "ernie_image/latent.h"
#include "ernie_image/pipeline.h"
#include "ernie_image/vae.h"
#include "ncnn_tensor_utils.h"

#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using ernie_image::HostTensor;
using ernie_image::ImageGeometry;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void require_throws(Function&& function, const std::string& message) {
    static_assert(
        std::is_base_of_v<std::exception, Exception>,
        "expected type must be an exception"
    );
    try {
        function();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(
            message + ": wrong exception type: " + error.what()
        );
    }
    throw std::runtime_error(message + ": no exception was thrown");
}

void test_tensor_round_trip(int height, int width) {
    constexpr int channels = 128;
    const std::size_t plane =
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    std::vector<float> values(static_cast<std::size_t>(channels) * plane);
    for (int channel = 0; channel < channels; ++channel) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t index =
                    static_cast<std::size_t>(channel) * plane +
                    static_cast<std::size_t>(y) * width + x;
                values[index] = static_cast<float>(
                    channel * 100000 + y * 100 + x
                );
            }
        }
    }

    HostTensor input;
    input.shape = {1, channels, height, width};
    input.storage = values;

    const ncnn::Mat packed = ernie_image::detail::to_ncnn_mat(input);
    const HostTensor output = ernie_image::detail::from_ncnn_mat(packed);
    require(output.shape == input.shape, "tensor round-trip shape mismatch");

    const auto* output_values = std::get_if<std::vector<float>>(&output.storage);
    require(output_values != nullptr, "tensor round-trip changed storage type");
    require(*output_values == values, "tensor round-trip values differ");
    std::cout << "tensor round-trip [1,128," << height << ',' << width
              << "]: plane=" << plane << ", cstep=" << packed.cstep
              << ", exact=PASS\n";
}

void test_geometry(
    int width,
    int height,
    int packed_width,
    int packed_height,
    int image_tokens
) {
    const ImageGeometry geometry =
        ernie_image::make_image_geometry(width, height);
    require(geometry.width == width, "geometry width mismatch");
    require(geometry.height == height, "geometry height mismatch");
    require(
        geometry.packed_width == packed_width,
        "geometry packed width mismatch"
    );
    require(
        geometry.packed_height == packed_height,
        "geometry packed height mismatch"
    );
    require(
        geometry.image_tokens == image_tokens,
        "geometry image token count mismatch"
    );
    require(
        geometry.dit_packed_latent_shape() ==
            std::vector<int>({1, 128, packed_height, packed_width}),
        "DiT packed latent shape mismatch"
    );
    require(
        geometry.vae_latent_shape() ==
            std::vector<int>({1, 32, height / 8, width / 8}),
        "VAE latent shape mismatch"
    );
    std::cout << "geometry " << width << 'x' << height << ": packed="
              << packed_width << 'x' << packed_height
              << ", image_tokens=" << image_tokens << ", PASS\n";
}

void test_invalid_geometry() {
    require_throws<std::invalid_argument>(
        [] { ernie_image::make_image_geometry(0, 1024); },
        "zero width must be rejected"
    );
    require_throws<std::invalid_argument>(
        [] { ernie_image::make_image_geometry(1024, 0); },
        "zero height must be rejected"
    );
    require_throws<std::invalid_argument>(
        [] { ernie_image::make_image_geometry(-16, 1024); },
        "negative width must be rejected"
    );
    require_throws<std::invalid_argument>(
        [] { ernie_image::make_image_geometry(1024, -16); },
        "negative height must be rejected"
    );
    require_throws<std::invalid_argument>(
        [] { ernie_image::make_image_geometry(1000, 1024); },
        "non-aligned width must be rejected"
    );
    require_throws<std::invalid_argument>(
        [] { ernie_image::make_image_geometry(1024, 1000); },
        "non-aligned height must be rejected"
    );

    const int largest_aligned_int =
        std::numeric_limits<int>::max() -
        std::numeric_limits<int>::max() %
            ernie_image::kImageDimensionAlignment;
    require_throws<std::overflow_error>(
        [largest_aligned_int] {
            ernie_image::make_image_geometry(
                largest_aligned_int,
                largest_aligned_int
            );
        },
        "overflowing image token count must be rejected"
    );
    std::cout << "invalid geometry rejection: PASS\n";
}

void test_generation_config_contract() {
    ernie_image::GenerationConfig fixed_reference;
    ernie_image::validate_generation_config(fixed_reference);

    ernie_image::GenerationConfig dynamic_portable;
    dynamic_portable.width = 1376;
    dynamic_portable.height = 768;
    dynamic_portable.seed = 123;
    dynamic_portable.initial_latent_mode =
        ernie_image::InitialLatentMode::PortablePhilox;
    ernie_image::validate_generation_config(dynamic_portable);

    ernie_image::GenerationConfig dynamic_reference = dynamic_portable;
    dynamic_reference.seed = 42;
    dynamic_reference.initial_latent_mode =
        ernie_image::InitialLatentMode::OfficialReference;
    require_throws<std::invalid_argument>(
        [&dynamic_reference] {
            ernie_image::validate_generation_config(dynamic_reference);
        },
        "dynamic resolution must reject reference RNG mode"
    );
    std::cout << "generation config contract: PASS\n";
}

void test_dynamic_vae_unpatchify() {
    const ImageGeometry geometry =
        ernie_image::make_image_geometry(1376, 768);
    const std::size_t packed_plane =
        static_cast<std::size_t>(geometry.image_tokens);
    std::vector<float> packed_values(
        static_cast<std::size_t>(ernie_image::kLatentChannels) * packed_plane
    );
    for (int channel = 0; channel < ernie_image::kLatentChannels; ++channel) {
        for (std::size_t offset = 0; offset < packed_plane; ++offset) {
            packed_values[static_cast<std::size_t>(channel) * packed_plane + offset] =
                static_cast<float>(channel * 100000) +
                static_cast<float>(offset);
        }
    }
    const HostTensor packed = ernie_image::make_latent_tensor(
        packed_values, geometry
    );
    std::vector<float> mean(ernie_image::kLatentChannels, 0.0f);
    std::vector<float> variance(
        ernie_image::kLatentChannels,
        1.0f - ernie_image::kVaeBatchNormEpsilon
    );
    const HostTensor unpacked = ernie_image::prepare_vae_decoder_input(
        packed, mean, variance, geometry
    );
    require(
        unpacked.shape == geometry.vae_latent_shape(),
        "dynamic VAE unpatchify shape mismatch"
    );
    const auto& output = std::get<std::vector<float>>(unpacked.storage);
    const std::size_t vae_plane =
        static_cast<std::size_t>(geometry.vae_latent_height) *
        geometry.vae_latent_width;
    for (int channel = 0; channel < ernie_image::kVaeLatentChannels; ++channel) {
        for (int y = 0; y < geometry.packed_height; ++y) {
            for (int x = 0; x < geometry.packed_width; ++x) {
                const std::size_t source_offset =
                    static_cast<std::size_t>(y) * geometry.packed_width + x;
                for (int patch_y = 0; patch_y < 2; ++patch_y) {
                    for (int patch_x = 0; patch_x < 2; ++patch_x) {
                        const int packed_channel =
                            channel * 4 + patch_y * 2 + patch_x;
                        const std::size_t source =
                            static_cast<std::size_t>(packed_channel) *
                                packed_plane +
                            source_offset;
                        const std::size_t destination =
                            static_cast<std::size_t>(channel) * vae_plane +
                            static_cast<std::size_t>(y * 2 + patch_y) *
                                geometry.vae_latent_width +
                            x * 2 + patch_x;
                        require(
                            output[destination] == packed_values[source],
                            "dynamic VAE unpatchify value mismatch"
                        );
                    }
                }
            }
        }
    }
    std::cout << "dynamic VAE unpatchify 1376x768: exact=PASS\n";
}

}  // namespace

int main() {
    try {
        test_tensor_round_trip(64, 64);
        test_tensor_round_trip(33, 33);
        test_geometry(1024, 1024, 64, 64, 4096);
        test_geometry(1376, 768, 86, 48, 4128);
        test_invalid_geometry();
        test_generation_config_contract();
        test_dynamic_vae_unpatchify();
        std::cout << "foundation tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "foundation tests: FAIL: " << error.what() << '\n';
        return 1;
    }
}
