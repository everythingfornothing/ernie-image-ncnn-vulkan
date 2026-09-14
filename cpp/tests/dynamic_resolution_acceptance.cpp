#include "ernie_image/dit.h"
#include "ernie_image/image_io.h"
#include "ernie_image/latent.h"
#include "ernie_image/model_layout.h"
#include "ernie_image/pipeline.h"
#include "ernie_image/random.h"
#include "ernie_image/runtime.h"
#include "ernie_image/vae.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

double elapsed_seconds(const std::chrono::steady_clock::time_point& started) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
}

void write_bytes_atomic(
    const std::filesystem::path& path,
    const void* data,
    std::size_t byte_count
) {
    std::filesystem::create_directories(path.parent_path());
    const std::filesystem::path temporary =
        path.parent_path() / ("." + path.filename().string() + ".partial");
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        stream.write(
            static_cast<const char*>(data),
            static_cast<std::streamsize>(byte_count)
        );
        if (!stream) {
            throw std::runtime_error("failed to write " + path.string());
        }
    }
    std::filesystem::rename(temporary, path);
}

void write_f32_atomic(
    const std::filesystem::path& path,
    const ernie_image::HostTensor& tensor
) {
    const auto* values = std::get_if<std::vector<float>>(&tensor.storage);
    if (values == nullptr) {
        throw std::runtime_error("acceptance tensor is not FP32");
    }
    for (float value : *values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("acceptance tensor contains NaN/Inf");
        }
    }
    write_bytes_atomic(path, values->data(), values->size() * sizeof(float));
}

ernie_image::HostTensor read_initial_latent(
    const std::filesystem::path& path,
    const ernie_image::ImageGeometry& geometry
) {
    const std::size_t count =
        static_cast<std::size_t>(ernie_image::kLatentChannels) *
        static_cast<std::size_t>(geometry.image_tokens);
    const std::uintmax_t expected_bytes = count * sizeof(float);
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            "initial latent is not a regular file: " + path.string()
        );
    }
    if (std::filesystem::file_size(path) != expected_bytes) {
        throw std::runtime_error(
            "initial latent byte size does not match requested geometry"
        );
    }

    std::vector<float> values(count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(expected_bytes)
    );
    if (!stream || stream.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("failed to read initial latent exactly");
    }
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("initial latent contains NaN/Inf");
        }
    }
    return ernie_image::make_latent_tensor(std::move(values), geometry);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr
            << "Usage: ernie_image_dynamic_acceptance MODEL_DIR PROMPT "
            << "OUTPUT_DIR WIDTH HEIGHT SEED THREADS "
            << "[--device cpu|vulkan] [--gpu-index N] "
            << "[--initial-latent FILE]\n";
        return 2;
    }
    try {
        const std::filesystem::path model_directory = argv[1];
        const std::string prompt_text = argv[2];
        const std::filesystem::path output_directory = argv[3];
        const int width = std::stoi(argv[4]);
        const int height = std::stoi(argv[5]);
        const std::uint64_t seed = std::stoull(argv[6]);
        const int threads = std::stoi(argv[7]);
        if (threads <= 0) {
            throw std::invalid_argument("THREADS must be positive");
        }
        ernie_image::NcnnComputeDevice compute_device =
            ernie_image::NcnnComputeDevice::Cpu;
        int gpu_device_index = 0;
        std::filesystem::path initial_latent_path;
        for (int index = 8; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--device") {
                if (++index >= argc) {
                    throw std::invalid_argument("--device requires a value");
                }
                compute_device =
                    ernie_image::parse_ncnn_compute_device(argv[index]);
                continue;
            }
            if (argument == "--gpu-index") {
                if (++index >= argc) {
                    throw std::invalid_argument(
                        "--gpu-index requires a value"
                    );
                }
                gpu_device_index = std::stoi(argv[index]);
                if (gpu_device_index < 0) {
                    throw std::invalid_argument(
                        "--gpu-index must not be negative"
                    );
                }
                continue;
            }
            if (argument == "--initial-latent") {
                if (++index >= argc) {
                    throw std::invalid_argument(
                        "--initial-latent requires a file"
                    );
                }
                initial_latent_path = argv[index];
                continue;
            }
            throw std::invalid_argument("unknown argument: " + argument);
        }

        ernie_image::GenerationConfig config;
        config.width = width;
        config.height = height;
        config.seed = seed;
        config.initial_latent_mode =
            ernie_image::InitialLatentMode::PortablePhilox;
        ernie_image::validate_generation_config(config);
        const ernie_image::ImageGeometry geometry =
            ernie_image::make_image_geometry(width, height);

        std::filesystem::create_directories(output_directory);
        ernie_image::NcnnRuntimeOptions options;
        options.num_threads = threads;
        options.light_mode = true;
        options.compute_device = compute_device;
        options.gpu_device_index = gpu_device_index;
        if (compute_device == ernie_image::NcnnComputeDevice::Vulkan) {
            if (!ernie_image::ncnn_vulkan_compiled()) {
                throw std::runtime_error(
                    "--device vulkan requires the Vulkan ncnn build"
                );
            }
            if (gpu_device_index >= ernie_image::ncnn_vulkan_gpu_count()) {
                throw std::out_of_range(
                    "--gpu-index is outside the available Vulkan GPU range"
                );
            }
        }

        const auto total_started = std::chrono::steady_clock::now();
        const ernie_image::PipelineModelPaths paths =
            ernie_image::PipelineModelPaths::from_model_directory(
                model_directory
            );
        ernie_image::ErnieImagePipeline pipeline(paths, options);
        pipeline.load();

        auto started = std::chrono::steady_clock::now();
        ernie_image::PreparedPrompt prompt =
            pipeline.prepare_prompt(prompt_text);
        const double prompt_seconds = elapsed_seconds(started);

        started = std::chrono::steady_clock::now();
        ernie_image::HostTensor initial_latent;
        std::string initial_policy;
        if (!initial_latent_path.empty()) {
            initial_latent = read_initial_latent(
                initial_latent_path, geometry
            );
            initial_policy = "supplied_f32_file";
        } else {
            ernie_image::InitialLatentProvider provider(
                paths.canonicalized().initial_latent_seed42
            );
            ernie_image::InitialLatentResult initial = provider.create(
                seed,
                ernie_image::InitialLatentMode::PortablePhilox,
                geometry
            );
            initial_latent = std::move(initial.latent);
            initial_policy = initial.policy;
        }
        write_f32_atomic(
            output_directory / "initial_latent.f32", initial_latent
        );
        const double initial_seconds = elapsed_seconds(started);

        ernie_image::DitDenoiseInputs denoise_inputs;
        denoise_inputs.initial_sample = std::move(initial_latent);
        denoise_inputs.text_embeddings.shape = {
            1, prompt.text_length, ernie_image::kTextEncoderHiddenSize,
        };
        denoise_inputs.text_embeddings.storage =
            std::move(prompt.text_embeddings);
        denoise_inputs.geometry = geometry;
        denoise_inputs.start_step = 0;
        denoise_inputs.num_inference_steps = ernie_image::kNumInferenceSteps;

        started = std::chrono::steady_clock::now();
        const auto on_step = [](const ernie_image::DitDenoiseStepResult& step) {
            std::cout << "step_" << step.step_index << ".status=computed\n";
            std::cout << "step_" << step.step_index
                      << ".seconds=" << step.total_seconds << '\n';
            std::cout.flush();
        };
        ernie_image::DitDenoiseResult denoised =
            pipeline.denoise_dit(std::move(denoise_inputs), on_step);
        const double denoise_seconds = elapsed_seconds(started);
        write_f32_atomic(
            output_directory / "final_latent.f32", denoised.final_sample
        );

        started = std::chrono::steady_clock::now();
        ernie_image::VaeDecodeResult decoded =
            pipeline.decode_vae(denoised.final_sample, geometry);
        const double vae_seconds = elapsed_seconds(started);
        write_f32_atomic(
            output_directory / "vae_decoder_input.f32",
            decoded.decoder_input
        );
        write_f32_atomic(
            output_directory / "vae_decoder_output.f32",
            decoded.decoder_output
        );
        write_bytes_atomic(
            output_directory / "rgb_u8.raw",
            decoded.image.pixels.data(),
            decoded.image.pixels.size()
        );
        ernie_image::write_png_atomic(
            output_directory / "image.png", decoded.image
        );

        std::cout << std::setprecision(10);
        std::cout << "device="
                  << ernie_image::ncnn_compute_device_name(compute_device)
                  << '\n';
        if (compute_device == ernie_image::NcnnComputeDevice::Vulkan) {
            std::cout << "gpu_index=" << gpu_device_index << '\n';
            std::cout << "gpu_name="
                      << ernie_image::ncnn_vulkan_gpu_name(gpu_device_index)
                      << '\n';
        }
        std::cout << "seed=" << seed << '\n';
        std::cout << "rng_mode="
                  << (initial_latent_path.empty() ? "portable" : "supplied")
                  << '\n';
        std::cout << "initial_latent_policy=" << initial_policy << '\n';
        std::cout << "width=" << geometry.width << '\n';
        std::cout << "height=" << geometry.height << '\n';
        std::cout << "packed_width=" << geometry.packed_width << '\n';
        std::cout << "packed_height=" << geometry.packed_height << '\n';
        std::cout << "image_tokens=" << geometry.image_tokens << '\n';
        std::cout << "text_length=" << prompt.text_length << '\n';
        std::cout << "sequence_length=" << denoised.sequence_length << '\n';
        std::cout << "completed_steps=" << denoised.num_inference_steps << '\n';
        std::cout << "initial_shape=1,128," << geometry.packed_height
                  << ',' << geometry.packed_width << '\n';
        std::cout << "final_shape=1,128," << geometry.packed_height
                  << ',' << geometry.packed_width << '\n';
        std::cout << "vae_input_shape=1,32," << geometry.vae_latent_height
                  << ',' << geometry.vae_latent_width << '\n';
        std::cout << "vae_output_shape=1,3," << geometry.height
                  << ',' << geometry.width << '\n';
        std::cout << "prompt_seconds=" << prompt_seconds << '\n';
        std::cout << "initial_seconds=" << initial_seconds << '\n';
        std::cout << "denoise_seconds=" << denoise_seconds << '\n';
        std::cout << "vae_seconds=" << vae_seconds << '\n';
        std::cout << "elapsed_seconds=" << elapsed_seconds(total_started)
                  << '\n';
        std::cout << "output="
                  << std::filesystem::canonical(
                         output_directory / "image.png"
                     ) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
