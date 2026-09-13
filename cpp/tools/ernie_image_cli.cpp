#include "ernie_image/pipeline.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Arguments {
    std::filesystem::path model_directory;
    int threads = 16;
    ernie_image::NcnnComputeDevice compute_device =
        ernie_image::NcnnComputeDevice::Cpu;
    int gpu_device_index = 0;
    std::string encode_prompt;
    std::string generation_prompt;
    std::filesystem::path output_path;
    std::uint64_t seed = 42;
    ernie_image::InitialLatentMode rng_mode =
        ernie_image::InitialLatentMode::OfficialReference;
    bool encode = false;
    bool generate = false;
    bool help = false;
};

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  ernie_image_cli --model-dir DIR [--threads N] [--device cpu|vulkan] "
        << "[--gpu-index N]\n"
        << "  ernie_image_cli --model-dir DIR [backend options] --encode PROMPT\n"
        << "  ernie_image_cli --model-dir DIR [backend options] --prompt PROMPT "
        << "--output IMAGE.png [--rng-mode reference|portable] "
        << "[--seed SEED]\n\n"
        << "Backend defaults to CPU; the first Vulkan path is FP32 only.\n"
        << "Without a prompt option, validates the deployment model layout.\n"
        << "--encode runs only the variable-length C++/ncnn Text Encoder.\n"
        << "--prompt runs Text Encoder -> initial latent -> eight-step "
        << "DiT/Scheduler -> VAE -> PNG. RNG mode defaults to reference, "
        << "which accepts only seed=42. Arbitrary seeds require "
        << "--rng-mode portable.\n";
}

std::uint64_t parse_seed(const std::string& text) {
    std::size_t parsed = 0;
    const std::uint64_t seed = std::stoull(text, &parsed, 10);
    if (parsed != text.size()) {
        throw std::invalid_argument("--seed must be an unsigned integer");
    }
    return seed;
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            result.help = true;
            continue;
        }
        if (argument == "--model-dir") {
            if (++index >= argc) {
                throw std::invalid_argument("--model-dir requires a value");
            }
            result.model_directory = argv[index];
            continue;
        }
        if (argument == "--threads") {
            if (++index >= argc) {
                throw std::invalid_argument("--threads requires a value");
            }
            result.threads = std::stoi(argv[index]);
            continue;
        }
        if (argument == "--device") {
            if (++index >= argc) {
                throw std::invalid_argument("--device requires a value");
            }
            result.compute_device =
                ernie_image::parse_ncnn_compute_device(argv[index]);
            continue;
        }
        if (argument == "--gpu-index") {
            if (++index >= argc) {
                throw std::invalid_argument("--gpu-index requires a value");
            }
            result.gpu_device_index = std::stoi(argv[index]);
            if (result.gpu_device_index < 0) {
                throw std::invalid_argument("--gpu-index must not be negative");
            }
            continue;
        }
        if (argument == "--encode") {
            if (++index >= argc) {
                throw std::invalid_argument("--encode requires a UTF-8 prompt");
            }
            result.encode_prompt = argv[index];
            result.encode = true;
            continue;
        }
        if (argument == "--prompt") {
            if (++index >= argc) {
                throw std::invalid_argument("--prompt requires UTF-8 text");
            }
            result.generation_prompt = argv[index];
            result.generate = true;
            continue;
        }
        if (argument == "--output") {
            if (++index >= argc) {
                throw std::invalid_argument("--output requires a PNG path");
            }
            result.output_path = argv[index];
            continue;
        }
        if (argument == "--seed") {
            if (++index >= argc) {
                throw std::invalid_argument("--seed requires a value");
            }
            result.seed = parse_seed(argv[index]);
            continue;
        }
        if (argument == "--rng-mode") {
            if (++index >= argc) {
                throw std::invalid_argument("--rng-mode requires a value");
            }
            result.rng_mode =
                ernie_image::parse_initial_latent_mode(argv[index]);
            continue;
        }
        throw std::invalid_argument("unknown argument: " + argument);
    }
    if (result.encode && result.generate) {
        throw std::invalid_argument(
            "--encode and --prompt are mutually exclusive"
        );
    }
    if (result.generate != !result.output_path.empty()) {
        throw std::invalid_argument(
            "--prompt and --output must be specified together"
        );
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        if (arguments.help) {
            print_usage();
            return 0;
        }
        if (arguments.model_directory.empty()) {
            print_usage();
            return 2;
        }

        ernie_image::NcnnRuntimeOptions options;
        options.num_threads = arguments.threads;
        options.compute_device = arguments.compute_device;
        options.gpu_device_index = arguments.gpu_device_index;

        const bool vulkan_compiled = ernie_image::ncnn_vulkan_compiled();
        const int vulkan_gpu_count =
            vulkan_compiled ? ernie_image::ncnn_vulkan_gpu_count() : 0;
        if (options.compute_device == ernie_image::NcnnComputeDevice::Vulkan) {
            if (!vulkan_compiled) {
                throw std::runtime_error(
                    "--device vulkan requires the Vulkan ncnn build"
                );
            }
            if (options.gpu_device_index >= vulkan_gpu_count) {
                throw std::out_of_range(
                    "--gpu-index is outside the available Vulkan GPU range"
                );
            }
        }

        ernie_image::ErnieImagePipeline pipeline(
            ernie_image::PipelineModelPaths::from_model_directory(
                arguments.model_directory
            ),
            options
        );
        pipeline.load();

        std::cout << std::setprecision(10);
        std::cout << "loaded=true\n";
        std::cout << "model_directory="
                  << pipeline.model_paths().model_directory << '\n';
        std::cout << "threads=" << pipeline.runtime_options().num_threads
                  << '\n';
        std::cout << std::boolalpha;
        std::cout << "device="
                  << ernie_image::ncnn_compute_device_name(
                         pipeline.runtime_options().compute_device
                     ) << '\n';
        std::cout << "compute_precision=fp32\n";
        std::cout << "ncnn_vulkan_compiled=" << vulkan_compiled << '\n';
        std::cout << "vulkan_gpu_count=" << vulkan_gpu_count << '\n';
        if (options.compute_device == ernie_image::NcnnComputeDevice::Vulkan) {
            std::cout << "gpu_index=" << options.gpu_device_index << '\n';
            std::cout << "gpu_name="
                      << ernie_image::ncnn_vulkan_gpu_name(
                             options.gpu_device_index
                         ) << '\n';
        }
        std::cout << "model_layout_validated=true\n";
        std::cout << "vae_decode_ready=true\n";
        std::cout << "png_output_ready=true\n";
        std::cout << "generation_ready=true\n";
        std::cout.flush();

        if (arguments.encode) {
            ernie_image::PreparedPrompt prompt =
                pipeline.prepare_prompt(arguments.encode_prompt);
            std::cout << "text_length=" << prompt.text_length << '\n';
            std::cout << "sequence_length=" << prompt.sequence_length << '\n';
            std::cout << "text_shape=1," << prompt.text_length
                      << ",3072\n";
            std::cout << "text_f32_count="
                      << prompt.text_embeddings.size() << '\n';
        }

        if (arguments.generate) {
            ernie_image::GenerationRequest request;
            request.prompt = arguments.generation_prompt;
            request.output_path = arguments.output_path;
            request.config.seed = arguments.seed;
            request.config.initial_latent_mode = arguments.rng_mode;

            const auto report_step = [](
                const ernie_image::DitDenoiseStepResult& step
            ) {
                std::cout << "step_" << step.step_index
                          << ".status=computed\n";
                std::cout << "step_" << step.step_index
                          << ".timestep=" << step.scheduler_timestep << '\n';
                std::cout << "step_" << step.step_index
                          << ".dit_seconds=" << step.dit_seconds << '\n';
                std::cout << "step_" << step.step_index
                          << ".scheduler_seconds="
                          << step.scheduler_seconds << '\n';
                std::cout << "step_" << step.step_index
                          << ".total_seconds=" << step.total_seconds << '\n';
                std::cout.flush();
            };

            const ernie_image::GenerationResult result =
                pipeline.generate(request, report_step);
            std::cout << "seed=" << result.seed << '\n';
            std::cout << "rng_mode="
                      << ernie_image::initial_latent_mode_name(
                             result.initial_latent_mode
                         ) << '\n';
            std::cout << "initial_latent_policy="
                      << result.initial_latent_policy << '\n';
            std::cout << "text_length=" << result.text_length << '\n';
            std::cout << "sequence_length=" << result.sequence_length << '\n';
            std::cout << "completed_steps="
                      << result.num_inference_steps << '\n';
            std::cout << "image_width=" << result.width << '\n';
            std::cout << "image_height=" << result.height << '\n';
            std::cout << "prompt_seconds=" << result.prompt_seconds << '\n';
            std::cout << "initial_latent_seconds="
                      << result.initial_latent_seconds << '\n';
            std::cout << "denoise_seconds="
                      << result.denoise_seconds << '\n';
            std::cout << "vae_seconds=" << result.vae_seconds << '\n';
            std::cout << "png_seconds=" << result.png_seconds << '\n';
            std::cout << "elapsed_seconds=" << result.elapsed_seconds << '\n';
            std::cout << "output=" << result.output_path << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
