#include "ernie_image/pipeline.h"

#include "ernie_image/dit.h"
#include "ernie_image/image_io.h"
#include "ernie_image/random.h"
#include "ernie_image/runtime.h"
#include "ernie_image/text_encoder.h"
#include "ernie_image/vae.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace ernie_image {

void validate_generation_config(const GenerationConfig& config) {
    if (config.batch_size != 1) {
        throw std::invalid_argument("first C++ runtime requires batch_size=1");
    }
    const ImageGeometry geometry =
        make_image_geometry(config.width, config.height);
    if (config.num_inference_steps != kNumInferenceSteps) {
        throw std::invalid_argument("first C++ runtime requires 8 steps");
    }
    if (config.guidance_scale != 1.0f) {
        throw std::invalid_argument(
            "first C++ runtime requires guidance_scale=1.0"
        );
    }
    if (config.use_prompt_enhancer) {
        throw std::invalid_argument(
            "Prompt Enhancer is disabled in the first C++ runtime"
        );
    }
    if (config.initial_latent_mode == InitialLatentMode::OfficialReference) {
        if (config.seed != kOfficialReferenceSeed) {
            throw std::invalid_argument(
                "reference RNG mode supports only seed=42; use portable RNG "
                "mode explicitly for arbitrary seeds"
            );
        }
        if (geometry.width != kImageWidth || geometry.height != kImageHeight) {
            throw std::invalid_argument(
                "reference RNG mode supports only resolution 1024x1024; "
                "use portable RNG mode for dynamic resolutions"
            );
        }
    }
}

struct ErnieImagePipeline::Impl {
    Impl(PipelineModelPaths paths, NcnnRuntimeOptions runtime_options)
        : requested_paths(std::move(paths)), options(runtime_options) {
        if (options.num_threads <= 0) {
            throw std::invalid_argument(
                "Pipeline ncnn num_threads must be positive"
            );
        }
    }

    PipelineModelPaths requested_paths;
    PipelineModelPaths paths;
    NcnnRuntimeOptions options;
    std::unique_ptr<DynamicTextEncoder> text_encoder;
    std::unique_ptr<DynamicDitModel> dit;
    std::unique_ptr<DynamicVaeDecoder> vae_decoder;
    std::unique_ptr<InitialLatentProvider> initial_latent;
    bool is_loaded = false;
};

ErnieImagePipeline::ErnieImagePipeline(
    PipelineModelPaths model_paths,
    NcnnRuntimeOptions options
) : impl_(std::make_unique<Impl>(std::move(model_paths), options)) {}

ErnieImagePipeline::~ErnieImagePipeline() = default;
ErnieImagePipeline::ErnieImagePipeline(ErnieImagePipeline&&) noexcept = default;
ErnieImagePipeline& ErnieImagePipeline::operator=(
    ErnieImagePipeline&&
) noexcept = default;

void ErnieImagePipeline::load() {
    PipelineModelPaths paths = impl_->requested_paths.canonicalized();
    auto text_encoder = std::make_unique<DynamicTextEncoder>(
        paths.tokenizer_json,
        paths.text_encoder_directory,
        impl_->options
    );
    auto dit = std::make_unique<DynamicDitModel>(paths.dit, impl_->options);
    auto vae_decoder = std::make_unique<DynamicVaeDecoder>(
        paths.vae_decoder,
        paths.vae_running_mean,
        paths.vae_running_var,
        impl_->options
    );
    auto initial_latent = std::make_unique<InitialLatentProvider>(
        paths.initial_latent_seed42
    );

    impl_->paths = std::move(paths);
    impl_->text_encoder = std::move(text_encoder);
    impl_->dit = std::move(dit);
    impl_->vae_decoder = std::move(vae_decoder);
    impl_->initial_latent = std::move(initial_latent);
    impl_->is_loaded = true;
}

bool ErnieImagePipeline::loaded() const noexcept {
    return impl_->is_loaded;
}

PreparedPrompt ErnieImagePipeline::prepare_prompt(
    const std::string& utf8_prompt
) const {
    if (!loaded()) {
        throw std::logic_error("Pipeline must be loaded before prepare_prompt");
    }
    TextEncoderBatchResult encoded =
        impl_->text_encoder->encode_batch({utf8_prompt}, false);
    if (encoded.cases.size() != 1) {
        throw std::runtime_error(
            "Text Encoder did not return exactly one prompt result"
        );
    }
    TextEncoderCaseResult& item = encoded.cases.front();
    if (item.text_length <= 0 || item.text_length > kTokenizerMaxLength) {
        throw std::runtime_error("Text Encoder returned invalid token length");
    }
    const std::size_t expected =
        static_cast<std::size_t>(item.text_length) * kTextEncoderHiddenSize;
    if (item.hidden_states.size() != expected) {
        throw std::runtime_error(
            "Text Encoder B04 element count does not match [1,N,3072]"
        );
    }
    PreparedPrompt result;
    result.text_length = item.text_length;
    result.sequence_length = kImageTokens + item.text_length;
    result.text_embeddings = std::move(item.hidden_states);
    return result;
}

DitFrontendOutputs ErnieImagePipeline::run_dit_frontend(
    DitFrontendInputs inputs
) const {
    if (!loaded()) {
        throw std::logic_error(
            "Pipeline must be loaded before run_dit_frontend"
        );
    }
    return impl_->dit->run_frontend(std::move(inputs));
}

DitPredictResult ErnieImagePipeline::predict_dit(
    DitFrontendInputs inputs,
    const DitChunkCallback& on_chunk
) const {
    if (!loaded()) {
        throw std::logic_error(
            "Pipeline must be loaded before predict_dit"
        );
    }
    return impl_->dit->predict(std::move(inputs), on_chunk);
}

DitDenoiseResult ErnieImagePipeline::denoise_dit(
    DitDenoiseInputs inputs,
    const DitDenoiseStepCallback& on_step
) const {
    if (!loaded()) {
        throw std::logic_error(
            "Pipeline must be loaded before denoise_dit"
        );
    }
    return impl_->dit->denoise(std::move(inputs), on_step);
}

VaeDecodeResult ErnieImagePipeline::decode_vae(
    const HostTensor& packed_latent
) const {
    if (!loaded()) {
        throw std::logic_error("Pipeline must be loaded before decode_vae");
    }
    return impl_->vae_decoder->decode(packed_latent);
}

VaeDecodeResult ErnieImagePipeline::decode_vae(
    const HostTensor& packed_latent,
    const ImageGeometry& geometry
) const {
    if (!loaded()) {
        throw std::logic_error("Pipeline must be loaded before decode_vae");
    }
    return impl_->vae_decoder->decode(packed_latent, geometry);
}

GenerationResult ErnieImagePipeline::generate(
    const GenerationRequest& request,
    const DitDenoiseStepCallback& on_step
) const {
    if (!loaded()) {
        throw std::logic_error("Pipeline must be loaded before generate");
    }
    validate_generation_config(request.config);
    if (request.output_path.empty()) {
        throw std::invalid_argument("generation output path must not be empty");
    }

    const ImageGeometry geometry = make_image_geometry(
        request.config.width, request.config.height
    );
    const auto total_started = std::chrono::steady_clock::now();
    auto started = std::chrono::steady_clock::now();
    PreparedPrompt prompt = prepare_prompt(request.prompt);

    GenerationResult result;
    result.seed = request.config.seed;
    result.initial_latent_mode = request.config.initial_latent_mode;
    result.width = request.config.width;
    result.height = request.config.height;
    result.num_inference_steps = request.config.num_inference_steps;
    result.text_length = prompt.text_length;
    result.image_tokens = geometry.image_tokens;
    result.sequence_length = geometry.image_tokens + prompt.text_length;
    result.prompt_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();

    InitialLatentResult initial =
        impl_->initial_latent->create(
            request.config.seed,
            request.config.initial_latent_mode,
            geometry
        );
    result.initial_latent_policy = initial.policy;
    result.initial_latent_seconds = initial.load_seconds;

    DitDenoiseInputs denoise_inputs;
    denoise_inputs.initial_sample = std::move(initial.latent);
    denoise_inputs.text_embeddings.shape = {
        1, prompt.text_length, kTextEncoderHiddenSize,
    };
    denoise_inputs.text_embeddings.storage =
        std::move(prompt.text_embeddings);
    denoise_inputs.start_step = 0;
    denoise_inputs.num_inference_steps = request.config.num_inference_steps;
    denoise_inputs.geometry = geometry;

    started = std::chrono::steady_clock::now();
    DitDenoiseResult denoised =
        impl_->dit->denoise(std::move(denoise_inputs), on_step);
    result.denoise_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();

    started = std::chrono::steady_clock::now();
    VaeDecodeResult decoded =
        impl_->vae_decoder->decode(denoised.final_sample, geometry);
    result.vae_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();

    started = std::chrono::steady_clock::now();
    write_png_atomic(request.output_path, decoded.image);
    result.png_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    result.output_path = std::filesystem::canonical(request.output_path);
    result.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - total_started
    ).count();
    return result;
}

const PipelineModelPaths& ErnieImagePipeline::model_paths() const {
    if (!loaded()) {
        throw std::logic_error("Pipeline model paths are not resolved before load");
    }
    return impl_->paths;
}

const NcnnRuntimeOptions&
ErnieImagePipeline::runtime_options() const noexcept {
    return impl_->options;
}

}  // namespace ernie_image
