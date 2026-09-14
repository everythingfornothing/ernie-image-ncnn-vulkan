#include "ernie_image/dit.h"

#include "ernie_image/latent.h"
#include "ernie_image/runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ernie_image {

DynamicDitModel::DynamicDitModel(
    DitModelFiles model_files,
    NcnnRuntimeOptions options
) : model_files_(std::move(model_files)), options_(options) {
    if (options_.num_threads <= 0) {
        throw std::invalid_argument("DiT ncnn num_threads must be positive");
    }
    if (
        model_files_.chunks.front().first_layer != 0 ||
        model_files_.chunks.back().last_layer != 35
    ) {
        throw std::invalid_argument("DiT model files must cover layers 0-35");
    }
}

const DitModelFiles& DynamicDitModel::model_files() const noexcept {
    return model_files_;
}

const NcnnRuntimeOptions& DynamicDitModel::runtime_options() const noexcept {
    return options_;
}

namespace {

double elapsed_seconds(
    const std::chrono::steady_clock::time_point& started
) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
}

void require_fp32(
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

const std::vector<std::string>& frontend_output_names() {
    static const std::vector<std::string> names = {
        "out0", "out1", "out2", "out3", "out4",
        "out5", "out6", "out7", "out8",
    };
    return names;
}

void require_contract(
    const ModelIoContract& contract
) {
    const std::vector<std::string> expected_inputs = {
        "in0", "in1", "in2",
    };
    if (contract.input_names != expected_inputs) {
        throw std::runtime_error(
            "DiT frontend ncnn input contract is not in0,in1,in2"
        );
    }

    const auto& expected_outputs = frontend_output_names();
    if (contract.output_names.size() != expected_outputs.size()) {
        throw std::runtime_error(
            "DiT frontend ncnn graph must expose exactly 9 outputs"
        );
    }
    for (const std::string& name : expected_outputs) {
        if (
            std::find(
                contract.output_names.begin(),
                contract.output_names.end(),
                name
            ) == contract.output_names.end()
        ) {
            throw std::runtime_error(
                "DiT frontend ncnn graph is missing output " + name
            );
        }
    }
}

}  // namespace

HostTensor make_joint_tokens(
    const HostTensor& image_tokens,
    const HostTensor& text_tokens,
    int text_length,
    int image_token_count
) {
    require_fp32(
        image_tokens,
        {1, image_token_count, kDitHiddenSize},
        "DiT image tokens"
    );
    require_fp32(
        text_tokens,
        {1, text_length, kDitHiddenSize},
        "DiT text tokens"
    );
    const auto& image_values =
        std::get<std::vector<float>>(image_tokens.storage);
    const auto& text_values =
        std::get<std::vector<float>>(text_tokens.storage);
    std::vector<float> values(image_values.size() + text_values.size());
    std::memcpy(
        values.data(),
        image_values.data(),
        image_values.size() * sizeof(float)
    );
    std::memcpy(
        values.data() + image_values.size(),
        text_values.data(),
        text_values.size() * sizeof(float)
    );

    HostTensor joint;
    joint.shape = {1, image_token_count + text_length, kDitHiddenSize};
    joint.storage = std::move(values);
    return joint;
}

HostTensor unpack_output_head_prediction(
    HostTensor projected_tokens,
    int sequence_length,
    const ImageGeometry& geometry
) {
    const std::vector<int> packed_shape =
        geometry.dit_packed_latent_shape();
    if (projected_tokens.shape == packed_shape) {
        // Compatibility with the original fixed 1024x1024 output-head graph.
        validate_latent_tensor(projected_tokens, geometry);
        return projected_tokens;
    }

    require_fp32(
        projected_tokens,
        {1, sequence_length, kLatentChannels},
        "DiT projected output tokens"
    );
    const auto& projected =
        std::get<std::vector<float>>(projected_tokens.storage);
    std::vector<float> packed(
        static_cast<std::size_t>(kLatentChannels) * geometry.image_tokens
    );
    for (int token = 0; token < geometry.image_tokens; ++token) {
        for (int channel = 0; channel < kLatentChannels; ++channel) {
            packed[
                static_cast<std::size_t>(channel) * geometry.image_tokens +
                token
            ] = projected[
                static_cast<std::size_t>(token) * kLatentChannels + channel
            ];
        }
    }
    return make_latent_tensor(std::move(packed), geometry);
}

HostTensor make_runtime_tensor(
    std::vector<int> shape,
    const std::vector<float>& values
) {
    HostTensor tensor;
    tensor.shape = std::move(shape);
    tensor.storage = values;
    return tensor;
}

void require_out0_contract(
    const ModelIoContract& contract,
    int input_count,
    const std::string& stage
) {
    std::vector<std::string> expected_inputs;
    expected_inputs.reserve(static_cast<std::size_t>(input_count));
    for (int index = 0; index < input_count; ++index) {
        expected_inputs.push_back("in" + std::to_string(index));
    }
    if (contract.input_names != expected_inputs) {
        throw std::runtime_error(
            stage + " has an unexpected ncnn input contract"
        );
    }
    if (
        std::find(
            contract.output_names.begin(),
            contract.output_names.end(),
            "out0"
        ) == contract.output_names.end()
    ) {
        throw std::runtime_error(
            stage + " does not expose required ncnn output out0"
        );
    }
}

HostTensor run_out0_positional(
    const NcnnModel& model,
    std::vector<HostTensor> inputs
) {
    std::vector<NamedHostTensor> named_inputs;
    named_inputs.reserve(inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        named_inputs.push_back({
            "in" + std::to_string(index),
            std::move(inputs[index]),
        });
    }
    std::vector<NamedHostTensor> outputs = model.run_many(
        named_inputs, {"out0"}
    );
    return std::move(outputs.front().tensor);
}

std::string chunk_name(const LayerChunkModelFiles& chunk) {
    return "layers_" + std::to_string(chunk.first_layer) + "_" +
        std::to_string(chunk.last_layer);
}

DitFrontendOutputs DynamicDitModel::run_frontend(
    DitFrontendInputs inputs
) const {
    const ImageGeometry geometry = make_image_geometry(
        inputs.geometry.width, inputs.geometry.height
    );
    validate_latent_tensor(inputs.latent, geometry);
    require_fp32(inputs.timestep, {1, 1}, "DiT timestep");
    if (
        inputs.text_embeddings.shape.size() != 3 ||
        inputs.text_embeddings.shape[0] != 1 ||
        inputs.text_embeddings.shape[2] != kTextEncoderHiddenSize
    ) {
        throw std::invalid_argument(
            "DiT text embeddings must have shape [1,N,3072]"
        );
    }
    const int text_length = inputs.text_embeddings.shape[1];
    if (text_length <= 0 || text_length > kTokenizerMaxLength) {
        throw std::invalid_argument("DiT text length must be in [1,2048]");
    }
    require_fp32(
        inputs.text_embeddings,
        {1, text_length, kTextEncoderHiddenSize},
        "DiT text embeddings"
    );

    NcnnRuntimeOptions frontend_options = options_;
    frontend_options.light_mode = false;
    NcnnModel model(frontend_options);
    auto started = std::chrono::steady_clock::now();
    model.load(model_files_.frontend.param, model_files_.frontend.bin);
    const double load_seconds = elapsed_seconds(started);
    require_contract(model.io_contract());

    started = std::chrono::steady_clock::now();
    std::vector<NamedHostTensor> named_outputs = model.run_many(
        {
            {"in0", std::move(inputs.latent)},
            {"in1", std::move(inputs.timestep)},
            {"in2", std::move(inputs.text_embeddings)},
        },
        frontend_output_names()
    );
    const double inference_seconds = elapsed_seconds(started);
    if (named_outputs.size() != 9) {
        throw std::runtime_error("DiT frontend did not return 9 outputs");
    }
    std::vector<HostTensor> outputs;
    outputs.reserve(named_outputs.size());
    for (NamedHostTensor& output : named_outputs) {
        outputs.push_back(std::move(output.tensor));
    }

    // ncnn Mat has no explicit batch axis. For the unsqueezed AdaLN graph
    // outputs, dims=3 represents logical [B=1,H=1,W=4096], while the generic
    // HostTensor converter conservatively interprets it as [B=1,C=1,H=1,W].
    // Normalize only these model-specific singleton axes here.
    for (int index = 0; index < kDitAdaLnInputCount; ++index) {
        HostTensor& value = outputs[static_cast<std::size_t>(index + 2)];
        if (value.shape == std::vector<int>({1, 1, 1, kDitHiddenSize})) {
            value.shape = {1, 1, kDitHiddenSize};
        }
    }
    if (outputs[8].shape == std::vector<int>({1, 1, kDitHiddenSize})) {
        outputs[8].shape = {1, kDitHiddenSize};
    }

    require_fp32(
        outputs[0],
        {1, geometry.image_tokens, kDitHiddenSize},
        "DiT image tokens"
    );
    require_fp32(
        outputs[1],
        {1, text_length, kDitHiddenSize},
        "DiT text tokens"
    );
    for (int index = 0; index < kDitAdaLnInputCount; ++index) {
        require_fp32(
            outputs[static_cast<std::size_t>(index + 2)],
            {1, 1, kDitHiddenSize},
            kDitAdaLnNames[static_cast<std::size_t>(index)]
        );
    }
    require_fp32(
        outputs[8],
        {1, kDitHiddenSize},
        "DiT conditioning"
    );

    DitFrontendOutputs result;
    result.geometry = geometry;
    result.text_length = text_length;
    result.sequence_length = geometry.image_tokens + text_length;
    result.image_tokens = std::move(outputs[0]);
    result.text_tokens = std::move(outputs[1]);
    for (int index = 0; index < kDitAdaLnInputCount; ++index) {
        result.adaln[static_cast<std::size_t>(index)] =
            std::move(outputs[static_cast<std::size_t>(index + 2)]);
    }
    result.conditioning = std::move(outputs[8]);
    result.model_load_seconds = load_seconds;
    result.inference_seconds = inference_seconds;
    return result;
}

DitPredictResult DynamicDitModel::predict(
    DitFrontendInputs inputs,
    const DitChunkCallback& on_chunk
) const {
    const auto total_started = std::chrono::steady_clock::now();
    DitFrontendOutputs frontend = run_frontend(std::move(inputs));
    const int text_length = frontend.text_length;
    const int sequence_length = frontend.sequence_length;
    const ImageGeometry geometry = frontend.geometry;

    DitPredictResult result;
    result.geometry = geometry;
    result.text_length = text_length;
    result.sequence_length = sequence_length;
    result.stages.reserve(2 + model_files_.chunks.size());
    result.stages.push_back({
        "frontend",
        -1,
        -1,
        model_files_.frontend.param,
        model_files_.frontend.bin,
        frontend.model_load_seconds,
        frontend.inference_seconds,
    });

    HostTensor joint = make_joint_tokens(
        frontend.image_tokens,
        frontend.text_tokens,
        text_length,
        geometry.image_tokens
    );
    require_fp32(
        joint,
        {1, sequence_length, kDitHiddenSize},
        "DiT joint tokens"
    );
    frontend.image_tokens = HostTensor();
    frontend.text_tokens = HostTensor();

    const DitRuntimeTensors runtime =
        make_dit_runtime_tensors(
            text_length,
            geometry.packed_height,
            geometry.packed_width
        );
    HostTensor rotary = make_runtime_tensor(
        {1, sequence_length, 1, kDitHeadDim},
        runtime.rotary_frequencies
    );
    HostTensor mask = make_runtime_tensor(
        {1, 1, sequence_length},
        runtime.additive_attention_mask
    );
    require_fp32(
        rotary,
        {1, sequence_length, 1, kDitHeadDim},
        "DiT rotary frequencies"
    );
    require_fp32(
        mask,
        {1, 1, sequence_length},
        "DiT attention mask"
    );

    for (const LayerChunkModelFiles& chunk : model_files_.chunks) {
        const std::string name = chunk_name(chunk);
        NcnnModel model(options_);
        auto started = std::chrono::steady_clock::now();
        model.load(chunk.model.param, chunk.model.bin);
        const double load_seconds = elapsed_seconds(started);
        require_out0_contract(model.io_contract(), 9, name);

        std::vector<HostTensor> chunk_inputs;
        chunk_inputs.reserve(9);
        chunk_inputs.push_back(std::move(joint));
        chunk_inputs.push_back(rotary);
        for (const HostTensor& base : frontend.adaln) {
            chunk_inputs.push_back(base);
        }
        chunk_inputs.push_back(mask);

        started = std::chrono::steady_clock::now();
        joint = run_out0_positional(model, std::move(chunk_inputs));
        const double inference_seconds = elapsed_seconds(started);
        require_fp32(
            joint,
            {1, sequence_length, kDitHiddenSize},
            name.c_str()
        );
        result.stages.push_back({
            name,
            chunk.first_layer,
            chunk.last_layer,
            chunk.model.param,
            chunk.model.bin,
            load_seconds,
            inference_seconds,
        });
        if (on_chunk) {
            on_chunk(result.stages.back(), joint);
        }
    }

    NcnnModel output_head(options_);
    auto started = std::chrono::steady_clock::now();
    output_head.load(
        model_files_.output_head.param,
        model_files_.output_head.bin
    );
    const double head_load_seconds = elapsed_seconds(started);
    require_out0_contract(output_head.io_contract(), 2, "output_head");

    std::vector<HostTensor> head_inputs;
    head_inputs.reserve(2);
    head_inputs.push_back(std::move(joint));
    head_inputs.push_back(std::move(frontend.conditioning));
    started = std::chrono::steady_clock::now();
    result.prediction = unpack_output_head_prediction(
        run_out0_positional(output_head, std::move(head_inputs)),
        sequence_length,
        geometry
    );
    const double head_inference_seconds = elapsed_seconds(started);
    require_fp32(
        result.prediction,
        geometry.dit_packed_latent_shape(),
        "DiT prediction"
    );
    result.stages.push_back({
        "output_head",
        -1,
        -1,
        model_files_.output_head.param,
        model_files_.output_head.bin,
        head_load_seconds,
        head_inference_seconds,
    });
    result.total_seconds = elapsed_seconds(total_started);
    return result;
}

DitDenoiseResult DynamicDitModel::denoise(
    DitDenoiseInputs inputs,
    const DitDenoiseStepCallback& on_step
) const {
    const auto total_started = std::chrono::steady_clock::now();
    if (inputs.num_inference_steps != kNumInferenceSteps) {
        throw std::invalid_argument("DiT denoise requires exactly 8 steps");
    }
    if (
        inputs.start_step < 0 ||
        inputs.start_step > inputs.num_inference_steps
    ) {
        throw std::invalid_argument(
            "DiT denoise start_step must be in [0,8]"
        );
    }
    const ImageGeometry geometry = make_image_geometry(
        inputs.geometry.width, inputs.geometry.height
    );
    validate_latent_tensor(inputs.initial_sample, geometry);
    if (
        inputs.text_embeddings.shape.size() != 3 ||
        inputs.text_embeddings.shape[0] != 1 ||
        inputs.text_embeddings.shape[2] != kTextEncoderHiddenSize
    ) {
        throw std::invalid_argument(
            "DiT denoise text embeddings must have shape [1,N,3072]"
        );
    }
    const int text_length = inputs.text_embeddings.shape[1];
    if (text_length <= 0 || text_length > kTokenizerMaxLength) {
        throw std::invalid_argument(
            "DiT denoise text length must be in [1,2048]"
        );
    }
    require_fp32(
        inputs.text_embeddings,
        {1, text_length, kTextEncoderHiddenSize},
        "DiT denoise text embeddings"
    );

    const FlowMatchSchedule schedule = make_flow_match_schedule(
        inputs.num_inference_steps
    );
    HostTensor sample = std::move(inputs.initial_sample);
    const HostTensor text_embeddings = std::move(inputs.text_embeddings);

    DitDenoiseResult result;
    result.geometry = geometry;
    result.text_length = text_length;
    result.sequence_length = geometry.image_tokens + text_length;
    result.start_step = inputs.start_step;
    result.num_inference_steps = inputs.num_inference_steps;
    result.steps.reserve(static_cast<std::size_t>(
        inputs.num_inference_steps - inputs.start_step
    ));

    for (
        int index = inputs.start_step;
        index < inputs.num_inference_steps;
        ++index
    ) {
        const auto step_started = std::chrono::steady_clock::now();
        DitFrontendInputs predict_inputs;
        predict_inputs.latent = sample;
        predict_inputs.timestep.shape = {1, 1};
        predict_inputs.timestep.storage = std::vector<float>{
            schedule.model_timesteps_bf16_as_fp32[
                static_cast<std::size_t>(index)
            ]
        };
        predict_inputs.text_embeddings = text_embeddings;
        predict_inputs.geometry = geometry;

        DitPredictResult prediction = predict(std::move(predict_inputs));
        const auto scheduler_started = std::chrono::steady_clock::now();
        const auto& sample_values =
            std::get<std::vector<float>>(sample.storage);
        const auto& prediction_values =
            std::get<std::vector<float>>(prediction.prediction.storage);
        std::vector<float> next_values = flow_match_euler_step(
            sample_values,
            prediction_values,
            schedule.sigmas[static_cast<std::size_t>(index)],
            schedule.sigmas[static_cast<std::size_t>(index + 1)]
        );
        const double scheduler_seconds = elapsed_seconds(scheduler_started);

        HostTensor next_sample;
        next_sample.shape = geometry.dit_packed_latent_shape();
        next_sample.storage = std::move(next_values);
        validate_latent_tensor(next_sample, geometry);

        DitDenoiseStepResult step;
        step.step_index = index;
        step.scheduler_timestep =
            schedule.timesteps[static_cast<std::size_t>(index)];
        step.model_timestep_bf16_as_fp32 =
            schedule.model_timesteps_bf16_as_fp32[
                static_cast<std::size_t>(index)
            ];
        step.sigma = schedule.sigmas[static_cast<std::size_t>(index)];
        step.sigma_next =
            schedule.sigmas[static_cast<std::size_t>(index + 1)];
        step.prediction = std::move(prediction.prediction);
        step.sample = next_sample;
        step.stages = std::move(prediction.stages);
        step.dit_seconds = prediction.total_seconds;
        step.scheduler_seconds = scheduler_seconds;
        step.total_seconds = elapsed_seconds(step_started);
        if (on_step) {
            on_step(step);
        }
        result.steps.push_back(std::move(step));
        sample = std::move(next_sample);
    }

    result.final_sample = std::move(sample);
    validate_latent_tensor(result.final_sample, geometry);
    result.total_seconds = elapsed_seconds(total_started);
    return result;
}


}  // namespace ernie_image
