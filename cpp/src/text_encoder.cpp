#include "ernie_image/text_encoder.h"

#include "ernie_image/tokenizer.h"

#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ernie_image {

namespace {

struct ChunkRange {
    int first;
    int last;
};

constexpr std::array<ChunkRange, 7> kChunks = {{
    {0, 0},
    {1, 4},
    {5, 8},
    {9, 12},
    {13, 16},
    {17, 20},
    {21, 24},
}};

struct ModelFiles {
    std::filesystem::path param;
    std::filesystem::path model;
};

struct WorkingCase {
    TextEncoderCaseResult result;
    TextRuntimeTensors runtime;
    std::vector<float> hidden;
};

std::string two_digits(int value) {
    if (value < 0 || value > 99) {
        throw std::invalid_argument("layer index is outside two-digit range");
    }
    std::string result = std::to_string(value);
    if (result.size() == 1) {
        result.insert(result.begin(), '0');
    }
    return result;
}

ModelFiles model_files(
    const std::filesystem::path& directory,
    const std::string& stem
) {
    ModelFiles files{
        directory / (stem + ".ncnn.param"),
        directory / (stem + ".ncnn.bin"),
    };
    if (!std::filesystem::is_regular_file(files.param)) {
        throw std::runtime_error(
            "Text Encoder param file not found: " + files.param.string()
        );
    }
    if (!std::filesystem::is_regular_file(files.model)) {
        throw std::runtime_error(
            "Text Encoder model file not found: " + files.model.string()
        );
    }
    return files;
}

ModelFiles embedding_files(const std::filesystem::path& directory) {
    return model_files(directory, "embedding");
}

ModelFiles chunk_files(
    const std::filesystem::path& directory,
    const ChunkRange& chunk
) {
    return model_files(
        directory,
        "layers_" + two_digits(chunk.first) + "_" + two_digits(chunk.last)
    );
}

HostTensor make_float_tensor(
    std::vector<int> shape,
    const std::vector<float>& values
) {
    HostTensor result;
    result.shape = std::move(shape);
    result.storage = values;
    return result;
}

std::vector<float> take_hidden(
    HostTensor& output,
    int text_length,
    const std::string& stage
) {
    const std::vector<int> expected_shape = {1, text_length, 3072};
    if (output.shape != expected_shape) {
        throw std::runtime_error(
            stage + " returned an unexpected logical shape"
        );
    }
    auto* values = std::get_if<std::vector<float>>(&output.storage);
    if (values == nullptr) {
        throw std::runtime_error(stage + " did not return FP32");
    }
    for (float value : *values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(stage + " returned NaN or Inf");
        }
    }
    return std::move(*values);
}

double elapsed_seconds(
    const std::chrono::steady_clock::time_point& started
) {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
}

}  // namespace

struct DynamicTextEncoder::Impl {
    Impl(
        const std::filesystem::path& tokenizer_path,
        const std::filesystem::path& models,
        NcnnRuntimeOptions runtime_options
    )
        : tokenizer(tokenizer_path),
          model_dir(std::filesystem::canonical(models)),
          options(runtime_options) {
        if (!std::filesystem::is_directory(model_dir)) {
            throw std::runtime_error(
                "Text Encoder model directory not found: " + model_dir.string()
            );
        }
        static_cast<void>(embedding_files(model_dir));
        for (const ChunkRange& chunk : kChunks) {
            static_cast<void>(chunk_files(model_dir, chunk));
        }
    }

    ErnieTokenizer tokenizer;
    std::filesystem::path model_dir;
    NcnnRuntimeOptions options;
};

DynamicTextEncoder::DynamicTextEncoder(
    const std::filesystem::path& tokenizer_json,
    const std::filesystem::path& model_directory,
    NcnnRuntimeOptions options
) : impl_(std::make_unique<Impl>(
        tokenizer_json, model_directory, options
    )) {}

DynamicTextEncoder::~DynamicTextEncoder() = default;
DynamicTextEncoder::DynamicTextEncoder(DynamicTextEncoder&&) noexcept = default;
DynamicTextEncoder& DynamicTextEncoder::operator=(
    DynamicTextEncoder&&
) noexcept = default;

TextEncoderBatchResult DynamicTextEncoder::encode_batch(
    const std::vector<std::string>& utf8_prompts,
    bool capture_intermediates
) const {
    if (utf8_prompts.empty()) {
        throw std::invalid_argument("Text Encoder prompt batch must not be empty");
    }

    std::vector<WorkingCase> working;
    working.reserve(utf8_prompts.size());
    for (const std::string& prompt : utf8_prompts) {
        WorkingCase item;
        item.result.input_ids = impl_->tokenizer.encode(prompt);
        item.result.text_length =
            static_cast<int>(item.result.input_ids.size());
        item.runtime = make_text_runtime_tensors(item.result.text_length);
        if (capture_intermediates) {
            item.result.runtime_tensors = item.runtime;
        } else {
            item.result.runtime_tensors.text_length = item.result.text_length;
        }
        working.push_back(std::move(item));
    }

    TextEncoderBatchResult batch;
    batch.stages.reserve(1 + kChunks.size());

    {
        const ModelFiles files = embedding_files(impl_->model_dir);
        TextEncoderStageTiming timing;
        timing.name = "embedding";
        timing.param_path = std::filesystem::canonical(files.param);
        timing.model_path = std::filesystem::canonical(files.model);
        timing.inference_seconds.reserve(working.size());

        NcnnModel model(impl_->options);
        auto started = std::chrono::steady_clock::now();
        model.load(files.param, files.model);
        timing.load_seconds = elapsed_seconds(started);
        for (WorkingCase& item : working) {
            HostTensor ids;
            ids.shape = {1, item.result.text_length};
            ids.storage = item.result.input_ids;
            started = std::chrono::steady_clock::now();
            HostTensor output = model.run_positional({std::move(ids)});
            timing.inference_seconds.push_back(elapsed_seconds(started));
            item.hidden = take_hidden(
                output, item.result.text_length, timing.name
            );
            if (capture_intermediates) {
                item.result.embedding_fp32 = item.hidden;
            }
        }
        batch.stages.push_back(std::move(timing));
    }

    for (const ChunkRange& chunk : kChunks) {
        const ModelFiles files = chunk_files(impl_->model_dir, chunk);
        TextEncoderStageTiming timing;
        timing.name =
            "layers_" + two_digits(chunk.first) + "_" + two_digits(chunk.last);
        timing.param_path = std::filesystem::canonical(files.param);
        timing.model_path = std::filesystem::canonical(files.model);
        timing.inference_seconds.reserve(working.size());

        NcnnModel model(impl_->options);
        auto started = std::chrono::steady_clock::now();
        model.load(files.param, files.model);
        timing.load_seconds = elapsed_seconds(started);
        for (WorkingCase& item : working) {
            HostTensor hidden;
            hidden.shape = {1, item.result.text_length, 3072};
            hidden.storage = std::move(item.hidden);
            HostTensor cosine = make_float_tensor(
                {1, item.result.text_length, kTextHeadDim},
                item.runtime.rotary_cos
            );
            HostTensor sine = make_float_tensor(
                {1, item.result.text_length, kTextHeadDim},
                item.runtime.rotary_sin
            );
            HostTensor mask = make_float_tensor(
                {1, item.result.text_length, item.result.text_length},
                item.runtime.causal_mask
            );

            started = std::chrono::steady_clock::now();
            HostTensor output = model.run_positional({
                std::move(hidden),
                std::move(cosine),
                std::move(sine),
                std::move(mask),
            });
            timing.inference_seconds.push_back(elapsed_seconds(started));
            item.hidden = take_hidden(
                output, item.result.text_length, timing.name
            );
            if (capture_intermediates) {
                item.result.boundaries.push_back({
                    chunk.first,
                    chunk.last,
                    item.hidden,
                });
            }
        }
        batch.stages.push_back(std::move(timing));
    }

    batch.cases.reserve(working.size());
    for (WorkingCase& item : working) {
        item.result.hidden_states = std::move(item.hidden);
        batch.cases.push_back(std::move(item.result));
    }
    return batch;
}

const std::filesystem::path&
DynamicTextEncoder::tokenizer_json_path() const noexcept {
    return impl_->tokenizer.tokenizer_json_path();
}

const std::filesystem::path&
DynamicTextEncoder::model_directory() const noexcept {
    return impl_->model_dir;
}

}  // namespace ernie_image
