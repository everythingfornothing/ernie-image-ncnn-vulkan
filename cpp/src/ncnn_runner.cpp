#include "ernie_image/ncnn_runner.h"
#include "ncnn_tensor_utils.h"

#include <gpu.h>
#include <net.h>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ernie_image {

struct NcnnModel::Impl {
    explicit Impl(NcnnRuntimeOptions value) : options(value) {}

    NcnnRuntimeOptions options;
    std::unique_ptr<ncnn::Net> net;
    ModelIoContract contract;
    std::filesystem::path param;
    std::filesystem::path model;
};

namespace {

void validate_options(const NcnnRuntimeOptions& options) {
    if (options.num_threads <= 0) {
        throw std::invalid_argument("ncnn num_threads must be positive");
    }
    if (options.gpu_device_index < 0) {
        throw std::invalid_argument("ncnn gpu_device_index must not be negative");
    }
}

void configure(ncnn::Net& net, const NcnnRuntimeOptions& options) {
    net.opt.num_threads = options.num_threads;
    net.opt.lightmode = options.light_mode;
    net.opt.use_packing_layout = options.use_packing_layout;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_fp16_uniform = false;
    net.opt.use_bf16_storage = false;
    net.opt.use_bf16_packed = false;
    net.opt.use_int8_inference = false;
    net.opt.use_int8_storage = false;
    net.opt.use_int8_packed = false;
    net.opt.use_int8_arithmetic = false;
    net.opt.use_int8_uniform = false;
    net.opt.use_cooperative_matrix = false;

    if (options.compute_device == NcnnComputeDevice::Cpu) {
        net.opt.use_vulkan_compute = false;
        return;
    }

#if NCNN_VULKAN
    const int gpu_count = ncnn::get_gpu_count();
    if (gpu_count <= 0) {
        throw std::runtime_error(
            "Vulkan was requested, but ncnn found no Vulkan compute device"
        );
    }
    if (options.gpu_device_index >= gpu_count) {
        throw std::out_of_range(
            "Vulkan GPU index " + std::to_string(options.gpu_device_index) +
            " is outside [0, " + std::to_string(gpu_count - 1) + "]"
        );
    }
    net.set_vulkan_device(options.gpu_device_index);
    net.opt.use_vulkan_compute = true;
#else
    throw std::runtime_error(
        "Vulkan was requested, but this executable is linked against a "
        "CPU-only ncnn build"
    );
#endif
}

std::vector<std::string> copy_names(const std::vector<const char*>& names) {
    std::vector<std::string> result;
    result.reserve(names.size());
    for (const char* name : names) {
        if (name == nullptr) {
            throw std::runtime_error("ncnn returned a null blob name");
        }
        result.emplace_back(name);
    }
    return result;
}

}  // namespace

const char* ncnn_compute_device_name(NcnnComputeDevice device) noexcept {
    switch (device) {
    case NcnnComputeDevice::Cpu:
        return "cpu";
    case NcnnComputeDevice::Vulkan:
        return "vulkan";
    }
    return "unknown";
}

NcnnComputeDevice parse_ncnn_compute_device(const std::string& value) {
    if (value == "cpu") {
        return NcnnComputeDevice::Cpu;
    }
    if (value == "vulkan") {
        return NcnnComputeDevice::Vulkan;
    }
    throw std::invalid_argument(
        "unsupported ncnn device '" + value + "' (expected cpu or vulkan)"
    );
}

bool ncnn_vulkan_compiled() noexcept {
#if NCNN_VULKAN
    return true;
#else
    return false;
#endif
}

int ncnn_vulkan_gpu_count() {
#if NCNN_VULKAN
    return ncnn::get_gpu_count();
#else
    return 0;
#endif
}

std::string ncnn_vulkan_gpu_name(int device_index) {
#if NCNN_VULKAN
    const int gpu_count = ncnn::get_gpu_count();
    if (device_index < 0 || device_index >= gpu_count) {
        throw std::out_of_range("Vulkan GPU index is out of range");
    }
    return ncnn::get_gpu_info(device_index).device_name();
#else
    (void)device_index;
    throw std::runtime_error(
        "this executable is linked against a CPU-only ncnn build"
    );
#endif
}

NcnnModel::NcnnModel(NcnnRuntimeOptions options)
    : impl_(std::make_unique<Impl>(options)) {
    validate_options(options);
}

NcnnModel::~NcnnModel() = default;
NcnnModel::NcnnModel(NcnnModel&&) noexcept = default;
NcnnModel& NcnnModel::operator=(NcnnModel&&) noexcept = default;

void NcnnModel::load(
    const std::filesystem::path& param_path,
    const std::filesystem::path& model_path
) {
    if (!std::filesystem::is_regular_file(param_path)) {
        throw std::runtime_error("ncnn param file not found: " + param_path.string());
    }
    if (!std::filesystem::is_regular_file(model_path)) {
        throw std::runtime_error("ncnn model file not found: " + model_path.string());
    }

    auto candidate = std::make_unique<ncnn::Net>();
    configure(*candidate, impl_->options);
    const int param_result = candidate->load_param(param_path.c_str());
    if (param_result != 0) {
        throw std::runtime_error(
            "ncnn load_param failed with code " + std::to_string(param_result) +
            ": " + param_path.string()
        );
    }
    if (
        impl_->options.compute_device == NcnnComputeDevice::Vulkan &&
        !candidate->opt.use_vulkan_compute
    ) {
        throw std::runtime_error(
            "ncnn disabled Vulkan while loading model: " + param_path.string()
        );
    }
    const int model_result = candidate->load_model(model_path.c_str());
    if (model_result != 0) {
        throw std::runtime_error(
            "ncnn load_model failed with code " + std::to_string(model_result) +
            ": " + model_path.string()
        );
    }

    ModelIoContract contract;
    contract.input_names = copy_names(candidate->input_names());
    contract.output_names = copy_names(candidate->output_names());
    if (contract.input_names.empty() || contract.output_names.empty()) {
        throw std::runtime_error("ncnn model has an empty input/output contract");
    }

    impl_->net = std::move(candidate);
    impl_->contract = std::move(contract);
    impl_->param = std::filesystem::canonical(param_path);
    impl_->model = std::filesystem::canonical(model_path);
}

bool NcnnModel::loaded() const noexcept {
    return static_cast<bool>(impl_->net);
}

const ModelIoContract& NcnnModel::io_contract() const {
    if (!loaded()) {
        throw std::logic_error("ncnn model is not loaded");
    }
    return impl_->contract;
}

std::vector<NamedHostTensor> NcnnModel::run_many(
    const std::vector<NamedHostTensor>& inputs,
    const std::vector<std::string>& output_names
) const {
    if (!loaded()) {
        throw std::logic_error("ncnn model is not loaded");
    }
    if (inputs.size() != impl_->contract.input_names.size()) {
        throw std::invalid_argument("ncnn input count does not match model contract");
    }
    if (output_names.empty()) {
        throw std::invalid_argument("ncnn output list must not be empty");
    }

    std::vector<std::string> seen_names;
    seen_names.reserve(inputs.size());
    ncnn::Extractor extractor = impl_->net->create_extractor();
    if (output_names.size() > 1) {
        // Multiple graph outputs must remain available until each one has been
        // copied into its owning HostTensor.
        extractor.set_light_mode(false);
    }
    for (const auto& input : inputs) {
        if (
            std::find(
                impl_->contract.input_names.begin(),
                impl_->contract.input_names.end(),
                input.name
            ) == impl_->contract.input_names.end()
        ) {
            throw std::invalid_argument("unknown ncnn input name: " + input.name);
        }
        if (
            std::find(seen_names.begin(), seen_names.end(), input.name) !=
            seen_names.end()
        ) {
            throw std::invalid_argument("duplicate ncnn input name: " + input.name);
        }
        seen_names.push_back(input.name);
        ncnn::Mat value = detail::to_ncnn_mat(input.tensor);
        const int input_result = extractor.input(input.name.c_str(), value);
        if (input_result != 0) {
            throw std::runtime_error(
                "ncnn input failed with code " + std::to_string(input_result) +
                ": " + input.name
            );
        }
    }

    std::vector<std::string> seen_outputs;
    seen_outputs.reserve(output_names.size());
    std::vector<NamedHostTensor> outputs;
    outputs.reserve(output_names.size());
    for (const std::string& output_name : output_names) {
        // Intermediate graph blobs are valid here; Extractor reports unknown
        // names with a non-zero error code below.
        if (
            std::find(
                seen_outputs.begin(), seen_outputs.end(), output_name
            ) != seen_outputs.end()
        ) {
            throw std::invalid_argument(
                "duplicate ncnn output name: " + output_name
            );
        }
        seen_outputs.push_back(output_name);
        ncnn::Mat output;
        const int output_result =
            extractor.extract(output_name.c_str(), output);
        if (output_result != 0) {
            throw std::runtime_error(
                "ncnn extract failed with code " +
                std::to_string(output_result) + ": " + output_name
            );
        }
        outputs.push_back({
            output_name,
            detail::from_ncnn_mat(output),
        });
    }
    return outputs;
}

HostTensor NcnnModel::run(
    const std::vector<NamedHostTensor>& inputs,
    const std::string& output_name
) const {
    std::vector<NamedHostTensor> outputs =
        run_many(inputs, {output_name});
    return std::move(outputs.front().tensor);
}

HostTensor NcnnModel::run_with_internal_inputs(
    const std::vector<NamedHostTensor>& inputs,
    const std::string& output_name
) const {
    if (!loaded()) {
        throw std::logic_error("ncnn model is not loaded");
    }
    if (inputs.empty()) {
        throw std::invalid_argument("diagnostic ncnn inputs must not be empty");
    }
    if (output_name.empty()) {
        throw std::invalid_argument("diagnostic ncnn output must not be empty");
    }

    std::vector<std::string> seen_names;
    seen_names.reserve(inputs.size());
    ncnn::Extractor extractor = impl_->net->create_extractor();
    for (const auto& input : inputs) {
        if (
            std::find(seen_names.begin(), seen_names.end(), input.name) !=
            seen_names.end()
        ) {
            throw std::invalid_argument("duplicate ncnn input name: " + input.name);
        }
        seen_names.push_back(input.name);
        ncnn::Mat value = detail::to_ncnn_mat(input.tensor);
        const int input_result = extractor.input(input.name.c_str(), value);
        if (input_result != 0) {
            throw std::runtime_error(
                "diagnostic ncnn input failed with code " +
                std::to_string(input_result) + ": " + input.name
            );
        }
    }

    ncnn::Mat output;
    const int output_result = extractor.extract(output_name.c_str(), output);
    if (output_result != 0) {
        throw std::runtime_error(
            "diagnostic ncnn extract failed with code " +
            std::to_string(output_result) + ": " + output_name
        );
    }
    return detail::from_ncnn_mat(output);
}

std::vector<HostTensor> NcnnModel::run_positional_all(
    std::vector<HostTensor> inputs
) const {
    if (!loaded()) {
        throw std::logic_error("ncnn model is not loaded");
    }
    if (inputs.size() != impl_->contract.input_names.size()) {
        throw std::invalid_argument(
            "ncnn positional input count does not match model contract"
        );
    }
    std::vector<NamedHostTensor> named;
    named.reserve(inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        named.push_back({
            impl_->contract.input_names[index],
            std::move(inputs[index]),
        });
    }
    std::vector<NamedHostTensor> named_outputs =
        run_many(named, impl_->contract.output_names);
    std::vector<HostTensor> outputs;
    outputs.reserve(named_outputs.size());
    for (NamedHostTensor& output : named_outputs) {
        outputs.push_back(std::move(output.tensor));
    }
    return outputs;
}

HostTensor NcnnModel::run_positional(std::vector<HostTensor> inputs) const {
    if (!loaded()) {
        throw std::logic_error("ncnn model is not loaded");
    }
    if (impl_->contract.output_names.size() != 1) {
        throw std::logic_error(
            "ncnn positional run requires exactly one model output"
        );
    }
    std::vector<HostTensor> outputs =
        run_positional_all(std::move(inputs));
    return std::move(outputs.front());
}

const std::filesystem::path& NcnnModel::param_path() const {
    if (!loaded()) {
        throw std::logic_error("ncnn model is not loaded");
    }
    return impl_->param;
}

const std::filesystem::path& NcnnModel::model_path() const {
    if (!loaded()) {
        throw std::logic_error("ncnn model is not loaded");
    }
    return impl_->model;
}

}  // namespace ernie_image
