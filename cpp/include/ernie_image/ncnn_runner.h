#pragma once

#include <filesystem>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace ernie_image {

enum class NcnnComputeDevice {
    Cpu,
    Vulkan,
};

const char* ncnn_compute_device_name(NcnnComputeDevice device) noexcept;
NcnnComputeDevice parse_ncnn_compute_device(const std::string& value);
bool ncnn_vulkan_compiled() noexcept;
int ncnn_vulkan_gpu_count();
std::string ncnn_vulkan_gpu_name(int device_index);

struct NcnnRuntimeOptions {
    int num_threads = 16;
    bool light_mode = true;
    bool use_packing_layout = true;
    NcnnComputeDevice compute_device = NcnnComputeDevice::Cpu;
    int gpu_device_index = 0;
};

struct ModelIoContract {
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

using HostTensorStorage = std::variant<
    std::vector<float>,
    std::vector<std::int32_t>
>;

struct HostTensor {
    std::vector<int> shape;
    HostTensorStorage storage;
};

struct NamedHostTensor {
    std::string name;
    HostTensor tensor;
};

// The only project-level class allowed to own or expose ncnn lifecycle.
// Model-specific tensor names stay behind this boundary.
class NcnnModel {
public:
    explicit NcnnModel(NcnnRuntimeOptions options = {});
    ~NcnnModel();

    NcnnModel(NcnnModel&&) noexcept;
    NcnnModel& operator=(NcnnModel&&) noexcept;

    NcnnModel(const NcnnModel&) = delete;
    NcnnModel& operator=(const NcnnModel&) = delete;

    void load(
        const std::filesystem::path& param_path,
        const std::filesystem::path& model_path
    );

    bool loaded() const noexcept;
    const ModelIoContract& io_contract() const;
    HostTensor run(
        const std::vector<NamedHostTensor>& inputs,
        const std::string& output_name
    ) const;
    std::vector<NamedHostTensor> run_many(
        const std::vector<NamedHostTensor>& inputs,
        const std::vector<std::string>& output_names
    ) const;
    // Diagnostic-only entry point for injecting saved tensors at named graph
    // blobs. Production inference must use run()/run_many() so the declared
    // model input contract remains enforced.
    HostTensor run_with_internal_inputs(
        const std::vector<NamedHostTensor>& inputs,
        const std::string& output_name
    ) const;
    HostTensor run_positional(std::vector<HostTensor> inputs) const;
    std::vector<HostTensor> run_positional_all(
        std::vector<HostTensor> inputs
    ) const;
    const std::filesystem::path& param_path() const;
    const std::filesystem::path& model_path() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ernie_image
