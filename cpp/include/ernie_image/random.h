#pragma once

#include "ernie_image/latent.h"
#include "ernie_image/ncnn_runner.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace ernie_image {

inline constexpr std::uint64_t kOfficialReferenceSeed = 42;
inline constexpr const char* kOfficialReferenceLatentPolicy =
    "official_cuda_bf16_seed42_reference";
inline constexpr const char* kPortableLatentPolicy =
    "portable_philox4x32_10_box_muller_bf16_rne_v1";

enum class InitialLatentMode {
    OfficialReference,
    PortablePhilox,
};

const char* initial_latent_mode_name(InitialLatentMode mode) noexcept;
InitialLatentMode parse_initial_latent_mode(const std::string& name);

struct InitialLatentResult {
    HostTensor latent;
    std::uint64_t seed = kOfficialReferenceSeed;
    InitialLatentMode mode = InitialLatentMode::OfficialReference;
    std::string policy = kOfficialReferenceLatentPolicy;
    double load_seconds = 0.0;
};

// Initial-noise boundary with two intentionally separate policies:
// - OfficialReference loads the exact packaged CUDA/BF16 seed=42 tensor.
// - PortablePhilox generates a deterministic CPU stream for arbitrary seeds.
// PortablePhilox is not claimed to match PyTorch CUDA's RNG stream.
class InitialLatentProvider {
public:
    explicit InitialLatentProvider(std::filesystem::path seed42_asset);

    InitialLatentResult create(
        std::uint64_t seed,
        InitialLatentMode mode = InitialLatentMode::OfficialReference
    ) const;
    InitialLatentResult create(
        std::uint64_t seed,
        InitialLatentMode mode,
        const ImageGeometry& geometry
    ) const;
    const std::filesystem::path& seed42_asset() const noexcept;

private:
    std::filesystem::path seed42_asset_;
};

}  // namespace ernie_image
