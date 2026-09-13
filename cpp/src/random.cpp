#include "ernie_image/random.h"

#include "ernie_image/latent.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ernie_image {

namespace {

constexpr std::size_t kLatentElements =
    static_cast<std::size_t>(kBatchSize) * kLatentChannels *
    kLatentHeight * kLatentWidth;

using PhiloxCounter = std::array<std::uint32_t, 4>;
using PhiloxKey = std::array<std::uint32_t, 2>;

constexpr std::uint32_t kPhiloxMultiplier0 = 0xd2511f53U;
constexpr std::uint32_t kPhiloxMultiplier1 = 0xcd9e8d57U;
constexpr std::uint32_t kPhiloxWeyl0 = 0x9e3779b9U;
constexpr std::uint32_t kPhiloxWeyl1 = 0xbb67ae85U;
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kUniformScale = 1.0 / 4294967296.0;

std::uint32_t multiply_high(
    std::uint32_t lhs,
    std::uint32_t rhs,
    std::uint32_t& low
) {
    const std::uint64_t product =
        static_cast<std::uint64_t>(lhs) * rhs;
    low = static_cast<std::uint32_t>(product);
    return static_cast<std::uint32_t>(product >> 32U);
}

PhiloxCounter philox4x32_10(PhiloxCounter counter, PhiloxKey key) {
    for (int round = 0; round < 10; ++round) {
        std::uint32_t low0 = 0;
        std::uint32_t low1 = 0;
        const std::uint32_t high0 = multiply_high(
            kPhiloxMultiplier0, counter[0], low0
        );
        const std::uint32_t high1 = multiply_high(
            kPhiloxMultiplier1, counter[2], low1
        );
        counter = {
            high1 ^ counter[1] ^ key[0],
            low1,
            high0 ^ counter[3] ^ key[1],
            low0,
        };
        key[0] += kPhiloxWeyl0;
        key[1] += kPhiloxWeyl1;
    }
    return counter;
}

double uint32_to_open_uniform(std::uint32_t value) {
    return (static_cast<double>(value) + 0.5) * kUniformScale;
}

float round_float_to_bf16_rne(float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding_bias =
        0x00007fffU + ((bits >> 16U) & 1U);
    bits = (bits + rounding_bias) & 0xffff0000U;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::vector<float> make_portable_latent(std::uint64_t seed) {
    std::vector<float> values;
    values.reserve(kLatentElements);
    const PhiloxKey key = {
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32U),
    };

    for (std::uint64_t block = 0; values.size() < kLatentElements; ++block) {
        const PhiloxCounter random = philox4x32_10(
            {
                static_cast<std::uint32_t>(block),
                static_cast<std::uint32_t>(block >> 32U),
                0U,
                0U,
            },
            key
        );
        for (std::size_t pair = 0; pair < 4; pair += 2) {
            const double u1 = uint32_to_open_uniform(random[pair]);
            const double u2 = uint32_to_open_uniform(random[pair + 1]);
            const double radius = std::sqrt(-2.0 * std::log(u1));
            const double angle = kTwoPi * u2;
            const std::array<double, 2> normal = {
                radius * std::cos(angle),
                radius * std::sin(angle),
            };
            for (double sample : normal) {
                if (values.size() == kLatentElements) {
                    break;
                }
                values.push_back(round_float_to_bf16_rne(
                    static_cast<float>(sample)
                ));
            }
        }
    }
    return values;
}

bool host_is_little_endian() {
    const std::uint16_t value = 1;
    return *reinterpret_cast<const std::uint8_t*>(&value) == 1;
}

}  // namespace

const char* initial_latent_mode_name(InitialLatentMode mode) noexcept {
    switch (mode) {
    case InitialLatentMode::OfficialReference:
        return "reference";
    case InitialLatentMode::PortablePhilox:
        return "portable";
    }
    return "unknown";
}

InitialLatentMode parse_initial_latent_mode(const std::string& name) {
    if (name == "reference") {
        return InitialLatentMode::OfficialReference;
    }
    if (name == "portable") {
        return InitialLatentMode::PortablePhilox;
    }
    throw std::invalid_argument(
        "RNG mode must be 'reference' or 'portable': " + name
    );
}

InitialLatentProvider::InitialLatentProvider(
    std::filesystem::path seed42_asset
) : seed42_asset_(std::move(seed42_asset)) {
    if (!std::filesystem::is_regular_file(seed42_asset_)) {
        throw std::runtime_error(
            "official seed=42 latent asset not found: " +
            seed42_asset_.string()
        );
    }
    const std::uintmax_t expected_bytes = kLatentElements * sizeof(float);
    if (std::filesystem::file_size(seed42_asset_) != expected_bytes) {
        throw std::runtime_error(
            "official seed=42 latent asset has an invalid byte count: " +
            seed42_asset_.string()
        );
    }
}

InitialLatentResult InitialLatentProvider::create(
    std::uint64_t seed,
    InitialLatentMode mode
) const {
    const auto started = std::chrono::steady_clock::now();
    std::vector<float> values;
    std::string policy;

    if (mode == InitialLatentMode::OfficialReference) {
        if (seed != kOfficialReferenceSeed) {
            throw std::invalid_argument(
                "reference RNG mode supports only seed=42"
            );
        }
        if (!host_is_little_endian()) {
            throw std::runtime_error(
                "official seed=42 latent asset requires a little-endian host"
            );
        }
        values.resize(kLatentElements);
        std::ifstream stream(seed42_asset_, std::ios::binary);
        stream.read(
            reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float))
        );
        if (!stream || stream.peek() != std::ifstream::traits_type::eof()) {
            throw std::runtime_error(
                "failed to read exact official seed=42 latent asset: " +
                seed42_asset_.string()
            );
        }
        policy = kOfficialReferenceLatentPolicy;
    } else if (mode == InitialLatentMode::PortablePhilox) {
        values = make_portable_latent(seed);
        policy = kPortableLatentPolicy;
    } else {
        throw std::invalid_argument("unknown initial latent RNG mode");
    }

    InitialLatentResult result;
    result.latent = make_latent_tensor(std::move(values));
    result.seed = seed;
    result.mode = mode;
    result.policy = std::move(policy);
    result.load_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started
    ).count();
    return result;
}

const std::filesystem::path&
InitialLatentProvider::seed42_asset() const noexcept {
    return seed42_asset_;
}

}  // namespace ernie_image
