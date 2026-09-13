#pragma once

#include "ernie_image/ncnn_runner.h"

#include <net.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ernie_image::detail {

inline std::size_t checked_element_count(const std::vector<int>& shape) {
    if (shape.size() < 2 || shape.size() > 5) {
        throw std::invalid_argument(
            "HostTensor shape must be [batch=1] plus 1 to 4 dimensions"
        );
    }
    if (shape.front() != 1) {
        throw std::invalid_argument("HostTensor batch must be 1");
    }
    std::size_t count = 1;
    for (int dimension : shape) {
        if (dimension <= 0) {
            throw std::invalid_argument("HostTensor dimensions must be positive");
        }
        const auto value = static_cast<std::size_t>(dimension);
        if (count > std::numeric_limits<std::size_t>::max() / value) {
            throw std::overflow_error("HostTensor element count overflow");
        }
        count *= value;
    }
    return count;
}

inline ncnn::Mat to_ncnn_mat(const HostTensor& tensor) {
    const std::size_t count = checked_element_count(tensor.shape);
    const std::size_t storage_size = std::visit(
        [](const auto& values) { return values.size(); }, tensor.storage
    );
    if (storage_size != count) {
        throw std::invalid_argument("HostTensor storage size does not match shape");
    }

    const auto& shape = tensor.shape;
    const std::size_t rank = shape.size() - 1;
    ncnn::Mat result;
    if (rank == 1) {
        result.create(shape[1]);
    } else if (rank == 2) {
        result.create(shape[2], shape[1]);
    } else if (rank == 3) {
        result.create(shape[3], shape[2], shape[1]);
    } else {
        result.create(shape[4], shape[3], shape[2], shape[1]);
    }
    if (result.empty()) {
        throw std::runtime_error("ncnn Mat allocation failed");
    }

    std::visit(
        [&result, count](const auto& values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            static_assert(sizeof(Value) == 4, "ncnn adapter requires 4-byte values");
            std::memcpy(result.data, values.data(), count * sizeof(Value));
        },
        tensor.storage
    );
    return result;
}

inline HostTensor from_ncnn_mat(const ncnn::Mat& tensor) {
    if (tensor.empty()) {
        throw std::runtime_error("ncnn returned an empty tensor");
    }
    if (tensor.elempack != 1 || tensor.elemsize != sizeof(float)) {
        throw std::runtime_error("ncnn output is not unpacked FP32");
    }
    if (tensor.dims < 1 || tensor.dims > 4) {
        throw std::runtime_error("ncnn output rank is outside 1 to 4");
    }

    HostTensor result;
    if (tensor.dims == 1) {
        result.shape = {1, tensor.w};
    } else if (tensor.dims == 2) {
        result.shape = {1, tensor.h, tensor.w};
    } else if (tensor.dims == 3) {
        result.shape = {1, tensor.c, tensor.h, tensor.w};
    } else {
        result.shape = {1, tensor.c, tensor.d, tensor.h, tensor.w};
    }

    const std::size_t count = checked_element_count(result.shape);
    std::vector<float> values(count);
    const std::size_t plane =
        static_cast<std::size_t>(tensor.w) * tensor.h * tensor.d;
    if (tensor.dims <= 2) {
        std::memcpy(values.data(), tensor.data, count * sizeof(float));
    } else {
        for (int channel = 0; channel < tensor.c; ++channel) {
            const ncnn::Mat source = tensor.channel(channel);
            std::memcpy(
                values.data() + static_cast<std::size_t>(channel) * plane,
                source.data,
                plane * sizeof(float)
            );
        }
    }
    result.storage = std::move(values);
    return result;
}

}  // namespace ernie_image::detail
