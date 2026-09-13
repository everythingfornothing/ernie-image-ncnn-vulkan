#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ernie_image {

// Adapter for the exact ERNIE-Image-Turbo Hugging Face tokenizer.json.
// The output is the stable B01 contract: int32 input_ids with logical shape [1,N].
class ErnieTokenizer {
public:
    explicit ErnieTokenizer(const std::filesystem::path& tokenizer_json);
    ~ErnieTokenizer();

    ErnieTokenizer(ErnieTokenizer&&) noexcept;
    ErnieTokenizer& operator=(ErnieTokenizer&&) noexcept;

    ErnieTokenizer(const ErnieTokenizer&) = delete;
    ErnieTokenizer& operator=(const ErnieTokenizer&) = delete;

    std::vector<std::int32_t> encode(const std::string& utf8_prompt) const;
    std::size_t vocab_size() const noexcept;
    std::int32_t bos_token_id() const noexcept;
    std::size_t model_max_length() const noexcept;
    const std::filesystem::path& tokenizer_json_path() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ernie_image
