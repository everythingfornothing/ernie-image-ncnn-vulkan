#include "ernie_image/tokenizer.h"

#include <tokenizers_cpp.h>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace ernie_image {

namespace {

constexpr std::int32_t kExpectedBosTokenId = 1;
constexpr std::size_t kExpectedVocabSize = 131072;
constexpr std::size_t kModelMaxLength = 2048;

std::string read_file(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("tokenizer.json not found: " + path.string());
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("failed to open tokenizer.json: " + path.string());
    }
    std::string contents{
        std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()
    };
    if (stream.bad() || contents.empty()) {
        throw std::runtime_error("failed to read tokenizer.json: " + path.string());
    }
    return contents;
}

}  // namespace

struct ErnieTokenizer::Impl {
    std::filesystem::path path;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    std::size_t vocab = 0;
};

ErnieTokenizer::ErnieTokenizer(const std::filesystem::path& tokenizer_json)
    : impl_(std::make_unique<Impl>()) {
    impl_->path = std::filesystem::canonical(tokenizer_json);
    impl_->tokenizer = tokenizers::Tokenizer::FromBlobJSON(read_file(impl_->path));
    if (!impl_->tokenizer) {
        throw std::runtime_error("tokenizers-cpp failed to load tokenizer.json");
    }
    impl_->vocab = impl_->tokenizer->GetVocabSize();
    if (impl_->vocab != kExpectedVocabSize) {
        throw std::runtime_error(
            "unexpected tokenizer vocabulary size: " + std::to_string(impl_->vocab)
        );
    }
    const std::int32_t actual_bos = impl_->tokenizer->TokenToId("<s>");
    if (actual_bos != kExpectedBosTokenId) {
        throw std::runtime_error(
            "unexpected <s> token id: " + std::to_string(actual_bos)
        );
    }
}

ErnieTokenizer::~ErnieTokenizer() = default;
ErnieTokenizer::ErnieTokenizer(ErnieTokenizer&&) noexcept = default;
ErnieTokenizer& ErnieTokenizer::operator=(ErnieTokenizer&&) noexcept = default;

std::vector<std::int32_t> ErnieTokenizer::encode(
    const std::string& utf8_prompt
) const {
    // tokenizers-cpp's public Encode intentionally disables post-processing.
    // ERNIE's single-sequence TemplateProcessing only prepends <s>, so apply
    // that stable model-specific contract here and truncate on the right.
    std::vector<std::int32_t> core = impl_->tokenizer->Encode(utf8_prompt);
    std::vector<std::int32_t> result;
    result.reserve(std::min(kModelMaxLength, core.size() + 1));
    result.push_back(kExpectedBosTokenId);
    const std::size_t remaining = kModelMaxLength - result.size();
    const std::size_t copied = std::min(remaining, core.size());
    result.insert(result.end(), core.begin(), core.begin() + copied);
    return result;
}

std::size_t ErnieTokenizer::vocab_size() const noexcept {
    return impl_->vocab;
}

std::int32_t ErnieTokenizer::bos_token_id() const noexcept {
    return kExpectedBosTokenId;
}

std::size_t ErnieTokenizer::model_max_length() const noexcept {
    return kModelMaxLength;
}

const std::filesystem::path& ErnieTokenizer::tokenizer_json_path() const noexcept {
    return impl_->path;
}

}  // namespace ernie_image
