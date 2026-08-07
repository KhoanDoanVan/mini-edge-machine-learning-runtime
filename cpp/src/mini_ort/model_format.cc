#include "mini_ort/model_format.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace mini_ort {

    namespace {

        constexpr std::array<std::uint8_t, 8> kMagic {
            'M', 'E', 'R', 'M', 'D', 'L', '1', 0
        };
        constexpr std::uint32_t kFormatVersion = 1;
        constexpr std::uint32_t kLinearLayer = 1;
        constexpr std::uint32_t kReluLayer = 2;
        constexpr std::uint32_t kMaximumLayerCount = 1U << 20U;


        class BinaryReader final {
            public:
                explicit BinaryReader(const std::filesystem::path& path) : stream_(path, std::ios::binary), remaining_(std::filesystem::file_size(path)) {
                    if (!stream_) {
                        throw std::runtime_error("failed to open model file: " + path.string());
                    }
                }

                std::vector<std::uint8_t> ReadBytes(const std::size_t count) {
                    RequireRemaining(count);
                    std::vector<std::uint8_t> bytes(count);

                    if (count != 0) {
                        stream_.read(
                            reinterpret_cast<char*>(bytes.data()),
                            static_cast<std::streamsize>(count)
                        );

                        if (!stream_) {
                            throw std::runtime_error("failed to read model file");
                        }
                    }

                    remaining_ -= count;
                    return bytes;
                }

                std::uint32_t ReadU32() {
                    const auto bytes = ReadBytes(4);
                    return static_cast<std::uint32_t>(bytes[0]) |
                            (static_cast<std::uint32_t>(bytes[1]) << 8U) |
                            (static_cast<std::uint32_t>(bytes[2]) << 16U) |
                            (static_cast<std::uint32_t>(bytes[3]) << 24U);
                }

                std::uint64_t ReadU64() {
                    const auto bytes = ReadBytes(8);
                    std::uint64_t value = 0;
                    for (std::size_t index = 0; index < bytes.size(); ++index) {
                        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
                    }
                    return value;
                }

                std::vector<float> ReadFloat32(const std::size_t count) {
                    if (count > remaining_ / sizeof(float)) {
                        throw std::runtime_error("model tensor exceeds the remaining file size");
                    }

                    std::vector<float> values;
                    values.reserve(count);
                    
                    for (std::size_t index = 0; index < count; ++index) {
                        values.push_back(std::bit_cast<float>(
                            ReadU32()
                        ));
                    }

                    return values;
                }

                void RequireFinished() const {
                    if (remaining_ != 0) {
                        throw std::runtime_error("model file contains trailing bytes");
                    }
                }


            private:
                void RequireRemaining(const std::size_t count) const {
                    if (count > remaining_) {
                        throw std::runtime_error("model file is truncated");
                    }

                    if (count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
                        throw std::runtime_error("model read is too large");
                    }
                }

                std::ifstream stream_;
                std::uintmax_t remaining_;
        };


        std::size_t CheckedSize(
            const std::uint64_t value,
            const char* field
        ) {
            if (value > std::numeric_limits<std::size_t>::max()) {
                throw std::runtime_error(std::string(field) + " does not fit size_t");
            }
            return static_cast<std::size_t>(value);
        }

        std::int64_t CheckedDimension(const std::size_t value) {
            if (value > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
                throw std::runtime_error("model dimension does not fit int64");
            }
            return static_cast<std::int64_t>(value);
        }

        std::size_t CheckedProduct(
            const std::size_t lhs,
            const std::size_t rhs
        ) {
            if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
                throw std::runtime_error("model tensor size overflows size_t");
            }
            return lhs * rhs;
        }

    }
        
    SequentialModel LoadModel(const std::filesystem::path& path) {
        BinaryReader reader(path);

        const auto magic = reader.ReadBytes(kMagic.size());

        if (!std::equal(
                magic.begin(),
                magic.end(),
                kMagic.begin(),
                kMagic.end()
            )
        ) {
            throw std::runtime_error("invalid MER model magic");
        }

        if (reader.ReadU32() != kFormatVersion) {
            throw std::runtime_error("unsupported MER model version");
        }

        const auto layer_count = reader.ReadU32();

        if (layer_count == 0 || layer_count > kMaximumLayerCount) {
            throw std::runtime_error("invalid MER model layer count");
        }

        std::vector<Layer> layers;
        layers.reserve(layer_count);

        for (std::uint32_t index = 0; index < layer_count; ++index) {
            const auto type = reader.ReadU32();
            const auto reserved = reader.ReadU32();
            const auto in_features = CheckedSize(
                reader.ReadU64(),
                "in_features"
            );
            const auto out_features = CheckedSize(
                reader.ReadU64(),
                "out_features"
            );
            const auto weight_count = CheckedSize(
                reader.ReadU64(),
                "weight_count"
            );
            const auto bias_count = CheckedSize(
                reader.ReadU64(),
                "bias_count"
            );

            if (reserved != 0) {
                throw std::runtime_error("MER layer reserved field must be zero");
            }

            // Linear Layer
            if (type == kLinearLayer) {
                if (
                    in_features == 0 || 
                    out_features == 0 || 
                    weight_count != CheckedProduct(in_features, out_features) ||
                    (bias_count != 0 && bias_count != out_features)
                ) {
                    throw std::runtime_error("invalid Linear Layer metadata");
                }

                Tensor weight(
                    {
                        CheckedDimension(in_features),
                        CheckedDimension(out_features)
                    },
                    reader.ReadFloat32(weight_count)
                );

                std::optional<Tensor> bias;

                if (bias_count != 0) {
                    bias.emplace(
                        Shape {
                            CheckedDimension(out_features)
                        },
                        reader.ReadFloat32(bias_count)
                    );
                }

                layers.emplace_back(
                    LinearLayer {
                        in_features,
                        out_features,
                        std::move(weight),
                        std::move(bias)
                    }
                );
            // ReLU
            } else if (type == kReluLayer) {
                if (in_features != 0 || out_features != 0 || weight_count != 0 || bias_count != 0) {
                    throw std::runtime_error("invalid Relu layer metadata");
                }
                layers.emplace_back(
                    ReluLayer{}
                );
            // error
            } else {
                throw std::runtime_error("unsupported MER layer type"); 
            }
        }

        reader.RequireFinished();
        
        return SequentialModel(
            std::move(layers)
        );
    }

}
