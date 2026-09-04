#pragma once

#include <filesystem>
#include <cstddef>
#include <span>

#include "mini_ort/model.h"

namespace mini_ort {

    // Loads the little-endian MER model format produced by the Python API.
    /**
     * @brief Deserialize a MER model from an in-memory byte buffer.
     * @param data Complete little-endian MER payload.
     * @return Validated sequential model owning its tensors.
     * @throws std::invalid_argument for malformed or unsupported data.
     */
    SequentialModel LoadModel(std::span<const std::byte> data);
    /**
     * @brief Deserialize a MER model from a filesystem path.
     * @param path Path to the model file.
     * @return Validated sequential model owning its tensors.
     */
    SequentialModel LoadModel(const std::filesystem::path& path);
}
