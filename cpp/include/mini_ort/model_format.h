#pragma once

#include <filesystem>

#include "mini_ort/model.h"

namespace mini_ort {

    // Loads the little-endian MER model format produced by the Python API.
    SequentialModel LoadModel(const std::filesystem::path& path);
}