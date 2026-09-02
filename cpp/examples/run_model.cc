#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>

#include "mini_ort/session.h"
#include "mini_ort/tensor.h"

int main(
    const int argc,
    const char* const* argv
) {
    if (argc != 2) {
        std::cerr << "usage: mini_ort_run_model <model.mer>\n";
        return 2;
    }

    try {
        const mini_ort::InferenceSession session{std::filesystem::path(argv[1])};
        const mini_ort::Tensor input({1, 3}, {1.0F, 2.0F, 3.0F});

        mini_ort::Tensor output(session.OutputShape(input.view()));

        std::cout << "output=[" << output.data()[0] << ", " << output.data()[1] << "]\n";

        if (output.shape() != mini_ort::Shape({1, 2}) ||
                std::fabs(output.data()[0] - 4.75F) > 1e-6F ||
                std::fabs(output.data()[1] - 4.5F) > 1e-6F
        ) {
            std::cerr << "model output does not match the expected result\n";
            return 1;
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "failed to run model: " << error.what() << '\n';
        return 1;
    }
}