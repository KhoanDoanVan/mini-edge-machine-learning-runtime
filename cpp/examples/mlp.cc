#include <cmath>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

#include "mini_ort/model.h"
#include "mini_ort/session.h"

int main() {
  try {
    const mini_ort::Tensor input({1, 3}, {1.0F, 2.0F, 3.0F});
    const mini_ort::Tensor w1(
        {3, 4}, {1.0F, 0.0F, -1.0F, 0.5F, 0.0F, 1.0F,
                 1.0F, -0.5F, 1.0F, 1.0F, 0.0F, 1.0F});
    const mini_ort::Tensor b1({4}, {0.0F, -1.0F, 0.5F, 0.0F});
    const mini_ort::Tensor w2(
        {4, 2}, {1.0F, -1.0F, 0.5F, 1.0F,
                 -1.0F, 0.0F, 0.0F, 2.0F});
    const mini_ort::Tensor b2({2}, {0.25F, -0.5F});

    std::vector<mini_ort::Layer> layers;
    layers.emplace_back(mini_ort::LinearLayer{3, 4, w1, b1});
    layers.emplace_back(mini_ort::ReluLayer{});
    layers.emplace_back(mini_ort::LinearLayer{4, 2, w2, b2});
    mini_ort::InferenceSession session(
        mini_ort::SequentialModel(std::move(layers)));
    const auto output = session.Run(input);

    std::cout << "output=[" << output.data()[0] << ", " << output.data()[1] << "]\n";
    if (output.shape() != mini_ort::Shape({1, 2}) ||
        std::fabs(output.data()[0] - 4.75F) > 1e-6F ||
        std::fabs(output.data()[1] - 4.5F) > 1e-6F
    ) {
      std::cerr << "native output does not match the expected result\n";
      return 1;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native MLP failed: " << error.what() << '\n';
    return 1;
  }
}
