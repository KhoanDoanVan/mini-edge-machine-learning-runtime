#include "mini_ort/kernels/cpu.h"

#include <algorithm>
#include <utility>
#include <vector>


namespace mini_ort::cpu {

    Tensor Relu(const ConstTensorView input) {
    
        std::vector<float> output(input.size());
        // reads elements from an input range, applies a function to each element, and writes the results into an output range.
        std::transform(
            input.data().begin(),
            input.data().end(),
            output.begin(),
            [](const float value) {
                return std::max(
                    0.0F,
                    value
                );
            }
        );

        return Tensor(
            Shape(
                input.shape().begin(),
                input.shape().end()
            ),
            std::move(output)
        );

    }

}