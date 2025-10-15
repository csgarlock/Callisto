#ifndef HOST_LINEAR_H_INCLUDED
#define HOST_LINEAR_H_INCLUDED

#include "../types.h"
#include "../kernels/activation.h"

#include <cassert>

template <typename Activation = Identity>
void cpu_linear_forward(Tensor<float, 1> &input, Tensor<float, 2> &weights, Tensor<float, 1> &biases, Tensor<float, 1> &output) {
    assert(input.shape[0] == weights.shape[0]);
    assert(weights.shape[1] == biases.shape[0]);
    assert(weights.shape[1] = output.shape[0]);
    for (int row = 0; row < weights.shape[1]; row++) {
        float acc = 0.0f;
        for (int col = 0; col < weights.shape[0]; col++) {
            acc += weights[row * weights.shape[0] + col] * input[col];
        }
        output.data[row] = Activation::apply(acc + biases[row]);
    }
}

#endif