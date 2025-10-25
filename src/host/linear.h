#ifndef HOST_LINEAR_H_INCLUDED
#define HOST_LINEAR_H_INCLUDED

#include "../types/tensor.h"
#include "../kernels/activation.h"

#include <cassert>
#include <iostream>

template <typename Activation = Identity>
void cpu_linear_forward(Tensor<float> &input, Tensor<float> &weights, Tensor<float> &biases, Tensor<float> &output) {
    assert(input.shape[0] == weights.shape[0]);
    assert(weights.shape[1] == biases.shape[0]);
    assert(weights.shape[1] = output.shape[0]);
    for (int row = 0; row < weights.shape[1]; row++) {
        float acc = 0.0f;
        for (int col = 0; col < weights.shape[0]; col++) {
            acc += weights.data[row * weights.shape[0] + col] * input.data[col];
        }
        output.data[row] = Activation::host_forward(acc + biases.data[row]);
    }
}

template <typename Activation = Identity>
void cpu_linear_forward_batch(Tensor<float> &input, Tensor<float> &weights, Tensor<float> &biases, Tensor<float> &output) {
    assert(input.shape[0] == weights.shape[0]);
    assert(weights.shape[1] == biases.shape[0]);
    assert(weights.shape[1] = output.shape[0]);
    assert(input.shape[1] == output.shape[1]);
    for (int depth = 0; depth < input.shape[1]; depth++) {
        for (int row = 0; row < weights.shape[1]; row++) {
            float acc = 0.0f;
            for (int col = 0; col < weights.shape[0]; col++) {
                acc += weights.data[row * weights.shape[0] + col] * input.data[depth * weights.shape[0] + col];
            }
            output.data[depth * output.shape[0] + row] = Activation::host_forward(acc + biases.data[row]);
        }
    }
}

#endif