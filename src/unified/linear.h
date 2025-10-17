#ifndef UNIFIED_LINEAR_H_INCLUDED
#define UNIFIED_LINEAR_H_INCLUDED

#include "../types.h"
#include "../types/activation_types.h"
#include "../host/linear.h"
#include "../kernels/linear.h"
#include "../util.h"

#include <iostream>

template <typename Activation = Identity>
void linear_forward(Tensor<float, 1> &input, Tensor<float, 2> &weights, Tensor<float, 1> &biases, Tensor<float, 1> &output, MemoryLocation location = MemoryLocation::Device) {
    
    const int n = weights.shape[0];
    const int m = weights.shape[1];
    assert(input.shape[0] == n);
    assert(biases.shape[0] == m);
    assert(output.shape[0] == m);

    input.change_memory_location(location);
    weights.change_memory_location(location);
    biases.change_memory_location(location);
    output.change_memory_location(location);

    if (location == MemoryLocation::Device) {
        MatrixMultShape shape{n, m};
        if (m % 32 == 0 && n % 32 == 0) {
            linear_forward_mtm<Activation><<<std::min(m / 32, 256), 256>>>(input.data, weights.data, biases.data, output.data, shape);
        } else if (m == 1) {
            linear_forward_mto<Activation><<<1, 512>>>(input.data, weights.data, biases.data, output.data, shape);
        } else {
            linear_forward_general<Activation><<<std::max(1, std::min(m / 32, 256)), 256>>>(input.data, weights.data, biases.data, output.data, shape);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        cpu_linear_forward<Activation>(input, weights, biases, output);
    }
}

template <typename Activation = Identity>
void linear_forward_batch(Tensor<float, 2> &input, Tensor<float, 2> &weights, Tensor<float, 1> &biases, Tensor<float, 2> &output, MemoryLocation location = MemoryLocation::Device) {

    const int n = weights.shape[0];
    const int m = weights.shape[1];
    const int k = input.shape[1];
    assert(input.shape[0] == n);
    assert(biases.shape[0] == m);
    assert(output.shape[0] == m);
    assert(output.shape[1] == k);

    input.change_memory_location(location);
    weights.change_memory_location(location);
    biases.change_memory_location(location);
    output.change_memory_location(location);

    if (location == MemoryLocation::Device) {
        MatrixMultShape shape{n, m, k};
        if (m % 32 == 0 && n % 32 == 0 && k >= 32) {
            int batch_k = k - (k % 32);
            shape.batch_size = batch_k;
            dim3 blocks(
                std::min(m / 32, 32),   // tile rows
                std::min(k / 32, 8) // tile depth (batch dimension)
            );
            linear_forward_mtm_batch<Activation><<<blocks, 256>>>(input.data, weights.data, biases.data, output.data, shape);
            int remainder = k % 32;
            shape.batch_size = 1;
            int input_offset = n * batch_k;
            int output_offset = m * batch_k;
            for (int i = 0; i < remainder; i++) {
                linear_forward_mtm<Activation><<<std::min(m / 32, 256), 256>>>(input.data + input_offset, weights.data, biases.data, output.data + output_offset, shape);
                input_offset += n;
                output_offset += m;
            }
        } else if (m == 1) {
            linear_forward_mto<Activation><<<std::max(1, std::min(k, 256)), 256>>>(input.data, weights.data, biases.data, output.data, shape);
        } else {
            linear_forward_general<Activation><<<std::max(1, std::min(m / 32, 256)), 256>>>(input.data, weights.data, biases.data, output.data, shape);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        cpu_linear_forward_batch<Activation>(input, weights, biases, output);
    }

}

#endif