#ifndef KERNEL_ACTIVATION_H_INCLUDED
#define KERNEL_ACTIVATION_H_INCLUDED

#include <cuda_runtime.h>
#include "../types/activation_types.h"

template <typename Activation, bool Backwards = false>
__global__ void activation(const float *input, float *output, int n) {
    int n4 = n / 4;
    const float4 *input4 = reinterpret_cast<const float4*>(input);
    float4 *output4 = reinterpret_cast<float4*>(output);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 v = input4[i];
        if constexpr (Backwards) {
            v.x = Activation::device_derivative(v.x);
            v.y = Activation::device_derivative(v.y);
            v.z = Activation::device_derivative(v.z);
            v.w = Activation::device_derivative(v.w);
        } else {
            v.x = Activation::device_forward(v.x);
            v.y = Activation::device_forward(v.y);
            v.z = Activation::device_forward(v.z);
            v.w = Activation::device_forward(v.w);
        }
        output4[i] = v;
    }

    if (blockIdx.x == 0 && threadIdx.x < n % 4) {
        int cleanup_idx = (n & ~0b11) + threadIdx.x;
        if constexpr (Backwards) {
            output[cleanup_idx] = Activation::device_derivative(input[cleanup_idx]);
        } else {
            output[cleanup_idx] = Activation::device_forward(input[cleanup_idx]);
        }
    }
}


#endif