#ifndef ACTIVATION_H_INCLUDED
#define ACTIVATION_H_INCLUDED

#include <cuda_runtime.h>

struct ReLU {
    __device__ static float apply(float x) { return fmaxf(0.0f, x); }

    static float cpu(float x) { return std::fmax(0.0f, x); }
};

struct LogisticSigmoid {
    __device__ static float apply(float x) { return 1.0f / (1.0f + expf(-x)); }

    static float cpu(float x) { return 1.0f / (1.0f + std::exp(-x)); }
};

template <typename Activation>
__global__ void activation(float *__restrict__ vec, int n) {
    int n4 = n / 4;
    float4 *vec4 = reinterpret_cast<float4*>(vec);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 v = vec4[i];
        v.x = Activation::apply(v.x);
        v.y = Activation::apply(v.y);
        v.z = Activation::apply(v.z);
        v.w = Activation::apply(v.w);
        vec4[i] = v;
    }

    if (blockIdx.x == 0 && threadIdx.x < n % 4) {
        int cleanup_idx = (n & ~0b11) + threadIdx.x;
        vec[cleanup_idx] = Activation::apply(vec[cleanup_idx]);
    }
}


#endif