#ifndef KERNEL_ACTIVATION_H_INCLUDED
#define KERNEL_ACTIVATION_H_INCLUDED

#include <cuda_runtime.h>

struct ReLU {
    __device__ static float device_forward(float x) { return fmaxf(0.0f, x); }
    __device__ static float device_derivative(float x) { if (x > 0.0f) { return 1.0f; } else { return 0.0f; } }

    static float host_forward(float x) { return std::fmax(0.0f, x); }
    static float host_derivative(float x) { if (x > 0.0f) { return 1.0f; } else { return 0.0f; } }
};

struct LogisticSigmoid {
    __device__ static float device_forward(float x) { return 1.0f / (1.0f + expf(-x)); }
    __device__ static float device_derivative(float x) { float sig = 1.0f / (1.0f + expf(-x)); return sig * (1 - sig); }

    static float host_forward(float x) { return 1.0f / (1.0f + std::exp(-x)); }
    static float host_derivative(float x) { float sig = 1.0f / (1.0f + std::exp(-x)); return sig * (1 - sig); }
};

struct Identity {
    __device__ static float device_forward(float x) { return x; }
    __device__ static float device_derivative(float x) { return 1.0f; }

    static float host_forward(float x) { return x; }
    static float host_derivative(float x) { return 1.0f; }
};

template <typename Activation>
__global__ void activation(float *__restrict__ vec, int n) {
    int n4 = n / 4;
    float4 *vec4 = reinterpret_cast<float4*>(vec);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n4; i += stride) {
        float4 v = vec4[i];
        v.x = Activation::device_forward(v.x);
        v.y = Activation::device_forward(v.y);
        v.z = Activation::device_forward(v.z);
        v.w = Activation::device_forward(v.w);
        vec4[i] = v;
    }

    if (blockIdx.x == 0 && threadIdx.x < n % 4) {
        int cleanup_idx = (n & ~0b11) + threadIdx.x;
        vec[cleanup_idx] = Activation::device_forward(vec[cleanup_idx]);
    }
}


#endif