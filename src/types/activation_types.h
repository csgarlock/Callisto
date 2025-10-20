#ifndef ACTIVATION_TYPES_H_INCLUDED
#define ACTIVATION_TYPES_H_INCLUDED

#include <cuda_runtime.h>

struct ReLU {
    __device__ __forceinline__ static float device_forward(float x) { return fmaxf(0.0f, x); }
    __device__ __forceinline__ static float device_derivative(float x) { return x > 0.0f ? 1.0f : 0.0f; }

    static float host_forward(float x) { return std::fmax(0.0f, x); }
    static float host_derivative(float x) { return x > 0.0f ? 1.0f : 0.0f; }
};

struct LogisticSigmoid {
    __device__ __forceinline__ static float device_forward(float x) { return 1.0f / (1.0f + expf(-x)); }
    __device__ __forceinline__ static float device_derivative(float x) { float sig = 1.0f / (1.0f + expf(-x)); return sig * (1.0f - sig); }

    static float host_forward(float x) { return 1.0f / (1.0f + std::exp(-x)); }
    static float host_derivative(float x) { float sig = 1.0f / (1.0f + std::exp(-x)); return sig * (1.0f - sig); }
};

struct Identity {
    __device__ __forceinline__ static float device_forward(float x) { return x; }
    __device__ __forceinline__ static float device_derivative(float x) { return 1.0f; }

    static float host_forward(float x) { return x; }
    static float host_derivative(float x) { return 1.0f; }
};

#endif