#include "loss.h"

#include "../types.h"
#include "../helpers/reductions.h"
#include "../util.h"

#include <cassert>
#include <cmath>
#include <array>

#define SHUFFLE_MASK 0xffffffffu

__device__ void warp_reduction(float sum, float *output);
__device__ float diff_and_square(float predicted, float actual);

float mean_squared_error(Tensor<float, 1> &predicted, Tensor<float, 1> &actual) {
    if (predicted.shape[0] != actual.shape[0]) {
        // actual errors later
        return std::nanf("");
    }
    int size = predicted.shape[0];
    if (size < MSE_SMALL_THRESHOLD) {
        predicted.change_memory_location(MemoryLocation::Host);
        actual.change_memory_location(MemoryLocation::Host);
        return mse_cpu<float>(predicted.data, actual.data, size);
    } else if (size < MSE_MEDIUM_THRESHOLD) {
        Tensor<float, 0> result(MemoryLocation::Device);
        predicted.change_memory_location(MemoryLocation::Device);
        actual.change_memory_location(MemoryLocation::Device);
        mse_gpu_single_block<<<1, 1024>>>(predicted.data, actual.data, result.data, size);
        CUDA_CHECK(cudaDeviceSynchronize());
        result.change_memory_location(MemoryLocation::Host);
        return result.data[0];
    } else {
        int blocks = (size + 16384 - 1) / 16384;
        Tensor<float, 1> result({blocks}, MemoryLocation::Device);
        predicted.change_memory_location(MemoryLocation::Device);
        actual.change_memory_location(MemoryLocation::Device);
        mse_gpu_multi_block<<<blocks, 1024>>>(predicted.data, actual.data, result.data, size);
        CUDA_CHECK(cudaDeviceSynchronize());
        // callee handles changing memory location back to Host
        return cpu_sum_reduction<float>(result) / size;
    }
}

template <typename T>
float mse_cpu(const T *__restrict__ predicted, const T *__restrict__ actual, int n) {
    T acc = 0;
    for (int i = 0; i < n; i++) {
        T diff = actual[i] - predicted[i];
        acc += diff * diff;
    }
    return acc;
}

// Must be called with 1 block and 1024 threads per block
__global__ void mse_gpu_single_block(float *__restrict__ predicted, float *__restrict__ actual, float *__restrict__ loss, int n) {
    int n4 = n / 4;

    float4 *predicted4 = reinterpret_cast<float4*>(predicted);
    float4 *actual4 = reinterpret_cast<float4*>(actual);
    
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    __shared__ float warp_sums[32];
    __shared__ float final_sum;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 pred_vals = predicted4[i];
        float4 actual_vals = actual4[i];
        sum += diff_and_square(pred_vals.x, actual_vals.x);
        sum += diff_and_square(pred_vals.y, actual_vals.y);
        sum += diff_and_square(pred_vals.z, actual_vals.z);
        sum += diff_and_square(pred_vals.w, actual_vals.w);
    }
    
    // bring last few values in if n not divisible by 4
    if (threadIdx.x < n % 4) {
        int cleanup_idx = (n & ~0b11) + threadIdx.x;
        sum += diff_and_square(predicted[cleanup_idx], actual[cleanup_idx]);
    }

    warp_reduction(sum, warp_sums);
    __syncthreads();
    if (warp == 0) {
        warp_reduction(warp_sums[lane], &final_sum);
        if (threadIdx.x == 0) {
            *loss = final_sum / float (n);
        }
    }

}

__global__ void mse_gpu_multi_block(float *__restrict__ predicted, float *__restrict__ actual, float *__restrict__ loss, int n) {

}

__device__ void warp_reduction(float sum, float *output) {
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(SHUFFLE_MASK, sum, offset);
    }
    if (threadIdx.x % 32 == 0) {
        output[threadIdx.x / 32] = sum;
    }
}

__device__ float diff_and_square(float predicted, float actual) {
    return (actual - predicted) * (actual - predicted);
}
