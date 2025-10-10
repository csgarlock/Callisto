#ifndef KERNEL_LOSS_H_INCLUDED
#define KERNEL_LOSS_H_INCLUDED

#include "../util.h"
#include "../types.h"

#include <cuda_runtime.h>

#define SHUFFLE_MASK 0xffffffffu

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

// Must be called with 1024 threads per block
// If calling with more than one block, MultiBlock must be true
template <bool MultiBlock = false>
__global__ void se_gpu(float *__restrict__ predicted, float *__restrict__ actual, float *__restrict__ loss, int n) {
    int n4 = n / 4;

    float4 *predicted4 = reinterpret_cast<float4*>(predicted);
    float4 *actual4 = reinterpret_cast<float4*>(actual);
    
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    __shared__ float warp_sums[32];
    __shared__ float final_sum;

    int idx = threadIdx.x;
    int stride = blockDim.x;
    if constexpr (MultiBlock) {
        idx += blockDim.x * blockIdx.x;
        stride *= gridDim.x;
    }

    float sum = 0.0f;
    for (int i = idx; i < n4; i += stride) {
        float4 pred_vals = predicted4[i];
        float4 actual_vals = actual4[i];
        sum += diff_and_square(pred_vals.x, actual_vals.x);
        sum += diff_and_square(pred_vals.y, actual_vals.y);
        sum += diff_and_square(pred_vals.z, actual_vals.z);
        sum += diff_and_square(pred_vals.w, actual_vals.w);
    }
    
    // bring last few values in if n not divisible by 4
    if (blockIdx.x == 0 && threadIdx.x < n % 4) {
        int cleanup_idx = (n & ~0b11) + threadIdx.x;
        sum += diff_and_square(predicted[cleanup_idx], actual[cleanup_idx]);
    }

    warp_reduction(sum, warp_sums);
    __syncthreads();
    if (warp == 0) {
        warp_reduction(warp_sums[lane], &final_sum);
        if (threadIdx.x == 0) {
            loss[blockIdx.x] = final_sum;
        }
    }
}

#endif