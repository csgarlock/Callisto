#ifndef KERNEL_LINEAR_H_INCLUDED
#define KERNEL_LINEAR_H_INCLUDED

#include "activation.h"

#include <cuda_runtime.h>

#define SHUFFLE_MASK 0xffffffffu

struct MatrixMultShape {
    int input;
    int output;
    int batch_size = 1;
};

__device__ __forceinline__ void fill_weights(const float *__restrict__ src, float *__restrict__ des, int x_offset, int y_offset, int n) {
    const float4 *src4 = reinterpret_cast<const float4*>(src);
    float4 *des4 = reinterpret_cast<float4*>(des);
    int effective_y = (threadIdx.x / 8) + y_offset;
    int effective_x = (threadIdx.x % 8) + (x_offset / 4);
    des4[threadIdx.x] = src4[effective_y * (n / 4) + effective_x];
}

template <bool Batch>
__device__ __forceinline__ void fill_input(const float *__restrict__ src, float *__restrict__ des, int x_offset, int k_offset, int n) {
    const float4 *src4 = reinterpret_cast<const float4*>(src);
    float4 *des4 = reinterpret_cast<float4*>(des);
    if constexpr (!Batch) {
        if (threadIdx.x < 32) {
            des[threadIdx.x] = src[threadIdx.x + x_offset];
        }
    }
}

template <bool Batch>
__device__ __forceinline__ void accumulate(const float *__restrict__ weights, float *__restrict__ input, float *__restrict__ sums) {
    int row = (threadIdx.x / 8);
    int col = (threadIdx.x % 8) * 4;
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        sum += weights[row * 32 + col + i] * input[col + i];
    }
    for (int offset = 4; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(SHUFFLE_MASK, sum, offset);
    }
    if (threadIdx.x % 8 == 0) {
        sums[row] += sum;
    }
}

// shape.n must be divisible by 32, shape.m must be divisible by 32.
// blockDim.x must equal 256
// Failing to meet any of the required precondition will result in undefined behavior
template <typename Activation = Identity>
__global__ void linear_forward_mtm(const float *__restrict__ input, const float *__restrict__ weights, const float *__restrict__ biases, float *__restrict__ output, MatrixMultShape shape) {
    
    // setup
    const int THREADS_PER_BLOCK = 256;
    const int TILE_WIDTH = 32;
    const int TILE_HEIGHT = 32;

    const int m = shape.output;
    const int n = shape.input;

    const int tile_m = m / TILE_HEIGHT;
    const int tile_n = n / TILE_WIDTH;
    
    __shared__ float weight_cache[TILE_WIDTH * TILE_HEIGHT];
    __shared__ float input_cache[TILE_WIDTH];
    __shared__ float sums[TILE_HEIGHT];

    // Each block takes care of a row of tiles at a time then moves to another row
    for (int tile_y = blockIdx.x; tile_y < tile_m; tile_y += gridDim.x) {
        // clear accumulators
        if (threadIdx.x < TILE_HEIGHT) {
            sums[threadIdx.x] = 0.0f;
        }

        for (int tile_x = 0; tile_x < tile_n; tile_x++) {

            int tile_y_offset = tile_y * TILE_HEIGHT;
            int tile_x_offset = tile_x * TILE_WIDTH;
            int tile_k_offset;

            fill_weights(weights, weight_cache, tile_x_offset, tile_y_offset, n);
            fill_input<false>(input, input_cache, tile_x_offset, tile_k_offset, n);
            __syncthreads();
            accumulate<false>(weight_cache, input_cache, sums);
            __syncthreads();
        }
        
        // The entire row of tiles now has it's sum in sums.
        // Move to output and add bias
        if (threadIdx.x < TILE_HEIGHT) {
            int output_index = tile_y * TILE_HEIGHT + threadIdx.x;
            output[output_index] = Activation::apply(sums[threadIdx.x] + biases[output_index]);
        }
    }   
}

#endif