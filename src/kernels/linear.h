#ifndef KERNEL_LINEAR_H_INCLUDED
#define KERNEL_LINEAR_H_INCLUDED

#include "../types/activation_types.h"
#include "../types.h"

#include <cuda_runtime.h>

#define SHUFFLE_MASK 0xffffffffu

template <bool Batch>
__device__ __forceinline__ void clear_accumulators(float *__restrict__ sums) {
    if constexpr (Batch) {
        for (int i = 0; i < 4; i++) {
            sums[threadIdx.x * 4 + i] = 0.0f;
        }
    } else {
        if (threadIdx.x < 32) {
            sums[threadIdx.x] = 0.0f;
        }
    }
}

__device__ __forceinline__ void fill_weights(const float *__restrict__ src, float *__restrict__ des, int x_offset, int y_offset, int n) {
    const float4 *src4 = reinterpret_cast<const float4*>(src);
    float4 *des4 = reinterpret_cast<float4*>(des);
    int effective_y = (threadIdx.x / 8) + y_offset;
    int effective_x = (threadIdx.x % 8) + (x_offset / 4);
    des4[threadIdx.x] = src4[effective_y * (n / 4) + effective_x];
}

template <bool Batch>
__device__ __forceinline__ void fill_input(const float *__restrict__ src, float *__restrict__ des, int x_offset, int k_offset, int n) {
    if constexpr (Batch) {
        const float4 *src4 = reinterpret_cast<const float4*>(src);
        float4 *des4 = reinterpret_cast<float4*>(des);
        int effective_k = (threadIdx.x / 8) + k_offset;
        int effective_x = (threadIdx.x % 8) + (x_offset / 4);
        des4[threadIdx.x] = src4[effective_k * (n / 4) + effective_x];
    } else {
        if (threadIdx.x < 32) {
            des[threadIdx.x] = src[threadIdx.x + x_offset];
        }
    }
}

template <bool Batch>
__device__ __forceinline__ void accumulate(const float *__restrict__ weights, float *__restrict__ input, float *__restrict__ sums) {
    if constexpr (Batch) {
        // 1 thread per 4 rows
        int start_row = (threadIdx.x % 8) * 4;
        int depth = (threadIdx.x / 8);
        for (int row = start_row; row < start_row + 4; row++) {
            float sum = 0.0f;
            for (int col = 0; col < 32; col++) {
                sum += weights[row * 32 + col] * input[depth * 32 + col];
            }
            sums[depth * 32 + row] += sum;
        }
    } else {
        // 8 threads per 1 row
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
}

template <bool Batch, typename Activation>
__device__ __forceinline__ void move_to_output(const float *__restrict__ sums, const float *__restrict__ biases, float *__restrict__ output, int y_offset, int k_offset, int m) {
    if constexpr (Batch) {
        int start_row = (threadIdx.x % 8) * 4 + y_offset;
        int depth = (threadIdx.x / 8) + k_offset;
        for (int i = 0; i < 4; i++) {
            int row = start_row + i;
            output[depth * m + row] = Activation::device_forward(sums[threadIdx.x * 4 + i] + biases[row]);
        }
    } else {
        if (threadIdx.x < 32) {
            int output_index = y_offset + threadIdx.x;
            output[output_index] = Activation::device_forward(sums[threadIdx.x] + biases[output_index]);
        }
    }
}

// shape.n must be divisible by 32, shape.m must be divisible by 32.
// blockDim.x must equal 256
// Failing to meet any of the required precondition will result in undefined behavior
template <typename Activation = Identity>
__global__ void linear_forward_mtm(const float *__restrict__ input, const float *__restrict__ weights, const float *__restrict__ biases, float *__restrict__ output, MatrixMultShape shape) {
    
    // setup
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
        int tile_y_offset = tile_y * TILE_HEIGHT;
        
        clear_accumulators<false>(sums);

        for (int tile_x = 0; tile_x < tile_n; tile_x++) {
            int tile_x_offset = tile_x * TILE_WIDTH;

            fill_weights(weights, weight_cache, tile_x_offset, tile_y_offset, n);
            fill_input<false>(input, input_cache, tile_x_offset, 0, n);
            __syncthreads();
            accumulate<false>(weight_cache, input_cache, sums);
            __syncthreads();
        }
        
        move_to_output<false, Activation>(sums, biases, output, tile_y_offset, 0, m);
    }   
}

// shape.n must be divisible by 32, shape.m must be divisible by 32, shape.batch_size must be divisible by 32.
// blockDim.x must equal 256
// Failing to meet any of the required precondition will result in undefined behavior
template <typename Activation = Identity>
__global__ void linear_forward_mtm_batch(const float *__restrict__ input, const float *__restrict__ weights, const float *__restrict__ biases, float *__restrict__ output, MatrixMultShape shape) {
    
    // setup
    const int TILE_WIDTH = 32;
    const int TILE_HEIGHT = 32;
    const int TILE_DEPTH = 32;

    const int m = shape.output;
    const int n = shape.input;
    const int k = shape.batch_size;

    const int tile_m = m / TILE_HEIGHT;
    const int tile_n = n / TILE_WIDTH;
    const int tile_k = k / TILE_DEPTH;
    
    __shared__ float weight_cache[TILE_WIDTH * TILE_HEIGHT];
    __shared__ float input_cache[TILE_WIDTH * TILE_DEPTH];
    __shared__ float sums[TILE_HEIGHT * TILE_DEPTH];

    for (int tile_z = blockIdx.y; tile_z < tile_k; tile_z += gridDim.y) {

        int tile_k_offset = tile_z * TILE_DEPTH;
        // Each block takes care of a row of tiles at a time then moves to another row
        for (int tile_y = blockIdx.x; tile_y < tile_m; tile_y += gridDim.x) {
            int tile_y_offset = tile_y * TILE_HEIGHT;
            
            clear_accumulators<true>(sums);

            for (int tile_x = 0; tile_x < tile_n; tile_x++) {

                int tile_x_offset = tile_x * TILE_WIDTH;

                fill_weights(weights, weight_cache, tile_x_offset, tile_y_offset, n);
                fill_input<true>(input, input_cache, tile_x_offset, tile_k_offset, n);
                __syncthreads();
                accumulate<true>(weight_cache, input_cache, sums);
                __syncthreads();
            }

            move_to_output<true, Activation>(sums, biases, output, tile_y_offset, tile_k_offset, m);
        }
    }
}

__device__ void linear_warp_reduction(float sum, float *output) {
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(SHUFFLE_MASK, sum, offset);
    }
    if (threadIdx.x % 32 == 0) {
        output[threadIdx.x / 32] = sum;
    }
}

template <typename Activation = Identity>
__global__ void linear_forward_mto(const float *__restrict__ input, const float *__restrict__ weights, const float *__restrict__ bias, float *__restrict__ output, MatrixMultShape shape) {
    
    const int n = shape.input;
    const int k = shape.batch_size;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    __shared__ float warp_sums[32];
    __shared__ float final_sum;

    for (int depth = blockIdx.x; depth < k; depth += gridDim.x) {
        float sum = 0.0f;
        for (int col = threadIdx.x; col < n; col += blockDim.x) {
            sum += weights[col] * input[depth * n + col];
        }
        linear_warp_reduction(sum, warp_sums);
        __syncthreads();
        if (threadIdx.x == 0) {
            linear_warp_reduction(warp_sums[lane], &final_sum);
            if (threadIdx.x == 0) {
                output[depth] = Activation::device_forward(final_sum + *bias);
            }
        }
        __syncthreads();
    }    
}

template <typename Activation = Identity>
__global__ void linear_forward_ftm(const float *__restrict__ input, const float *__restrict__ weights, const float *__restrict__ biases, float *__restrict__ output, MatrixMultShape shape) {

    const int m = shape.output;
    const int n = shape.input;
    const int k = shape.batch_size;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    __shared__ float warp_sums[32];
    __shared__ float final_sum;

    for (int depth = 0; depth < k; depth++) {

        for (int row = blockIdx.x; row < m; row += gridDim.x) {
            float sum = 0.0f;
            for (int col = threadIdx.x; col < n; col += blockDim.x) {
                sum += weights[row * n + col] * input[depth * n + col];
            }
            linear_warp_reduction(sum, warp_sums);
            __syncthreads();
            if (threadIdx.x == 0) {
                linear_warp_reduction(warp_sums[lane], &final_sum);
                if (threadIdx.x == 0) {
                    output[depth * m + row] = Activation::device_forward(final_sum + biases[row]);
                }
            }
            __syncthreads();
        }
    }
    
}

#endif