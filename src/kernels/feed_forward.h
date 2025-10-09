#ifndef FEED_FORWARD_H_INCLUDED
#define FEED_FORWARD_H_INCLUDED

#include "activation.h"

#include <cuda_runtime.h>

struct MatrixMultShape {
    int input;
    int output;
};
 
// shape.n must be divisible by 32, shape.m must be divisible by 32.
// blockDim.x must equal 256
// Failing to meet any of the required precondition will result in undefined behavior
template <typename Activation = Identity>
__global__ void feed_forward(const float *__restrict__ input, const float *__restrict__ weights, const float *__restrict__ biases, float *__restrict__ output, MatrixMultShape shape) {
    
    // setup
    const int THREADS_PER_BLOCK = 256;
    const int TILE_WIDTH = 32;
    const int TILE_HEIGHT = 32;
    const int TILE_SIZE = TILE_WIDTH * TILE_HEIGHT;
    const int WARP_COUNT = THREADS_PER_BLOCK / 32;

    const int m = shape.output;
    const int n = shape.input;

    const int tile_m = m / TILE_HEIGHT;
    const int tile_n = n / TILE_WIDTH;
    
    __shared__ float weight_cache[TILE_WIDTH * TILE_HEIGHT];
    __shared__ float input_cache[TILE_WIDTH];
    __shared__ float sums[TILE_HEIGHT];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    
    const unsigned int shuffle_mask = 0xffffffff;

    // Each block takes care of a row of tiles at a time then moves to another row
    for (int tile_y = blockIdx.x; tile_y < tile_m; tile_y += gridDim.x) {
        // clear accumulators
        if (threadIdx.x < TILE_HEIGHT) {
            sums[threadIdx.x] = 0.0f;
        }

        for (int tile_x = 0; tile_x < tile_n; tile_x++) {

            int tile_y_offset = tile_y * TILE_HEIGHT;
            int tile_x_offset = tile_x * TILE_WIDTH;

            // fill caches 
            for (int i = threadIdx.x; i < TILE_SIZE; i += THREADS_PER_BLOCK) {
                int effective_y = (i / TILE_WIDTH) + tile_y_offset;
                int effective_x = (i % TILE_WIDTH) + tile_x_offset;
                int effective_address = effective_y * n + effective_x;
                weight_cache[i] = weights[effective_address];
            }

            if (threadIdx.x < TILE_WIDTH) {
                input_cache[threadIdx.x] = input[threadIdx.x + tile_x_offset];
            }

            __syncthreads();

            // Accumulation
            float sum;
            for (int tile_row = warp; tile_row < TILE_HEIGHT; tile_row += WARP_COUNT) {

                sum = weight_cache[tile_row * TILE_WIDTH + lane] * input_cache[lane];

                // Reduction
                for (int offset = 16; offset > 0; offset /= 2) {
                    sum += __shfl_down_sync(shuffle_mask, sum, offset);
                }
            
                if (lane == 0) {
                    sums[tile_row] += sum;
                }
            }

            __syncthreads();
        }

        // The entire row of tiles now has it's sum in sums.
        // Move to output and add bias
        if (threadIdx.x < TILE_HEIGHT) {
            int output_index = tile_y * TILE_HEIGHT + threadIdx.x;
            output[output_index] = Activation::apply(sums[threadIdx.x] + biases[output_index]);
        }

        __syncthreads();
    }   
}

#endif