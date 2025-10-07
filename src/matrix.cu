#include "matrix.h"

__global__ void feed_forward(const float *__restrict__ input, const float *__restrict__ weights, const float *__restrict__ biases, float *__restrict__ output, MatrixMultShape shape) {
    
    int m = shape.output;
    int n = shape.input;
    __shared__ float warp_sums[32];
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    const unsigned int shuffle_mask = 0xffffffff;

    for (int row = blockIdx.x; row < m; row += gridDim.x) {
        float sum = 0.0f;
        int row_offset = row * n;
        for (int col = threadIdx.x; col < n; col += blockDim.x) {
            sum += weights[row_offset + col] * input[col];
        }

        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(shuffle_mask, sum, offset);
        }


        if (lane == 0) {
            warp_sums[warp] = sum;
        }

        __syncthreads();

        if (warp == 0) {
            float warp_sum = 0.0f;
            if (threadIdx.x < (blockDim.x / 32)) {
                warp_sum = warp_sums[threadIdx.x];
            }
            for (int offset = 16; offset > 0; offset /= 2) {
                warp_sum += __shfl_down_sync(shuffle_mask, warp_sum, offset);
            }
            if (threadIdx.x == 0) {
                output[row] = warp_sum + biases[row];
            }
        }
    }
}