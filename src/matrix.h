#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <cstdlib>
#include <cuda_runtime.h>

struct MatrixMultShape {
    int input;
    int output;
};

__global__ void feed_forward(const float *__restrict__ input, const float *__restrict__ weights, const float *__restrict__ biases, float *__restrict__ output, MatrixMultShape shape);

#endif