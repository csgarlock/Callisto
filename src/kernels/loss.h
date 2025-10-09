#ifndef LOSS_H_INCLUDED
#define LOSS_H_INCLUDED

#include "../util.h"
#include "../types.h"

#include <cuda_runtime.h>

#define MSE_SMALL_THRESHOLD 1024
#define MSE_MEDIUM_THRESHOLD 65536

float mean_squared_error(Tensor<float, 1> &predicted, Tensor<float, 1> &actual);

template <typename T = float>
float mse_cpu(const T *__restrict__ predicted, const T *__restrict__ actual, int n);

template <bool MultiBlock = false>
__global__ void se_gpu(float *__restrict__ predicted, float *__restrict__ actual, float *__restrict__ loss, int n);

#endif