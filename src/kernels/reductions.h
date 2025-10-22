#ifndef KERNEL_REDUCTIONS_H_INCLUDED
#define KERNEL_REDUCTIONS_H_INCLUDED

#include <cuda_runtime.h>

__device__ void reduction_gpu(float *__restrict__ input, float *output);

#endif