#ifndef UNIFIED_LOSS_H_INCLUDED
#define UNIFIED_LOSS_H_INCLUDED

#include "../types/tensor.h"
#include <cuda_runtime.h>

float mean_squared_error(Tensor<float> &predicted, Tensor<float> &actual);

#endif