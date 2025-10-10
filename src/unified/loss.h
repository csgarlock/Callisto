#ifndef UNIFIED_LOSS_H_INCLUDED
#define UNIFIED_LOSS_H_INCLUDED

#include "../types.h"
#include <cuda_runtime.h>

float mean_squared_error(Tensor<float, 1> &predicted, Tensor<float, 1> &actual);

#endif