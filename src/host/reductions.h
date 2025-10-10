#ifndef HOST_REDUCTIONS_H_INCLUDED
#define HOST_REDUCTIONS_H_INCLUDED

#include "../types.h"

template <typename T>
T cpu_sum_reduction(Tensor<T, 1> &tensor) {
    tensor.change_memory_location(MemoryLocation::Host);
    T sum = 0;
    for (size_t i = 0; i < tensor.shape[0]; i++)
        sum += tensor.data[i];
    return sum;
}

#endif