#ifndef HOST_ACTIVATION_H_INCLUDED
#define HOST_ACTIVATION_H_INCLUDED

#include "../types.h"

template <typename Activation>
void cpu_activation(Tensor<float, 1> &vec) {
    for (int i = 0; i < vec.shape[0]; i++) {
        vec.data[i] = Activation::host_forward(vec.data[i]);
    }
}


#endif