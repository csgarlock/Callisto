#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include "module.h"
#include "../types/tensor.h"

#include <vector>

class Network {
    public:

    std::vector<Module> layers;
    std::vector<Tensor<float>> results;

};

#endif