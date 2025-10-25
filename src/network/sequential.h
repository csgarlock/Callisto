#ifndef SEQUENTIAL_H_INCLUDED
#define SEQUENTIAL_H_INCLUDED

#include "module.h"
#include "../types/tensor.h"

#include <vector>

class Sequential : Module {

    public:

    std::vector<Module> layers;
    
    void forward() override;
    
    void propagate_error() override;
    void find_gradients() override;

    void update_parameters() override;
    void zero_grads() override;


};

#endif