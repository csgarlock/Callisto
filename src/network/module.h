#ifndef MODEL_H_INCLUDED
#define MODEL_H_INCLUDED

#include "../types/tensor.h"

enum ForwardMode {
    Inference,
    Training,
};

class Module {

    public:

    MemoryLocation mem_location = MemoryLocation::Device;
    ForwardMode forward_mode = ForwardMode::Training;

    Tensor<float> *inputs = nullptr;
    Tensor<float> *input_errors = nullptr;

    Tensor<float> *outputs = nullptr;
    Tensor<float> *output_errors = nullptr;

    virtual ~Module() = default;

    virtual void forward() = 0;
    
    virtual void propagate_error() = 0;
    virtual void find_gradients() = 0;

    virtual void update_parameters() = 0;
    virtual void zero_grads() = 0;

};

#endif