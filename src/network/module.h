#ifndef MODEL_H_INCLUDED
#define MODEL_H_INCLUDED

#include "../types/tensor.h"
#include <cstdlib>

enum ForwardMode {
    Inference,
    Training,
};

template <int InputDim = 1, int OutputDim = 1>
class Module {

    public:

    static constexpr int BatchInputDim = InputDim+1;
    static constexpr int BatchOutputDim = OutputDim+1;

    MemoryLocation mem_location = MemoryLocation::Device;
    ForwardMode forward_mode = ForwardMode::Training;

    Tensor<float> *inputs = nullptr;
    Tensor<float> *input_errors = nullptr;

    Tensor<float> outputs;
    Tensor<float> output_errors;

    virtual ~Module() = default;

    virtual Tensor<float, BatchOutputDim>& forward(const Tensor<float> &input) = 0;
    
    virtual void update_error() = 0;
    virtual void update_gradients() = 0;

    virtual void update_parameters() = 0;
    virtual void zero_grads() = 0;

    void to(MemoryLocation new_mem_loc) {
        inputs.change_memory_location(new_mem_loc);
        input_errors.change_memory_location(new_mem_loc);
    }
};

#endif