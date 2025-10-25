#ifndef TENSOR_H_INCLUDED
#define TENSOR_H_INCLUDED

#include "../util.h"

#include <cuda_runtime.h>
#include <cstdlib>
#include <vector>

enum MemoryLocation {
    Host,
    Device,
};

template <typename T = float>
class Tensor {
    
    public:
    
    T *data;
    size_t dim;
    std::vector<size_t> shape;
    MemoryLocation memory_location;

    Tensor(const std::vector<size_t> &shape, T *src_data, MemoryLocation loc = MemoryLocation::Host);
    Tensor(const std::vector<size_t> &shape, MemoryLocation loc = MemoryLocation::Host);
    Tensor(MemoryLocation loc);
    ~Tensor();
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    size_t size() const;
    size_t mem_size() const;
    void change_memory_location(MemoryLocation new_location);

};

#endif