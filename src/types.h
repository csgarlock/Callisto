#ifndef TYPES_H_INCLUDED
#define TYPES_H_INCLUDED

#include "util.h"

#include <cuda_runtime.h>
#include <cstdlib>
#include <array>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>

enum MemoryLocation {
    Host,
    Device,
};

template <typename T = float, int Dim = 1>
struct Tensor {
    T *data;
    int shape[Dim];
    MemoryLocation memory_location;
    
    Tensor(const std::array<int, Dim>& shape, MemoryLocation loc = MemoryLocation::Host) : memory_location(loc) {
        std::copy(shape.begin(), shape.end(), this->shape);
        if (loc == MemoryLocation::Device) {
            CUDA_CHECK(cudaMalloc(&data, mem_size()));
        }
        else {
            CUDA_CHECK(cudaMallocHost(&data, mem_size()));
        }
    }

    Tensor(MemoryLocation loc) : memory_location(loc) {
        if (loc == MemoryLocation::Device) {
            CUDA_CHECK(cudaMalloc(&data, sizeof(T)));
        }
        else {
            CUDA_CHECK(cudaMallocHost(&data, sizeof(T)));
        }
    }

    ~Tensor() {
        if (data) {
            if (memory_location == MemoryLocation::Host) {
                CUDA_CHECK(cudaFreeHost(data));
            } else {
                CUDA_CHECK(cudaFree(data));
            }
        }
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept : data(other.data), memory_location(other.memory_location) {
        std::copy(other.shape, other.shape + Dim, shape);
        other.data = nullptr;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            this->~Tensor();
            new (this) Tensor(std::move(other));
        }
        return *this;
    }

    size_t size() const {
        size_t total = 1;
        for (int i = 0; i < Dim; i++) {
            total *= shape[i];
        }
        return total;
    }
    
    size_t mem_size() const { return size() * sizeof(T); }
    
    void change_memory_location(MemoryLocation new_location) {
        if (new_location == memory_location || !data) {
            return;
        }

        T* new_data = nullptr;
        if (new_location == MemoryLocation::Device) {
            CUDA_CHECK(cudaMalloc(&new_data, mem_size()));
            CUDA_CHECK(cudaMemcpy(new_data, data, mem_size(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaFreeHost(data));
        } else {
            CUDA_CHECK(cudaMallocHost(&new_data, mem_size()));
            CUDA_CHECK(cudaMemcpy(new_data, data, mem_size(), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(data));
        }

        data = new_data;
        memory_location = new_location;
    }

    void copy_vec(std::vector<T> &vec) {
        if (vec.size() != size() || !data) {
            return;
        }
        if (memory_location == MemoryLocation::Device) {
            CUDA_CHECK(cudaMemcpy(data, vec.data(), mem_size(), cudaMemcpyHostToDevice));
        } else if (memory_location == MemoryLocation::Host) {
            std::copy(vec.begin(), vec.end(), data);
        }
    }

};

struct MatrixMultShape {
    int input;
    int output;
    int batch_size = 1;
};
    

#endif