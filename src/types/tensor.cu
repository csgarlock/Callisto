#include "tensor.h"

#include <iostream>
#include <algorithm>
#include <cstring>

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t> &shape, T *src_data, MemoryLocation loc) : memory_location(loc) {
    dim = shape.size();
    this->shape = shape;
    if (loc == MemoryLocation::Device) {
        CUDA_CHECK(cudaMalloc(&data, mem_size()));
        CUDA_CHECK(cudaMemcpy(data, src_data, mem_size(), cudaMemcpyHostToDevice));
    }
    else {
        data = new T[size()];
        memcpy(data, src_data, mem_size());
    }
}

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t> &shape, MemoryLocation loc) : memory_location(loc) {
    dim = shape.size();
    this->shape = shape;
    if (loc == MemoryLocation::Device) {
        CUDA_CHECK(cudaMalloc(&data, mem_size()));
    }
    else {
        data = new T[size()];
    }
}

template <typename T>
Tensor<T>::Tensor(MemoryLocation loc) : memory_location(loc) {
    dim = 0;
    if (loc == MemoryLocation::Device) {
        CUDA_CHECK(cudaMalloc(&data, sizeof(T)));
    }
    else {
        data = new T[1];
    }
}

template <typename T>
Tensor<T>::~Tensor() {
    if (data) {
        if (memory_location == MemoryLocation::Host) {
            delete[] data;
        } else {
            CUDA_CHECK(cudaFree(data));
        }
    }
}

template <typename T>
Tensor<T>::Tensor(Tensor&& other) noexcept : data(other.data), memory_location(other.memory_location) {
    dim = other.dim; 
    shape = other.shape;
    other.data = nullptr;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        this->~Tensor();
        new (this) Tensor(std::move(other));
    }
    return *this;
}

template <typename T>
size_t Tensor<T>::size() const {
    size_t total = 1;
    for (int i = 0; i < dim; i++) {
        total *= shape[i];
    }
    return total;
}

template <typename T>
size_t Tensor<T>::mem_size() const { return size() * sizeof(T); }

template <typename T>
void Tensor<T>::change_memory_location(MemoryLocation new_location) {
    if (new_location == memory_location || !data) {
        return;
    }

    T* new_data = nullptr;
    if (new_location == MemoryLocation::Device) {
        CUDA_CHECK(cudaMalloc(&new_data, mem_size()));
        CUDA_CHECK(cudaMemcpy(new_data, data, mem_size(), cudaMemcpyHostToDevice));
        delete[] data;
    } else {
        new_data = new T[size()];
        CUDA_CHECK(cudaMemcpy(new_data, data, mem_size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(data));
    }
    data = new_data;
    memory_location = new_location;
}

template class Tensor<float>;
template class Tensor<int>;