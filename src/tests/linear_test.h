#ifndef LINEAR_TEST_H_INCLUDED
#define LINEAR_TEST_H_INCLUDED

#include "../types/activation_types.h"
#include "../unified/linear.h"
#include "../types.h"
#include "../util.h"

#include <vector>
#include <iostream>
#include <random>
#include <array>
#include <thread>
#include <chrono>

template <typename Activation = Identity>
void linear_forward_test(int m, int n) {
    std::cout << "Testing linear_forward with m=" << m << ", n=" << n << std::endl;
    
    Tensor<float, 1> t_input({n}, MemoryLocation::Host);
    Tensor<float, 2> t_weights({n, m}, MemoryLocation::Host);
    Tensor<float, 1> t_biases({m}, MemoryLocation::Host);

    // Initialize data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) t_input.data[i] = dist(rng);
    for (int i = 0; i < m * n; ++i) t_weights.data[i] = dist(rng);
    for (int i = 0; i < m; ++i) t_biases.data[i] = dist(rng);

    Tensor<float, 1> h_output({m}, MemoryLocation::Host);
    Tensor<float, 1> d_output({m}, MemoryLocation::Device);

    linear_forward<Activation>(t_input, t_weights, t_biases, h_output, MemoryLocation::Host);
    std::cout << "Host Finished..." << std::endl;
    linear_forward<Activation>(t_input, t_weights, t_biases, d_output, MemoryLocation::Device);
    std::cout << "Device Finished..." << std::endl;

    d_output.change_memory_location(MemoryLocation::Host);

    float max_error = 0.0f;
    for (int i = 0; i < m; i++) {
        max_error = std::max(max_error, std::abs(h_output.data[i] - d_output.data[i]));
    }
    std::cout << "Max Error: " << max_error << std::endl;
    CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Activation = Identity>
void linear_forward_test_batch(int m, int n, int k) {
    std::cout << "Testing linear_forward with m=" << m << ", n=" << n <<", k=" << k << std::endl;
    
    Tensor<float, 2> t_input({n, k}, MemoryLocation::Host);
    Tensor<float, 2> t_weights({n, m}, MemoryLocation::Host);
    Tensor<float, 1> t_biases({m}, MemoryLocation::Host);

    // Initialize data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n * k; ++i) t_input.data[i] = dist(rng);
    for (int i = 0; i < m * n; ++i) t_weights.data[i] = dist(rng);
    for (int i = 0; i < m; ++i) t_biases.data[i] = dist(rng);
    
    Tensor<float, 2> h_output({m, k}, MemoryLocation::Host);
    Tensor<float, 2> d_output({m, k}, MemoryLocation::Device);
    
    linear_forward_batch<Activation>(t_input, t_weights, t_biases, h_output, MemoryLocation::Host);
    std::cout << "Host Finished..." << std::endl;

    linear_forward_batch<Activation>(t_input, t_weights, t_biases, d_output, MemoryLocation::Device);
    std::cout << "Device Finished..." << std::endl;
    
    d_output.change_memory_location(MemoryLocation::Host);

    float max_error = 0.0f;
    for (int i = 0; i < m * k; i++) {
        max_error = std::max(max_error, std::abs(h_output.data[i] - d_output.data[i]));
    }

    std::cout << "Max Error: " << max_error << std::endl;
    CUDA_CHECK(cudaDeviceSynchronize());
}

#endif