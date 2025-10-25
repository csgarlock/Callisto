#include "loss_test.h"

#include "../unified/loss.h"
#include "../host/loss.h"
#include "../types/tensor.h"

#include <vector>
#include <iostream>
#include <random>

void test_mse(size_t size) {
    std::cout << "Testing Mean Square error with: " << size << std::endl;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> predicted(size);
    std::vector<float> actual(size);

    for (int i = 0; i < size; i++) {
        predicted[i] = dist(rng);
        actual[i] = dist(rng);
    }

    std::vector<size_t> size_vec = {size};

    Tensor<float> predicted_tensor(size_vec, predicted.data(), MemoryLocation::Device);
    Tensor<float> actual_tensor(size_vec, actual.data(), MemoryLocation::Device);

    float reference = mse_cpu<float>(predicted.data(), actual.data(), size);
    
    float test_result = mean_squared_error(predicted_tensor, actual_tensor);

    std::cout << "Reference: " << reference << ", Test Result: " << test_result << ", Difference: " << std::abs(reference - test_result) << std::endl;

}