#include "loss_test.h"

#include "../unified/loss.h"
#include "../host/loss.h"
#include "../types.h"

#include <vector>
#include <iostream>
#include <random>
#include <array>

void test_mse(int size) {
    std::cout << "Testing Mean Square error with: " << size << std::endl;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> predicted(size);
    std::vector<float> actual(size);

    for (int i = 0; i < size; i++) {
        predicted[i] = dist(rng);
        actual[i] = dist(rng);
        // std::cout << "i: " << i << ", pred: " << predicted[i] << ", act: " << actual[i] << std::endl;
    }

    Tensor<float, 1> predicted_tensor({size}, MemoryLocation::Device);
    Tensor<float, 1> actual_tensor({size}, MemoryLocation::Device);

    predicted_tensor.copy_vec(predicted);
    actual_tensor.copy_vec(actual);

    float reference = mse_cpu<float>(predicted.data(), actual.data(), size);
    
    float test_result = mean_squared_error(predicted_tensor, actual_tensor);

    std::cout << "Reference: " << reference << ", Test Result: " << test_result << ", Difference: " << std::abs(reference - test_result) << std::endl;

}