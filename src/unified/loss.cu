#include "loss.h"

#include "../host/reductions.h"
#include "../host/loss.h"
#include "../kernels/loss.h"
#include "../util.h"

#include <cmath>
#include <vector>

#define MSE_SMALL_THRESHOLD 1024
#define MSE_MEDIUM_THRESHOLD 65536

float mean_squared_error(Tensor<float> &predicted, Tensor<float> &actual) {
    if (predicted.shape[0] != actual.shape[0]) {
        // actual errors later
        return std::nanf("");
    }
    size_t size = predicted.shape[0];
    if (size < MSE_SMALL_THRESHOLD && (predicted.memory_location == MemoryLocation::Host || actual.memory_location == MemoryLocation::Host)) {
        predicted.change_memory_location(MemoryLocation::Host);
        actual.change_memory_location(MemoryLocation::Host);
        return mse_cpu(predicted.data, actual.data, size);
    } else if (size < MSE_MEDIUM_THRESHOLD) {
        Tensor<float> result(MemoryLocation::Device);
        predicted.change_memory_location(MemoryLocation::Device);
        actual.change_memory_location(MemoryLocation::Device);
        se_gpu<<<1, 1024>>>(predicted.data, actual.data, result.data, size);
        CUDA_CHECK(cudaDeviceSynchronize());
        result.change_memory_location(MemoryLocation::Host);
        return result.data[0] / size;
    } else {
        size_t blocks = (size + 16384 - 1) / 16384;
        std::vector<size_t> size_vec = {blocks};
        Tensor<float> result(size_vec, MemoryLocation::Device);
        predicted.change_memory_location(MemoryLocation::Device);
        actual.change_memory_location(MemoryLocation::Device);
        se_gpu<true><<<blocks, 1024>>>(predicted.data, actual.data, result.data, size);
        CUDA_CHECK(cudaDeviceSynchronize());
        // callee handles changing memory location back to Host
        return cpu_sum_reduction<float>(result) / size;
    }
}