#ifndef INFERENCE_TEST_H_INCLUDED
#define INFERENCE_TEST_H_INCLUDED

#include "../util.h"
#include "../kernels/activation.h"

#include <vector>
#include <iostream>
#include <random>

template <typename Activation>
void test_activation(const int n) {
    std::cout << "Testing activation with n = " << n << std::endl;

    std::vector<float> host_in(n);
    std::vector<float> host_out_cpu(n);

    // Fill input with simple pattern
    for (int i = 0; i < n; ++i)
        host_in[i] = 0.0f;
        // host_in[i] = std::sin(i * 0.01f) * 5.0f;

    float *dev_vec = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_vec, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev_vec, host_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Launch config
    const int threads = 256;
    const int blocks = (n / 4 + threads - 1) / threads;

    // Timing with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop)); 

    CUDA_CHECK(cudaEventRecord(start));
    activation<Activation><<<blocks, threads>>>(dev_vec, n);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Copy back results
    std::vector<float> host_out_gpu(n);
    CUDA_CHECK(cudaMemcpy(host_out_gpu.data(), dev_vec, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute reference on CPU
    for (int i = 0; i < n; ++i)
        host_out_cpu[i] = Activation::cpu(host_in[i]);

    // Validate
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::abs(host_out_cpu[i] - host_out_gpu[i]));
    }
    // Bandwidth calculation
    double bytes = static_cast<double>(n) * sizeof(float) * 2; // read + write
    double gbps = bytes / (elapsed_ms / 1e3) / 1e9;

    std::cout << "Max error: " << max_err << std::endl;
    std::cout << "Elapsed: " << elapsed_ms << " ms" << std::endl;
    std::cout << "Throughput: " << gbps << " GB/s" << std::endl;

    CUDA_CHECK(cudaFree(dev_vec));
}

void test_feed_forward(int m, int n);

#endif