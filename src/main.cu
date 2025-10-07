#include "activation.h"
#include "util.h"
#include "matrix.h"

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
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

void test_feed_forward(int m, int n) {
    std::cout << "Testing feed_forward with m=" << m << ", n=" << n << std::endl;

    // Host memory
    std::vector<float> h_input(n);
    std::vector<float> h_weights(m * n);
    std::vector<float> h_biases(m);
    std::vector<float> h_output_cpu(m);
    std::vector<float> h_output_gpu(m);

    // Initialize data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) h_input[i] = dist(rng);
    for (int i = 0; i < m * n; ++i) h_weights[i] = dist(rng);
    for (int i = 0; i < m; ++i) h_biases[i] = dist(rng);

    // CPU reference
    for (int row = 0; row < m; ++row) {
        float sum = h_biases[row];
        for (int col = 0; col < n; ++col)
            sum += h_weights[row * n + col] * h_input[col];
        h_output_cpu[row] = sum;
    }

    // Device memory
    float *d_input, *d_weights, *d_biases, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_biases, m * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, m * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_biases, h_biases.data(), m * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel configuration
    const int threads = 256;
    const int blocks = std::min(m, 128);  // up to 128 rows in parallel

    MatrixMultShape shape{n, m};

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    feed_forward<<<blocks, threads>>>(d_input, d_weights, d_biases, d_output, shape);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, m * sizeof(float), cudaMemcpyDeviceToHost));

    // Validation
    float max_err = 0.0f;
    for (int i = 0; i < m; ++i)
        max_err = std::max(max_err, std::abs(h_output_cpu[i] - h_output_gpu[i]));

    // Performance metrics
    double flops = static_cast<double>(m) * n * 2.0; // mul + add per element
    double gflops = (flops / (elapsed_ms / 1e3)) / 1e9;

    std::cout << "Max error: " << max_err << std::endl;
    std::cout << "Elapsed: " << elapsed_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_biases));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    // Example: 4096×4096 matvec (big enough for stable timing)
    test_feed_forward(4096, 4096);
    test_feed_forward(8192, 32768);
    test_feed_forward(32768, 8192);
    return 0;
}

// int main() {
//     int n = 1 << 26;  // 16 million elements (64 MB)
//     test_activation<ReLU>(n);
//     test_activation<LogisticSigmoid>(n);
//     return 0;
// }