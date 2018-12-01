#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "network.h"

const int BLK_SIZE = 1024;

void fully_connected(float *input, int n_inputs, float *weights, float *output, int n_outputs) {
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Device alloc
    float *d_weights, *d_input, *d_output;
    cudaMalloc((void**)&d_weights, n_outputs * n_inputs * sizeof(float));
    cudaMalloc((void**)&d_input, n_inputs * sizeof(float));
    cudaMalloc((void**)&d_output, n_outputs * sizeof(float));

    // Copy to device
    stat = cublasSetMatrix(n_outputs, n_inputs, sizeof(float), weights, n_outputs, d_weights, n_outputs);
    stat = cublasSetVector(n_inputs, sizeof(float), input, 1, d_input, 1);
    stat = cublasSetVector(n_outputs, sizeof(float), output, 1, d_output, 1);

    // Device compute
    float alpha = 1.0;
    float beta = 0.0;
    stat = cublasSgemv(handle, CUBLAS_OP_N, n_outputs, n_inputs, &alpha, d_weights, n_outputs, d_input, 1, &beta, d_output, 1);

    // Copy answer from device
    stat = cublasGetVector(n_outputs, sizeof(float), d_output, 1, output, 1);

    // Clean up
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    cublasDestroy(handle);
}

void convolution(float *input, int input_width, int input_height, int input_channels, float *weights, int filter_size, int num_filters, float *output, int stride, int padding) {

}

__global__ void relu(float *input, int n_inputs) {
    int index = blockIdx.x * BLK_SIZE + threadIdx.x;

    if (index < n_inputs && input[index] < 0) {
        input[index] = 0;
    }
}

void batch_normalization(float *input, int n_inputs) {

}

void max_pooling(float *input, int input_width, int input_height, int input_channels, int filter_size, int stride, float *output) {

}
