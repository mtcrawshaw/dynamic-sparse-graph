#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "network.h"

const int BLK_SIZE = 1024;

const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status)
    {
	case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

void fully_connected(float *input, int n_inputs, float *weights, float *biases, float *output, int n_outputs) {
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);

    // Device compute
    float alpha = 1.0;
    float beta = 1.0;
    cudaMemcpy(output, biases, n_outputs * sizeof(float), cudaMemcpyDeviceToDevice);
    stat = cublasSgemv(handle, CUBLAS_OP_N, n_outputs, n_inputs, &alpha, weights, n_outputs, input, 1, &beta, output, 1);

    cublasDestroy(handle);
}

void sparse_fully_connected(float *input_values, unsigned int *input_indices, unsigned int input_nnz, float *weights, float *biases, float *output, int n_outputs) {
    dim3 threads(BLK_SIZE);
    dim3 grid((int)ceil((float)n_outputs/BLK_SIZE));
    sparse_fully_connected_kernel<<<grid, threads>>>(input_values, input_indices, input_nnz, weights, biases, output, n_outputs);
}

__global__ void sparse_fully_connected_kernel(float *input_values, unsigned int *input_indices, unsigned int input_nnz, float *weights, float *biases, float *output, int n_outputs) {
    unsigned int index = blockIdx.x * BLK_SIZE + threadIdx.x;
    unsigned int element_index;

    if (index < n_outputs) {
	float ans = 0;
	for (int i = 0; i < input_nnz; i++) {
	    element_index = input_indices[i];
	    ans += input_values[i] * weights[element_index * n_outputs + index];
        }

	output[index] = ans + biases[index];
    }
}

void convolution(float *input, int input_width, int input_height, int input_channels, float *weights, int filter_size, int num_filters, float *output, int stride, int padding) {

}

void relu(float *input, int n_inputs) {
    dim3 threads(BLK_SIZE);
    dim3 grid((int)ceil((float)n_inputs/BLK_SIZE));
    relu_kernel<<<grid, threads>>>(input, n_inputs);
}

__global__ void relu_kernel(float *input, int n_inputs) {
    int index = blockIdx.x * BLK_SIZE + threadIdx.x;

    if (index < n_inputs && input[index] < 0) {
        input[index] = 0;
    }
}

void batch_normalization(float *input, int n_inputs) {

}

void max_pooling(float *input, int input_width, int input_height, int input_channels, int filter_size, int stride, float *output) {

}
