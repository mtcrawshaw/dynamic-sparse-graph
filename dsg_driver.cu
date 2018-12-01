#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "network.h"

void fill_weights(layer &l) {
    l.weights = (float*) malloc(l.n_units * l.n_inputs * sizeof(float));
    for (int i = 0; i < l.n_units; i++) {
	for (int j = 0; j < l.n_inputs; j++) {
	    l.weights[i * l.n_inputs + j] = 2 * i - j;
	}
    }
}

void forward(layer &l, float *input, float *output) {
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Device alloc
    float *d_weights, *d_input, *d_output;
    cudaMalloc((void**)&d_weights, l.n_units * l.n_inputs * sizeof(float));
    cudaMalloc((void**)&d_input, l.n_inputs * sizeof(float));
    cudaMalloc((void**)&d_output, l.n_units * sizeof(float));

    // Copy to device
    stat = cublasSetMatrix(l.n_units, l.n_inputs, sizeof(float), l.weights, l.n_units, d_weights, l.n_units);
    stat = cublasSetVector(l.n_inputs, sizeof(float), input, 1, d_input, 1);
    stat = cublasSetVector(l.n_units, sizeof(float), output, 1, d_output, 1);

    // Device compute
    float alpha = 1.0;
    float beta = 0.0;
    stat = cublasSgemv(handle, CUBLAS_OP_N, l.n_units, l.n_inputs, &alpha, d_weights, l.n_units, d_input, 1, &beta, d_output, 1);

    // Copy answer from device
    stat = cublasGetVector(l.n_units, sizeof(float), d_output, 1, output, 1);

    // Clean up
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    cublasDestroy(handle);
}

void print_vector(float *vec, int len) {
    for (int i = 0; i < len; i++) {
	std::cout << vec[i] << ' ';
    }
    std::cout << std::endl;
}

void print_matrix(float *mat, int n_rows, int n_cols) {
    for (int i = 0; i < n_rows; i++){
	for (int j = 0; j < n_cols; j++) {
	    std::cout << mat[j * n_rows + i] << ' ';
        }
	std::cout << std::endl;
    }
}

int main(int argc, char *argv[]) {

    int n_units = 5;
    int n_inputs = 4;

    layer l1;
    l1.n_units = n_units;
    l1.n_inputs = n_inputs;
    fill_weights(l1);

    float *input, *output;
    input = (float*) malloc(n_inputs * sizeof(float));
    output = (float*) malloc(n_units * sizeof(float));

    for (int i = 0; i < n_inputs; i++) {
	input[i] = i;
    }
    
    forward(l1, input, output);
    std::cout << "Weights:" << std::endl;
    print_matrix(l1.weights, l1.n_units, l1.n_inputs);
    std::cout << "Input:" << std::endl;
    print_vector(input, n_inputs);
    std::cout << "Output:" << std::endl;
    print_vector(output, n_units);

    return 0;
}

    // Host alloc
    // Device alloc
    // Host initialization
    // Host compute
    // Declare events and array to hold exeuction times
    // Declare streams and grid, threads
    // Copy to gpu
    // Device compute
    // Calculate and output GFLOPS
    // Calculate min, max, and average stream time
    // Host vs device validation
    // Free memory

