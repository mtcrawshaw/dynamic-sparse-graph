#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "network.h"
#include "utils.hpp"

void fill_weights(float *weights, int n_inputs, int n_outputs) {
    for (int i = 0; i < n_outputs; i++) {
	for (int j = 0; j < n_inputs; j++) {
	    weights[j * n_outputs + i] = 2 * i - j;
	}
    }
}

int main(int argc, char *argv[]) {

    int n_outputs = 5;
    int n_inputs = 4;
    float *weights = (float*) malloc(n_inputs * n_outputs * sizeof(float));
    fill_weights(weights, n_inputs, n_outputs);

    float *input, *output;
    input = (float*) malloc(n_inputs * sizeof(float));
    output = (float*) malloc(n_outputs * sizeof(float));
    for (int i = 0; i < n_inputs; i++) {
	input[i] = i;
    }
    
    fully_connected(input, n_inputs, weights, output, n_outputs);
    std::cout << "Weights:" << std::endl;
    print_matrix(weights, n_outputs, n_inputs);
    std::cout << "Input:" << std::endl;
    print_vector(input, n_inputs);
    std::cout << "Output:" << std::endl;
    print_vector(output, n_outputs);

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

