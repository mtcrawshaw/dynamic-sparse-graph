#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "network.h"
#include "utils.hpp"

const int BLK_SIZE = 1024;

void fill_weights(float *weights, int n_inputs, int n_outputs) {
    for (int i = 0; i < n_outputs; i++) {
	for (int j = 0; j < n_inputs; j++) {
	    weights[j * n_outputs + i] = 2 * i - j;
	}
    }
}

void load_network(float *fc1_weights, float *fc2_weights, float *fc3_weights, const char *path) {
    FILE *ptr;
    ptr = fopen("models/mlp_weights.bin", "rb");

    unsigned int fc1_rows = 256;
    unsigned int fc1_cols = 784;
    unsigned int fc2_rows = 256;
    unsigned int fc2_cols = 256;
    unsigned int fc3_rows = 256;
    unsigned int fc3_cols = 10;

    fread(fc1_weights, fc1_rows * fc1_cols, 1, ptr);
    fread(fc2_weights, fc2_rows * fc2_cols, 1, ptr);
    fread(fc3_weights, fc3_rows * fc3_cols, 1, ptr);

    fclose(ptr);
}

void load_mnist(float *mnist_data) {
    int num_images = 60000;
    int image_len = 28 * 28;
    const char *path = "data/mnist/train-images-idx3-ubyte";

    FILE *ptr;
    ptr = fopen(path, "rb");

    // Eat up header information
    // START HERE TOMORROW!!!
    unsigned char* bytes = (unsigned char*) malloc(num_images * image_len * sizeof(unsigned char));
    fseek(ptr, sizeof(int) * 4, SEEK_SET);
    fread(bytes, num_images * image_len, 1, ptr);
    fclose(ptr);

    for (int n = 0; n < num_images; n++) {
	for (int i = 0; i < image_len; i++) {
	    mnist_data[n * image_len + i] = bytes[n * image_len + i] / 255.0;
	}
    }
}

int main(int argc, char *argv[]) {

    // Declare and load in network
    int n_inputs = 784;
    int n_hidden1 = 256;
    int n_hidden2 = 256;
    int n_outputs = 10;
    
    float *fc1_weights = (float*) malloc(n_inputs * n_hidden1 * sizeof(float));
    float *fc2_weights = (float*) malloc(n_hidden1 * n_hidden2 * sizeof(float));
    float *fc3_weights = (float*) malloc(n_hidden2 * n_outputs * sizeof(float));

    load_network(fc1_weights, fc2_weights, fc3_weights, "models/mlp_weights.bin");

    // Declare and load in data
    int num_images = 60000;
    int image_len = 28 * 28;
    float *mnist_data = (float*) malloc(num_images * image_len * sizeof(float));
    load_mnist(mnist_data);

    // Forward pass for single image
    float *fc1_activations = (float*) malloc(n_hidden1 * sizeof(float));
    float *fc2_activations = (float*) malloc(n_hidden2 * sizeof(float));
    float *net_output = (float*) malloc(n_outputs * sizeof(float));

    dim3 threads(BLK_SIZE);
    dim3 grid1((int)ceil((float)n_hidden1/BLK_SIZE));
    fully_connected(mnist_data, n_inputs, fc1_weights, fc1_activations, n_hidden1);
    relu<<<grid1, threads>>>(fc1_activations, n_hidden1);
    printf("made it\n");
    fully_connected(fc1_activations, n_hidden1, fc2_weights, fc2_activations, n_hidden2);
    printf("made it\n");
    dim3 grid2((int)ceil((float)n_hidden2/BLK_SIZE));
    relu<<<grid2, threads>>>(fc2_activations, n_hidden2);
    printf("made it\n");
    fully_connected(fc2_activations, n_hidden2, fc3_weights, net_output, n_outputs);
    printf("made it\n");

    for (int i = 0; i < n_outputs; i++) {
	printf("%f ", net_output[i]);
    }

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

