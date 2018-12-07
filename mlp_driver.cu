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

void load_network(float *fc1_weights, float *fc1_biases, float *fc2_weights, float *fc2_biases, float *fc3_weights, float *fc3_biases, const char *path) {
    FILE *ptr;
    ptr = fopen("models/mlp_weights.bin", "rb");

    unsigned int fc1_rows = 256;
    unsigned int fc1_cols = 784;
    unsigned int fc2_rows = 256;
    unsigned int fc2_cols = 256;
    unsigned int fc3_rows = 10;
    unsigned int fc3_cols = 256;

    fread(fc1_weights, sizeof(float), fc1_rows * fc1_cols, ptr);
    fread(fc1_biases, sizeof(float), fc1_rows, ptr);
    fread(fc2_weights, sizeof(float), fc2_rows * fc2_cols, ptr);
    fread(fc2_biases, sizeof(float), fc2_rows, ptr);
    fread(fc3_weights, sizeof(float), fc3_rows * fc3_cols, ptr);
    fread(fc3_biases, sizeof(float), fc3_rows, ptr);

    fclose(ptr);
}

void load_mnist_data(float *mnist_data) {
    int num_images = 60000;
    int image_len = 28 * 28;
    const char *path = "data/mnist/train-images-idx3-ubyte";

    FILE *ptr;
    ptr = fopen(path, "rb");

    unsigned char* bytes = (unsigned char*) malloc(num_images * image_len * sizeof(unsigned char));
    fseek(ptr, sizeof(int) * 4, SEEK_SET);
    size_t n1 = fread(bytes, sizeof(unsigned char), num_images * image_len, ptr);
    size_t n2 = fread(bytes, 1, 1, ptr);
    fclose(ptr);

    for (int n = 0; n < num_images; n++) {
	for (int i = 0; i < image_len; i++) {
	    mnist_data[n * image_len + i] = bytes[n * image_len + i] / 255.0;
	}
    }
}

void load_mnist_labels(unsigned char *mnist_labels) {
    int num_images = 60000;
    const char *path = "data/mnist/train-labels-idx1-ubyte";

    FILE *ptr;
    ptr = fopen(path, "rb");

    fseek(ptr, sizeof(int) * 2, SEEK_SET);
    fread(mnist_labels, sizeof(unsigned char), num_images, ptr);
    fclose(ptr);
}

int main(int argc, char *argv[]) {

    // Declare and load in network weights
    int n_inputs = 784;
    int n_hidden1 = 256;
    int n_hidden2 = 256;
    int n_outputs = 10;
    
    float *fc1_weights = (float*) malloc(n_inputs * n_hidden1 * sizeof(float));
    float *fc1_biases = (float*) malloc(n_hidden1 * sizeof(float));
    float *fc2_weights = (float*) malloc(n_hidden1 * n_hidden2 * sizeof(float));
    float *fc2_biases = (float*) malloc(n_hidden2 * sizeof(float));
    float *fc3_weights = (float*) malloc(n_hidden2 * n_outputs * sizeof(float));
    float *fc3_biases = (float*) malloc(n_outputs * sizeof(float));

    float *d_fc1_weights, *d_fc2_weights, *d_fc3_weights;
    float *d_fc1_biases, *d_fc2_biases, *d_fc3_biases;
    cudaMalloc(&d_fc1_weights, n_inputs * n_hidden1 * sizeof(float));
    cudaMalloc(&d_fc1_biases, n_hidden1 * sizeof(float));
    cudaMalloc(&d_fc2_weights, n_hidden1 * n_hidden2 * sizeof(float));
    cudaMalloc(&d_fc2_biases, n_hidden2 * sizeof(float));
    cudaMalloc(&d_fc3_weights, n_hidden2 * n_outputs * sizeof(float));
    cudaMalloc(&d_fc3_biases, n_outputs * sizeof(float));

    load_network(fc1_weights, fc1_biases, fc2_weights, fc2_biases, fc3_weights, fc3_biases, "models/mlp_weights.bin");

    cudaMemcpy(d_fc1_weights, fc1_weights, n_inputs * n_hidden1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_biases, fc1_biases, n_hidden1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_weights, fc2_weights, n_hidden1 * n_hidden2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_biases, fc2_biases, n_hidden2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc3_weights, fc3_weights, n_hidden2 * n_outputs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc3_biases, fc3_biases, n_outputs * sizeof(float), cudaMemcpyHostToDevice);

    // Declare and load in data
    int num_images = 60000;
    int image_len = 28 * 28;
    float *input_data = (float*) malloc(num_images * image_len * sizeof(float));

    float *d_input_data;
    cudaMalloc(&d_input_data, num_images * image_len * sizeof(float));

    load_mnist_data(input_data); // Since the storage requirements are small, we can move entire dataset to GPU at once. This should change once we add streams
    cudaMemcpy(d_input_data, input_data, num_images * image_len * sizeof(float), cudaMemcpyHostToDevice);

    // Declare and load in labels
    unsigned char *labels = (unsigned char*) malloc(num_images * sizeof(unsigned char));
    load_mnist_labels(labels);

    // Declare network activations
    float *d_fc1_activations, *d_fc2_activations, *d_net_output;
    cudaMalloc(&d_fc1_activations, n_hidden1 * sizeof(float));
    cudaMalloc(&d_fc2_activations, n_hidden2 * sizeof(float));
    cudaMalloc(&d_net_output, n_outputs * sizeof(float));

    dim3 threads(BLK_SIZE);
    dim3 grid1((int)ceil((float)n_hidden1/BLK_SIZE));
    dim3 grid2((int)ceil((float)n_hidden2/BLK_SIZE));

    float *net_output = (float*) malloc(n_outputs * sizeof(float));
    int num_correct = 0;
    unsigned char pred = 0;
    float max_output = -1000;
    int msg_freq = 5000;

    for (int n = 0; n < num_images; n++) {
        // Device compute
        fully_connected(d_input_data + n * image_len, n_inputs, d_fc1_weights, d_fc1_biases, d_fc1_activations, n_hidden1);
        relu<<<grid1, threads>>>(d_fc1_activations, n_hidden1);
        fully_connected(d_fc1_activations, n_hidden1, d_fc2_weights, d_fc2_biases, d_fc2_activations, n_hidden2);
        relu<<<grid2, threads>>>(d_fc2_activations, n_hidden2);
        fully_connected(d_fc2_activations, n_hidden2, d_fc3_weights, d_fc1_biases, d_net_output, n_outputs);

        // Copy network output to host
        cudaMemcpy(net_output, d_net_output, n_outputs * sizeof(float), cudaMemcpyDeviceToHost);

        pred = 0;
        max_output = 0;
        for (unsigned char i = 0; i < n_outputs; i++) {
            if (net_output[i] > max_output) {
                pred = i;
                max_output = net_output[i];
            }
        }

        if (pred == labels[n]) 
	    num_correct++;

        if (n % msg_freq == 0)
            printf("Done with %d instances.\n", n);
    }

    printf("%d / %d correct.\n", num_correct, num_images);
    float accuracy = (float)num_correct / num_images;
    printf("Accuracy: %f\n", accuracy);

    return 0;
}

