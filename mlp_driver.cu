#include <iostream>
#include <fstream>

#include "network.h"
#include "utils.hpp"

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

    load_mlp_weights(fc1_weights, fc1_biases, fc2_weights, fc2_biases, fc3_weights, fc3_biases, "models/mlp_weights.bin");

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

    float *net_output = (float*) malloc(n_outputs * sizeof(float));
    int num_correct = 0;
    unsigned char pred = 0;
    float max_output = -1000;
    int msg_freq = 5000;

    int test_size = 60000;
    for (int n = 0; n < test_size; n++) {
        // Device compute
        fully_connected(d_input_data + n * image_len, n_inputs, d_fc1_weights, d_fc1_biases, d_fc1_activations, n_hidden1);
        relu(d_fc1_activations, n_hidden1);
        fully_connected(d_fc1_activations, n_hidden1, d_fc2_weights, d_fc2_biases, d_fc2_activations, n_hidden2);
        relu(d_fc2_activations, n_hidden2);
        fully_connected(d_fc2_activations, n_hidden2, d_fc3_weights, d_fc3_biases, d_net_output, n_outputs);

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

        if ((n + 1) % msg_freq == 0)
            printf("Done with %d instances.\n", n + 1);
    }

    printf("%d / %d correct.\n", num_correct, num_images);
    float accuracy = (float)num_correct / test_size;
    printf("Accuracy: %f\n", accuracy);

    return 0;
}

