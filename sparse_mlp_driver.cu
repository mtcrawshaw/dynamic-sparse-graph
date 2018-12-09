#include <iostream>
#include <fstream>

#include "network.h"
#include "utils.h"
#include "sparse_representation.hpp"

const int num_images = 60000;
const int image_len = 28 * 28;
const float sparsity = 0.8;

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
    float *input_data = (float*) malloc(num_images * image_len * sizeof(float));
    load_mnist_data(input_data); 

    // Declare and load in labels
    unsigned char *labels = (unsigned char*) malloc(num_images * sizeof(unsigned char));
    load_mnist_labels(labels);

    // Declare network activations
    SparseVector s_input, s_fc1_activations, s_fc2_activations;
    float *d_fc1_activations, *d_fc2_activations, *d_net_output;
    cudaMalloc(&d_fc1_activations, n_hidden1 * sizeof(float));
    cudaMalloc(&d_fc2_activations, n_hidden2 * sizeof(float));
    cudaMalloc(&d_net_output, n_outputs * sizeof(float));

    float *net_output = (float*) malloc(n_outputs * sizeof(float));
    int num_correct = 0;
    unsigned char pred = 0;
    float max_output = -1000;
    int msg_freq = 5000;

    float *temp_fc1_activations = (float*) malloc(n_hidden1 * sizeof(float));
    float *temp_fc2_activations = (float*) malloc(n_hidden2 * sizeof(float));

    int test_size = 60000;
    for (int n = 0; n < test_size; n++) {
        // Copy network input to gpu
	s_input = dense_to_SparseVector(input_data + n * image_len, image_len, 1);

        // Device compute
        sparse_fully_connected(s_input.values, s_input.indices, s_input.nnz, d_fc1_weights, d_fc1_biases, d_fc1_activations, n_hidden1);
        cudaMemcpy(temp_fc1_activations, d_fc1_activations, n_hidden1 * sizeof(float), cudaMemcpyDeviceToHost);
	s_fc1_activations = dense_to_SparseVector(temp_fc1_activations, n_hidden1, 1);
        relu(s_fc1_activations.values, s_fc1_activations.nnz);

        sparse_fully_connected(s_fc1_activations.values, s_fc1_activations.indices, s_fc2_activations.nnz, d_fc2_weights, d_fc2_biases, d_fc2_activations, n_hidden2);
        cudaMemcpy(temp_fc2_activations, d_fc2_activations, n_hidden2 * sizeof(float), cudaMemcpyDeviceToHost);
        s_fc2_activations = dense_to_SparseVector(temp_fc2_activations, n_hidden2, 1);
        relu(s_fc2_activations.values, s_fc2_activations.nnz);

        sparse_fully_connected(s_fc2_activations.values, s_fc2_activations.indices, s_fc2_activations.nnz, d_fc3_weights, d_fc3_biases, d_net_output, n_outputs);

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

    // Clean up
    free(fc1_weights);
    free(fc1_biases);
    free(fc2_weights);
    free(fc2_biases);
    free(fc3_weights);
    free(fc3_biases);
    free(input_data);
    free(labels);
    free(net_output);

    cudaFree(d_fc1_weights);
    cudaFree(d_fc1_biases);
    cudaFree(d_fc2_weights);
    cudaFree(d_fc2_biases);
    cudaFree(d_fc3_weights);
    cudaFree(d_fc3_biases);
    cudaFree(d_fc1_activations);
    cudaFree(d_fc2_activations);
    cudaFree(d_net_output);
    cudaFree(s_input.values);
    cudaFree(s_input.indices);
    cudaFree(s_fc1_activations.values);
    cudaFree(s_fc1_activations.indices);
    cudaFree(s_fc2_activations.values);
    cudaFree(s_fc2_activations.indices);

    return 0;
}

