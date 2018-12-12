#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <float.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"

#include "network.h"
#include "utils.h"

const int num_images = 60000;
const int image_len = 28 * 28;
const int msg_freq = 5000;

const int n_inputs = 784;
const int n_hidden1 = 256;
const int n_hidden2 = 256;
const int n_outputs = 10;

float sparsity = 0.5;

float dense_infer(MLP mlp, float *input_data, unsigned char *labels) {

    printf("Starting dense inference...\n");

    // Declare network activations
    float *d_input, *d_fc1_activations, *d_fc2_activations, *d_net_output;
    cudaMalloc(&d_input, n_inputs * sizeof(float));
    cudaMalloc(&d_fc1_activations, n_hidden1 * sizeof(float));
    cudaMalloc(&d_fc2_activations, n_hidden2 * sizeof(float));
    cudaMalloc(&d_net_output, n_outputs * sizeof(float));

    float *net_output = (float*) malloc(n_outputs * sizeof(float));
    int num_correct = 0;
    unsigned char pred = 0;
    float max_output = -1000;

    int test_size = 60000;
    for (int n = 0; n < test_size; n++) {

	// Copy input to gpu
	cudaMemcpy(d_input, input_data + n * image_len, image_len * sizeof(float), cudaMemcpyHostToDevice);

        // Device compute
        fully_connected(d_input, n_inputs, mlp.fc1_weights, mlp.fc1_biases, d_fc1_activations, n_hidden1);
        relu(d_fc1_activations, n_hidden1);
        fully_connected(d_fc1_activations, n_hidden1, mlp.fc2_weights, mlp.fc2_biases, d_fc2_activations, n_hidden2);
        relu(d_fc2_activations, n_hidden2);
        fully_connected(d_fc2_activations, n_hidden2, mlp.fc3_weights, mlp.fc3_biases, d_net_output, n_outputs);

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

    // Clean up
    free(net_output);
    cudaFree(d_input);
    cudaFree(d_fc1_activations);
    cudaFree(d_fc2_activations);
    cudaFree(d_net_output);

    printf("Finished dense inference.\n");

    return (float)num_correct / test_size;
}


float dsg_infer(MLP mlp, float *input_data, unsigned char *labels, float sparsity) {
    printf("Starting dsg inference...\n");

    // Declare network activations and output
    SparseVector s_input, s_fc1_activations, s_fc2_activations;
    float *net_output = (float*) malloc(n_outputs * sizeof(float));
    float *d_net_output;
    cudaMalloc(&d_net_output, n_outputs * sizeof(float));

    int num_correct = 0;
    unsigned char pred = 0;
    float max_output = -FLT_MAX;

    // Generate linear projections for each layer
    CSR projection1 = get_random_projection(150, n_inputs, 3, 1);
    CSR projection2 = get_random_projection(50, n_hidden1, 3, 1);

    int test_size = 10000;

    struct SparseVector inputs[test_size];
    for (int n = 0; n < test_size; n++) {
	inputs[n] = dense_to_SparseVector(input_data + n * image_len, image_len, 1);
    }

    for (int n = 0; n < test_size; n++) {

        // Copy network input to gpu
        //s_input = dense_to_SparseVector(input_data + n * image_len, image_len, 1);

        // Device compute
        s_fc1_activations = dsg_fully_connected(inputs[n], mlp.fc1_weights, mlp.fc1_biases, n_hidden1, projection1, sparsity);
        relu(s_fc1_activations.values, s_fc1_activations.nnz);
        s_fc2_activations = dsg_fully_connected(s_fc1_activations, mlp.fc2_weights, mlp.fc2_biases, n_hidden2, projection2, sparsity);
        relu(s_fc2_activations.values, s_fc2_activations.nnz);
        sparse_fully_connected(s_fc2_activations, mlp.fc3_weights, mlp.fc3_biases, d_net_output, n_outputs);

        // Copy network output to host
        cudaMemcpy(net_output, d_net_output, n_outputs * sizeof(float), cudaMemcpyDeviceToHost);

	// Compare network output with true label
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

    // Clean up
    free(net_output);
    cudaFree(s_input.values);
    cudaFree(s_input.indices);
    cudaFree(s_fc1_activations.values);
    cudaFree(s_fc1_activations.indices);
    cudaFree(projection1.values);
    cudaFree(projection1.col_ids);
    cudaFree(projection1.row_indx);
    cudaFree(projection2.values);
    cudaFree(projection2.col_ids);
    cudaFree(projection2.row_indx);

    printf("Finished dsg inference.\n");

    return (float)num_correct / test_size;
}

int main(int argc, char *argv[]) {

    // Load model parameters
    MLP mlp = load_mlp("models/mlp_weights.bin", n_inputs, n_hidden1, n_hidden2, n_outputs);

    // Declare and load in data
    float *input_data = (float*) malloc(num_images * image_len * sizeof(float));
    load_mnist_data(input_data);

    // Declare and load in labels
    unsigned char *labels = (unsigned char*) malloc(num_images * sizeof(unsigned char));
    load_mnist_labels(labels);

    //float dense_accuracy = dense_infer(mlp, input_data, labels);
    //printf("Dense accuracy: %f\n\n", dense_accuracy);
    float dsg_accuracy = dsg_infer(mlp, input_data, labels, sparsity);
    printf("DSG accuracy: %f\n\n", dsg_accuracy);

    // Clean up
    free(input_data);
    free(labels);

    cudaFree(mlp.fc1_weights);
    cudaFree(mlp.fc1_biases);
    cudaFree(mlp.fc2_weights);
    cudaFree(mlp.fc2_biases);
    cudaFree(mlp.fc3_weights);
    cudaFree(mlp.fc3_biases);

    return 0;
}

