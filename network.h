#ifndef NETWORK_H
#define NETWORK_H

#include "sparse_representation.hpp"

struct MLP
{
    float *fc1_weights{};
    float *fc1_biases{};
    float *fc2_weights{};
    float *fc2_biases{};
    float *fc3_weights{};
    float *fc3_biases{};

    unsigned int n_inputs{};
    unsigned int n_hidden1{};
    unsigned int n_hidden2{};
    unsigned int n_outputs{};
};

void sparse_fully_connected(float *input_values, unsigned int *input_indices, unsigned int input_nnz, float *weights, float *biases, float *output, int n_outputs);
__global__ void sparse_fully_connected_kernel(float *input_values, unsigned int *input_indices, unsigned int input_nnz, float *weights, float *biases, float *output, int n_outputs);
void fully_connected(float *input, int n_inputs, float *weights, float *biases, float *output, int n_outputs);
void convolution(float *input, int input_width, int input_height, int input_channels, float *weights, int filter_size, int num_filters, float *output, int stride, int padding);

void relu(float *input, int n_inputs);
__global__ void relu_kernel(float *input, int n_inputs);

void batch_normalization(float *input, int n_inputs, float *output);
void max_pooling(float *input, int input_width, int input_height, int input_channels, int filter_size, int stride, float *output);

#endif /* !NETWORK_H */
