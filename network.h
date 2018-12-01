#ifndef NETWORK_H
#define NETWORK_H

void fully_connected(float *input, int n_inputs, float *weights, float *output, int n_outputs);
void convolution(float *input, int input_width, int input_height, int input_channels, float *weights, int filter_size, int num_filters, float *output, int stride, int padding);

__global__ void relu(float *input, int n_inputs);

void batch_normalization(float *input, int n_inputs, float *output);
void max_pooling(float *input, int input_width, int input_height, int input_channels, int filter_size, int stride, float *output);

#endif /* !NETWORK_H */
