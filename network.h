#ifndef NETWORK_H
#define NETWORK_H

#include "cublas_v2.h"
#include "cusparse.h"

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

void checkCublasError(cublasStatus_t stat);
void checkCusparseError(cusparseStatus_t stat);

void fully_connected(float *input, int n_inputs, float *weights, float *biases, float *output, int n_outputs);
void sparse_fully_connected(SparseVector input, float *weights, float *biases, float *output, int n_outputs);
__global__ void sparse_fully_connected_kernel(float *input_values, int *input_indices, unsigned int input_nnz, float *weights, float *biases, float *output, int n_outputs);
SparseVector dsg_fully_connected(SparseVector input, float *weights, float *biases, int n_outputs, CSR projection, float sparsity);
__global__ void spm_spv(float *mat_values, int *mat_col_ids, int *mat_row_indx, float *vec_values, int *vec_indices, int vec_nnz, float *output, int output_len, float alpha);
__device__ float atomicMaxf(float *address, float val);
__device__ float atomicMinf(float *address, float val);
__global__ void max_reduce(float *values, int num_elements, float *max);
__global__ void min_reduce(float *values, int num_elements, float *min);
__global__ void filter_activations(float *values, int num_elements, float *min, float *max, float sparsity, int *significant_unit_indices, int *num_significant_units);
__global__ void filtered_product(float *weights, float *input_values, int *input_indices, int input_nnz, float *biases, int n_outputs, int *significant_unit_indices, int *num_significant_units, float *output_values);

void relu(float *input, int n_inputs);
__global__ void relu_kernel(float *input, int n_inputs);

void batch_normalization(float *input, int n_inputs, float *output);

#endif /* !NETWORK_H */
