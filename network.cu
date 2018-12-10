#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"

#include "network.h"

const int BLK_SIZE = 1024;

const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status)
    {
	case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

const char* cusparseGetErrorString(cusparseStatus_t status) {
    switch(status)
    {
	case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

void fully_connected(float *input, int n_inputs, float *weights, float *biases, float *output, int n_outputs) {
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);

    // Device compute
    float alpha = 1.0;
    float beta = 1.0;
    cudaMemcpy(output, biases, n_outputs * sizeof(float), cudaMemcpyDeviceToDevice);
    stat = cublasSgemv(handle, CUBLAS_OP_N, n_outputs, n_inputs, &alpha, weights, n_outputs, input, 1, &beta, output, 1);

    cublasDestroy(handle);
}

void sparse_fully_connected(SparseVector input, float *weights, float *biases, float *output, int n_outputs) {
    dim3 threads(BLK_SIZE);
    dim3 grid((int)ceil((float)n_outputs/BLK_SIZE));
    sparse_fully_connected_kernel<<<grid, threads>>>(input.values, input.indices, input.nnz, weights, biases, output, n_outputs);
}

__global__ void sparse_fully_connected_kernel(float *input_values, int *input_indices, unsigned int input_nnz, float *weights, float *biases, float *output, int n_outputs) {
    unsigned int index = blockIdx.x * BLK_SIZE + threadIdx.x;
    unsigned int element_index;

    if (index < n_outputs) {
	float ans = 0;
	for (int i = 0; i < input_nnz; i++) {
	    element_index = input_indices[i];
	    ans += input_values[i] * weights[element_index * n_outputs + index];
        }

	output[index] = ans + biases[index];
    }
}

void dsg_fully_connected(SparseVector input, float *weights, float *biases, float *output, int n_outputs, CSR projection, float sparsity) {
    // Create CUBLAS context for later
    cublasHandle_t dense_handle;
    cublasCreate(&dense_handle);

    // Reduce dimension of weight matrix by projecting it with projection
    float alpha = 1.0 / sqrt(projection.nrows);
    float beta = 0.0;
    float *reduced_weights;
    cudaMalloc(&reduced_weights, projection.nrows * n_outputs * sizeof(float));
    cusparseHandle_t sparse_handle;
    cusparseCreate(&sparse_handle);
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseScsrmm2(sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, projection.nrows, n_outputs, projection.ncols, projection.nnz, &alpha, descrA, projection.values, projection.row_indx, projection.col_ids, weights, n_outputs, &beta, reduced_weights, projection.nrows);
    cusparseDestroy(sparse_handle);
    cusparseDestroyMatDescr(descrA);

    // Reduce dimension of input vector by projecting it with projection
    float *reduced_input;
    cudaMalloc(&reduced_input, projection.nrows * sizeof(float));
    dim3 threads(BLK_SIZE);
    dim3 grid((int)ceil((float)projection.nrows/BLK_SIZE));
    spm_spv<<<grid, threads>>>(projection.values, projection.col_ids, projection.row_indx, input.values, input.indices, input.nnz, reduced_input, projection.nrows, alpha);

    // Multiply reduced weight with reduced vector
    alpha = 1.0;
    beta = 1.0;
    float *reduced_product;
    cudaMalloc(&reduced_product, n_outputs * sizeof(float));
    cudaMemcpy(reduced_product, biases, n_outputs * sizeof(float), cudaMemcpyDeviceToDevice);
    cublasSgemv(dense_handle, CUBLAS_OP_T, n_outputs, projection.nrows, &alpha, reduced_weights, n_outputs, reduced_input, 1, &beta, reduced_product, 1);
    cublasDestroy(dense_handle);
    
    // Top k search to find output units with large approximate activation
    int K = (int)ceil(n_outputs * (1.0 - sparsity));
    int *top_indices = (int*) malloc(K * sizeof(int));
    
}

__global__ void spm_spv(float *mat_values, int *mat_col_ids, int *mat_row_indx, float *vec_values, int *vec_indices, int vec_nnz, float *output, int output_len, float alpha) {
    unsigned int index = blockIdx.x * BLK_SIZE + threadIdx.x;
    float ans = 0;

    if (index < output_len) {
        int mat_row_end = mat_row_indx[index + 1];

        int mat_pos = mat_row_indx[index];
        int vec_pos = 0;
        int mat_col, vec_row;

	while (mat_pos < mat_row_end && vec_pos < vec_nnz) {
	    mat_col = mat_col_ids[mat_pos];
	    vec_row = vec_indices[vec_pos];
	    if (index == 1)
		printf("%d %d %f %d %d %f\n", mat_pos, mat_col, mat_values[mat_pos], vec_pos, vec_row, vec_values[vec_pos]);
	    if (mat_col == vec_row) {
		ans += mat_values[mat_pos] * vec_values[vec_pos];
		mat_pos += 1;
		vec_pos += 1;
	    } else if (mat_col < vec_row) {
		mat_pos += 1;
	    } else {
		vec_pos += 1;
	    }
        }

        output[index] = ans * alpha;
    }
}

void relu(float *input, int n_inputs) {
    dim3 threads(BLK_SIZE);
    dim3 grid((int)ceil((float)n_inputs/BLK_SIZE));
    relu_kernel<<<grid, threads>>>(input, n_inputs);
}

__global__ void relu_kernel(float *input, int n_inputs) {
    int index = blockIdx.x * BLK_SIZE + threadIdx.x;

    if (index < n_inputs && input[index] < 0) {
        input[index] = 0;
    }
}

void batch_normalization(float *input, int n_inputs) {

}

