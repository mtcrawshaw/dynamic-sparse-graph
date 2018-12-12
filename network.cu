#include <iostream>
#include <math.h>
#include <float.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse.h"

#include "network.h"
#include "utils.h"

const int BLK_SIZE = 128;

void fully_connected(float *input, int n_inputs, float *weights, float *biases, float *output, int n_outputs) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Device compute
    float alpha = 1.0;
    float beta = 1.0;
    cudaMemcpy(output, biases, n_outputs * sizeof(float), cudaMemcpyDeviceToDevice);
    cublasSgemv(handle, CUBLAS_OP_N, n_outputs, n_inputs, &alpha, weights, n_outputs, input, 1, &beta, output, 1);

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

SparseVector dsg_fully_connected(SparseVector input, float *weights, float *biases, int n_outputs, CSR projection, float sparsity) {
    // Create CUBLAS context for later
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // Reduce dimension of weight matrix by projection x weights^T
    float alpha = 1.0 / sqrt(projection.nrows);
    float beta = 0.0;
    float *reduced_weights;
    cudaMalloc(&reduced_weights, projection.nrows * n_outputs * sizeof(float));
    cusparseHandle_t cusparse_handle;
    cusparseCreate(&cusparse_handle);
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseScsrmm2(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, projection.nrows, n_outputs, projection.ncols, projection.nnz, &alpha, descrA, projection.values, projection.row_indx, projection.col_ids, weights, n_outputs, &beta, reduced_weights, projection.nrows);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroy(cusparse_handle);

    // Reduce dimension of input vector by projection x input
    float *reduced_input;
    cudaMalloc(&reduced_input, projection.nrows * sizeof(float));
    dim3 threads(BLK_SIZE);
    dim3 grid1((int)ceil((float)projection.nrows/BLK_SIZE));
    cudaCheckErr();
    spm_spv<<<grid1, threads>>>(projection.values, projection.col_ids, projection.row_indx, input.values, input.indices, input.nnz, projection.ncols, reduced_input, projection.nrows, alpha);
    cudaCheckErr();
    
    // Approximate weights * input by (reduced weight)^T * (reduced input)
    alpha = 1.0;
    beta = 1.0;
    float *reduced_product;
    cudaMalloc(&reduced_product, n_outputs * sizeof(float));
    cudaMemcpy(reduced_product, biases, n_outputs * sizeof(float), cudaMemcpyDeviceToDevice);
    cublasSgemv(cublas_handle, CUBLAS_OP_T, projection.nrows, n_outputs, &alpha, reduced_weights, projection.nrows, reduced_input, 1, &beta, reduced_product, 1);
    cublasDestroy(cublas_handle);

    // Get min and max to estimate threshold for top values
    float *reduced_min, *reduced_max;
    cudaMalloc((void**)&reduced_min, sizeof(float));
    cudaMalloc((void**)&reduced_max, sizeof(float));
    dim3 grid2((int)ceil((float)n_outputs/BLK_SIZE));
    max_reduce<<<grid2, threads>>>(reduced_product, n_outputs, reduced_max);
    min_reduce<<<grid2, threads>>>(reduced_product, n_outputs, reduced_min);

    // Get list of indices of output units whose predicted activations are larger than estimated threshold
    int *significant_unit_indices;
    int *num_significant_units;
    cudaMalloc(&significant_unit_indices, n_outputs * sizeof(int));
    cudaMalloc((void**)&num_significant_units, sizeof(int));
    filter_activations<<<grid2, threads>>>(reduced_product, n_outputs, reduced_min, reduced_max, sparsity, significant_unit_indices, num_significant_units);
  
    // Multiply weights with input vector according to binary mask
    SparseVector output;
    cudaMemcpy(&output.nnz, num_significant_units, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMalloc(&output.values, output.nnz * sizeof(float));
    cudaMalloc(&output.indices, output.nnz * sizeof(int));
    cudaMemcpy(output.indices, significant_unit_indices, output.nnz * sizeof(int), cudaMemcpyDeviceToDevice);
    dim3 grid3((int)ceil((float)output.nnz/BLK_SIZE));
    filtered_product<<<grid2, threads>>>(weights, input.values, input.indices, input.nnz, biases, n_outputs, significant_unit_indices, num_significant_units, output.values);

    // Add freeing stuff up here 
    cudaFree(reduced_weights);
    cudaFree(reduced_input);
    cudaFree(reduced_product);
    cudaFree(reduced_min);
    cudaFree(reduced_max);
    cudaFree(significant_unit_indices);
    cudaFree(num_significant_units);

    return output;
}

__global__ void spm_spv(float *mat_values, int *mat_col_ids, int *mat_row_indx, float *vec_values, int *vec_indices, int vec_nnz, int vec_len, float *output, int output_len, float alpha) {
    unsigned int index = blockIdx.x * BLK_SIZE + threadIdx.x;
    float ans = 0;
    __shared__ float s_vec_values[BLK_SIZE];
    __shared__ int s_vec_indices[BLK_SIZE];

    int mat_row_end = index < output_len ? mat_row_indx[index + 1] : 0;

    int mat_pos = index <= output_len ? mat_row_indx[index] : 0;
    int vec_pos = 0;
    int mat_col, vec_row;
    int vec_val_index;
    int mat_col_end;

    // We load the vector values and indices into shared memory in intervals
    int num_intervals = ceil((float)vec_nnz / BLK_SIZE);
    for (int interval = 0; interval < num_intervals; interval++) {
	vec_val_index = interval * BLK_SIZE + threadIdx.x;
	s_vec_values[threadIdx.x] = vec_val_index < vec_nnz ? vec_values[vec_val_index] : 0;
	s_vec_indices[threadIdx.x] = vec_val_index < vec_nnz ? vec_indices[vec_val_index] : vec_len;
	__syncthreads();

	// This loop calculates the dot product of the vector interval and the corresponding matrix row interval
	vec_pos = 0;
        mat_col_end = s_vec_indices[BLK_SIZE - 1];
        mat_col = mat_pos < mat_row_end ? mat_col_ids[mat_pos] : vec_len;

	while (mat_col <= mat_col_end && vec_pos < BLK_SIZE) {
	    vec_row = s_vec_indices[vec_pos];

	    if (mat_col == vec_row) {
	        ans += mat_values[mat_pos] * s_vec_values[vec_pos];
		mat_pos += 1;
		vec_pos += 1;
	    } else if (mat_col < vec_row) {
		mat_pos += 1;
	    } else {
		vec_pos += 1;
	    }
	    mat_col = mat_pos < mat_row_end ? mat_col_ids[mat_pos] : vec_len + 1;
        }
	    
        __syncthreads();
    }

    if (index < vec_len)
        output[index] = ans * alpha;
    
}

__device__ float atomicMaxf(float *address, float val) {
    int *address_as_int = (int*) address;
    int old = *address_as_int, assumed;
    do {
	assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
__device__ float atomicMinf(float *address, float val) {
    int *address_as_int = (int*) address;
    int old = *address_as_int, assumed;
    do {
	assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void max_reduce(float *values, int num_elements, float *max) {
    __shared__ float shared_max[BLK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid == 0)
	*max = -FLT_MAX;

    shared_max[tid] = -FLT_MAX;
    if (gid < num_elements) 
	shared_max[tid] = values[gid];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
	if (tid < s && gid < num_elements) 
	    shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
	__syncthreads();
    }

    if (tid == 0) 
	atomicMaxf(max, shared_max[0]);
}

__global__ void min_reduce(float *values, int num_elements, float *min) {
    __shared__ float shared_min[BLK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid == 0)
	*min = FLT_MAX;

    shared_min[tid] = FLT_MAX;
    if (gid < num_elements) 
	shared_min[tid] = values[gid];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
	if (tid < s && gid < num_elements) 
	    shared_min[tid] = fminf(shared_min[tid], shared_min[tid + s]);
	__syncthreads();
    }

    if (tid == 0) 
	atomicMinf(min, shared_min[0]);
}

__global__ void filter_activations(float *values, int num_elements, float *min, float *max, float sparsity, int *significant_unit_indices, int *num_significant_units) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int l_numfiltered_tb, g_pos;
    __shared__ float threshold;
    int l_pos = -1;

    float val = -FLT_MAX;
    if (id < num_elements) 
        val = values[id];

    if (id == 0)
	*num_significant_units = 0;

    // Init local counter and threshold
    if (threadIdx.x == 0) {
	l_numfiltered_tb = 0;
        threshold = (*max + *min) / 2.0 + (*max - *min) / M_PI * asinf(2 * sparsity - 1);
    }
    __syncthreads();

    // Evaluate threshold comparison
    if (val > threshold)
	l_pos = atomicAdd(&l_numfiltered_tb, 1);
    __syncthreads();

    // Get global index
    if(threadIdx.x == 0)
	g_pos = atomicAdd(num_significant_units, l_numfiltered_tb);
    __syncthreads();

    if (l_pos != -1)
	significant_unit_indices[g_pos + l_pos] = id;
}

__global__ void filtered_product(float *weights, float *input_values, int *input_indices, int input_nnz, float *biases, int n_outputs, int *significant_unit_indices, int *num_significant_units, float *output_values) {
    unsigned int val_index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float s_input_values[BLK_SIZE];
    __shared__ float s_input_indices[BLK_SIZE];
    
    int unit_index = val_index < *num_significant_units ? significant_unit_indices[val_index] : 0;
    int input_index = 0;
    double ans = 0;
    int input_val_index = 0;
    
    // We load the input_values into shared memory in intervals
    int num_intervals = (int)ceil((float)input_nnz/BLK_SIZE);
    for (int interval = 0; interval < num_intervals; interval++) {
	input_val_index = interval * BLK_SIZE + threadIdx.x;
	s_input_values[threadIdx.x] = input_val_index < input_nnz ? input_values[input_val_index] : 0;
        s_input_indices[threadIdx.x] = input_val_index < input_nnz ? input_indices[input_val_index] : 0;
	__syncthreads();

        for (int i = 0; i < BLK_SIZE; i++) {
            input_index = s_input_indices[i];
	    ans += weights[input_index * n_outputs + unit_index] * s_input_values[i];
        }
	__syncthreads();
    }

    if (val_index < *num_significant_units) {
        output_values[val_index] = ans + biases[unit_index];
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

