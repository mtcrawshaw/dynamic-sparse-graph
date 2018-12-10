#include <stdio.h>
#include <fstream>
#include <math.h>
#include <time.h>

#include "utils.h"
#include "sparse_representation.hpp"

void print_vector(float *vec, int len) {
    for (int i = 0; i < len; i++) {
        printf("%f ", vec[i]);
    }
    printf("\n");
}

// Assumes matrix is in column-major order
void print_matrix(float *mat, int n_rows, int n_cols) {
    for (int i = 0; i < n_rows; i++){
        for (int j = 0; j < n_cols; j++) {
            printf("%f ", mat[j * n_rows + i]);
        }
        printf("\n");
    }
}

void print_bsr_matrix(BSR &mat) {
    printf("Block size: %d\n", mat.block_size);
    printf("Block rows: %d\n", mat.block_rows);
    printf("Block cols: %d\n", mat.block_cols);
    printf("Nnzb: %d\n", mat.nnzb);
    printf("Values:\n");
    for (int i = 0; i < mat.nnzb * mat.block_size * mat.block_size; i++) {
	printf("%f ", mat.values[i]);
    }
    printf("\nColumn IDs:\n");
    for (int i = 0; i < mat.nnzb; i++) {
	printf("%d ", mat.col_ids[i]);
    }
    printf("\nRow indexes:\n");
    for (int i = 0; i < mat.block_rows + 1; i++) {
	printf("%d ", mat.row_indx[i]);
    }
    printf("\n");
}

void print_csr_matrix(CSR &mat) {
    printf("Nrows: %d\n", mat.nrows);
    printf("Ncols: %d\n", mat.ncols);
    printf("Nnz: %d\n", mat.nnz);
    printf("Values:\n");
    for (int i = 0; i < mat.nnz; i++)
        printf("%f ", mat.values[i]);
    printf("\nCol IDs:\n");
    for (int i = 0; i < mat.nnz; i++)
        printf("%d ", mat.col_ids[i]);
    printf("\nRow indexes:\n");
    for (int i = 0; i < mat.nrows + 1; i++)
        printf("%d ", mat.row_indx[i]);
    printf("\n");
}

void print_sparse_vector(SparseVector vec) {
    printf("Values:\n");
    for (int i = 0; i < vec.nnz; i++)
	printf("%f ", vec.values[i]);
    printf("\nIndices\n");
    for (int i = 0; i < vec.nnz; i++)
	printf("%d ", vec.indices[i]);
    printf("\n");
}

MLP load_mlp(const char *path, int n_inputs, int n_hidden1, int n_hidden2, int n_outputs) {
    float *fc1_weights = (float*) malloc(n_inputs * n_hidden1 * sizeof(float));
    float *fc1_biases = (float*) malloc(n_hidden1 * sizeof(float));
    float *fc2_weights = (float*) malloc(n_hidden1 * n_hidden2 * sizeof(float));
    float *fc2_biases = (float*) malloc(n_hidden2 * sizeof(float));
    float *fc3_weights = (float*) malloc(n_hidden2 * n_outputs * sizeof(float));
    float *fc3_biases = (float*) malloc(n_outputs * sizeof(float));

    FILE *ptr;
    ptr = fopen(path, "rb");

    // These temp matrices will hold the weights in column major order, as they are stored
    float *temp_fc1_weights = (float*) malloc(n_inputs * n_hidden1 * sizeof(float));
    float *temp_fc2_weights = (float*) malloc(n_hidden1 * n_hidden2 * sizeof(float));
    float *temp_fc3_weights = (float*) malloc(n_hidden2 * n_outputs * sizeof(float));

    fread(temp_fc1_weights, sizeof(float), n_inputs * n_hidden1, ptr);
    fread(fc1_biases, sizeof(float), n_hidden1, ptr);
    fread(temp_fc2_weights, sizeof(float), n_hidden1 * n_hidden2, ptr);
    fread(fc2_biases, sizeof(float), n_hidden2, ptr);
    fread(temp_fc3_weights, sizeof(float), n_hidden2 * n_outputs, ptr);
    fread(fc3_biases, sizeof(float), n_outputs, ptr);

    fclose(ptr);

    // Converting from column major order to row major order
    for (int i = 0; i < n_hidden1; i++) {
        for (int j = 0; j < n_inputs; j++) {
            fc1_weights[j * n_hidden1 + i] = temp_fc1_weights[i * n_inputs + j];
        }
    }
    for (int i = 0; i < n_hidden2; i++) {
        for (int j = 0; j < n_hidden1; j++) {
            fc2_weights[j * n_hidden2 + i] = temp_fc2_weights[i * n_hidden1 + j];
        }
    }
    for (int i = 0; i < n_outputs; i++) {
        for (int j = 0; j < n_hidden2; j++) {
            fc3_weights[j * n_outputs + i] = temp_fc3_weights[i * n_hidden2 + j];
        }
    }

    // Loading weights into MLP struct on device
    MLP mlp;
    mlp.n_inputs = n_inputs;
    mlp.n_hidden1 = n_hidden1;
    mlp.n_hidden2 = n_hidden2;
    mlp.n_outputs = n_outputs;
    cudaMalloc(&mlp.fc1_weights, n_inputs * n_hidden1 * sizeof(float));
    cudaMalloc(&mlp.fc1_biases, n_hidden1 * sizeof(float));
    cudaMalloc(&mlp.fc2_weights, n_hidden1 * n_hidden2 * sizeof(float));
    cudaMalloc(&mlp.fc2_biases, n_hidden2 * sizeof(float));
    cudaMalloc(&mlp.fc3_weights, n_hidden2 * n_outputs * sizeof(float));
    cudaMalloc(&mlp.fc3_biases, n_outputs * sizeof(float));

    cudaMemcpy(mlp.fc1_weights, fc1_weights, n_inputs * n_hidden1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mlp.fc1_biases, fc1_biases, n_hidden1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mlp.fc2_weights, fc2_weights, n_hidden1 * n_hidden2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mlp.fc2_biases, fc2_biases, n_hidden2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mlp.fc3_weights, fc3_weights, n_hidden2 * n_outputs * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mlp.fc3_biases, fc3_biases, n_outputs * sizeof(float), cudaMemcpyHostToDevice); 

    // Clean up
    free(fc1_weights);
    free(temp_fc1_weights);
    free(fc1_biases);
    free(fc2_weights);
    free(temp_fc2_weights);
    free(fc2_biases);
    free(fc3_weights);
    free(temp_fc3_weights);
    free(fc3_biases);

    return mlp;
}

void load_mnist_data(float *mnist_data) {
    int num_images = 60000;
    int image_len = 28 * 28;
    const char *path = "data/mnist/train-images-idx3-ubyte";

    FILE *ptr;
    ptr = fopen(path, "rb");

    unsigned char* bytes = (unsigned char*) malloc(num_images * image_len * sizeof(unsigned char));
    fseek(ptr, sizeof(int) * 4, SEEK_SET);
    fread(bytes, sizeof(unsigned char), num_images * image_len, ptr);
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

// copy_type = 0 means vec is on device and return value will be on device
// copy_type = 1 means vec is on device and return value will be on host
/*__global__ void device_dense_to_SparseVector(float *vec, unsigned int len, float *values, unsigned int *indices, int *nnz, unsigned int copy_type) {
    nnz = 0;

    for (int i = 0; i < len; i++) {
	if (vec[i] != 0) {
	    values[nnz] = vec[i];
	    indices[nnz] = i;
	    nnz += 1;
	}
    }

    float *s_vec.values;
    unsigned int *s_vec.indices;
    cudaMalloc(&s_vec.values, nnz * sizeof(float));
    cudaMalloc(&s_vec.indices, nnz * sizeof(unsigned int));

    cudaMemcpyKind copy_kind;
    if (copy_type == 0)
	copy_kind = cudaMemcpyDeviceToDevice;
    else if (copy_type == 1)
	copy_kind = cudaMemcpyDeviceToHost;

    cudaMemcpy(s_vec.values, temp_values, nnz * sizeof(float), copy_kind);
    cudaMemcpy(s_vec.indices, temp_indices, nnz * sizeof(unsigned int), copy_kind);

    free(temp_values);
    free(temp_indices);

    return s_vec;
}*/

// copy_type = 0 means vec is on host and return value will be on host
// copy_type = 1 means vec is on host and return value will be on device
SparseVector dense_to_SparseVector(float *vec, unsigned int len, unsigned int copy_type) {
    float *temp_values = (float*) malloc(len * sizeof(float));
    unsigned int *temp_indices = (unsigned int*) malloc(len * sizeof(unsigned int));
    unsigned int nnz = 0;

    for (int i = 0; i < len; i++) {
	if (vec[i] != 0) {
	    temp_values[nnz] = vec[i];
	    temp_indices[nnz] = i;
	    nnz += 1;
	}
    }

    SparseVector s_vec;

    s_vec.nnz = nnz;
    if (copy_type == 0) {
        s_vec.values = (float*) malloc(nnz * sizeof(float));
        s_vec.indices = (int*) malloc(nnz * sizeof(int));
        std::copy(temp_values, temp_values + nnz, s_vec.values);
        std::copy(temp_indices, temp_indices + nnz, s_vec.indices);
    } else {
	cudaMalloc(&s_vec.values, nnz * sizeof(float));
        cudaMalloc(&s_vec.indices, nnz * sizeof(unsigned int));
        cudaMemcpy(s_vec.values, temp_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(s_vec.indices, temp_indices, nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }

    free(temp_values);
    free(temp_indices);

    return s_vec;
}

CSR get_random_projection(unsigned int nrows, unsigned int ncols, float s) {
    CSR projection;
    projection.nrows = nrows;
    projection.ncols = ncols;

    float* temp_values = (float*) malloc(nrows * ncols * sizeof(float));
    int* temp_col_ids = (int*) malloc(nrows * ncols * sizeof(int));
    projection.row_indx = (int*) malloc((nrows + 1) * sizeof(int));
    unsigned int nnz = 0;
    
    float random = 0;
    float base = 2 * s;
    srand(time(0));

    for (int i = 0; i < nrows; i++) {
	projection.row_indx[i] = nnz;

	for (int j = 0; j < ncols; j++) {
	    random = (float)rand() / (float)(RAND_MAX);
	    if (random <= 1.0 / s)  {
		temp_col_ids[nnz] = j;
		if (random <= 1.0 / base)
		    temp_values[nnz] = sqrt(s);
		else
		    temp_values[nnz] = -1 * sqrt(s);
		nnz += 1;
            }
        }
    }
    projection.row_indx[nrows] = nnz;

    projection.nnz = nnz;
    projection.values = (float*) malloc(nnz * sizeof(float));
    projection.col_ids = (int*) malloc(nnz * sizeof(int));
    std::copy(temp_values, temp_values + nnz, projection.values);
    std::copy(temp_col_ids, temp_col_ids + nnz, projection.col_ids);

    return projection;
}

BSR dense_to_BSR(float *mat, unsigned int nrows, unsigned int ncols, unsigned int block_size) {
    int nnzb = 0;
    int block_rows = (int)ceil((float)nrows / block_size);
    int block_cols = (int)ceil((float)ncols / block_size);

    BSR bsr_mat;
    float *temp_values = (float*) malloc(block_rows * block_size * block_cols * block_size * sizeof(float));
    unsigned int *temp_col_ids = (unsigned int*) malloc(block_rows * block_cols * sizeof(unsigned int));
    bsr_mat.row_indx = (unsigned int*) malloc((block_rows + 1) * sizeof(unsigned int));

    float *block = (float*) malloc(block_size * block_size * sizeof(float));
    int found_nz = 0;
    float element = 0;
    int row = 0;
    int col = 0;

    for (int b_row = 0; b_row < block_rows; b_row++) {
        bsr_mat.row_indx[b_row] = nnzb;

	for (int b_col = 0; b_col < block_cols; b_col++) {
	    found_nz = 0;

            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < block_size; j++) {
                    row = b_row * block_size + i;
                    col = b_col * block_size + j;
		    if (row < nrows && col < ncols)
		    	element = mat[col * nrows + row];
		    else
			element = 0;

		    block[i * block_size + j] = element;
		    if (element != 0) {
			found_nz = 1;
		    }
		}
            }

            if (found_nz == 1) {
		std::copy(block, block + block_size * block_size, temp_values + nnzb * block_size * block_size);
		temp_col_ids[nnzb] = b_col;
                nnzb += 1;
            }
	}
    }

    bsr_mat.row_indx[block_rows] = nnzb;

    bsr_mat.block_size = block_size;
    bsr_mat.block_rows = block_rows;
    bsr_mat.block_cols = block_cols;
    bsr_mat.nnzb = nnzb;
    
    bsr_mat.values = (float*) malloc(nnzb * block_size * block_size * sizeof(float));
    bsr_mat.col_ids = (unsigned int*) malloc(nnzb * sizeof(unsigned int));
    std::copy(temp_values, temp_values + nnzb * block_size * block_size, bsr_mat.values);
    std::copy(temp_col_ids, temp_col_ids + nnzb, bsr_mat.col_ids);

    return bsr_mat;
}
