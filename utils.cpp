#include <stdio.h>
#include <fstream>
#include <math.h>

#include "utils.hpp"
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

void load_mlp_weights(float *fc1_weights, float *fc1_biases, float *fc2_weights, float *fc2_biases, float *fc3_weights, float *fc3_biases, const char *path) {
    FILE *ptr;
    ptr = fopen("models/mlp_weights.bin", "rb");

    unsigned int fc1_rows = 256;
    unsigned int fc1_cols = 784;
    unsigned int fc2_rows = 256;
    unsigned int fc2_cols = 256;
    unsigned int fc3_rows = 10;
    unsigned int fc3_cols = 256;

    float *temp_fc1_weights = (float*) malloc(fc1_rows * fc1_cols * sizeof(float));
    float *temp_fc2_weights = (float*) malloc(fc2_rows * fc2_cols * sizeof(float));
    float *temp_fc3_weights = (float*) malloc(fc3_rows * fc3_cols * sizeof(float));

    fread(temp_fc1_weights, sizeof(float), fc1_rows * fc1_cols, ptr);
    fread(fc1_biases, sizeof(float), fc1_rows, ptr);
    fread(temp_fc2_weights, sizeof(float), fc2_rows * fc2_cols, ptr);
    fread(fc2_biases, sizeof(float), fc2_rows, ptr);
    fread(temp_fc3_weights, sizeof(float), fc3_rows * fc3_cols, ptr);
    fread(fc3_biases, sizeof(float), fc3_rows, ptr);

    for (int i = 0; i < fc1_rows; i++) {
        for (int j = 0; j < fc1_cols; j++) {
            fc1_weights[j * fc1_rows + i] = temp_fc1_weights[i * fc1_cols + j];
        }
    }
    for (int i = 0; i < fc2_rows; i++) {
        for (int j = 0; j < fc2_cols; j++) {
            fc2_weights[j * fc2_rows + i] = temp_fc2_weights[i * fc2_cols + j];
        }
    }
    for (int i = 0; i < fc3_rows; i++) {
        for (int j = 0; j < fc3_cols; j++) {
            fc3_weights[j * fc3_rows + i] = temp_fc3_weights[i * fc3_cols + j];
        }
    }

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
