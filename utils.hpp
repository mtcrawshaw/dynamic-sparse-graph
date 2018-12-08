#ifndef UTILS_H
#define UTILS_H

#include "sparse_representation.hpp"

void print_vector(float *vec, int len);
void print_matrix(float *mat, int n_rows, int ncols);
void print_bsr_matrix(BSR &mat);

void load_mlp_weights(float *fc1_weights, float *fc1_biases, float *fc2_weights, float *fc2_biases, float *fc3_weights, float *fc3_biases, const char *path);
void load_mnist_data(float *mnist_data);
void load_mnist_labels(unsigned char *mnist_labels);

BSR dense_to_BSR(float *mat, unsigned int nrows, unsigned int ncols, unsigned int block_size);

#endif /* !UTILS_H */
