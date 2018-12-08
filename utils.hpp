#ifndef UTILS_H
#define UTILS_H

void print_vector(float *vec, int len);
void print_matrix(float *mat, int n_rows, int ncols);

void load_mlp_weights(float *fc1_weights, float *fc1_biases, float *fc2_weights, float *fc2_biases, float *fc3_weights, float *fc3_biases, const char *path);
void load_mnist_data(float *mnist_data);
void load_mnist_labels(unsigned char *mnist_labels);

#endif /* !UTILS_H */
