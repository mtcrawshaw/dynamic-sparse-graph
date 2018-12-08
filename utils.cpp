#include <stdio.h>
#include <fstream>

#include "utils.hpp"

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
