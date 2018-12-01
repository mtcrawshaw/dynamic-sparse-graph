#include <stdio.h>

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
