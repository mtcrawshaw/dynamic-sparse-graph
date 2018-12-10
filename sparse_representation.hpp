#ifndef SPARSE_REPRESENTATION_HPP
#define SPARSE_REPRESENTATION_HPP

struct SparseVector
{
    float *values{};
    int *indices{};
    int nnz{};
};

struct BSR
{
    unsigned int block_size{};
    unsigned int block_rows{};
    unsigned int block_cols{};
    unsigned int nnzb{};

    float *values{};
    unsigned int *col_ids{};
    unsigned int *row_indx{};
};

struct CSR
{
    unsigned int nrows{};
    unsigned int ncols{};
    unsigned int nnz{};

    float *values{};
    int *col_ids{};
    int *row_indx{};
};

#endif /* SPARSE_REPRESENTATION_HPP */
