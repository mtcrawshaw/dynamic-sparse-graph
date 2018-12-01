# dynamic-sparse-graph
This is an implementation of the paper "Dynamic Sparse Graph for Efficient Deep Learning" using CUDA. 

## Installation
1. Edit Makefile so that NVCCFLAGS contains the correct value of --gpu-architecture and --gpu-code corresponding to your GPU. A list of the correct values for each generation can be found here: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list.
2. Run 'make'.

## Usage
./dsg_driver
