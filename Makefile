#
# Makefile
#

CC=nvcc
NVCCFLAGS=--gpu-architecture=compute_60 --gpu-code=sm_60 -O3 -lcublas
HCC=gcc

dense_mlp_driver: dense_mlp_driver.o network.o utils.o
	$(CC) $(NVCCFLAGS) $^ -o $@
sparse_mlp_driver: sparse_mlp_driver.o network.o utils.o
	$(CC) $(NVCCFLAGS) $^ -o $@

%.o: %.cu 
	$(CC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(HCC) -O3 -c $< -o $@

# vim:ft=make
#
