#
# Makefile
#

CC=nvcc
NVCCFLAGS=--gpu-architecture=compute_60 --gpu-code=sm_60 -O3 -lcublas

dsg_driver: dsg_driver.o
	$(CC) $(NVCCFLAGS) $^ -o $@

%.o: %.cu
	$(CC) $(NVCCFLAGS) -c $< -o $@

# vim:ft=make
#
