#include <cuda.h>

extern __device__ int warp_cumsum(int val, unsigned int mask);
extern __device__ int block_cumsum(int val);
