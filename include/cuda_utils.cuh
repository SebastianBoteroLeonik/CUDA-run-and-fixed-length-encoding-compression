#ifndef RLE_UTILS
#define RLE_UTILS
#include <stdio.h>
#define BLOCK_SIZE 1024
#define WARP_SIZE 32

__device__ int warp_cumsum(int val, unsigned int mask);

__device__ int block_cumsum(int val);

#define CUDA_ERROR_CHECK(expr)                                                 \
  do {                                                                         \
    cudaError_t cudaStatus = expr;                                             \
    if (cudaStatus != cudaSuccess) {                                           \
      fprintf(stderr, "%s failed! At line %d, in %s\nError: %s\n\t %s\n",      \
              #expr, __LINE__, __FILE__, cudaGetErrorName(cudaStatus),         \
              cudaGetErrorString(cudaStatus));                                 \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#endif // !RLE_UTILS
