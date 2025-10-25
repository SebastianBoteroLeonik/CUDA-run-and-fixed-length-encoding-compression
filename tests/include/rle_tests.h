#include "rle.h"
#include <cuda.h>

extern __device__ int warp_cumsum(int val, unsigned int mask);
extern __device__ int block_cumsum(int val);
#define BLOCK_SIZE 1024
extern __device__ void
find_indexes_after_compression(const unsigned char *data,
                               int new_indexes[BLOCK_SIZE],
                               const int block_length);
