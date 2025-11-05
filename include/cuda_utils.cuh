#ifndef RLE_UTILS
#define RLE_UTILS
#define BLOCK_SIZE 1024
#define WARP_SIZE 32

__device__ int warp_cumsum(int val, unsigned int mask);

__device__ int block_cumsum(int val);

#endif // !RLE_UTILS
