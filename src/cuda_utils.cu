#include "cuda_utils.cuh"
// #include "rle_tests.h"
#include <cuda.h>
#include <device_launch_parameters.h>

/* A function to be run warpwise.
 * It sums all of the values val beetween the threads
 * and returns the cumulative sum of val.
 * mask is the mask passed down to __shfl_up_sync
 *
 * Example:
 * the value of val in subsequent threads: 2, 1, 5
 * after calling warp_cumsum(val) the threads return: 2, 3, 8
 */
__device__ int warp_cumsum(int val, unsigned int mask) {

  int laneId = threadIdx.x % WARP_SIZE;
  for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
    int n = __shfl_up_sync(mask, val, offset);
    if (laneId >= offset)
      val += n;
  }
  return val;
}

/* A function to be run blockwise.
 * It sums all of the values val beetween the threads
 * and returns the cumulative sum of val
 *
 * Example:
 * the value of val in subsequent threads: 2, 1, 5
 * after calling block_cumsum(val) the threads return: 2, 3, 8
 */
__device__ int block_cumsum(int val) {
  __shared__ int partial_sums[BLOCK_SIZE / WARP_SIZE];
  int mask = 0xffffffff;
  val = warp_cumsum(val, mask);
  int id = threadIdx.x;
  if (id % WARP_SIZE == WARP_SIZE - 1) {
    partial_sums[id / WARP_SIZE] = val;
  }
  __syncthreads();
  if (id < WARP_SIZE) {
    partial_sums[id] = warp_cumsum(partial_sums[id], mask);
  }
  __syncthreads();
  if (id >= WARP_SIZE) {
    val += partial_sums[id / WARP_SIZE - 1];
  }
  return val;
}
