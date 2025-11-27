#include "cuda_utils.cuh"
#include <cuda.h>
#include <device_launch_parameters.h>
#include <stdio.h>

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

__global__ void run_cumsum(unsigned int *array,
                           unsigned int *last_sums_in_chunks,
                           unsigned int array_length) {
  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_thread_id >= array_length) {
    return;
  }
  const unsigned int block_length =
      blockDim.x * ((blockIdx.x + 1) * blockDim.x <= array_length) +
      (array_length % blockDim.x) *
          ((blockIdx.x + 1) * blockDim.x > array_length);
  array[global_thread_id] = block_cumsum(array[global_thread_id]);
  if (threadIdx.x == block_length - 1) {
    last_sums_in_chunks[blockIdx.x] = array[global_thread_id];
  }
}

__global__ void down_propagate_cumsum(unsigned int *array,
                                      unsigned int *last_sums_in_chunks,
                                      unsigned int array_length) {
  const long long global_thread_id =
      blockDim.x * (blockIdx.x + 1) + threadIdx.x;
  if (global_thread_id >= array_length) {
    return;
  }
  array[global_thread_id] += last_sums_in_chunks[blockIdx.x];
}

__host__ void recursive_cumsum(unsigned int *array, unsigned int array_len) {
  if (array_len <= 1) {
    return;
  }
  unsigned int next_arr_len = CEIL_DEV(array_len, BLOCK_SIZE);
  unsigned int *next_array;
  CUDA_ERROR_CHECK(cudaMalloc(&next_array, next_arr_len * sizeof(*next_array)));
  run_cumsum<<<next_arr_len, BLOCK_SIZE>>>(array, next_array, array_len);
  recursive_cumsum(next_array, next_arr_len);
  down_propagate_cumsum<<<next_arr_len - 1, BLOCK_SIZE>>>(array, next_array,
                                                          array_len);
  CUDA_ERROR_CHECK(cudaFree(next_array));
}
