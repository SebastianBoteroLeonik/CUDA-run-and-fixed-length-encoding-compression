#include "rle.h"
#include <alloca.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define BLOCK_SIZE 1024
#define WARP_SIZE 32
#define EXEC_AND_ERR(expr)                                                     \
  cudaStatus = expr;                                                           \
  if (cudaStatus != cudaSuccess) {                                             \
    fprintf(stderr, "%s failed! At line %d, in %s\n", #expr, __LINE__,         \
            __FILE__);                                                         \
    goto Error;                                                                \
  }

__device__ int cumsum(int val) {

  int laneId = threadIdx.x % WARP_SIZE;
  unsigned int mask = 0xffffffff;
  for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
    int n = __shfl_up_sync(mask, val, offset);
    if (laneId >= offset)
      val += n;
    // printf("lane: %d, val: %d, n: %d, offset: %d\n", laneId, val, n, offset);
  }
  // printf("%d, ", val);
  return val;
}

__global__ void compression_kernel(unsigned char *data, unsigned int data_len,
                                   unsigned char jump_len,
                                   struct rle_chunk **chunks) {
  int id = threadIdx.x;
  int id_in_block = id % BLOCK_SIZE;
  int lane_id = id % WARP_SIZE;
#define BLOCK_LEN                                                              \
  ((blockIdx.x == gridDim.x - 1) * ((data_len - 1) % BLOCK_SIZE + 1) +         \
   (blockIdx.x != gridDim.x - 1) * BLOCK_SIZE)
  printf("BLOCK_LEN = %d\n", BLOCK_LEN);

  if (id >= data_len) {
    return;
  }
  char is_different;
  if (id > 0) {
    is_different = (data[id] != data[id - 1]);
  } else {
    is_different = 0;
  }
  int cs = cumsum(is_different);
#ifdef DEBUG
  printf("cumsum: %d; != : %d; thread_id = %d\n", cs, is_different, id);
#endif /* ifdef DEBUG */
  __shared__ int partial_sums[BLOCK_SIZE / WARP_SIZE];
  if (lane_id == WARP_SIZE - 1 || id == data_len - 1) {
#ifdef DEBUG
    printf("last in chunk: %d\n", id);
#endif /* ifdef DEBUG */
    partial_sums[id / WARP_SIZE] = cs;
  }
  __syncthreads();
#ifdef DEBUG
  if (id_in_block == 0) {
    printf("cumsums = [");
    for (int i = 0; i < 32 + 1; i++) {
      printf("%d, ", partial_sums[i]);
    }
    printf("]\n");
  }
#endif /* ifdef DEBUG */
  if (id < WARP_SIZE) {
    partial_sums[id] = cumsum(partial_sums[id]);
  }
  __syncthreads();
#ifdef DEBUG
  if (id_in_block == 0) {
    printf("cumsums updated = [");
    for (int i = 0; i < 32 + 1; i++) {
      printf("%d, ", partial_sums[i]);
    }
    printf("]\n");
  }
#endif /* ifdef DEBUG */
  __shared__ int cumsums[BLOCK_SIZE + 1];
  cumsums[BLOCK_SIZE] = -1;
  if (id != data_len && id >= WARP_SIZE) {
    cs += partial_sums[id / WARP_SIZE - 1];
  }
  cumsums[id] = cs;
  __syncthreads();
#ifdef DEBUG
  if (id_in_block == 0) {
    printf("all cumsums = [");
    for (int i = 0; i < BLOCK_LEN; i++) {
      printf("%d, ", cumsums[i]);
    }
    printf("]\n");
  }
#endif /* ifdef DEBUG */
  bool is_last_in_seq = (cumsums[id_in_block] != cumsums[id_in_block + 1]);
#ifdef DEBUG
  if (id_in_block == 0) {
    printf("diffs = [");
    for (int i = 0; i < BLOCK_LEN; i++) {
      printf("%d, ", cumsums[i] != cumsums[i + 1]);
    }
    printf("]\n");
  }
#endif /* ifdef DEBUG */
  struct rle_chunk *my_chunk = chunks[blockIdx.x];
  if (id_in_block == BLOCK_LEN - 1) {
    my_chunk->array_lenght = cumsums[id_in_block];
    my_chunk->lengths = (unsigned char *)alloca(my_chunk->array_lenght);
    my_chunk->values = (unsigned char *)alloca(my_chunk->array_lenght);
  }
  if (is_last_in_seq) {
  }
}

__host__ struct rle_data *compress_rle(unsigned char *data,
                                       unsigned int data_len,
                                       unsigned char jump_len) {
  unsigned int number_of_blocks =
      data_len / BLOCK_SIZE + (data_len % BLOCK_SIZE != 0);
  printf("number_of_blocks = %d\n", number_of_blocks);
  printf("data_len = %d\n", data_len);
  printf("BLOCK_SIZE = %d\n", BLOCK_SIZE);
  cudaError_t cudaStatus;
  unsigned char *dev_data_arr;
  EXEC_AND_ERR(cudaSetDevice(0))
  EXEC_AND_ERR(cudaMalloc(&dev_data_arr, data_len))
  EXEC_AND_ERR(cudaMemcpy(dev_data_arr, data, data_len, cudaMemcpyHostToDevice))
  compression_kernel<<<number_of_blocks, BLOCK_SIZE>>>(dev_data_arr, data_len,
                                                       jump_len, NULL);
  EXEC_AND_ERR(cudaGetLastError())
  EXEC_AND_ERR(cudaDeviceSynchronize())
Error:
  cudaFree(dev_data_arr);
#ifdef DEBUG
  printf("Done\n");
#endif /* ifdef DEBUG */
  return NULL;
}
