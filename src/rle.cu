#include "rle.h"
// #include "rle_tests.h"
#include <cuda.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

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

/*A kernel function used for run length encoding
 * params:
 * data - the data to be encoded
 * data_len - the number of bytes in data
 * rle - output variable
 */
__global__ void rle_compression_kernel(unsigned char *data,
                                       unsigned int data_len,
                                       struct rle_chunk *rle) {
  int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  int id = threadIdx.x;
  int block_id = blockIdx.x;
  unsigned int block_length =
      blockDim.x * ((blockIdx.x + 1) * blockDim.x <= data_len) +
      (data_len % blockDim.x) * ((blockIdx.x + 1) * blockDim.x > data_len);
  // if (block_length < BLOCK_SIZE) {
  //   printf("[%d]: len: %d, dim: %d, dl mod dim: %d\neq: %d, dl: %d, grid:
  //   %d\n",
  //          block_id, block_length, blockDim.x, data_len % blockDim.x,
  //          (blockIdx.x + 1) * blockDim.x, data_len, gridDim.x);
  // }
  bool is_diff_from_prev;

  if (global_thread_id >= data_len) {
    return;
  }
  if (id) {
    is_diff_from_prev = (data[global_thread_id] != data[global_thread_id - 1]);
  } else {
    is_diff_from_prev = 0;
  }
  __shared__ int cumsums[BLOCK_SIZE + 1];
  cumsums[id] = block_cumsum(is_diff_from_prev);
  if (id == 0) {
    cumsums[block_length] = -1;
  }
  __syncthreads();
  bool is_last_in_chunk = (cumsums[id] != cumsums[id + 1]);
  bool is_first_in_chunk =
      (cumsums[id] != cumsums[(id - 1) % (block_length + 1)]);
  __shared__ int chunk_lengths[BLOCK_SIZE];
  if (is_last_in_chunk) {
    chunk_lengths[cumsums[id]] = id + 1;
  }
  __syncthreads();
  if (is_first_in_chunk) {
    chunk_lengths[cumsums[id]] -= id;
    rle[block_id].values[cumsums[id]] = data[global_thread_id];
    rle[block_id].lengths[cumsums[id]] = chunk_lengths[cumsums[id]];
  }
  if (id == block_length - 1) {
    rle[block_id].array_length = cumsums[id] + 1;
  }
}
// __global__ void compression_kernel(unsigned char *data, unsigned int
// data_len,
//                                    unsigned char jump_len,
//                                    struct rle_data *rle) {
//   int id = threadIdx.x + blockIdx.x * blockDim.x;
//   int id_in_block = id % threadIdx.x;
//   int lane_id = id % WARP_SIZE;
// #define BLOCK_LEN \
//   ((blockIdx.x == gridDim.x - 1) * ((data_len - 1) % BLOCK_SIZE + 1) + \
//    (blockIdx.x != gridDim.x - 1) * BLOCK_SIZE)
//   // printf("BLOCK_LEN = %d\n", BLOCK_LEN);
//
//   if (id >= data_len) {
//     return;
//   }
//   char is_different;
//   if (id > 0 && id < data_len) {
//     is_different = (data[id] != data[id - 1]);
//   } else {
//     is_different = 0;
//   }
//   cumsums[id] = block_cumsum(is_different);
//   __syncthreads();
// #ifdef DEBUG
//   if (id_in_block == 0) {
//     printf("all cumsums = [");
//     for (int i = 0; i < BLOCK_LEN; i++) {
//       printf("%d, ", cumsums[i]);
//     }
//     printf("]\n");
//   }
// #endif /* ifdef DEBUG */
//   bool is_last_in_seq = (cumsums[id_in_block] != cumsums[id_in_block + 1]);
//   bool is_first_in_seq =
//       (cumsums[id_in_block] != cumsums[(id_in_block - 1) % BLOCK_SIZE]);
// #ifdef DEBUG
//   if (id_in_block == 0) {
//     printf("diffs = [");
//     for (int i = 0; i < BLOCK_LEN; i++) {
//       printf("%d, ", cumsums[i] != cumsums[i + 1]);
//     }
//     printf("]\n");
//   }
// #endif /* ifdef DEBUG */
//   struct rle_chunk *my_chunk = rle->chunks + blockIdx.x;
//   if (id_in_block == BLOCK_LEN - 1) {
//     my_chunk->array_length = cumsums[id_in_block] + 1;
//     // my_chunk->lengths = (unsigned char *)malloc(my_chunk->array_length);
//     // my_chunk->values = (unsigned char *)malloc(my_chunk->array_length);
//     // printf("my_chunk->array_length = %d\n", my_chunk->array_length);
//   }
//   __shared__ int lengths[BLOCK_SIZE];
//   __syncthreads();
//   if (is_last_in_seq) {
//     my_chunk->values[cumsums[id_in_block]] = data[id];
//     // my_chunk->lengths[cumsums[id_in_block]] = id_in_block + 1;
//     lengths[cumsums[id_in_block]] = id_in_block + 1;
//     // printf("%d, %d\n", cumsums[id_in_block], data[id]);
//   }
//   __syncthreads();
//   if (is_first_in_seq) {
//     // my_chunk->lengths[cumsums[id_in_block]] -= id_in_block;
//     lengths[cumsums[id_in_block]] -= id_in_block;
//     // printf("%d, %d\n", cumsums[id_in_block], data[id]);
//   }
//
//   // if (id_in_block < my_chunk->array_length) {
//   //   short to_shift = lengths[id_in_block] > 255;
//   //   cumsum(int val)
//   // }
// #ifdef DEBUG
//   __syncthreads();
//   if (id_in_block == 0) {
//     printf("rle values = [");
//     for (int i = 0; i < my_chunk->array_length; i++) {
//       printf("%d, ", my_chunk->values[i]);
//     }
//     printf("]\n");
//   }
//   if (id_in_block == 0) {
//     printf("rle lengths = [");
//     for (int i = 0; i < my_chunk->array_length; i++) {
//       printf("%d, ", my_chunk->lengths[i]);
//     }
//     printf("]\n");
//   }
// #endif /* ifdef DEBUG */
// }

__global__ void choose_every_nth(unsigned char *data_in,
                                 unsigned char *data_out, unsigned int n,
                                 unsigned int data_len, unsigned int offset) {
  if (threadIdx.x * n + offset < data_len) {
    data_out[threadIdx.x] = data_in[threadIdx.x * n + offset] & 0b11111110;
  }
}

__host__ struct rle_data *compress_rle(unsigned char *data,
                                       unsigned int data_len) {
  unsigned int number_of_blocks =
      data_len / BLOCK_SIZE + (data_len % BLOCK_SIZE != 0);
  printf("number_of_blocks = %d\n", number_of_blocks);
  printf("data_len = %d\n", data_len);
  printf("BLOCK_SIZE = %d\n", BLOCK_SIZE);
  cudaError_t cudaStatus;
  unsigned char *dev_data_arr_all;
  unsigned char *dev_data_arr;
  struct rle_data *rle = (struct rle_data *)malloc(sizeof(rle_data));
  ;
  struct rle_chunk *chunks_produced;
  CUDA_ERROR_CHECK(cudaSetDevice(0));
  CUDA_ERROR_CHECK(cudaMalloc(&dev_data_arr_all, data_len));
  CUDA_ERROR_CHECK(
      cudaMemcpy(dev_data_arr_all, data, data_len, cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(cudaMalloc(&dev_data_arr, data_len / 3));
  int all_n_o_b = number_of_blocks;
  int all_d_l = data_len;
  number_of_blocks /= 3;
  number_of_blocks++;
  data_len /= 3;
  CUDA_ERROR_CHECK(
      cudaMalloc(&(rle->chunks), sizeof(struct rle_chunk) * number_of_blocks));

  struct rle_data *output =
      (struct rle_data *)malloc(sizeof(struct rle_data) * 3);
  for (int ofst = 0; ofst < 3; ofst++) {
    choose_every_nth<<<all_n_o_b, BLOCK_SIZE>>>(dev_data_arr_all, dev_data_arr,
                                                3, all_d_l, ofst);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    rle->number_of_chunks = number_of_blocks;
    output[ofst].number_of_chunks = number_of_blocks;

    rle_compression_kernel<<<number_of_blocks, BLOCK_SIZE>>>(
        dev_data_arr, data_len, rle->chunks);
    CUDA_ERROR_CHECK(cudaGetLastError());
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    chunks_produced = ((struct rle_chunk *)malloc(sizeof(struct rle_chunk) *
                                                  number_of_blocks));
    CUDA_ERROR_CHECK(
        cudaMemcpy(chunks_produced, rle->chunks,
                   rle->number_of_chunks * sizeof(struct rle_chunk),
                   cudaMemcpyDeviceToHost));

    // rle->chunks = chunks_produced;
#ifdef DEBUG
    printf("Done:\n");
    printf("lengths:\n");
    for (int i = 0; i < rle->number_of_chunks; i++) {
      for (int j = 0; j < chunks_produced[i].array_length; j++) {
        printf("%d ", chunks_produced[i].lengths[j]);
      }
    }
    printf("\n");
    printf("values:\n");
    for (int i = 0; i < rle->number_of_chunks; i++) {
      for (int j = 0; j < chunks_produced[i].array_length; j++) {
        printf("%d ", chunks_produced[i].values[j]);
      }
    }
#endif /* ifdef DEBUG */
    output[ofst].chunks = chunks_produced;
  }
  cudaFree(dev_data_arr_all);
  cudaFree(dev_data_arr);
  cudaFree(rle->chunks);

  unsigned int sum = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < output[i].number_of_chunks; j++) {
      sum += output[i].chunks[j].array_length;
    }
  }
  printf("sum = %dkB\n", sum / 1024);
  printf("data_len = %dkB\n", all_d_l / 1024);
  printf("rate = %f\n", (float)sum / (float)all_d_l);
  for (int i = 0; i < output[0].chunks[0].array_length; i++) {
    printf("output[0].chunks[0].lengths[%d] = %d\n ", i,
           output[0].chunks[0].lengths[i]);
    // printf("
    //        output[0]
    //            .chunks[0]
    //            .array_length = % d\n ",
    //                              output[0]
    //                                  .chunks[0]
    //                                  .array_length);
  }

  return output;
}
