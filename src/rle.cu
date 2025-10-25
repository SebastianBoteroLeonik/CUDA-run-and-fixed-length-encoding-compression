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

#define DECLARE_IDS                                                            \
  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;    \
  const int id = threadIdx.x;                                                  \
  const int block_id = blockIdx.x;

__device__ void find_indexes_after_compression(const unsigned char *data,
                                               int new_indexes[BLOCK_SIZE],
                                               const int block_length) {
  // int id = threadIdx.x;
  // int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  // int block_id = blockIdx.x;
  DECLARE_IDS
  bool is_diff_from_prev;
  if (id) {
    is_diff_from_prev = (data[global_thread_id] != data[global_thread_id - 1]);
  } else {
    is_diff_from_prev = 0;
  }

  new_indexes[id] = block_cumsum(is_diff_from_prev);
  if (id == 0) {
    new_indexes[block_length] = -1;
  }
}

/*A kernel function used for run length encoding
 * params:
 * data - the data to be encoded
 * data_len - the number of bytes in data
 * rle - output variable
 */
__global__ void rle_compression_kernel(const unsigned char *data,
                                       unsigned int data_len,
                                       struct rle_chunk *rle) {
  // const int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  // const int id = threadIdx.x;
  // const int block_id = blockIdx.x;
  DECLARE_IDS
  const unsigned int block_length =
      blockDim.x * ((blockIdx.x + 1) * blockDim.x <= data_len) +
      (data_len % blockDim.x) * ((blockIdx.x + 1) * blockDim.x > data_len);

  if (global_thread_id >= data_len) {
    return;
  }

  __shared__ int cumsums[BLOCK_SIZE + 1];
  // bool is_diff_from_prev;
  // if (id) {
  //   is_diff_from_prev = (data[global_thread_id] != data[global_thread_id -
  //   1]);
  // } else {
  //   is_diff_from_prev = 0;
  // }
  // cumsums[id] = block_cumsum(is_diff_from_prev);

  // if (id == 0) {
  //   cumsums[block_length] = -1;
  // }
  find_indexes_after_compression(data, cumsums, block_length);

  __syncthreads();

  bool is_last_in_chunk = (cumsums[id] != cumsums[id + 1]);
  bool is_first_in_chunk;
  if (id) {
    is_first_in_chunk = (cumsums[id] != cumsums[id - 1]);
  } else {
    is_first_in_chunk = 1;
  }

  __shared__ int chunk_lengths[BLOCK_SIZE];
  __shared__ unsigned char values[BLOCK_SIZE];

  if (is_last_in_chunk) {
    chunk_lengths[cumsums[id]] = id + 1;
  }
  __syncthreads();

  if (is_first_in_chunk) {
    chunk_lengths[cumsums[id]] -= id;
    values[cumsums[id]] = data[global_thread_id];
  }
  __syncthreads();

  __shared__ int previous_number_of_chunks;
  if (id == block_length - 1) {
    previous_number_of_chunks = cumsums[id] + 1;
  }
  __syncthreads();
  if (id >= previous_number_of_chunks) {
    return;
  }
  constexpr unsigned long long capacity =
      (2 << (8 * sizeof(rle->lengths[0]) - 1));
  int previous_chunk_len = chunk_lengths[id];
  if (previous_chunk_len < 0) {
    return;
  }
  char additional_bytes_needed = ((previous_chunk_len - 1) / capacity);
  int offset = block_cumsum(additional_bytes_needed) - additional_bytes_needed;

  __syncthreads();
  for (int i = 0;
       i <= additional_bytes_needed && i + offset + id < BLOCK_SIZE &&
       id < previous_number_of_chunks;
       i++) {
    rle[block_id].values[id + i + offset] = values[id];
    rle[block_id].lengths[id + i + offset] =
        ((i + 1) * capacity > previous_chunk_len) *
            (previous_chunk_len % capacity) +
        ((i + 1) * capacity <= previous_chunk_len) * (capacity)-1;
  }
  __syncthreads();

  if (id == previous_number_of_chunks - 1) {
    rle[block_id].array_length =
        previous_number_of_chunks + offset + additional_bytes_needed;
  }
}

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
