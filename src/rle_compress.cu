#include "cuda_utils.cuh"
#include "rle.h"
// #include "rle_tests.h"
#include <cuda.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__device__ void find_indexes_after_compression(const unsigned char *data,
                                               int new_indexes[BLOCK_SIZE],
                                               const int block_length) {
  // int id = threadIdx.x;
  // int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  // int block_id = blockIdx.x;
  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  const int id = threadIdx.x;
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
                                       size_t data_len, struct rle_chunk *rle) {
  // const int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  // const int id = threadIdx.x;
  // const int block_id = blockIdx.x;
  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  const int id = threadIdx.x;
  const int block_id = blockIdx.x;
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

#define CEIL_DEV(num, div) (((num) / (div)) + ((num) % (div) != 0))

__host__ struct rle_data *compress_rle(unsigned char *data, size_t data_len) {
  struct rle_data *out;
  out = (struct rle_data *)malloc(sizeof(struct rle_data));
  if (!out) {
    perror("malloc rle_data");
    return NULL;
  }
  out->number_of_chunks = CEIL_DEV(data_len, BLOCK_SIZE);

  CUDA_ERROR_CHECK(cudaSetDevice(0));

  unsigned char *dev_data;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_data, sizeof(*data) * data_len));
  CUDA_ERROR_CHECK(cudaMemcpy(dev_data, data, sizeof(*data) * data_len,
                              cudaMemcpyHostToDevice));
  struct rle_chunk *dev_chunks;
  CUDA_ERROR_CHECK(
      cudaMalloc(&dev_chunks, sizeof(*dev_chunks) * out->number_of_chunks));
  rle_compression_kernel<<<out->number_of_chunks, BLOCK_SIZE>>>(data, data_len,
                                                                dev_chunks);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaFree(dev_data));
  out->chunks =
      (struct rle_chunk *)malloc(sizeof(*out->chunks) * out->number_of_chunks);
  if (!out->chunks) {
    perror("malloc chunks");
    return NULL;
  }
  CUDA_ERROR_CHECK(cudaMemcpy(out->chunks, dev_chunks,
                              sizeof(*dev_chunks) * out->number_of_chunks,
                              cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaFree(dev_chunks));
  return out;
}
