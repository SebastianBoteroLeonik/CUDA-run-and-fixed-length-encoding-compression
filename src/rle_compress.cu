#include "cuda_utils.cuh"
#include "rle.h"
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
                                       size_t data_len,
                                       struct rle_chunks *rle) {
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
      (2 << (8 * sizeof(rle->repetitions[0]) - 1));
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
    rle->values[block_id * blockDim.x + id + i + offset] = values[id];
    rle->repetitions[block_id * blockDim.x + id + i + offset] =
        ((i + 1) * capacity > previous_chunk_len) *
            (previous_chunk_len % capacity) +
        ((i + 1) * capacity <= previous_chunk_len) * (capacity)-1;
  }
  __syncthreads();

  if (id == previous_number_of_chunks - 1) {
    rle->chunk_lengths[block_id] =
        previous_number_of_chunks + offset + additional_bytes_needed;
    rle->chunk_starts[block_id] = block_id * blockDim.x;
  }
}

// __global__ void choose_every_nth(unsigned char *data_in,
//                                  unsigned char *data_out, unsigned int n,
//                                  unsigned int data_len, unsigned int offset)
//                                  {
//   if (threadIdx.x * n + offset < data_len) {
//     data_out[threadIdx.x] = data_in[threadIdx.x * n + offset] & 0b11111110;
//   }
// }

__host__ struct rle_data *compress_rle(unsigned char *data, size_t data_len) {
  struct rle_data *out;
  out = (struct rle_data *)malloc(sizeof(struct rle_data));
  if (!out) {
    perror("malloc rle_data");
    return NULL;
  }
  INITIALIZE_CUDA_PERFORMANCE_CHECK(10)
  out->number_of_chunks = CEIL_DEV(data_len, BLOCK_SIZE);
  out->total_data_length = data_len;

  CUDA_ERROR_CHECK(cudaSetDevice(0));

  unsigned char *dev_data;
  CUDA_PERFORMANCE_CHECKPOINT(binary_data_alloc)
  CUDA_ERROR_CHECK(cudaMalloc(&dev_data, sizeof(*data) * data_len));
  CUDA_PERFORMANCE_CHECKPOINT(binary_data_memcpy)
  CUDA_ERROR_CHECK(cudaMemcpy(dev_data, data, sizeof(*data) * data_len,
                              cudaMemcpyHostToDevice));
  CUDA_PERFORMANCE_CHECKPOINT(rle_alloc)
  struct rle_chunks *dev_chunks =
      make_device_rle_chunks(out->number_of_chunks, BLOCK_SIZE);
  // unsigned char *dev_arena;
  // CUDA_ERROR_CHECK(
  //     cudaMalloc(&dev_arena, 2 * BLOCK_SIZE * out->number_of_chunks));
  // CUDA_ERROR_CHECK(
  //     cudaMalloc(&dev_chunks, sizeof(*dev_chunks) * out->number_of_chunks));
  // for (int i = 0; i < out->number_of_chunks; i++) {
  //   unsigned char *ptr;
  //   // CUDA_ERROR_CHECK(cudaMalloc(&ptr, BLOCK_SIZE));
  //   ptr = dev_arena + 2 * BLOCK_SIZE * i;
  //   CUDA_ERROR_CHECK(cudaMemcpy(&(dev_chunks[i].lengths), &ptr, sizeof(ptr),
  //                               cudaMemcpyHostToDevice));
  //   // CUDA_ERROR_CHECK(cudaMalloc(&ptr, BLOCK_SIZE));
  //   ptr = dev_arena + 2 * BLOCK_SIZE * i + BLOCK_SIZE;
  //   CUDA_ERROR_CHECK(cudaMemcpy(&(dev_chunks[i].values), &ptr, sizeof(ptr),
  //                               cudaMemcpyHostToDevice));
  // }
  CUDA_PERFORMANCE_CHECKPOINT(before_kernel)
  rle_compression_kernel<<<out->number_of_chunks, BLOCK_SIZE>>>(data, data_len,
                                                                dev_chunks);
  CUDA_PERFORMANCE_CHECKPOINT(after_kernel)
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaFree(dev_data));
  CUDA_PERFORMANCE_CHECKPOINT(after_kernel)
  out->chunks = make_host_rle_chunks(out->number_of_chunks, BLOCK_SIZE);

  // if (!out->chunks) {
  //   perror("malloc chunks");
  //   return NULL;
  // }
  // CUDA_ERROR_CHECK(cudaMemcpy(out->chunks, dev_chunks,
  //                             sizeof(*dev_chunks) * out->number_of_chunks,
  //                             cudaMemcpyDeviceToHost));
  // for (int i = 0; i < out->number_of_chunks; i++) {
  //   unsigned char *old = out->chunks[i].repetitions;
  //   out->chunks[i].repetitions =
  //       (unsigned char *)malloc(out->chunks[i].chunk_lengths[i]);
  //   CUDA_ERROR_CHECK(cudaMemcpy(out->chunks[i].repetitions, old,
  //        compile_commands.json out->chunks[i].chunk_lengths,
  //                               cudaMemcpyDeviceToHost));
  //   old = out->chunks[i].values;
  //   out->chunks[i].values =
  //       (unsigned char *)malloc(out->chunks[i].chunk_lengths);
  //   CUDA_ERROR_CHECK(cudaMemcpy(out->chunks[i].values, old,
  //                               out->chunks[i].chunk_lengths,
  //                               cudaMemcpyDeviceToHost));
  // }
  CUDA_PERFORMANCE_CHECKPOINT(before_rle_copy)
  copy_rle_chunks(dev_chunks, out->chunks, DeviceToHost, out->number_of_chunks,
                  out->number_of_chunks * BLOCK_SIZE);

  CUDA_PERFORMANCE_CHECKPOINT(after_rle_copy)
  CUDA_ERROR_CHECK(cudaFree(dev_chunks));
  PRINT_AND_TERMINATE_CUDA_PERFORMANCE_CHECK()
  return out;
}
