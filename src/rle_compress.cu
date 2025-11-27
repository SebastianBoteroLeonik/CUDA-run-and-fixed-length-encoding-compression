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

__host__ struct rle_data *compress_rle2(unsigned char *data, size_t data_len) {
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
  CUDA_PERFORMANCE_CHECKPOINT(before_kernel)
  rle_compression_kernel<<<out->number_of_chunks, BLOCK_SIZE>>>(data, data_len,
                                                                dev_chunks);
  CUDA_PERFORMANCE_CHECKPOINT(after_kernel)
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaFree(dev_data));
  CUDA_PERFORMANCE_CHECKPOINT(after_kernel)
  out->chunks = make_host_rle_chunks(out->number_of_chunks, BLOCK_SIZE);

  CUDA_PERFORMANCE_CHECKPOINT(before_rle_copy)
  copy_rle_chunks(dev_chunks, out->chunks, DeviceToHost, out->number_of_chunks,
                  out->number_of_chunks * BLOCK_SIZE);

  CUDA_PERFORMANCE_CHECKPOINT(after_rle_copy)
  CUDA_ERROR_CHECK(cudaFree(dev_chunks));
  PRINT_AND_TERMINATE_CUDA_PERFORMANCE_CHECK()
  return out;
}

__global__ void
find_differing_neighbours(unsigned char *data,
                          unsigned int *diff_from_prev_indicators, size_t len) {
  const int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_thread_id >= len) {
    return;
  }
  diff_from_prev_indicators[global_thread_id] =
      global_thread_id ? (data[global_thread_id] != data[global_thread_id - 1])
                       : 0;
}

__global__ void find_segment_end(unsigned int *scan_result,
                                 unsigned *segment_ends, size_t len) {

  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_thread_id >= len) {
    return;
  }
  if (global_thread_id == len - 1) {
    segment_ends[scan_result[global_thread_id]] = global_thread_id;
  } else if (scan_result[global_thread_id] !=
             scan_result[global_thread_id + 1]) {
    segment_ends[scan_result[global_thread_id]] = global_thread_id;
  }
}

__global__ void subtract_segment_begining(unsigned int *scan_result,
                                          unsigned *segment_lengths_out,
                                          unsigned int *overflows,
                                          unsigned char *data,
                                          unsigned char *compressed_data_vals,
                                          size_t len) {

  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_thread_id >= len) {
    return;
  }
  int this_val = scan_result[global_thread_id];
  int previous_val;
  if (global_thread_id) {
    previous_val = scan_result[global_thread_id - 1];
  } else {
    previous_val = -1;
  }
  if (previous_val != this_val) {
    segment_lengths_out[this_val] -= global_thread_id;
    overflows[this_val] = segment_lengths_out[this_val] / 256;
    compressed_data_vals[this_val] = data[global_thread_id];
  }
}

__global__ void write_rle(unsigned char *values, unsigned int *og_lengths,
                          unsigned int *overflows, struct rle_chunks *chunk,
                          unsigned int len) {
  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_thread_id >= len) {
    return;
  }
  unsigned int offset;
  if (global_thread_id == 0) {
    offset = 0;
  } else {
    offset = overflows[global_thread_id - 1];
  }
  unsigned int og_length = og_lengths[global_thread_id];
  unsigned char value = values[global_thread_id];
  int i = 0;
  for (; i < og_length / 256; i++) {
    chunk->values[global_thread_id + offset + i] = value;
    chunk->repetitions[global_thread_id + offset + i] = 255;
  }
  chunk->values[global_thread_id + offset + i] = value;
  chunk->repetitions[global_thread_id + offset + i] = og_length % 256;
  if (global_thread_id == len - 1) {
    chunk->chunk_starts[0] = 0;
    chunk->chunk_lengths[0] = len + overflows[len - 1];
  }
}

__host__ struct rle_data *compress_rle(unsigned char *data, size_t data_len) {
  struct rle_data *out;
  out = (struct rle_data *)malloc(sizeof(struct rle_data));
  if (!out) {
    perror("malloc rle_data");
    return NULL;
  }
  INITIALIZE_CUDA_PERFORMANCE_CHECK(20)
  out->number_of_chunks = 1;
  out->total_data_length = data_len;
  int number_of_blocks = CEIL_DEV(data_len, BLOCK_SIZE);

  unsigned char *dev_data;
  CUDA_PERFORMANCE_CHECKPOINT(binary_data_alloc)
  CUDA_ERROR_CHECK(cudaMalloc(&dev_data, sizeof(*data) * data_len));
  CUDA_PERFORMANCE_CHECKPOINT(binary_data_memcpy)
  CUDA_ERROR_CHECK(cudaMemcpy(dev_data, data, sizeof(*data) * data_len,
                              cudaMemcpyHostToDevice));
  CUDA_PERFORMANCE_CHECKPOINT(malloc_scan_array)
  unsigned int *scan_array;
  CUDA_ERROR_CHECK(cudaMalloc(&scan_array, sizeof(*scan_array) * data_len));
  CUDA_PERFORMANCE_CHECKPOINT(diff_kernel)
  find_differing_neighbours<<<number_of_blocks, BLOCK_SIZE>>>(data, scan_array,
                                                              data_len);
  CUDA_PERFORMANCE_CHECKPOINT(recursive_cumsum)
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  recursive_cumsum(scan_array, data_len);
  unsigned int compressed_len;
  CUDA_PERFORMANCE_CHECKPOINT(memcpy_comp_len)
  CUDA_ERROR_CHECK(cudaMemcpy(&compressed_len, &(scan_array[data_len - 1]),
                              sizeof(compressed_len), cudaMemcpyDeviceToHost));
  compressed_len++;
  unsigned int *og_lengths;
  CUDA_PERFORMANCE_CHECKPOINT(og_len_malloc)
  CUDA_ERROR_CHECK(
      cudaMalloc(&og_lengths, sizeof(*og_lengths) * compressed_len));
  CUDA_PERFORMANCE_CHECKPOINT(find_end)
  find_segment_end<<<number_of_blocks, BLOCK_SIZE>>>(scan_array, og_lengths,
                                                     data_len);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  unsigned int *overflows;
  unsigned char *values;
  CUDA_PERFORMANCE_CHECKPOINT(malloc_overflows_and_vals)
  CUDA_ERROR_CHECK(cudaMalloc(&overflows, sizeof(*overflows) * compressed_len));
  CUDA_ERROR_CHECK(cudaMalloc(&values, sizeof(*values) * compressed_len));
  CUDA_PERFORMANCE_CHECKPOINT(sub_begining)
  subtract_segment_begining<<<number_of_blocks, BLOCK_SIZE>>>(
      scan_array, og_lengths, overflows, dev_data, values, data_len);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_PERFORMANCE_CHECKPOINT(recursive_cumsum_overflows)
  recursive_cumsum(overflows, compressed_len);
  CUDA_PERFORMANCE_CHECKPOINT(after_recursive_cumsum_overflows)
  CUDA_ERROR_CHECK(cudaFree(dev_data));
  CUDA_PERFORMANCE_CHECKPOINT(rle_malloc)
  struct rle_chunks *dev_chunks =
      make_device_rle_chunks(out->number_of_chunks, data_len);
  CUDA_PERFORMANCE_CHECKPOINT(write_rle)
  write_rle<<<number_of_blocks, BLOCK_SIZE>>>(values, og_lengths, overflows,
                                              dev_chunks, compressed_len);
  CUDA_PERFORMANCE_CHECKPOINT(after_write_rle)
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  out->chunks = make_host_rle_chunks(out->number_of_chunks, data_len);
  CUDA_ERROR_CHECK(cudaMemcpy(&compressed_len, &(overflows[compressed_len - 1]),
                              sizeof(compressed_len), cudaMemcpyDeviceToHost));
  compressed_len += compressed_len;

  CUDA_PERFORMANCE_CHECKPOINT(before_rle_copy)
  copy_rle_chunks(dev_chunks, out->chunks, DeviceToHost, out->number_of_chunks,
                  compressed_len);

  CUDA_PERFORMANCE_CHECKPOINT(after_rle_copy)
  CUDA_ERROR_CHECK(cudaFree(dev_chunks));
  PRINT_AND_TERMINATE_CUDA_PERFORMANCE_CHECK()
  return out;
}
