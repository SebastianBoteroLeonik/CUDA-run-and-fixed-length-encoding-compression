#include "cuda_utils.cuh"
#include "rle.h"
#include <cuda.h>
#include <device_launch_parameters.h>

__global__ void uchar_array_to_ullong_array(unsigned char *chars,
                                            unsigned long long *llongs,
                                            unsigned long long array_length) {
  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_thread_id >= array_length) {
    return;
  }
  llongs[global_thread_id] = chars[global_thread_id];
}

__global__ void run_cumsum(unsigned long long *array,
                           unsigned long long *last_sums_in_chunks,
                           unsigned long long array_length) {
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

__global__ void down_propagate_cumsum(unsigned long long *array,
                                      unsigned long long *last_sums_in_chunks,
                                      unsigned long long array_length) {
  const long long global_thread_id =
      blockDim.x * (blockIdx.x + 1) + threadIdx.x;
  if (global_thread_id >= array_length) {
    return;
  }
  array[global_thread_id] += last_sums_in_chunks[blockIdx.x];
}

__host__ void recursive_cumsum(unsigned long long *array,
                               unsigned long long array_len) {
  if (array_len <= 1) {
    return;
  }
  unsigned long long next_arr_len = CEIL_DEV(array_len, BLOCK_SIZE);
  unsigned long long *next_array;
  CUDA_ERROR_CHECK(cudaMalloc(&next_array, next_arr_len * sizeof(*next_array)));
  run_cumsum<<<next_arr_len, BLOCK_SIZE>>>(array, next_array, array_len);
  recursive_cumsum(next_array, next_arr_len);
  down_propagate_cumsum<<<next_arr_len - 1, BLOCK_SIZE>>>(array, next_array,
                                                          array_len);
  CUDA_ERROR_CHECK(cudaFree(next_array));
}

__host__ void cumsum_repetitions(unsigned long long *result,
                                 struct rle_chunks *chunks,
                                 unsigned long long compressed_array_length) {
  unsigned long long *repetition_cumsums;
  CUDA_ERROR_CHECK(
      cudaMalloc(&repetition_cumsums,
                 sizeof(*repetition_cumsums) * compressed_array_length));
  unsigned long number_of_blocks =
      CEIL_DEV(compressed_array_length, BLOCK_SIZE);
  uchar_array_to_ullong_array<<<number_of_blocks, BLOCK_SIZE>>>(
      chunks->repetitions, repetition_cumsums, compressed_array_length);
  recursive_cumsum(repetition_cumsums, compressed_array_length);
}

__global__ void
decompress_rle_kernel(char *uncompressed_data, struct rle_chunks *chunks,
                      unsigned long long compressed_array_length) {

  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  const int id = threadIdx.x;
  const int block_id = blockIdx.x;
  // const unsigned int block_length =
  //     blockDim.x * ((blockIdx.x + 1) * blockDim.x <= data_len) +
  //     (data_len % blockDim.x) * ((blockIdx.x + 1) * blockDim.x > data_len);
  if (global_thread_id >= compressed_array_length) {
    return;
  }
}

__host__ char *decompress_rle(struct rle_data *compressed_data) {
  char *dev_uncompressed_data;
  CUDA_ERROR_CHECK(
      cudaMalloc(&dev_uncompressed_data, compressed_data->total_data_length));

  char *uncompressed_data = (char *)malloc(compressed_data->total_data_length);
  struct rle_chunks *dev_chunks =
      make_device_rle_chunks(1, compressed_data->compressed_array_length);
  copy_rle_chunks(compressed_data->chunks, dev_chunks, HostToDevice,
                  compressed_data->number_of_chunks,
                  compressed_data->compressed_array_length);
  decompress_rle_kernel<<<1, 1024>>>(dev_uncompressed_data, dev_chunks,
                                     compressed_data->compressed_array_length);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaFree(dev_chunks));
  CUDA_ERROR_CHECK(cudaMemcpy(uncompressed_data, dev_uncompressed_data,
                              compressed_data->total_data_length,
                              cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaFree(dev_uncompressed_data));
  return uncompressed_data;
}
