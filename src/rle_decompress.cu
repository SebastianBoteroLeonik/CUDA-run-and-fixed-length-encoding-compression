#include "cuda_utils.cuh"
#include "rle.h"
#include <cuda.h>
#include <device_launch_parameters.h>

__global__ void uchar_array_to_uint_array(unsigned char *chars,
                                          unsigned int *llongs,
                                          unsigned int array_length) {
  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_thread_id >= array_length) {
    return;
  }
  llongs[global_thread_id] = chars[global_thread_id] + 1;
}

__host__ void cumsum_repetitions(unsigned int **result, struct rle_data *rle,
                                 unsigned int compressed_array_length) {
  unsigned int *repetition_cumsums;
  CUDA_ERROR_CHECK(
      cudaMalloc(&repetition_cumsums,
                 sizeof(*repetition_cumsums) * compressed_array_length));
  unsigned long number_of_blocks =
      CEIL_DEV(compressed_array_length, BLOCK_SIZE);
  struct rle_data dummy;
  CUDA_ERROR_CHECK(
      cudaMemcpy(&dummy, rle, sizeof(dummy), cudaMemcpyDeviceToHost));
  uchar_array_to_uint_array<<<number_of_blocks, BLOCK_SIZE>>>(
      dummy.repetitions, repetition_cumsums, compressed_array_length);
  recursive_cumsum(repetition_cumsums, compressed_array_length);
  *result = repetition_cumsums;
}

__global__ void decompress_rle_kernel(char *uncompressed_data,
                                      struct rle_data *chunks,
                                      unsigned int *repetitions_cumsums,
                                      unsigned int compressed_array_length) {

  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  const int id = threadIdx.x;
  const int block_id = blockIdx.x;
  // const unsigned int block_length =
  //     blockDim.x * ((blockIdx.x + 1) * blockDim.x <= data_len) +
  //     (data_len % blockDim.x) * ((blockIdx.x + 1) * blockDim.x > data_len);
  if (global_thread_id >= compressed_array_length) {
    return;
  }
  unsigned int offset;
  if (global_thread_id == 0) {
    offset = 0;
  } else {
    offset = repetitions_cumsums[global_thread_id - 1];
  }
  for (int i = 0; i <= chunks->repetitions[global_thread_id]; i++) {
    uncompressed_data[offset + i] = chunks->values[global_thread_id];
  }
}

__host__ unsigned char *decompress_rle(struct rle_data *compressed_data) {
  INITIALIZE_CUDA_PERFORMANCE_CHECK(10)
  char *dev_uncompressed_data;
  CUDA_PERFORMANCE_CHECKPOINT(before_binary_alloc)
  CUDA_ERROR_CHECK(
      cudaMalloc(&dev_uncompressed_data, compressed_data->total_data_length));

  unsigned char *uncompressed_data =
      (unsigned char *)malloc(compressed_data->total_data_length);
  CUDA_PERFORMANCE_CHECKPOINT(before_rle_alloc)
  struct rle_data *dev_rle =
      make_device_rle_data(compressed_data->compressed_array_length);
  CUDA_PERFORMANCE_CHECKPOINT(before_rle_copy)
  copy_rle_data(compressed_data, dev_rle, HostToDevice,
                compressed_data->compressed_array_length);
  unsigned int *repetitions_cumsums;
  CUDA_PERFORMANCE_CHECKPOINT(cumsum_repetitions)
  cumsum_repetitions(&repetitions_cumsums, dev_rle,
                     compressed_data->compressed_array_length);
  CUDA_PERFORMANCE_CHECKPOINT(decompression_kernel)
  decompress_rle_kernel<<<
      CEIL_DEV(compressed_data->compressed_array_length, BLOCK_SIZE), 1024>>>(
      dev_uncompressed_data, dev_rle, repetitions_cumsums,
      compressed_data->compressed_array_length);
  CUDA_PERFORMANCE_CHECKPOINT(after_kernel)
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaFree(dev_rle));
  CUDA_PERFORMANCE_CHECKPOINT(before_binary_memcpy)
  CUDA_ERROR_CHECK(cudaMemcpy(uncompressed_data, dev_uncompressed_data,
                              compressed_data->total_data_length,
                              cudaMemcpyDeviceToHost));
  CUDA_PERFORMANCE_CHECKPOINT(after_binary_memcpy)
  CUDA_ERROR_CHECK(cudaFree(dev_uncompressed_data));
  PRINT_AND_TERMINATE_CUDA_PERFORMANCE_CHECK()
  return uncompressed_data;
}
