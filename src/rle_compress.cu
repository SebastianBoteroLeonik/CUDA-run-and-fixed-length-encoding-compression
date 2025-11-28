#include "cuda_utils.cuh"
#include "rle.h"
#include <cuda.h>
#include <device_launch_parameters.h>
#include <stdio.h>

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
                          unsigned int *overflows, struct rle_data *rle,
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
    rle->values[global_thread_id + offset + i] = value;
    rle->repetitions[global_thread_id + offset + i] = 255;
  }
  rle->values[global_thread_id + offset + i] = value;
  rle->repetitions[global_thread_id + offset + i] = og_length % 256;
  rle->compressed_array_length = len + overflows[len - 1];
}

__host__ struct rle_data *compress_rle(unsigned char *data, size_t data_len) {
  INITIALIZE_CUDA_PERFORMANCE_CHECK(20)
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

  unsigned int last_overflow;
  CUDA_ERROR_CHECK(cudaMemcpy(&last_overflow, &(overflows[compressed_len - 1]),
                              sizeof(last_overflow), cudaMemcpyDeviceToHost));
  struct rle_data *dev_rle =
      make_device_rle_data(compressed_len + last_overflow);
  CUDA_PERFORMANCE_CHECKPOINT(write_rle)
  write_rle<<<number_of_blocks, BLOCK_SIZE>>>(values, og_lengths, overflows,
                                              dev_rle, compressed_len);
  CUDA_PERFORMANCE_CHECKPOINT(after_write_rle)
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  struct rle_data *out_rle = make_host_rle_data(compressed_len + last_overflow);

  CUDA_PERFORMANCE_CHECKPOINT(before_rle_copy)
  copy_rle_data(dev_rle, out_rle, DeviceToHost, compressed_len + last_overflow);

  CUDA_PERFORMANCE_CHECKPOINT(after_rle_copy)
  CUDA_ERROR_CHECK(cudaFree(dev_rle));
  PRINT_AND_TERMINATE_CUDA_PERFORMANCE_CHECK()
  out_rle->total_data_length = data_len;
  out_rle->compressed_array_length = compressed_len + last_overflow;
  return out_rle;
}
