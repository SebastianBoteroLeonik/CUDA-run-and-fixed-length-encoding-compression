#include "cuda_utils.cuh"
#include "rle.h"
#include <cuda.h>
#include <device_launch_parameters.h>

__global__ void decompress_rle_kernel(char *uncompressed_data) {}

__host__ char *decompress_rle(struct rle_data *compressed_data) {
  char *dev_uncompressed_data;
  CUDA_ERROR_CHECK(
      cudaMalloc(&dev_uncompressed_data, compressed_data->total_data_length));

  char *uncompressed_data = (char *)malloc(compressed_data->total_data_length);
  CUDA_ERROR_CHECK(cudaMemcpy(uncompressed_data, dev_uncompressed_data,
                              compressed_data->total_data_length,
                              cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaFree(dev_uncompressed_data));
  return uncompressed_data;
}
