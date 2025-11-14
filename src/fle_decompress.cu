
#include "cuda_utils.cuh"
#include "fle.h"
#include <cuda.h>
#include <device_launch_parameters.h>

__global__ void fle_decompress_kernel(struct fle_data *fle,
                                      unsigned char *binary_data) {
  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_thread_id > fle->total_data_length) {
    return;
  }
  const int id = threadIdx.x;
  const int block_id = blockIdx.x;
  if (!id) {
    printf("fle->number_of_chunks = %lu\n", fle->number_of_chunks);
  }
  // __syncthreads();
  // if (fle->number_of_chunks != 1 || fle->total_data_length != 512) {
  //   printf("fle: %p\n", fle);
  //   __syncthreads();
  //   printf("noc: %lu\n", fle->number_of_chunks);
  //   printf("tdl: %lu\n", fle->total_data_length);
  //   printf("ces: %p\n", fle->chunk_element_size);
  //   printf("cd: %p\n", fle->chunk_data);
  //   return;
  // }
  // __syncthreads();
  unsigned char bits_necessary = fle->chunk_element_size[block_id];
  int bit_index = bits_necessary * id;
  bool does_overflow = (bit_index % 8 + bits_necessary > 8);
  constexpr unsigned char full_mask = 0xff;
  unsigned char mask = full_mask;
  mask <<= 8 - bits_necessary;
  mask >>= bit_index % 8;
  unsigned char masked = mask & (fle->chunk_data[block_id][bit_index / 8]);
  masked <<= bit_index % 8;
  masked >>= 8 - bits_necessary;
  binary_data[global_thread_id] = 0;
  binary_data[global_thread_id] |= masked;
  if (does_overflow) {
    mask = full_mask;
    mask <<= 8 - bit_index % 8;
    masked = mask & (fle->chunk_data[block_id][bit_index / 8 + 1]);
    masked >>= 8 - bit_index & 8;
    binary_data[global_thread_id] |= masked;
  }
}

__host__ unsigned char *fle_decompress(struct fle_data *compressed) {
  unsigned char *dev_binary_data;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_binary_data, compressed->total_data_length));
  struct fle_data *dev_fle = make_device_fle_data(compressed->number_of_chunks);
  copy_fle_data(compressed, dev_fle, HostToDevice);
  fle_decompress_kernel<<<compressed->number_of_chunks, BLOCK_SIZE>>>(
      dev_fle, dev_binary_data);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  unsigned char *binary_data =
      (unsigned char *)malloc(compressed->total_data_length);
  CUDA_ERROR_CHECK(cudaFree(dev_fle));
  CUDA_ERROR_CHECK(cudaMemcpy(binary_data, dev_binary_data,
                              compressed->total_data_length,
                              cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaFree(dev_binary_data));
  return binary_data;
}
