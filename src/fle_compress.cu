#include "cuda_utils.cuh"
#include "fle.h"
#include <cuda.h>
#include <device_launch_parameters.h>

__global__ void fle_compress_kernel(struct fle_data *fle,
                                    unsigned char *binary_data,
                                    unsigned long data_length) {

  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  const int id = threadIdx.x;
  const int block_id = blockIdx.x;
  unsigned char bits_necessary = 8;
  unsigned char this_byte = binary_data[global_thread_id];
  while (bits_necessary > 0 &&
         !__syncthreads_or(this_byte & (0b1 << (bits_necessary - 1)))) {
    bits_necessary--;
  }
  if (!id) {
    fle->chunk_element_size[block_id] = bits_necessary;
  }
  this_byte <<= (8 - bits_necessary);
  int bit_index = bits_necessary * id;
  bool does_overflow = ((bit_index / 8) != ((bit_index + bits_necessary) / 8));
  // clear array
  fle->chunk_data[block_id][id] = 0;
  __syncthreads();
  // write
  for (int i = 0; i < 8; i++) {
    if (id % 8 == i) {
      fle->chunk_data[block_id][bit_index / 8] |=
          (this_byte >> (bit_index % 8));
      if (does_overflow) {
        fle->chunk_data[block_id][bit_index / 8 + 1] |=
            (this_byte << (8 - bit_index % 8));
      }
    }
  }
}

__host__ struct fle_data *fle_compress(unsigned char *binary_data,
                                       unsigned long data_length) {

  int number_of_chunks = CEIL_DEV(data_length, BLOCK_SIZE);

  struct fle_data *output = make_host_fle_data(number_of_chunks);
  struct fle_data *dev_fle = make_device_fle_data(number_of_chunks);

  unsigned char *dev_binary_data;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_binary_data, data_length));
  CUDA_ERROR_CHECK(cudaMemcpy(dev_binary_data, binary_data, data_length,
                              cudaMemcpyHostToDevice));

  fle_compress_kernel<<<number_of_chunks, BLOCK_SIZE>>>(
      dev_fle, dev_binary_data, data_length);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaFree(dev_binary_data));
  copy_fle_data(dev_fle, output, DeviceToHost);
  CUDA_ERROR_CHECK(cudaFree(dev_fle));
  output->total_data_length = data_length;

  return output;
}
