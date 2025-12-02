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
  if (global_thread_id >= data_length) {
    return;
  }
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
  bool does_overflow = (bit_index % 8 + bits_necessary > 8);
  // clear array
  fle->chunk_data[block_id][id] = 0;
  // write
  for (int i = 0; i < 8; i++) {
    __syncthreads();
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

#ifdef CPU

__host__ struct fle_data *fle_compress(unsigned char *binary_data,
                                       unsigned long data_length) {

  int chunk_len = BLOCK_SIZE;
  int number_of_chunks = CEIL_DEV(data_length, BLOCK_SIZE);
  struct fle_data *output = make_host_fle_data(number_of_chunks);
  output->total_data_length = data_length;
  output->number_of_chunks = number_of_chunks;
  for (int chunk_number = 0; chunk_number < number_of_chunks; chunk_number++) {
    if ((chunk_number + 1) * BLOCK_SIZE > data_length) {
      chunk_len = data_length % BLOCK_SIZE;
    }
    char bits_needed = 0;
    unsigned char mask = 0xff;
    for (int i = 0; i < chunk_len && bits_needed < 8; i++) {
      while ((mask << bits_needed) &
             binary_data[chunk_number * BLOCK_SIZE + i]) {
        bits_needed++;
      }
    }
    output->chunk_element_size[chunk_number] = bits_needed;
    for (int i = 0; i < chunk_len; i++) {
      unsigned char this_byte = binary_data[chunk_number * BLOCK_SIZE + i];
      this_byte <<= (8 - bits_needed);
      int bit_index = bits_needed * i;
      bool does_overflow = (bit_index % 8 + bits_needed > 8);
      // clear array
      output->chunk_data[chunk_number][i] = 0;
      // write
      output->chunk_data[chunk_number][bit_index / 8] |=
          (this_byte >> (bit_index % 8));
      if (does_overflow) {
        output->chunk_data[chunk_number][bit_index / 8 + 1] |=
            (this_byte << (8 - bit_index % 8));
      }
    }
  }
  return output;
}

#else

__host__ struct fle_data *fle_compress(unsigned char *binary_data,
                                       unsigned long data_length) {

  INITIALIZE_CUDA_PERFORMANCE_CHECK(7)

  int number_of_chunks = CEIL_DEV(data_length, BLOCK_SIZE);

  CUDA_PERFORMANCE_CHECKPOINT(before_fle_allocation)

  struct fle_data *output = make_host_fle_data(number_of_chunks);
  struct fle_data *dev_fle = make_device_fle_data(number_of_chunks);

  CUDA_PERFORMANCE_CHECKPOINT(before_binary_allocation)

  unsigned char *dev_binary_data;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_binary_data, data_length));

  CUDA_PERFORMANCE_CHECKPOINT(before_binary_memcpy)

  CUDA_ERROR_CHECK(cudaMemcpy(dev_binary_data, binary_data, data_length,
                              cudaMemcpyHostToDevice));
  printf("Copied data onto gpu\n");

  CUDA_PERFORMANCE_CHECKPOINT(before_kernel)

  fle_compress_kernel<<<number_of_chunks, BLOCK_SIZE>>>(
      dev_fle, dev_binary_data, data_length);

  CUDA_PERFORMANCE_CHECKPOINT(after_kernel)

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  printf("Compressed\n");
  CUDA_ERROR_CHECK(cudaFree(dev_binary_data));

  CUDA_PERFORMANCE_CHECKPOINT(before_fle_copy)

  copy_fle_data(dev_fle, output, DeviceToHost);
  printf("Copied data onto cpu\n");

  CUDA_PERFORMANCE_CHECKPOINT(after_fle_copy)

  CUDA_ERROR_CHECK(cudaFree(dev_fle));
  output->total_data_length = data_length;

  PRINT_AND_TERMINATE_CUDA_PERFORMANCE_CHECK()

  return output;
}
#endif /* ifdef CPU */
