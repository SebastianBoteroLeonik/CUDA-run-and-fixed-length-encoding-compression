#include "cuda_utils.cuh"
#include "fle.h"

__global__ void fle_decompress_kernel(struct fle_data *fle,
                                      unsigned char *binary_data) {
  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_thread_id >= fle->total_data_length) {
    return;
  }
  const int id = threadIdx.x;
  const int block_id = blockIdx.x;

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
  char decoded_byte = 0;
  decoded_byte |= masked;
  if (does_overflow) {
    mask = full_mask;
    mask <<= 8 - bits_necessary;
    mask <<= 8 - bit_index % 8;
    masked = mask & (fle->chunk_data[block_id][bit_index / 8 + 1]);
    masked >>= 8 - bit_index % 8;
    masked >>= 8 - bits_necessary;
    decoded_byte |= masked;
  }
  binary_data[global_thread_id] = decoded_byte;
}

#ifdef CPU

__host__ unsigned char *fle_decompress(struct fle_data *compressed) {
  unsigned char *binary_data =
      (unsigned char *)malloc(compressed->total_data_length);
  int chunk_len = BLOCK_SIZE;
  for (int chunk_number = 0; chunk_number < compressed->number_of_chunks;
       chunk_number++) {
    if ((chunk_number + 1) * BLOCK_SIZE > compressed->total_data_length) {
      chunk_len = compressed->total_data_length % BLOCK_SIZE;
    }
    for (int id = 0; id < chunk_len; id++) {
      unsigned char bits_necessary =
          compressed->chunk_element_size[chunk_number];
      int bit_index = bits_necessary * id;
      bool does_overflow = (bit_index % 8 + bits_necessary > 8);
      constexpr unsigned char full_mask = 0xff;
      unsigned char mask = full_mask;
      mask <<= 8 - bits_necessary;
      mask >>= bit_index % 8;
      unsigned char masked =
          mask & (compressed->chunk_data[chunk_number][bit_index / 8]);
      masked <<= bit_index % 8;
      masked >>= 8 - bits_necessary;
      char decoded_byte = 0;
      decoded_byte |= masked;
      if (does_overflow) {
        mask = full_mask;
        mask <<= 8 - bits_necessary;
        mask <<= 8 - bit_index % 8;
        masked =
            mask & (compressed->chunk_data[chunk_number][bit_index / 8 + 1]);
        masked >>= 8 - bit_index % 8;
        masked >>= 8 - bits_necessary;
        decoded_byte |= masked;
      }
      binary_data[chunk_number * BLOCK_SIZE + id] = decoded_byte;
    }
  }
  return binary_data;
}

#else

__host__ unsigned char *fle_decompress(struct fle_data *compressed) {
  INITIALIZE_CUDA_PERFORMANCE_CHECK(10)
  unsigned char *dev_binary_data;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_binary_data, compressed->total_data_length));
  CUDA_PERFORMANCE_CHECKPOINT(before_fle_alloc)
  struct fle_data *dev_fle = make_device_fle_data(compressed->number_of_chunks);
  CUDA_PERFORMANCE_CHECKPOINT(before_fle_memcopy)
  copy_fle_data(compressed, dev_fle, HostToDevice);
  printf("Copied data onto gpu\n");
  CUDA_PERFORMANCE_CHECKPOINT(before_kernel)
  fle_decompress_kernel<<<compressed->number_of_chunks, BLOCK_SIZE>>>(
      dev_fle, dev_binary_data);
  CUDA_PERFORMANCE_CHECKPOINT(after_kernel)
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  printf("Decompressed\n");
  unsigned char *binary_data =
      (unsigned char *)malloc(compressed->total_data_length);
  CUDA_ERROR_CHECK(cudaFree(dev_fle));
  CUDA_PERFORMANCE_CHECKPOINT(before_binary_memcpy)
  CUDA_ERROR_CHECK(cudaMemcpy(binary_data, dev_binary_data,
                              compressed->total_data_length,
                              cudaMemcpyDeviceToHost));
  printf("Copied data onto cpu\n");
  CUDA_PERFORMANCE_CHECKPOINT(after_binary_memcpy)
  CUDA_ERROR_CHECK(cudaFree(dev_binary_data));
  PRINT_AND_TERMINATE_CUDA_PERFORMANCE_CHECK()
  return binary_data;
#define CPU
}
#endif /* ifdef CPU */
