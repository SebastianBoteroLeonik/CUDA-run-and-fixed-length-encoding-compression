#include "cuda_utils.cuh"
#include "fle.h"
#include <stdlib.h>

__host__ struct fle_data *make_device_fle_data(size_t number_of_chunks) {
  struct fle_data *fle;
  unsigned char *arena;
  CUDA_ERROR_CHECK(cudaMalloc(
      &arena, sizeof(*fle) +
                  sizeof(*fle->chunk_element_size) * number_of_chunks +
                  sizeof(*fle->chunk_data) * number_of_chunks));

  fle = (struct fle_data *)arena;
  arena += sizeof(*fle);
  unsigned long to_copy = number_of_chunks;

  CUDA_ERROR_CHECK(cudaMemcpy(&(fle->number_of_chunks), &to_copy,
                              sizeof(to_copy), cudaMemcpyHostToDevice));

  CUDA_ERROR_CHECK(cudaMemcpy(&(fle->chunk_element_size), &arena, sizeof(arena),
                              cudaMemcpyHostToDevice));
  arena += sizeof(*fle->chunk_element_size) * number_of_chunks;

  CUDA_ERROR_CHECK(cudaMemcpy(&(fle->chunk_data), &arena, sizeof(arena),
                              cudaMemcpyHostToDevice));

  return fle;
}

__host__ struct fle_data *make_host_fle_data(size_t number_of_chunks) {
  struct fle_data *fle;
  unsigned char *arena = (unsigned char *)malloc(
      sizeof(*fle) + sizeof(*fle->chunk_element_size) * number_of_chunks +
      sizeof(*fle->chunk_data) * number_of_chunks);

  fle = (struct fle_data *)arena;
  fle->number_of_chunks = number_of_chunks;

  arena += sizeof(*fle);

  fle->chunk_element_size = arena;
  arena += sizeof(*fle->chunk_element_size) * number_of_chunks;

  fle->chunk_data = (unsigned char (*)[1024])arena;

  return fle;
}

// __global__ void print_kernel(struct fle_data *fle) {
//   printf("number of chunks: %lu\n", fle->number_of_chunks);
//   printf("total_data_length: %lu\n", fle->total_data_length);
//   for (int i = 0; i < fle->number_of_chunks; i++) {
//     printf("chunk_element_size[%d]: %u\n", i, fle->chunk_element_size[i]);
//   }
// }
// __host__ void print_dev_fle(struct fle_data *fle) {
//   print_kernel<<<1, 1>>>(fle);
//   CUDA_ERROR_CHECK(cudaDeviceSynchronize());
// }

__host__ void copy_fle_data(struct fle_data *src, struct fle_data *dst,
                            enum cpyKind kind) {
  enum cudaMemcpyKind cudakind;
  switch (kind) {
  case DeviceToHost:
    cudakind = cudaMemcpyDeviceToHost;
    break;
  case HostToDevice:
    cudakind = cudaMemcpyHostToDevice;
    break;
  case HostToHost:
    cudakind = cudaMemcpyHostToHost;
    break;
  default:
    exit(EXIT_FAILURE);
  }
  struct fle_data *true_src, *true_dst;
  true_src = src;
  true_dst = dst;
  if (cudakind == cudaMemcpyDeviceToHost ||
      cudakind == cudaMemcpyDeviceToDevice) {
    struct fle_data dummy;
    CUDA_ERROR_CHECK(
        cudaMemcpy(&dummy, src, sizeof(dummy), cudaMemcpyDeviceToHost));
    src = &dummy;
  }
  if (cudakind == cudaMemcpyHostToDevice ||
      cudakind == cudaMemcpyDeviceToDevice) {
    struct fle_data dummy;
    CUDA_ERROR_CHECK(
        cudaMemcpy(&dummy, dst, sizeof(dummy), cudaMemcpyDeviceToHost));
    dst = &dummy;
  }

  CUDA_ERROR_CHECK(cudaMemcpy(
      dst->chunk_element_size, src->chunk_element_size,
      sizeof(*src->chunk_element_size) * src->number_of_chunks, cudakind));

  CUDA_ERROR_CHECK(cudaMemcpy(dst->chunk_data, src->chunk_data,
                              sizeof(*src->chunk_data) * src->number_of_chunks,
                              cudakind));

  CUDA_ERROR_CHECK(cudaMemcpy(&true_dst->number_of_chunks,
                              &true_src->number_of_chunks,
                              sizeof(src->number_of_chunks), cudakind));

  CUDA_ERROR_CHECK(cudaMemcpy(&true_dst->total_data_length,
                              &true_src->total_data_length,
                              sizeof(src->total_data_length), cudakind));
}
