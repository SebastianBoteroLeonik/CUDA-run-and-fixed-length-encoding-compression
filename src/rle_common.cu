#include "cuda_utils.cuh"
#include "rle.h"
#include <stddef.h>

__host__ struct rle_data *make_device_rle_data(unsigned int capacity) {
  struct rle_data *data;
  unsigned char *arena;
  CUDA_ERROR_CHECK(cudaMalloc(
      &arena,
      sizeof(*data) + (sizeof(*data->repetitions + *data->values)) * capacity));

  data = (struct rle_data *)arena;
  arena = arena + sizeof(*data);
  // CUDA_ERROR_CHECK(cudaMalloc(&arena,  * sizeof()));
  CUDA_ERROR_CHECK(cudaMemcpy(&(data->repetitions), &arena, sizeof(arena),
                              cudaMemcpyHostToDevice));
  arena += sizeof(*data->repetitions) * capacity;
  CUDA_ERROR_CHECK(cudaMemcpy(&(data->values), &arena, sizeof(arena),
                              cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(cudaMemcpy(&(data->compressed_array_length), &capacity,
                              sizeof(capacity), cudaMemcpyHostToDevice));
  // arena += sizeof(data->chunk_starts[0]) * number_of_data;
  // CUDA_ERROR_CHECK(cudaMemcpy(&(data->chunk_lengths), &arena, sizeof(arena),
  //                             cudaMemcpyHostToDevice));
  // arena += sizeof(data->chunk_lengths[0]) * number_of_data;
  // CUDA_ERROR_CHECK(cudaMemcpy(&(data->repetitions), &arena, sizeof(arena),
  //                             cudaMemcpyHostToDevice));
  // arena += sizeof(data->repetitions[0]) * number_of_data * chunk_capacity;
  // CUDA_ERROR_CHECK(cudaMemcpy(&(data->values), &arena, sizeof(arena),
  //                             cudaMemcpyHostToDevice));
  return data;
}

__host__ struct rle_data *make_host_rle_data(unsigned int capacity) {
  struct rle_data *data;
  unsigned char *arena = (unsigned char *)malloc(
      sizeof(*data) +
      (sizeof(*data->repetitions) + sizeof(*data->values)) * capacity);
  data = (struct rle_data *)arena;
  arena = arena + sizeof(*data);
  data->repetitions = arena;
  arena = arena + sizeof(*data->repetitions) * capacity;
  data->values = arena;
  data->compressed_array_length = capacity;
  // data->chunk_starts = (unsigned long *)arena;
  // arena = arena + sizeof(data->chunk_starts[0]) * number_of_data;
  // data->chunk_lengths = (unsigned int *)arena;
  // arena += sizeof(data->chunk_lengths[0]) * number_of_chunks;
  // data->repetitions = arena;
  // arena += sizeof(data->repetitions[0]) * number_of_chunks * chunk_capacity;
  // data->values = arena;

  return data;
}

__host__ void copy_rle_data(struct rle_data *src, struct rle_data *dst,
                            enum cpyKind kind, unsigned capacity) {
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
  CUDA_ERROR_CHECK(cudaMemcpy(&(dst->total_data_length),
                              &(src->total_data_length),
                              sizeof(src->total_data_length), cudakind));
  CUDA_ERROR_CHECK(cudaMemcpy(&(dst->compressed_array_length),
                              &src->compressed_array_length,
                              sizeof(src->compressed_array_length), cudakind));
  if (cudakind == cudaMemcpyDeviceToHost ||
      cudakind == cudaMemcpyDeviceToDevice) {
    struct rle_data dummy;
    CUDA_ERROR_CHECK(
        cudaMemcpy(&dummy, src, sizeof(dummy), cudaMemcpyDeviceToHost));
    src = &dummy;
  }
  if (cudakind == cudaMemcpyHostToDevice ||
      cudakind == cudaMemcpyDeviceToDevice) {
    struct rle_data dummy;
    CUDA_ERROR_CHECK(
        cudaMemcpy(&dummy, dst, sizeof(dummy), cudaMemcpyDeviceToHost));
    dst = &dummy;
  }
  CUDA_ERROR_CHECK(cudaMemcpy(dst->repetitions, src->repetitions,
                              sizeof(*src->repetitions) * capacity, cudakind));
  CUDA_ERROR_CHECK(cudaMemcpy(dst->values, src->values,
                              sizeof(*src->values) * capacity, cudakind));
}
