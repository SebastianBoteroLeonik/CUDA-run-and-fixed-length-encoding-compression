#include "cuda_utils.cuh"
#include "rle.h"
#include <cstddef>
#include <stddef.h>

__host__ struct rle_chunks *make_device_rle_chunks(size_t number_of_chunks,
                                                   size_t chunk_capacity) {
  struct rle_chunks *chunks;
  unsigned char *arena;
  CUDA_ERROR_CHECK(cudaMalloc(&arena, sizeof(struct rle_chunks) +
                                          (sizeof(chunks->chunk_lengths) +
                                           sizeof(chunks->chunk_starts) +
                                           2 * chunk_capacity) *
                                              number_of_chunks));

  chunks = (struct rle_chunks *)arena;
  arena = arena + sizeof(struct rle_chunks);
  // CUDA_ERROR_CHECK(cudaMalloc(&arena, 1024 * sizeof(long)));
  CUDA_ERROR_CHECK(cudaMemcpy(&(chunks->chunk_starts), &arena, sizeof(arena),
                              cudaMemcpyHostToDevice));
  arena += sizeof(chunks->chunk_starts[0]) * number_of_chunks;
  CUDA_ERROR_CHECK(cudaMemcpy(&(chunks->chunk_lengths), &arena, sizeof(arena),
                              cudaMemcpyHostToDevice));
  arena += sizeof(chunks->chunk_lengths[0]) * number_of_chunks;
  CUDA_ERROR_CHECK(cudaMemcpy(&(chunks->repetitions), &arena, sizeof(arena),
                              cudaMemcpyHostToDevice));
  arena += sizeof(chunks->repetitions[0]) * number_of_chunks * chunk_capacity;
  CUDA_ERROR_CHECK(cudaMemcpy(&(chunks->values), &arena, sizeof(arena),
                              cudaMemcpyHostToDevice));
  return chunks;
}

__host__ struct rle_chunks *make_host_rle_chunks(size_t number_of_chunks,
                                                 size_t chunk_capacity) {
  struct rle_chunks *chunks;
  unsigned char *arena = (unsigned char *)malloc(
      sizeof(struct rle_chunks) +
      (sizeof(chunks->chunk_lengths) + sizeof(chunks->chunk_starts) +
       2 * chunk_capacity) *
          number_of_chunks);
  chunks = (struct rle_chunks *)arena;
  arena = arena + sizeof(struct rle_chunks);
  chunks->chunk_starts = (unsigned long *)arena;
  arena = arena + sizeof(chunks->chunk_starts[0]) * number_of_chunks;
  chunks->chunk_lengths = (unsigned int *)arena;
  arena += sizeof(chunks->chunk_lengths[0]) * number_of_chunks;
  chunks->repetitions = arena;
  arena += sizeof(chunks->repetitions[0]) * number_of_chunks * chunk_capacity;
  chunks->values = arena;

  return chunks;
}

__host__ void copy_rle_chunks(struct rle_chunks *src, struct rle_chunks *dst,
                              enum cpyKind kind, size_t number_of_chunks,
                              ssize_t total_array_length) {
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
  if (cudakind == cudaMemcpyDeviceToHost ||
      cudakind == cudaMemcpyDeviceToDevice) {
    struct rle_chunks dummy;
    CUDA_ERROR_CHECK(
        cudaMemcpy(&dummy, src, sizeof(dummy), cudaMemcpyDeviceToHost));
    src = &dummy;
  }
  if (cudakind == cudaMemcpyHostToDevice ||
      cudakind == cudaMemcpyDeviceToDevice) {
    struct rle_chunks dummy;
    CUDA_ERROR_CHECK(
        cudaMemcpy(&dummy, dst, sizeof(dummy), cudaMemcpyDeviceToHost));
    dst = &dummy;
  }
  CUDA_ERROR_CHECK(cudaMemcpy(dst->chunk_starts, src->chunk_starts,
                              sizeof(*src->chunk_starts) * number_of_chunks,
                              cudakind));
  CUDA_ERROR_CHECK(cudaMemcpy(dst->chunk_lengths, src->chunk_lengths,
                              sizeof(*src->chunk_lengths) * number_of_chunks,
                              cudakind));
  CUDA_ERROR_CHECK(cudaMemcpy(dst->repetitions, src->repetitions,
                              total_array_length, cudakind));
  CUDA_ERROR_CHECK(
      cudaMemcpy(dst->values, src->values, total_array_length, cudakind));
}
