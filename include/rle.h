#ifndef RLE_H
#define RLE_H
#include <stdio.h>

struct rle_chunk {
  unsigned int array_length;
  unsigned char lengths[1024];
  unsigned char values[1024];
};

struct rle_data {
  unsigned long long total_data_length;
  unsigned int number_of_chunks;
  struct rle_chunk *chunks;
};

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
struct rle_data *compress_rle(unsigned char *data, size_t data_len);

#ifdef __cplusplus
}
#endif // __cplusplus

#define CUDA_ERROR_CHECK(expr)                                                 \
  do {                                                                         \
    cudaError_t cudaStatus = expr;                                             \
    if (cudaStatus != cudaSuccess) {                                           \
      fprintf(stderr, "%s failed! At line %d, in %s\nError: %s\n\t %s\n",      \
              #expr, __LINE__, __FILE__, cudaGetErrorName(cudaStatus),         \
              cudaGetErrorString(cudaStatus));                                 \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#endif // !RLE_H
