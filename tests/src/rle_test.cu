#include "cuda_utils.cuh"
#include "rle.h"
#include <gtest/gtest.h>
#include <rle_tests.h>

__global__ void run_warp_cumsum(int *vals) {
  vals[threadIdx.x] = warp_cumsum(vals[threadIdx.x], 0xffffffff);
}

__global__ void run_block_cumsum(int *vals) {
  vals[threadIdx.x] = block_cumsum(vals[threadIdx.x]);
}

__global__ void run_index_finder(const unsigned char *data,
                                 unsigned int data_len, int *new_indexes) {
  const unsigned int block_length =
      blockDim.x * ((blockIdx.x + 1) * blockDim.x <= data_len) +
      (data_len % blockDim.x) * ((blockIdx.x + 1) * blockDim.x > data_len);
  find_indexes_after_compression(data, new_indexes, block_length);
}

__global__ void rle_compression_kernel(const unsigned char *data,
                                       size_t data_len, struct rle_chunks *rle);

TEST(run_length_encoding, warp_sum) {
  int vals[32];
  for (int i = 0; i < 32; i++) {
    vals[i] = i;
  }
  int *dev_vals;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_vals, 32 * sizeof(int)));
  CUDA_ERROR_CHECK(
      cudaMemcpy(dev_vals, vals, 32 * sizeof(int), cudaMemcpyHostToDevice));
  run_warp_cumsum<<<1, 32>>>(dev_vals);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(
      cudaMemcpy(vals, dev_vals, 32 * sizeof(int), cudaMemcpyDeviceToHost));
  int counter = 0;
  for (int i = 0; i < 32; i++) {
    counter += i;
    EXPECT_EQ(counter, vals[i]);
  }
  CUDA_ERROR_CHECK(cudaFree(dev_vals));
}

#define BLOCK_SIZE 1024
TEST(run_length_encoding, block_sum) {
  int vals[BLOCK_SIZE];
  for (int i = 0; i < BLOCK_SIZE; i++) {
    vals[i] = i;
  }
  int *dev_vals;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_vals, BLOCK_SIZE * sizeof(int)));
  CUDA_ERROR_CHECK(cudaMemcpy(dev_vals, vals, BLOCK_SIZE * sizeof(int),
                              cudaMemcpyHostToDevice));
  run_block_cumsum<<<1, BLOCK_SIZE>>>(dev_vals);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaMemcpy(vals, dev_vals, BLOCK_SIZE * sizeof(int),
                              cudaMemcpyDeviceToHost));
  int counter = 0;
  for (int i = 0; i < BLOCK_SIZE; i++) {
    counter += i;
    EXPECT_EQ(counter, vals[i]);
  }
  CUDA_ERROR_CHECK(cudaFree(dev_vals));
}

TEST(run_length_encoding, partial_block_sum) {
  int vals[BLOCK_SIZE];
  int *dev_vals;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_vals, BLOCK_SIZE * sizeof(int)));
  for (int curr_block_size = 1; curr_block_size <= BLOCK_SIZE;
       curr_block_size++) {
    for (int i = 0; i < curr_block_size; i++) {
      vals[i] = i;
    }
    CUDA_ERROR_CHECK(cudaMemcpy(dev_vals, vals, curr_block_size * sizeof(int),
                                cudaMemcpyHostToDevice));
    run_block_cumsum<<<1, curr_block_size>>>(dev_vals);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    CUDA_ERROR_CHECK(cudaMemcpy(vals, dev_vals, curr_block_size * sizeof(int),
                                cudaMemcpyDeviceToHost));
    int counter = 0;
    for (int i = 0; i < curr_block_size; i++) {
      counter += i;
      EXPECT_EQ(counter, vals[i]);
    }
  }
  CUDA_ERROR_CHECK(cudaFree(dev_vals));
}

TEST(run_length_encoding, find_index_after_compression) {
  int LEN = 1020;
  unsigned char data[LEN];
  for (int i = 0; i < LEN; i++) {
    data[i] = i / 16 + 200;
  }
  int *new_indexes;
  CUDA_ERROR_CHECK(cudaMalloc(&new_indexes, (LEN + 1) * sizeof(int)));
  unsigned char *dev_data;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_data, LEN));
  CUDA_ERROR_CHECK(cudaMemcpy(dev_data, data, LEN, cudaMemcpyHostToDevice));
  run_index_finder<<<1, LEN>>>(dev_data, LEN, new_indexes);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  int new_indexes_host[LEN + 1];
  CUDA_ERROR_CHECK(cudaMemcpy(new_indexes_host, new_indexes,
                              (LEN + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < LEN; i++) {
    EXPECT_EQ(new_indexes_host[i], i / 16);
  }
}

#define CEIL_DEV(num, div) ((num / div) + (num % div != 0))
#define TEST_COMPRESSION_KERNEL_TEMPLATE(TEST_DATA_LEN, PERIOD)                \
  unsigned char *data = (unsigned char *)malloc(TEST_DATA_LEN);                \
  EXPECT_NE(data, nullptr);                                                    \
  unsigned char *dev_data;                                                     \
  for (int i = 0; i < TEST_DATA_LEN; i++) {                                    \
    data[i] = i / (PERIOD) + 2 + (i % 2909 == 0);                              \
  }                                                                            \
  CUDA_ERROR_CHECK(cudaMalloc(&dev_data, TEST_DATA_LEN));                      \
  CUDA_ERROR_CHECK(cudaMemcpy(dev_data, data,                                  \
                              sizeof(unsigned char) * TEST_DATA_LEN,           \
                              cudaMemcpyHostToDevice));                        \
  struct rle_data out;                                                         \
  out.number_of_chunks = CEIL_DEV(TEST_DATA_LEN, 1024);                        \
  out.chunks = make_host_rle_chunks(out.number_of_chunks, BLOCK_SIZE);         \
  EXPECT_NE(out.chunks, nullptr);                                              \
  struct rle_chunks *compressed =                                              \
      make_device_rle_chunks(out.number_of_chunks, BLOCK_SIZE);                \
  compressed = make_device_rle_chunks(out.number_of_chunks, BLOCK_SIZE);       \
  rle_compression_kernel<<<out.number_of_chunks, 1024>>>(                      \
      dev_data, TEST_DATA_LEN, compressed);                                    \
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());                                   \
  CUDA_ERROR_CHECK(cudaFree(dev_data));                                        \
  copy_rle_chunks(compressed, out.chunks, DeviceToHost, out.number_of_chunks,  \
                  out.number_of_chunks *BLOCK_SIZE);                           \
  CUDA_ERROR_CHECK(cudaFree(compressed));                                      \
  unsigned char decomp_data[TEST_DATA_LEN];                                    \
  int counter = 0;                                                             \
  EXPECT_GT(out.chunks->chunk_lengths[0], 0);                                  \
  /* EXPECT_GT(out.chunks[1].array_length, 0); */                              \
  /* EXPECT_GE((int)out.chunks[0].lengths[0], 0); */                           \
  /* EXPECT_GE((int)out.chunks[1].lengths[0], 0); */                           \
  for (int chunk = 0; chunk < out.number_of_chunks; chunk++) {                 \
    EXPECT_LE(out.chunks->chunk_lengths[chunk], 1024);                         \
    EXPECT_GT(out.chunks->chunk_lengths[chunk], 0);                            \
    for (int i = 0; i < out.chunks->chunk_lengths[chunk]; i++) {               \
      EXPECT_LT(out.chunks->values[out.chunks->chunk_starts[chunk] + i], 256); \
      EXPECT_GE(out.chunks->values[out.chunks->chunk_starts[chunk] + i], 0);   \
      EXPECT_GE(out.chunks->repetitions[out.chunks->chunk_starts[chunk] + i],  \
                0);                                                            \
      EXPECT_LT(out.chunks->repetitions[out.chunks->chunk_starts[chunk] + i],  \
                256);                                                          \
      for (int j = 0;                                                          \
           j <= out.chunks->repetitions[out.chunks->chunk_starts[chunk] + i];  \
           j++) {                                                              \
        decomp_data[counter] =                                                 \
            out.chunks->values[out.chunks->chunk_starts[chunk] + i];           \
        counter++;                                                             \
        EXPECT_LE(counter, TEST_DATA_LEN);                                     \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  for (int i = 0; i < TEST_DATA_LEN; i++) {                                    \
    EXPECT_EQ(decomp_data[i], data[i]);                                        \
    if (decomp_data[i] != data[i]) {                                           \
      printf("i:%d, period: %d\n", i, PERIOD);                                 \
    }                                                                          \
  }                                                                            \
  EXPECT_EQ(counter, TEST_DATA_LEN);                                           \
  free(out.chunks);                                                            \
  free(data);

TEST(run_length_encoding,
     short_freq_compression_kernel){TEST_COMPRESSION_KERNEL_TEMPLATE(1024, 19)}

TEST(run_length_encoding, long_freq_compression_kernel){
    TEST_COMPRESSION_KERNEL_TEMPLATE((2048 * 32 + 9), 19)}

TEST(run_length_encoding, short_infreq_compression_kernel){
    TEST_COMPRESSION_KERNEL_TEMPLATE(4096, 1091)}

TEST(run_length_encoding, long_infreq_compression_kernel){
    TEST_COMPRESSION_KERNEL_TEMPLATE((2048 * 32 + 9), 1091)}

TEST(run_length_encoding, host_compress_rle) {
  unsigned int TEST_DATA_LEN = (1 << 25);
  // unsigned int TEST_DATA_LEN = 2048 * 2048 ;
  printf("running on %dMB\n", TEST_DATA_LEN / (1024 * 1024));
  unsigned int PERIOD = 19;
  unsigned char *data = (unsigned char *)malloc(TEST_DATA_LEN);
  EXPECT_NE(data, nullptr);
  for (long long i = 0; i < TEST_DATA_LEN; i++) {
    data[i] = i / (PERIOD) + 2 + (i % 1091 == 0);
  }
  struct rle_data *rle = compress_rle(data, TEST_DATA_LEN);
  // EXPECT_NE(rle, nullptr);
  unsigned char *decomp_data = (unsigned char *)malloc(TEST_DATA_LEN);
  int counter = 0;
  EXPECT_GT(rle->chunks->chunk_lengths[0], 0);
  for (int chunk = 0; chunk < rle->number_of_chunks; chunk++) {
    EXPECT_LE(rle->chunks->chunk_lengths[chunk], 1024);
    EXPECT_GT(rle->chunks->chunk_lengths[chunk], 0);
    for (int i = 0; i < rle->chunks->chunk_lengths[chunk]; i++) {
      EXPECT_LT(rle->chunks->values[rle->chunks->chunk_starts[chunk] + i], 256);
      EXPECT_GE(rle->chunks->values[rle->chunks->chunk_starts[chunk] + i], 0);
      EXPECT_GE(rle->chunks->repetitions[rle->chunks->chunk_starts[chunk] + i],
                0);
      EXPECT_LT(rle->chunks->repetitions[rle->chunks->chunk_starts[chunk] + i],
                256);
      for (int j = 0;
           j <= rle->chunks->repetitions[rle->chunks->chunk_starts[chunk] + i];
           j++) {
        decomp_data[counter] =
            rle->chunks->values[rle->chunks->chunk_starts[chunk] + i];
        counter++;
        EXPECT_LE(counter, TEST_DATA_LEN);
      }
    }
  }
  for (long long i = 0; i < TEST_DATA_LEN; i++) {
    EXPECT_EQ(decomp_data[i], data[i]);
    if (decomp_data[i] != data[i]) {
      printf("i: %lld, period: %d\n", i, PERIOD);
    }
  }
  EXPECT_EQ(counter, TEST_DATA_LEN);
  free(data);
  free(decomp_data);
  free(rle->chunks);
  free(rle);
}

TEST(rle_utils, make_device_rle_chunk) {
  int number_og_chunks = 100;
  int chunk_cap = BLOCK_SIZE;
  struct rle_chunks *dev_chunks =
      make_device_rle_chunks(number_og_chunks, chunk_cap);
  CUDA_ERROR_CHECK(cudaFree(dev_chunks));
}

TEST(rle_utils, make_host_rle_chunk) {
  int number_og_chunks = 100;
  int chunk_cap = BLOCK_SIZE;
  struct rle_chunks *host_chunks =
      make_host_rle_chunks(number_og_chunks, chunk_cap);
  for (int i = 0; i < number_og_chunks; i++) {
    int dummy = host_chunks->chunk_lengths[i];
    for (int j = 0; j < host_chunks->chunk_lengths[i]; j++) {
      dummy = host_chunks->repetitions[i * BLOCK_SIZE + j];
      dummy = host_chunks->values[i * BLOCK_SIZE + j];
    }
  }
  free(host_chunks);
}
