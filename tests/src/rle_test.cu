#include <gtest/gtest.h>
#include <rle.h>
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
                                       unsigned int data_len,
                                       struct rle_chunk *rle);

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
  struct rle_chunk *compressed;                                                \
  struct rle_data out;                                                         \
  out.number_of_chunks = CEIL_DEV(TEST_DATA_LEN, 1024);                        \
  out.chunks = (struct rle_chunk *)malloc(sizeof(struct rle_chunk) *           \
                                          out.number_of_chunks);               \
  EXPECT_NE(out.chunks, nullptr);                                              \
  CUDA_ERROR_CHECK(cudaMalloc(&compressed, sizeof(struct rle_chunk) *          \
                                               out.number_of_chunks));         \
  rle_compression_kernel<<<out.number_of_chunks, 1024>>>(                      \
      dev_data, TEST_DATA_LEN, compressed);                                    \
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());                                   \
  CUDA_ERROR_CHECK(cudaFree(dev_data));                                        \
  CUDA_ERROR_CHECK(cudaMemcpy(out.chunks, compressed,                          \
                              sizeof(struct rle_chunk) * out.number_of_chunks, \
                              cudaMemcpyDeviceToHost));                        \
  CUDA_ERROR_CHECK(cudaFree(compressed));                                      \
  unsigned char decomp_data[TEST_DATA_LEN];                                    \
  int counter = 0;                                                             \
  EXPECT_GT(out.chunks[0].array_length, 0);                                    \
  /* EXPECT_GT(out.chunks[1].array_length, 0); */                              \
  /* EXPECT_GE((int)out.chunks[0].lengths[0], 0); */                           \
  /* EXPECT_GE((int)out.chunks[1].lengths[0], 0); */                           \
  for (int chunk = 0; chunk < out.number_of_chunks; chunk++) {                 \
    EXPECT_LE(out.chunks[chunk].array_length, 1024);                           \
    EXPECT_GT(out.chunks[chunk].array_length, 0);                              \
    for (int i = 0; i < out.chunks[chunk].array_length; i++) {                 \
      EXPECT_LT(out.chunks[chunk].values[i], 256);                             \
      EXPECT_GE(out.chunks[chunk].values[i], 0);                               \
      EXPECT_GE(out.chunks[chunk].lengths[i], 0);                              \
      EXPECT_LT(out.chunks[chunk].lengths[i], 256);                            \
      for (int j = 0; j <= out.chunks[chunk].lengths[i]; j++) {                \
        decomp_data[counter] = out.chunks[chunk].values[i];                    \
        if (decomp_data[counter] != data[counter] || counter == 16383) {       \
          /*printf("counter: %d, j: %d, i: %d, chunk: %d, len: %d, val: %d\n", \
                 counter, j, i, chunk, out.chunks[chunk].lengths[i],           \
                 out.chunks[chunk].values[i]);*/                               \
        }                                                                      \
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

TEST(run_length_encoding, long_infreq_compression_kernel) {
  TEST_COMPRESSION_KERNEL_TEMPLATE((2048 * 32 + 9), 1091)
}

// TEST(run_length_encoding, compression_kernel) {
//   // #define TEST_DATA_LEN (2048 * 2048 + 1)
// #define TEST_DATA_LEN (512 * 32 + 2)
// #define CEIL_DEV(num, div) ((num / div) + (num % div != 0))
//   unsigned char *data = (unsigned char *)malloc(TEST_DATA_LEN);
//   EXPECT_NE(data, nullptr);
//   unsigned char *dev_data;
//   for (int i = 0; i < TEST_DATA_LEN; i++) {
//     data[i] = i / 91 + 2 + (i % 2909 == 0);
//   }
//   CUDA_ERROR_CHECK(cudaMalloc(&dev_data, TEST_DATA_LEN));
//   CUDA_ERROR_CHECK(cudaMemcpy(dev_data, data,
//                               sizeof(unsigned char) * TEST_DATA_LEN,
//                               cudaMemcpyHostToDevice));
//
//   struct rle_chunk *compressed;
//   struct rle_data out;
//   out.number_of_chunks = CEIL_DEV(TEST_DATA_LEN, 1024);
//   out.chunks = (struct rle_chunk *)malloc(sizeof(struct rle_chunk) *
//                                           out.number_of_chunks);
//   EXPECT_NE(out.chunks, nullptr);
//   CUDA_ERROR_CHECK(
//       cudaMalloc(&compressed, sizeof(struct rle_chunk) *
//       out.number_of_chunks));
//   rle_compression_kernel<<<out.number_of_chunks, 1024>>>(
//       dev_data, TEST_DATA_LEN, compressed);
//   CUDA_ERROR_CHECK(cudaDeviceSynchronize());
//   CUDA_ERROR_CHECK(cudaFree(dev_data));
//   CUDA_ERROR_CHECK(cudaMemcpy(out.chunks, compressed,
//                               sizeof(struct rle_chunk) *
//                               out.number_of_chunks, cudaMemcpyDeviceToHost));
//   CUDA_ERROR_CHECK(cudaFree(compressed));
//   unsigned char decomp_data[TEST_DATA_LEN];
//   int counter = 0;
//   EXPECT_GT(out.chunks[0].array_length, 0);
//   // EXPECT_GT(out.chunks[1].array_length, 0);
//   // EXPECT_GE((int)out.chunks[0].lengths[0], 0);
//   // EXPECT_GE((int)out.chunks[1].lengths[0], 0);
//   for (int chunk = 0; chunk < out.number_of_chunks; chunk++) {
//     EXPECT_LE(out.chunks[chunk].array_length, 1024);
//     EXPECT_GT(out.chunks[chunk].array_length, 0);
//     for (int i = 0; i < out.chunks[chunk].array_length; i++) {
//       EXPECT_LT(out.chunks[chunk].values[i], 256);
//       EXPECT_GE(out.chunks[chunk].values[i], 0);
//       EXPECT_GE(out.chunks[chunk].lengths[i], 0);
//       EXPECT_LT(out.chunks[chunk].lengths[i], 256);
//       for (int j = 0; j <= out.chunks[chunk].lengths[i]; j++) {
//         decomp_data[counter] = out.chunks[chunk].values[i];
//         if (decomp_data[counter] != data[counter] || counter == 16383) {
//           printf("counter: %d, j: %d, i: %d, chunk: %d, len: %d, val: "
//                  "%d\n",
//                  counter, j, i, chunk, out.chunks[chunk].lengths[i],
//                  out.chunks[chunk].values[i]);
//         }
//         counter++;
//         EXPECT_LE(counter, TEST_DATA_LEN);
//       }
//     }
//   }
//   for (int i = 0; i < TEST_DATA_LEN; i++) {
//     EXPECT_EQ(decomp_data[i], data[i]);
//     // if (decomp_data[i] != data[i]) {
//     //   printf("i:%d, \n");
//     // }
//   }
//   EXPECT_EQ(counter, TEST_DATA_LEN);
//   free(out.chunks);
//   free(data);
// }
