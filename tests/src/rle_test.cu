#include "cuda_utils.cuh"
#include "rle.h"
#include <cstddef>
#include <gtest/gtest.h>

__global__ void run_warp_cumsum(int *vals) {
  vals[threadIdx.x] = warp_cumsum(vals[threadIdx.x], 0xffffffff);
}

__global__ void run_block_cumsum(int *vals) {
  vals[threadIdx.x] = block_cumsum(vals[threadIdx.x]);
}

__global__ void
find_differing_neighbours(unsigned char *data,
                          unsigned int *diff_from_prev_indicators, size_t len);

__global__ void find_segment_end(unsigned int *scan_result,
                                 unsigned *segment_ends, size_t len);

__global__ void subtract_segment_begining(unsigned int *scan_result,
                                          unsigned *segment_lengths_out,
                                          unsigned int *overflows,
                                          unsigned char *data,
                                          unsigned char *compressed_data_vals,
                                          size_t len);

__global__ void rle_compression_kernel(const unsigned char *data,
                                       size_t data_len, struct rle_data *rle);

TEST(rle_common, warp_sum) {
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

TEST(rle_common, block_sum) {
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

TEST(rle_common, partial_block_sum) {
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

TEST(run_length_encoding, find_diffs) {
  constexpr int len = 4000;
  constexpr int period = 23;
  unsigned char *dev_data;
  unsigned char data[len];
  unsigned int *dev_diffs;
  unsigned int diffs[len];
  for (int i = 0; i < len; i++) {
    data[i] = i / period;
  }
  CUDA_ERROR_CHECK(cudaMalloc(&dev_data, len * sizeof(*dev_data)));
  CUDA_ERROR_CHECK(
      cudaMemcpy(dev_data, data, sizeof(*data) * len, cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(cudaMalloc(&dev_diffs, len * sizeof(*dev_diffs)));
  find_differing_neighbours<<<CEIL_DEV(len, BLOCK_SIZE), BLOCK_SIZE>>>(
      dev_data, dev_diffs, len);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaFree(dev_data));
  CUDA_ERROR_CHECK(cudaMemcpy(diffs, dev_diffs, sizeof(*diffs) * len,
                              cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaFree(dev_diffs));
  for (int i = 0; i < len; i++) {
    EXPECT_EQ(diffs[i], i ? (i - 1) / period != i / period : 0);
  }
}

TEST(run_length_encoding, find_segment_ends) {
  constexpr int len = 4000;
  constexpr int period = 23;
  unsigned int *dev_scan_array;
  unsigned int scan_array[len];
  unsigned int *dev_segment_ends;
  unsigned int segment_ends[len];
  unsigned int true_segment_ends[len];
  int acc = 0;
  for (int i = 0; i < len; i++) {
    int change = i ? ((i - 1) / period != i / period) : 0;
    acc += change;
    scan_array[i] = acc;
    if (change) {
      true_segment_ends[acc - 1] = i - 1;
    }
  }
  CUDA_ERROR_CHECK(cudaMalloc(&dev_scan_array, len * sizeof(*dev_scan_array)));
  CUDA_ERROR_CHECK(cudaMemcpy(dev_scan_array, scan_array,
                              sizeof(*scan_array) * len,
                              cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(
      cudaMalloc(&dev_segment_ends, len * sizeof(*dev_segment_ends)));
  find_segment_end<<<CEIL_DEV(len, BLOCK_SIZE), BLOCK_SIZE>>>(
      dev_scan_array, dev_segment_ends, len);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaMemcpy(segment_ends, dev_segment_ends,
                              sizeof(*segment_ends) * len,
                              cudaMemcpyDeviceToHost));
  for (int i = 0; i < acc; i++) {
    EXPECT_EQ(segment_ends[i], true_segment_ends[i]);
  }
}

TEST(run_length_encoding, subtract_beginings) {
  constexpr int len = 4000;
  constexpr int period = 400;
  unsigned int *dev_scan_array;
  unsigned int scan_array[len];
  unsigned int segment_lengths[len];
  unsigned int *dev_segment_ends;
  unsigned int segment_ends[len];
  unsigned int *dev_overflows;
  unsigned int overflows[len];
  unsigned char *dev_data;
  unsigned char data[len];
  unsigned char *dev_vals;
  unsigned char vals[len];
  int acc = 0;
  for (int i = 0; i < len; i++) {
    data[i] = i / period;
    int change = i ? ((i - 1) / period != i / period) : 0;
    acc += change;
    scan_array[i] = acc;
    if (change) {
      segment_ends[acc - 1] = i - 1;
    }
  }
  CUDA_ERROR_CHECK(cudaMalloc(&dev_data, len * sizeof(*dev_data)));
  CUDA_ERROR_CHECK(
      cudaMemcpy(dev_data, data, sizeof(*data) * len, cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(cudaMalloc(&dev_scan_array, len * sizeof(*dev_scan_array)));
  CUDA_ERROR_CHECK(cudaMemcpy(dev_scan_array, scan_array,
                              sizeof(*scan_array) * len,
                              cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(
      cudaMalloc(&dev_segment_ends, len * sizeof(*dev_segment_ends)));
  CUDA_ERROR_CHECK(cudaMemcpy(dev_segment_ends, segment_ends,
                              sizeof(*segment_ends) * len,
                              cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(cudaMalloc(&dev_overflows, len * sizeof(*dev_overflows)));
  CUDA_ERROR_CHECK(cudaMalloc(&dev_vals, len * sizeof(*dev_vals)));
  subtract_segment_begining<<<CEIL_DEV(len, BLOCK_SIZE), BLOCK_SIZE>>>(
      dev_scan_array, dev_segment_ends, dev_overflows, dev_data, dev_vals, len);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaMemcpy(overflows, dev_overflows,
                              sizeof(*overflows) * acc,
                              cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(
      cudaMemcpy(vals, dev_vals, sizeof(*vals) * acc, cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaMemcpy(segment_lengths, dev_segment_ends,
                              sizeof(*segment_lengths) * acc,
                              cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaFree(dev_scan_array));
  CUDA_ERROR_CHECK(cudaFree(dev_segment_ends));
  CUDA_ERROR_CHECK(cudaFree(dev_overflows));
  CUDA_ERROR_CHECK(cudaFree(dev_data));
  CUDA_ERROR_CHECK(cudaFree(dev_vals));
  for (int i = 0; i < acc; i++) {
    EXPECT_EQ(segment_lengths[i], period - 1);
  }
  for (int i = 0; i < acc; i++) {
    EXPECT_EQ(overflows[i], period / 256);
  }
  for (int i = 0; i < acc; i++) {
    EXPECT_EQ(vals[i], i);
  }
}

TEST(run_length_encoding, compress_rle) {
  unsigned int TEST_DATA_LEN = (1 << 25);
  // unsigned int TEST_DATA_LEN = (1 << 16);
  unsigned int PERIOD = 19;
  unsigned char *data = (unsigned char *)malloc(TEST_DATA_LEN);
  EXPECT_NE(data, nullptr);
  for (unsigned int i = 0; i < TEST_DATA_LEN; i++) {
    data[i] = i / (PERIOD) + 2 + (i % 1091 == 0);
  }
  struct rle_data *rle = compress_rle(data, TEST_DATA_LEN);
  unsigned char *decomp_data = (unsigned char *)malloc(TEST_DATA_LEN);
  int counter = 0;
  for (int i = 0; i < rle->compressed_array_length; i++) {
    EXPECT_GE(rle->values[i], 0);
    EXPECT_LT(rle->values[i], 256);
    EXPECT_GE(rle->repetitions[i], 0);
    EXPECT_LT(rle->repetitions[i], 256);
    for (int j = 0; j <= rle->repetitions[i]; j++) {
      decomp_data[counter] = rle->values[i];
      counter++;
      EXPECT_LE(counter, TEST_DATA_LEN);
    }
  }
  for (unsigned int i = 0; i < TEST_DATA_LEN; i++) {
    EXPECT_EQ(decomp_data[i], data[i]);
    if (decomp_data[i] != data[i]) {
      printf("i: %u, period: %d\n", i, PERIOD);
    }
  }
  EXPECT_EQ(counter, TEST_DATA_LEN);
  free(data);
  free(decomp_data);
  free(rle);
}

TEST(rle_utils, make_device_rle_data) {
  int capacity = 100;
  struct rle_data *dev_data = make_device_rle_data(capacity);
  CUDA_ERROR_CHECK(cudaFree(dev_data));
}

TEST(rle_utils, make_host_rle_data) {
  int capacity = 100;
  struct rle_data *host_data = make_host_rle_data(capacity);
  int dummy = host_data->compressed_array_length;
  dummy = host_data->total_data_length;
  for (int i = 0; i < capacity; i++) {
    dummy = host_data->repetitions[i];
    dummy = host_data->values[i];
  }
  // To turn off unused warning about dummy
  capacity = dummy;
  free(host_data);
}

__global__ void uchar_array_to_uint_array(unsigned char *chars,
                                          unsigned int *ints,
                                          unsigned int array_length);

__global__ void run_cumsum(unsigned int *array,
                           unsigned int *last_sums_in_chunks,
                           unsigned int array_length);

__global__ void down_propagate_cumsum(unsigned int *array,
                                      unsigned int *last_sums_in_chunks,
                                      unsigned int array_length);

TEST(rle_decoding, char_to_llong) {
  size_t SIZE = 2000;
  unsigned char *chars = (unsigned char *)malloc(SIZE);
  for (int i = 0; i < SIZE; i++) {
    chars[i] = i;
  }
  unsigned char *dev_chars;
  unsigned int *ints;
  unsigned int *dev_ints;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_chars, SIZE));
  CUDA_ERROR_CHECK(cudaMalloc(&dev_ints, sizeof(unsigned int) * SIZE));
  CUDA_ERROR_CHECK(cudaMemcpy(dev_chars, chars, SIZE, cudaMemcpyHostToDevice));
  uchar_array_to_uint_array<<<CEIL_DEV(SIZE, BLOCK_SIZE), BLOCK_SIZE>>>(
      dev_chars, dev_ints, SIZE);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  ints = (unsigned int *)malloc(sizeof(unsigned int) * SIZE);
  CUDA_ERROR_CHECK(cudaMemcpy(ints, dev_ints, sizeof(unsigned int) * SIZE,
                              cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaFree(dev_chars));
  CUDA_ERROR_CHECK(cudaFree(dev_ints));
  for (int i = 0; i < SIZE; i++) {
    EXPECT_EQ(ints[i], chars[i] + 1);
  }
  free(chars);
  free(ints);
}

TEST(rle_decoding, run_cumsums) {
  constexpr size_t number_of_blocks = 3;
  size_t SIZE = number_of_blocks * BLOCK_SIZE;
  unsigned int *ints = (unsigned int *)malloc(sizeof(unsigned int) * SIZE);
  for (int i = 0; i < SIZE; i++) {
    ints[i] = 1;
  }
  unsigned int *dev_ints, *dev_finals;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_ints, sizeof(unsigned int) * SIZE));
  CUDA_ERROR_CHECK(
      cudaMalloc(&dev_finals, sizeof(unsigned int) * number_of_blocks));
  CUDA_ERROR_CHECK(cudaMemcpy(dev_ints, ints, sizeof(unsigned int) * SIZE,
                              cudaMemcpyHostToDevice));
  run_cumsum<<<number_of_blocks, BLOCK_SIZE>>>(dev_ints, dev_finals, SIZE);
  CUDA_ERROR_CHECK(cudaMemcpy(ints, dev_ints, sizeof(unsigned int) * SIZE,
                              cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaFree(dev_ints));
  unsigned int finals[number_of_blocks];
  CUDA_ERROR_CHECK(cudaMemcpy(finals, dev_finals,
                              sizeof(unsigned int) * number_of_blocks,
                              cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaFree(dev_finals));
  for (int i = 0; i < SIZE; i++) {
    EXPECT_EQ(ints[i], i % BLOCK_SIZE + 1);
  }
  for (int i = 0; i < number_of_blocks; i++) {
    EXPECT_EQ(finals[i], BLOCK_SIZE);
  }
  free(ints);
}

TEST(rle_decoding, down_propagate_cumsums) {
  constexpr size_t number_of_blocks = 3;
  size_t SIZE = number_of_blocks * BLOCK_SIZE;
  unsigned int *ints = (unsigned int *)malloc(sizeof(unsigned int) * SIZE);
  for (int i = 0; i < SIZE; i++) {
    ints[i] = i % BLOCK_SIZE + 1;
  }
  unsigned int *dev_ints, *dev_finals;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_ints, sizeof(unsigned int) * SIZE));
  CUDA_ERROR_CHECK(
      cudaMalloc(&dev_finals, sizeof(unsigned int) * number_of_blocks));
  CUDA_ERROR_CHECK(cudaMemcpy(dev_ints, ints, sizeof(unsigned int) * SIZE,
                              cudaMemcpyHostToDevice));
  unsigned int finals[number_of_blocks];
  for (int i = 0; i < number_of_blocks; i++) {
    finals[i] = BLOCK_SIZE * (i + 1);
  }
  CUDA_ERROR_CHECK(cudaMemcpy(dev_finals, finals,
                              sizeof(unsigned int) * number_of_blocks,
                              cudaMemcpyHostToDevice));
  down_propagate_cumsum<<<number_of_blocks - 1, BLOCK_SIZE>>>(dev_ints,
                                                              dev_finals, SIZE);
  CUDA_ERROR_CHECK(cudaMemcpy(ints, dev_ints, sizeof(unsigned int) * SIZE,
                              cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaFree(dev_ints));
  CUDA_ERROR_CHECK(cudaFree(dev_finals));
  for (int i = 0; i < SIZE; i++) {
    EXPECT_EQ(ints[i], i + 1);
  }
  free(ints);
}

TEST(rle_decoding, rec_cumsum) {
  size_t SIZE = 2000 * 1024;
  unsigned int *ints = (unsigned int *)malloc(sizeof(unsigned int) * SIZE);
  for (int i = 0; i < SIZE; i++) {
    ints[i] = 1;
  }
  unsigned int *dev_ints;
  CUDA_ERROR_CHECK(cudaMalloc(&dev_ints, sizeof(unsigned int) * SIZE));
  CUDA_ERROR_CHECK(cudaMemcpy(dev_ints, ints, sizeof(unsigned int) * SIZE,
                              cudaMemcpyHostToDevice));
  recursive_cumsum(dev_ints, SIZE);
  CUDA_ERROR_CHECK(cudaMemcpy(ints, dev_ints, sizeof(unsigned int) * SIZE,
                              cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaFree(dev_ints));
  for (int i = 0; i < SIZE; i++) {
    EXPECT_EQ(ints[i], i + 1);
  }
  free(ints);
}

TEST(rle_decoding, decode_rle) {
  int capacity = 2000;
  struct rle_data *data = make_host_rle_data(capacity);
  data->compressed_array_length = 2000;
  data->total_data_length = 0;
  for (int i = 0; i < data->compressed_array_length; i++) {
    data->repetitions[i] = i % 5;
    data->values[i] = i % 17;
    data->total_data_length += 1 + (i % 5);
  }
  unsigned char *decomp = decompress_rle(data);
  int counter = 0;
  for (int i = 0; i < data->compressed_array_length; i++) {
    for (int j = 0; j <= i % 5; j++) {
      EXPECT_EQ(decomp[counter], i % 17);
      counter++;
    }
  }
  EXPECT_EQ(counter, data->total_data_length);
}
