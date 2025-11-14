#include "cuda_utils.cuh"
#include "fle.h"
#include <cstdio>
#include <cuda.h>
#include <gtest/gtest.h>

__global__ void validate_device_fle(struct fle_data *fle,
                                    int number_of_chunks) {
  assert(fle->number_of_chunks == number_of_chunks);
  char dummy;
  for (int i = 0; i < fle->number_of_chunks; i++) {
    dummy = fle->chunk_element_size[i];
    for (int j = 0; j < 1024; j++) {
      dummy = fle->chunk_data[i][j];
    }
  }
}

TEST(fle_utils, make_device_fle_data) {
  int number_of_chunks = 50;
  struct fle_data *fle = make_device_fle_data(number_of_chunks);
  validate_device_fle<<<1, 1>>>(fle, number_of_chunks);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_ERROR_CHECK(cudaFree(fle));
}

TEST(fle_utils, make_host_fle_data) {
  int number_of_chunks = 50;
  struct fle_data *fle = make_host_fle_data(number_of_chunks);
  EXPECT_EQ(fle->number_of_chunks, number_of_chunks);
  EXPECT_GE(fle->total_data_length, 0);
  for (int i = 0; i < number_of_chunks; i++) {
    EXPECT_GE(fle->chunk_element_size[i], 0);
    for (int j = 0; j < 1024; j++) {
      EXPECT_GE(fle->chunk_data[i][j], 0);
    }
  }
}

TEST(fle_utils, copy_fle) {
  int number_of_chunks = 10;
  struct fle_data *host = make_host_fle_data(number_of_chunks);
  struct fle_data *device = make_device_fle_data(number_of_chunks);
  struct fle_data *host_again = make_host_fle_data(number_of_chunks);
  host->total_data_length = 42;
  host->number_of_chunks = number_of_chunks;
  for (int i = 0; i < host->number_of_chunks; i++) {
    host->chunk_element_size[i] = 37 + i;
    for (int j = 0; j < 1024; j++) {
      host->chunk_data[i][j] = j;
    }
  }
  copy_fle_data(host, device, HostToDevice);
  copy_fle_data(device, host_again, DeviceToHost);
  EXPECT_EQ(host->total_data_length, host_again->total_data_length);
  EXPECT_EQ(number_of_chunks, host_again->number_of_chunks);
  for (int i = 0; i < host_again->number_of_chunks; i++) {
    EXPECT_EQ(host->chunk_element_size[i], host_again->chunk_element_size[i]);
    for (int j = 0; j < 1024; j++) {
      EXPECT_EQ(host->chunk_data[i][j], host_again->chunk_data[i][j]);
    }
  }
  free(host);
  free(host_again);
  CUDA_ERROR_CHECK(cudaFree(device));
}

TEST(fle_encoding, fle_compression) {
  constexpr int size = 10000;
  unsigned char buf[size];
  for (int i = 0; i < size; i++) {
    buf[i] = (i % 16 + (i > 1500)) * (1 + (i > 5000));
  }
  struct fle_data *fle = fle_compress(buf, size);
  // printf("number of chunks: %lu\n", fle->number_of_chunks);
  for (int i = 0; i < fle->number_of_chunks; i++) {
    // printf("chunk_size[%d]: %d\n", i, fle->chunk_element_size[i]);
    if ((i + 1) * BLOCK_SIZE < 1500) {
      EXPECT_EQ(fle->chunk_element_size[i], 4);
    } else if ((i + 1) * BLOCK_SIZE < 5000) {
      EXPECT_EQ(fle->chunk_element_size[i], 5);
    } else {
      EXPECT_EQ(fle->chunk_element_size[i], 6);
    }

    for (int j = 0; j < 1024 && 1024 * i + j < size; j++) {
      unsigned char mask = 0xff;
      mask <<= 8 - fle->chunk_element_size[i];
      int bit_id = j * fle->chunk_element_size[i];
      unsigned char masked =
          (mask >> (bit_id % 8)) & (fle->chunk_data[i][bit_id / 8]);
      unsigned char alligned = (buf[i * 1024 + j])
                               << (8 - fle->chunk_element_size[i]);
      unsigned char condition = (masked ^ (alligned >> (bit_id % 8)));
      EXPECT_EQ(condition, 0);
      if (condition) {
        printf("i:%d j:%d cond:%#04x masked:%#04x size:%d alligned:%#02x, "
               "mask:%#02x\n",
               i, j, condition, masked, fle->chunk_element_size[i], alligned,
               (mask >> (bit_id % 8)));
      }
    }
  }
  fflush(stdout);
}

TEST(fle_encoding, fle_decompression) {
  size_t number_of_chunks = 1;
  struct fle_data *fle = make_host_fle_data(number_of_chunks);
  fle->number_of_chunks = number_of_chunks;
  fle->total_data_length = number_of_chunks * 512;
  for (int i = 0; i < number_of_chunks; i++) {
    fle->chunk_element_size[i] = 4;
    for (int j = 0; j < 512; j += 2) {
      char tmp = j % 16;
      tmp <<= 4;
      tmp |= (j + 1) % 16;
      fle->chunk_data[i][j / 2] = tmp;
    }
  }
  unsigned char *buf = fle_decompress(fle);
  for (int i = 0; i < fle->total_data_length; i++) {
    EXPECT_EQ(buf[i], i % 16);
  }
}
