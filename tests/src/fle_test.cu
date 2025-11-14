#include "cuda_utils.cuh"
#include "fle.h"
#include <cstdio>
#include <cuda.h>
#include <gtest/gtest.h>

TEST(fle_utils, make_device_fle_data) {
  int number_of_chunks = 50;
  struct fle_data *fle = make_device_fle_data(number_of_chunks);
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

TEST(fle_encoding, fle_compression) {
  constexpr int size = 10000;
  unsigned char buf[size];
  for (int i = 0; i < size; i++) {
    buf[i] = (i % 16 + (i > 1500)) * (1 + (i > 5000));
  }
  struct fle_data *fle = fle_compress(buf, size);
  printf("number of chunks: %lu\n", fle->number_of_chunks);
  for (int i = 0; i < fle->number_of_chunks; i++) {
    printf("chunk_size[%d]: %d\n", i, fle->chunk_element_size[i]);
    if ((i + 1) * BLOCK_SIZE < 1500) {
      EXPECT_EQ(fle->chunk_element_size[i], 4);
    } else if ((i + 1) * BLOCK_SIZE < 5000) {
      EXPECT_EQ(fle->chunk_element_size[i], 5);
    } else {
      EXPECT_EQ(fle->chunk_element_size[i], 6);
    }

    for (int j = 0; j < 1024 && 1024 * i + j < size; j++) {
      unsigned char mask = 0xff;
      mask >>= 8 - fle->chunk_element_size[i];
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
