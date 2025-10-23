#include <gtest/gtest.h>
#include <rle_tests.h>

__global__ void run_warp_cumsum(int *vals) {
  vals[threadIdx.x] = warp_cumsum(vals[threadIdx.x], 0xffffffff);
}

__global__ void run_block_cumsum(int *vals) {
  vals[threadIdx.x] = block_cumsum(vals[threadIdx.x]);
}

TEST(run_length_encoding, warp_sum) {
  int vals[32];
  for (int i = 0; i < 32; i++) {
    vals[i] = i;
  }
  int *dev_vals;
  cudaMalloc(&dev_vals, 32 * sizeof(int));
  cudaMemcpy(dev_vals, vals, 32 * sizeof(int), cudaMemcpyHostToDevice);
  run_warp_cumsum<<<1, 32>>>(dev_vals);
  cudaDeviceSynchronize();
  cudaMemcpy(vals, dev_vals, 32 * sizeof(int), cudaMemcpyDeviceToHost);
  int counter = 0;
  for (int i = 0; i < 32; i++) {
    counter += i;
    EXPECT_EQ(counter, vals[i]);
  }
}

#define BLOCK_SIZE 1024
TEST(run_length_encoding, block_sum) {
  int vals[BLOCK_SIZE];
  for (int i = 0; i < BLOCK_SIZE; i++) {
    vals[i] = i;
  }
  int *dev_vals;
  cudaMalloc(&dev_vals, BLOCK_SIZE * sizeof(int));
  cudaMemcpy(dev_vals, vals, BLOCK_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  run_block_cumsum<<<1, BLOCK_SIZE>>>(dev_vals);
  cudaDeviceSynchronize();
  cudaMemcpy(vals, dev_vals, BLOCK_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  int counter = 0;
  for (int i = 0; i < BLOCK_SIZE; i++) {
    counter += i;
    EXPECT_EQ(counter, vals[i]);
  }
}
