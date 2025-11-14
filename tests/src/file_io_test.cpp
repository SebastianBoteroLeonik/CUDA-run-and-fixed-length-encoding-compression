#include "fle.h"
#include "rle.h"
#include <gtest/gtest.h>
#include <stdio.h>

extern "C" {
#include "file_io.h"
}

void test_file_equality(char *filename1, char *filename2) {
  FILE *f1 = fopen(filename1, "r");
  EXPECT_NE(f1, nullptr);
  if (!f1) {
    perror("f1");
    return;
  }
  FILE *f2 = fopen(filename2, "r");
  EXPECT_NE(f2, nullptr);
  if (!f2) {
    perror("f2");
    return;
  }
  int char_read_f1 = 0;
  int char_read_f2 = 0;
  do {
    EXPECT_EQ(char_read_f1, char_read_f2);
  } while ((char_read_f1 = fgetc(f1)) != EOF &&
           (char_read_f2 = fgetc(f2)) != EOF);
  fclose(f1);
  fclose(f2);
}

TEST(file_io, write_binary) {
  constexpr int size = 40;
  unsigned char buf[size];
  for (int i = 0; i < size; i++) {
    buf[i] = i;
  }
  char file_name[50] = "test_outputs/write_binary";
  write_binary_file(file_name, buf, size);
  char reference_file_name[50] = "test_data/reference_binary";
  test_file_equality(file_name, reference_file_name);
}

TEST(file_io, read_binary) {
  char file_name[50] = "test_data/reference_binary";
  int size;
  unsigned char *data_read = read_binary_file(file_name, &size);
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(data_read[i], i);
  }
}

/*
[ 1, 1, 1, 2, 2, 255, 255, 124 ]
||
V
[1, 2] [255, 124]
[2, 1] [1, 0]
 */
TEST(file_io, write_rle) {
  char file_name[40] = "test_outputs/write_file.rle";
  struct rle_data data;
  data.total_data_length = 8;
  data.number_of_chunks = 2;
  struct rle_chunks *chunks = make_host_rle_chunks(data.number_of_chunks, 2);
  chunks->chunk_starts[0] = 0;
  chunks->chunk_starts[1] = 2;
  chunks->chunk_lengths[0] = 2;
  chunks->chunk_lengths[1] = 2;
  chunks->repetitions[chunks->chunk_starts[0]] = 2;
  chunks->repetitions[chunks->chunk_starts[0] + 1] = 1;
  chunks->repetitions[chunks->chunk_starts[1]] = 1;
  chunks->repetitions[chunks->chunk_starts[1] + 1] = 0;
  chunks->values[chunks->chunk_starts[0]] = 1;
  chunks->values[chunks->chunk_starts[0] + 1] = 2;
  chunks->values[chunks->chunk_starts[1]] = 255;
  chunks->values[chunks->chunk_starts[1] + 1] = 124;
  data.chunks = chunks;
  write_rle_to_file(&data, file_name);
  char true_file[40] = "test_data/reference_file.rle";
  test_file_equality(file_name, true_file);
}

TEST(file_io, read_rle) {
  char file_name[40] = "test_data/reference_file.rle";
  struct rle_data *data = read_rle_from_file(file_name);
  EXPECT_EQ(data->total_data_length, 8);
  EXPECT_EQ(data->compressed_array_length, 4);
  EXPECT_EQ(data->number_of_chunks, 1);
  EXPECT_EQ(data->chunks->chunk_lengths[0], 4);
  EXPECT_EQ(data->chunks->chunk_starts[0], 0);
  EXPECT_EQ(data->chunks->repetitions[0], 2);
  EXPECT_EQ(data->chunks->repetitions[1], 1);
  EXPECT_EQ(data->chunks->repetitions[2], 1);
  EXPECT_EQ(data->chunks->repetitions[3], 0);
  EXPECT_EQ(data->chunks->values[0], 1);
  EXPECT_EQ(data->chunks->values[1], 2);
  EXPECT_EQ(data->chunks->values[2], 255);
  EXPECT_EQ(data->chunks->values[3], 124);
}

TEST(file_io, write_fle) {
  size_t number_of_chunks = 2;
  struct fle_data *fle = make_host_fle_data(number_of_chunks);
  fle->number_of_chunks = number_of_chunks;
  fle->total_data_length = number_of_chunks * 256;
  for (int i = 0; i < number_of_chunks; i++) {
    fle->chunk_element_size[i] = 2;
    for (int j = 0; j < 1024; j += 4) {
      char tmp = j & 0b11;
      tmp <<= 2;
      tmp |= (j + 1) & 0b11;
      tmp <<= 2;
      tmp |= (j + 2) & 0b11;
      tmp <<= 2;
      tmp |= (j + 3) & 0b11;
      fle->chunk_data[i][j / 4] = tmp;
    }
  }
  char file_name[40] = "test_outputs/write_file.fle";
  write_fle_to_file(fle, file_name);
  char reference_file_name[40] = "test_data/reference_file.fle";
  test_file_equality(file_name, reference_file_name);
}

TEST(file_io, read_fle) {
  char file_name[40] = "test_data/reference_file.fle";
  struct fle_data *fle = read_fle_from_file(file_name);
  EXPECT_EQ(fle->number_of_chunks, 2);
  EXPECT_EQ(fle->total_data_length, 512);
  for (int i = 0; i < fle->number_of_chunks; i++) {
    EXPECT_EQ(fle->chunk_element_size[i], 2);
    for (int j = 0; j < 516; j += 4) {
      char tmp = j & 0b11;
      tmp <<= 2;
      tmp |= (j + 1) & 0b11;
      tmp <<= 2;
      tmp |= (j + 2) & 0b11;
      tmp <<= 2;
      tmp |= (j + 3) & 0b11;
      EXPECT_EQ(fle->chunk_data[i][j / 4], tmp);
    }
  }
}
