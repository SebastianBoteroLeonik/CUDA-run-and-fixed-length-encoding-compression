#include "file_io.h"
#include "fle.h"
#include "rle.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define ERR(source)                                                            \
  (fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), perror(source),             \
   exit(EXIT_FAILURE))

void write_binary_file(char *file_name, unsigned char *data, size_t data_len) {
  FILE *fileptr = fopen(file_name, "wb");
  if (!fileptr) {
    ERR("fopen");
  }
  if (fwrite(data, 1, data_len, fileptr) != data_len) {
    ERR("fwrite");
  }
  fclose(fileptr);
}

unsigned char *read_binary_file(char *file_name, int *data_len) {
  FILE *fileptr = fopen(file_name, "rb");
  if (!fileptr) {
    ERR("fopen");
  }
  fseek(fileptr, 0L, SEEK_END);
  int file_size = ftell(fileptr);
  rewind(fileptr);
  unsigned char *data = malloc(file_size);
  if (fread(data, 1, file_size, fileptr) != file_size) {
    ERR("fread");
  }
  fclose(fileptr);
  *data_len = file_size;
  return data;
}

void write_rle_to_file(struct rle_data *data, char *file_name) {
  FILE *fileptr = fopen(file_name, "wb");
  if (!fileptr) {
    ERR("fopen");
  }
  int wc; // written count
  wc = fwrite(&(data->total_data_length), sizeof(data->total_data_length), 1,
              fileptr);
  if (wc != 1)
    ERR("fwrite total_data_length");
  wc = fwrite(&(data->compressed_array_length),
              sizeof(data->compressed_array_length), 1, fileptr);
  if (wc != 1)
    ERR("fwrite compressed_array_length");
  wc = fwrite(data->repetitions, sizeof(*data->repetitions),
              data->compressed_array_length, fileptr);
  if (wc != data->compressed_array_length)
    ERR("fwrite data->repetitions");
  wc = fwrite(data->values, sizeof(*data->values),
              data->compressed_array_length, fileptr);
  if (wc != data->compressed_array_length)
    ERR("fwrite data->values");
  if (fclose(fileptr))
    ERR("fclose");
}

struct rle_data *read_rle_from_file(char *file_name) {
  FILE *fileptr = fopen(file_name, "rb");
  if (!fileptr) {
    ERR("fopen");
  }
  int rc; // read count
  unsigned int total_data_length;
  unsigned int compressed_array_length;
  rc = fread(&total_data_length, sizeof(total_data_length), 1, fileptr);
  if (rc != 1)
    ERR("fread total_data_length");
  rc = fread(&compressed_array_length, sizeof(compressed_array_length), 1,
             fileptr);
  if (rc != 1)
    ERR("fread compressed_array_length");
  struct rle_data *data = make_host_rle_data(compressed_array_length);
  data->compressed_array_length = compressed_array_length;
  data->total_data_length = total_data_length;
  rc = fread(data->repetitions, sizeof(*data->repetitions),
             data->compressed_array_length, fileptr);
  if (rc != data->compressed_array_length)
    ERR("fread data->repetitions");
  rc = fread(data->values, sizeof(*data->values), data->compressed_array_length,
             fileptr);
  if (rc != data->compressed_array_length)
    ERR("fread data->values");
  if (fclose(fileptr))
    ERR("fclose");
  return data;
  return NULL;
}

void write_fle_to_file(struct fle_data *data, char *file_name) {
  FILE *fileptr = fopen(file_name, "wb");
  if (!fileptr) {
    ERR("fopen");
  }
  int wc; // written count
  wc = fwrite(&(data->total_data_length), sizeof(data->total_data_length), 1,
              fileptr);
  if (wc != 1)
    ERR("fwrite total_data_length");

  wc = fwrite(&(data->number_of_chunks), sizeof(data->number_of_chunks), 1,
              fileptr);
  if (wc != 1)
    ERR("fwrite number_of_chunks");

  wc = fwrite(data->chunk_element_size, sizeof(*data->chunk_element_size),
              data->number_of_chunks, fileptr);
  if (wc != data->number_of_chunks)
    ERR("fwrite chunk_element_size");

  for (int i = 0; i < data->number_of_chunks; i++) {
    int full_len = BLOCK_SIZE;
    if ((i + 1) * BLOCK_SIZE > data->total_data_length) {
      full_len = data->total_data_length % BLOCK_SIZE;
    }
    int chunk_len = CEIL_DEV((data->chunk_element_size[i] * full_len), 8);
    wc = fwrite(data->chunk_data[i], sizeof(*data->chunk_data[i]), chunk_len,
                fileptr);
    if (wc != chunk_len)
      ERR("fwrite chunk_data");
  }
  fclose(fileptr);
}

struct fle_data *read_fle_from_file(char *file_name) {
  FILE *fileptr = fopen(file_name, "rb");
  if (!fileptr) {
    ERR("fopen");
  }
  int rc; // read count
  unsigned long total_data_length;
  rc = fread(&(total_data_length), sizeof(total_data_length), 1, fileptr);
  if (rc != 1)
    ERR("fread total_data_length");
  unsigned long number_of_chunks;
  rc = fread(&(number_of_chunks), sizeof(number_of_chunks), 1, fileptr);
  if (rc != 1)
    ERR("fread number_of_chunks");
  struct fle_data *data = make_host_fle_data(number_of_chunks);
  data->total_data_length = total_data_length;
  rc = fread(data->chunk_element_size, sizeof(*data->chunk_element_size),
             data->number_of_chunks, fileptr);
  if (rc != data->number_of_chunks)
    ERR("fread chunk_element_size");
  for (int i = 0; i < data->number_of_chunks; i++) {
    int full_len = BLOCK_SIZE;
    if ((i + 1) * BLOCK_SIZE > data->total_data_length) {
      full_len = data->total_data_length % BLOCK_SIZE;
    }
    int chunk_len = CEIL_DEV((data->chunk_element_size[i] * full_len), 8);
    rc = fread(data->chunk_data[i], sizeof(*data->chunk_data[i]), chunk_len,
               fileptr);
    if (rc != chunk_len) {
      fprintf(stderr, "expected length %d; got %d; element size %d\n",
              chunk_len, rc, data->chunk_element_size[i]);
      ERR("fread chunk_data");
    }
  }
  fclose(fileptr);
  return data;
}
