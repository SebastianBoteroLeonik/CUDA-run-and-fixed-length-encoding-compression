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
  data->compressed_array_length = 0;
  fpos_t position;
  fgetpos(fileptr, &position);
  wc = fwrite(&(data->compressed_array_length),
              sizeof(data->compressed_array_length), 1, fileptr);
  if (wc != 1)
    ERR("fwrite compressed_array_length");
  unsigned long long accumulator = 0;
  for (int i = 0; i < data->number_of_chunks; i++) {
    wc = fwrite(data->chunks->repetitions + data->chunks->chunk_starts[i],
                sizeof(*data->chunks->repetitions),
                data->chunks->chunk_lengths[i], fileptr);
    if (wc != data->chunks->chunk_lengths[i])
      ERR("fwrite data->chunk->lengths");
  }
  for (int i = 0; i < data->number_of_chunks; i++) {
    wc = fwrite(data->chunks->values + data->chunks->chunk_starts[i],
                sizeof(*data->chunks->values), data->chunks->chunk_lengths[i],
                fileptr);
    if (wc != data->chunks->chunk_lengths[i])
      ERR("fwrite data->chunk->lengths");
    accumulator += data->chunks->chunk_lengths[i];
  }
  data->compressed_array_length = accumulator;
  fsetpos(fileptr, &position);
  wc = fwrite(&(data->compressed_array_length),
              sizeof(data->compressed_array_length), 1, fileptr);
  if (wc != 1)
    ERR("fwrite compressed_array_length");
  if (fclose(fileptr))
    ERR("fclose");
}

struct rle_data *read_rle_from_file(char *file_name) {
  struct rle_data *data = (struct rle_data *)malloc(sizeof(struct rle_data));
  FILE *fileptr = fopen(file_name, "rb");
  if (!fileptr) {
    ERR("fopen");
  }
  int rc; // read count
  rc = fread(&(data->total_data_length), sizeof(data->total_data_length), 1,
             fileptr);
  if (rc != 1)
    ERR("fread total_data_length");
  rc = fread(&(data->compressed_array_length),
             sizeof(data->compressed_array_length), 1, fileptr);
  if (rc != 1)
    ERR("fread compressed_array_length");
  data->chunks = make_host_rle_chunks(1, data->compressed_array_length);
  data->number_of_chunks = 1;
  data->chunks->chunk_starts[0] = 0;
  data->chunks->chunk_lengths[0] = data->compressed_array_length;
  rc = fread(data->chunks->repetitions, sizeof(*data->chunks->repetitions),
             data->compressed_array_length, fileptr);
  if (rc != data->compressed_array_length)
    ERR("fread data->chunk->repetitions");
  rc = fread(data->chunks->values, sizeof(*data->chunks->values),
             data->compressed_array_length, fileptr);
  if (rc != data->compressed_array_length)
    ERR("fread data->chunk->values");
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
  printf("data->total_data_length = %lu\n", data->total_data_length);
  printf("data->total_data_length = %p\n", &data->total_data_length);

  wc = fwrite(&(data->number_of_chunks), sizeof(data->number_of_chunks), 1,
              fileptr);
  if (wc != 1)
    ERR("fwrite number_of_chunks");
  printf("data->number_of_chunks = %lu\n", data->number_of_chunks);
  printf("data->number_of_chunks = %p\n", &data->number_of_chunks);

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
    // printf("chunk_len = %d\n", chunk_len);
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
    ERR("fwrite chunk_element_size");
  for (int i = 0; i < data->number_of_chunks; i++) {
    int chunk_len = CEIL_DEV((data->chunk_element_size[i] * BLOCK_SIZE), 8);
    printf("chunk_len = %d\n", chunk_len);
    rc = fread(data->chunk_data[i], sizeof(*data->chunk_data[i]), chunk_len,
               fileptr);
    if (rc != chunk_len)
      ERR("fwrite chunk_data");
  }
  fclose(fileptr);
  return data;
}
