#ifndef FILE_IO
#define FILE_IO
#include "fle.h"
#include "rle.h"

#define BLOCK_SIZE 1024
#define CEIL_DEV(num, div) (((num) / (div)) + ((num) % (div) != 0))

void write_binary_file(char *file_name, unsigned char *data, size_t data_len);
unsigned char *read_binary_file(char *file_name, int *data_len);

void write_rle_to_file(struct rle_data *data, char *file_name);
struct rle_data *read_rle_from_file(char *file_name);

void write_fle_to_file(struct fle_data *data, char *file_name);
struct fle_data *read_fle_from_file(char *file_name);

#endif // !FILE_IO
