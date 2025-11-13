#ifndef FILE_IO
#define FILE_IO
#include "rle.h"

void write_binary_file(char *file_name, unsigned char *data, size_t data_len);
unsigned char *read_binary_file(char *file_name, int *data_len);

void write_rle_to_file(struct rle_data *data, char *file_name);
struct rle_data *read_rle_from_file(char *file_name);

#endif // !FILE_IO
