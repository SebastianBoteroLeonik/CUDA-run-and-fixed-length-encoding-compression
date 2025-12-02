#ifndef FILE_IO
#define FILE_IO
#include "fle.h"
#include "rle.h"

// Writing bytes to file
void write_binary_file(char *file_name, unsigned char *data, size_t data_len);
// Reading bytes from file
unsigned char *read_binary_file(char *file_name, int *data_len);

// Writing struct rle_data to file
void write_rle_to_file(struct rle_data *data, char *file_name);
// Reading struct rle_data from file
struct rle_data *read_rle_from_file(char *file_name);

// Writing struct fle_data to file
void write_fle_to_file(struct fle_data *data, char *file_name);
// Reading struct fle_data from file
struct fle_data *read_fle_from_file(char *file_name);

#endif // !FILE_IO
