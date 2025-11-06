#ifndef FILE_IO
#define FILE_IO
#include "rle.h"
void write_rle_to_file(struct rle_data *data, char *file_name);
struct rle_data *read_rle_from_file(char *file_name);
#endif // !FILE_IO
