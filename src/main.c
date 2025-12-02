#include "cli.h"
#include "file_io.h"
#include "fle.h"
#include "rle.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  enum args_options options;
  char *output_file_name = NULL;
  char *input_file_name = NULL;
  parse_args(argc, argv, &options, &output_file_name, &input_file_name);
  if (output_file_name) {
    printf("output file name: %s\n", output_file_name);
  }
  printf("input file name: %s\n", input_file_name);
  if (options & USE_FLE) {
    if (options & DECOMPRESS) {
      if (!output_file_name) {
        output_file_name = "fle.decompressed";
      }
      printf("decompressing fle\n");
      struct fle_data *fle = read_fle_from_file(input_file_name);
      unsigned char *binary_data = fle_decompress(fle);
      write_binary_file(output_file_name, binary_data, fle->total_data_length);
      free(fle);
      free(binary_data);
      printf("decompression succeded\n");
    } else {
      if (!output_file_name) {
        output_file_name = "output.fle";
      }
      printf("compressing fle\n");
      int data_length;
      unsigned char *binary_data =
          read_binary_file(input_file_name, &data_length);
      struct fle_data *fle = fle_compress(binary_data, data_length);
      write_fle_to_file(fle, output_file_name);
      free(fle);
      free(binary_data);
      printf("compression succeded\n");
    }
  } else {
    if (options & DECOMPRESS) {
      if (!output_file_name) {
        output_file_name = "rle.decompressed";
      }
      printf("decompressing rle\n");
      struct rle_data *rle = read_rle_from_file(input_file_name);
      unsigned char *binary_data = decompress_rle(rle);
      write_binary_file(output_file_name, binary_data, rle->total_data_length);
      free(rle);
      free(binary_data);
      printf("decompression succeded\n");
    } else {
      if (!output_file_name) {
        output_file_name = "output.rle";
      }
      printf("compressing rle\n");
      int data_length;
      unsigned char *binary_data =
          read_binary_file(input_file_name, &data_length);
      struct rle_data *rle = compress_rle(binary_data, data_length);
      write_rle_to_file(rle, output_file_name);
      free(rle);
      free(binary_data);
      printf("compression succeded\n");
    }
  }
}
