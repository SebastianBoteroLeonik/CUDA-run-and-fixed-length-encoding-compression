#include "cli.h"
// #include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ERR(source)                                                            \
  (fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), perror(source),             \
   exit(EXIT_FAILURE))

void show_help() {
  printf("fixed and run length encoding\n"
         "\n"
         "usage:\n"
         "    compress operation method input_file output_file\n"
         "\n"
         "    operation: [c|d] compress or decompress\n"
         "    method: [rl|fl] run or fixed length encoding\n");
}

void parse_args(int argc, char **argv, args_options_t *options,
                char **output_file_name, char **input_file_name) {
  *output_file_name = NULL;
  if (argc != 5 || (strcmp(argv[1], "c") && strcmp(argv[1], "d")) ||
      (strcmp(argv[2], "rl") && strcmp(argv[2], "fl"))) {
    show_help();
    exit(EXIT_FAILURE);
  }
  *options = 0;
  if (!strcmp(argv[1], "d")) {
    *options |= DECOMPRESS;
  }
  if (!strcmp(argv[2], "fl")) {
    *options |= USE_FLE;
  }
  *input_file_name = argv[3];
  FILE *fptr = fopen(*input_file_name, "r");
  if (!fptr) {
    fprintf(stderr, "Could not open input file - please make sure the file "
                    "exists and is readable\n");
    ERR("fopen");
  }
  fclose(fptr);
  *output_file_name = argv[4];
  fptr = fopen(*output_file_name, "w");
  if (!fptr) {
    fprintf(stderr, "Could not open output file - please make sure the file "
                    "exists and is writable\n");
    ERR("fopen");
  }
  fclose(fptr);
}
