#include "cli.h"
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define ERR(source)                                                            \
  (fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), perror(source),             \
   exit(EXIT_FAILURE))

void show_help() {
  // char help_file_name[30] = "doc/HELP.txt";
  // FILE *help_file = fopen(help_file_name, "r");
  // if (!help_file) {
  //   ERR("fopen(help_file)");
  // }
  // char buf[50];
  // int read_count;
  // while ((read_count = fread(buf, 1, sizeof(buf), help_file)) > 0) {
  //   fwrite(buf, 1, read_count, stdout);
  // }
  // fclose(help_file);

  printf(
      "fixed and run length encoding\n"
      "\n"
      "usage:\n"
      "    frle [OPTIONS] [FILE]\n"
      "\n"
      "options:\n"
      "    -r, --rle               use run length encoding\n"
      "    -f, --fle               use fixed length encoding\n"
      "    -c, --compress          compress FILE\n"
      "    -d, --decompress        decompress FILE\n"
      "    -o, --output_name       sec the file name for the outputted file\n"
      "    -h, --help              print help information\n");
}

void parse_args(int argc, char **argv, args_options_t *options,
                char **output_file_name, char **input_file_name) {
  *options = 0;
  optind = 1;
  bool has_chosen_compression_direction = false;
  bool has_chosen_compression_format = false;
  bool has_set_output_file_name = false;
  *output_file_name = NULL;
  // *input_file_name = NULL;
  while (1) {
    // int this_option_optind = optind ? optind : 1;
    static struct option long_options[] = {
        {"rle", no_argument, 0, 'r'},
        {"fle", no_argument, 0, 'f'},
        {"compress", no_argument, 0, 'c'},
        {"decompress", no_argument, 0, 'd'},
        {"output_name", required_argument, 0, 'o'},
        {"help", no_argument, 0, 'h'},
        {"add", no_argument, 0, 1},
        {0, 0, 0, 0}};

    int c = getopt_long(argc, argv, ":rfcdho:", long_options, NULL);
    if (c == -1)
      break;
    switch (c) {
    case 'r':
      if (has_chosen_compression_format && (*options & USE_FLE)) {
        fprintf(stderr, "Only one compression format can be used at a time. "
                        "Two were given\n");
        exit(EXIT_FAILURE);
      }
      has_chosen_compression_format = true;
      break;

    case 'f':
      if (has_chosen_compression_format && !(*options & USE_FLE)) {
        fprintf(stderr, "Only one compression format can be used at a time. "
                        "Two were given\n");
        exit(EXIT_FAILURE);
      }
      *options |= USE_FLE;
      has_chosen_compression_format = true;
      break;

    case 'c':
      if (has_chosen_compression_direction && (*options & DECOMPRESS)) {
        fprintf(stderr, "Cannot both compress and decompress at once. Please "
                        "choose one option\n");
        exit(EXIT_FAILURE);
      }
      has_chosen_compression_direction = true;
      break;

    case 'd':
      if (has_chosen_compression_direction && !(*options & DECOMPRESS)) {
        fprintf(stderr, "Cannot both compress and decompress at once. Please "
                        "choose one option\n");
        exit(EXIT_FAILURE);
      }
      *options |= DECOMPRESS;
      has_chosen_compression_direction = true;
      break;

    case 'h':
      show_help();
      exit(EXIT_SUCCESS);
      break;

    case 'o':
      has_set_output_file_name = true;
      *output_file_name = optarg;
      break;

    case ':':
      fprintf(stderr, "Missing file name after [-o|--output_name]\n");
      exit(EXIT_FAILURE);
      break;

    case '?':
      fprintf(stderr, "\nUNKNOWN OPTION: ");
      if (optopt) {
        fprintf(stderr, "-%c\n", optopt);
      } else {
        fprintf(stderr, "%s\n", argv[optind - 1]);
      }
      fprintf(stderr, "\n");
      show_help();
      exit(EXIT_FAILURE);
      break;

    default:
      fprintf(stderr, "?? getopt returned character code 0%o ??\n", c);
      fprintf(stderr, "This error is unexpected. Aboting\n");
      exit(EXIT_FAILURE);
    }
  }

  if (!has_chosen_compression_format) {
    fprintf(stderr, "No compression format has been specified. Please use the "
                    "necessary options to specify whether run or fixed length "
                    "encoding should be used\n");
    show_help();
    exit(EXIT_FAILURE);
  }
  if (!has_chosen_compression_direction) {
    fprintf(stderr, "Please specify whether to compress or decompress the "
                    "file. Use the necessary options\n");
    show_help();
    exit(EXIT_FAILURE);
  }

  if (optind + 1 != argc) {
    fprintf(stderr, "Incorrect number of arguments\n");
    show_help();
    exit(EXIT_FAILURE);
  }

  *input_file_name = argv[optind];
}
