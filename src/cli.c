#include "cli.h"
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>

#define ERR(source)                                                            \
  (fprintf(stderr, "%s:%d\n", __FILE__, __LINE__), perror(source),             \
   exit(EXIT_FAILURE))

void show_help() {
  char help_file_name[30] = "doc/HELP.txt";
  FILE *help_file = fopen(help_file_name, "r");
  if (!help_file) {
    ERR("fopen(help_file)");
  }
  char buf[50];
  int read_count;
  while ((read_count = fread(buf, 1, sizeof(buf), help_file)) > 0) {
    fwrite(buf, 1, read_count, stdout);
  }
  fclose(help_file);
}

void parse_args(int argc, char **argv, args_options_t *options) {
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
      printf("option r [TODO]\n");
      break;

    case 'f':
      printf("option f [TODO]\n");
      break;

    case 'c':
      printf("option c [TODO]\n");
      break;

    case 'd':
      printf("option d [TODO]\n");
      break;

    case 'h':
      // printf("option h [TODO]\n");
      show_help();
      exit(EXIT_SUCCESS);
      break;

    case 'o':
      printf("option o with value '%s' [TODO]\n", optarg);
      break;

    case ':':
      printf("Missing argument [TODO]\n");
      break;

    case '?':
      printf("\n");
      if (optopt) {
        printf("UNKNOWN OPTION: -%c\n", optopt);
      } else {
        printf("UNKNOWN OPTION: %s\n", argv[optind - 1]);
      }
      printf("\n");
      show_help();
      exit(EXIT_FAILURE);
      break;

    default:
      fprintf(stderr, "?? getopt returned character code 0%o ??\n", c);
      fprintf(stderr, "This error is unexpected. Aboting\n");
      exit(EXIT_FAILURE);
    }
  }

  if (optind < argc) {
    printf("non-option ARGV-elements: ");
    while (optind < argc)
      printf("%s ", argv[optind++]);
    printf("\n");
  }
}
