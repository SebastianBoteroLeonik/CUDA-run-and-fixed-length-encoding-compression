#ifndef CLI_H
#define CLI_H

typedef enum args_options { USE_FLE = 0b1, DECOMPRESS = 0b10 } args_options_t;

void show_help();

void parse_args(int argc, char **argv, enum args_options *options);

#endif // !CLI_H
