#include "gtest/gtest.h"
#include <gtest/gtest.h>
#include <stdio.h>
#include <unistd.h>

extern "C" {
#include "cli.h"
}
// Zrobione pod poprzednią wersję podawania argumentów
/*

TEST(argument_parsing, correct_execution_full) {
  const char *argv[] = {"frle", "-rco", "output", "input", NULL};
  const char argc = sizeof(argv) / sizeof(*argv) - 1;
  args_options_t options;
  char *output;
  char *input;
  parse_args(argc, (char **)argv, &options, &output, &input);
  EXPECT_TRUE(!options | USE_FLE);
  EXPECT_TRUE(!options | DECOMPRESS);
  EXPECT_STREQ(output, "output");
  EXPECT_STREQ(input, "input");
}

TEST(argument_parsing, correct_execution_no_output) {
  const char *argv[] = {"frle", "-rc", "input", NULL};
  const char argc = sizeof(argv) / sizeof(*argv) - 1;
  args_options_t options;
  char *output;
  char *input;
  parse_args(argc, (char **)argv, &options, &output, &input);
  EXPECT_TRUE(!options | USE_FLE);
  EXPECT_TRUE(!options | DECOMPRESS);
  EXPECT_EQ(output, nullptr);
  EXPECT_STREQ(input, "input");
}

TEST(argument_parsing, no_input_name) {
  const char *argv[] = {"frle", "-rc", NULL};
  const char argc = sizeof(argv) / sizeof(*argv) - 1;
  args_options_t options;
  char *output;
  char *input;
  ASSERT_EXIT(parse_args(argc, (char **)argv, &options, &output, &input),
              ::testing::ExitedWithCode(EXIT_FAILURE),
              "Incorrect number of arguments");
}

TEST(argument_parsing, missing_output_name) {
  const char *argv[] = {"frle", "-rco", NULL};
  const char argc = sizeof(argv) / sizeof(*argv) - 1;
  args_options_t options;
  char *output;
  char *input;
  ASSERT_EXIT(parse_args(argc, (char **)argv, &options, &output, &input),
              ::testing::ExitedWithCode(EXIT_FAILURE),
              "Missing file name after \\[-o|--output_name\\]");
}

TEST(argument_parsing, multiple_compression_formats) {
  const char *argv[] = {"frle", "-rfc", "input", NULL};
  const char argc = sizeof(argv) / sizeof(*argv) - 1;
  args_options_t options;
  char *output;
  char *input;
  ASSERT_EXIT(parse_args(argc, (char **)argv, &options, &output, &input),
              ::testing::ExitedWithCode(EXIT_FAILURE),
              "Only one compression format can be used at a time. "
              "Two were given");
}

TEST(argument_parsing, multiple_compression_directions) {
  const char *argv[] = {"frle", "-cd", "input", NULL};
  const char argc = sizeof(argv) / sizeof(*argv) - 1;
  args_options_t options;
  char *output;
  char *input;
  ASSERT_EXIT(parse_args(argc, (char **)argv, &options, &output, &input),
              ::testing::ExitedWithCode(EXIT_FAILURE),
              "Cannot both compress and decompress at once. Please "
              "choose one option");
}

TEST(argument_parsing, unknown_short_option) {
  const char *argv[] = {"frle", "-x", "input", NULL};
  const char argc = sizeof(argv) / sizeof(*argv) - 1;
  args_options_t options;
  char *output;
  char *input;
  ASSERT_EXIT(parse_args(argc, (char **)argv, &options, &output, &input),
              ::testing::ExitedWithCode(EXIT_FAILURE), "UNKNOWN OPTION: -x");
}

TEST(argument_parsing, unknown_long_option) {
  const char *argv[] = {"frle", "--foo", "input", NULL};
  const char argc = sizeof(argv) / sizeof(*argv) - 1;
  args_options_t options;
  char *output;
  char *input;
  ASSERT_EXIT(parse_args(argc, (char **)argv, &options, &output, &input),
              ::testing::ExitedWithCode(EXIT_FAILURE), "UNKNOWN OPTION: --foo");
}

TEST(argument_parsing, no_compression_format) {
  const char *argv[] = {"frle", "-c", "input", NULL};
  const char argc = sizeof(argv) / sizeof(*argv) - 1;
  args_options_t options;
  char *output;
  char *input;
  ASSERT_EXIT(parse_args(argc, (char **)argv, &options, &output, &input),
              ::testing::ExitedWithCode(EXIT_FAILURE),
              "No compression format has been specified");
}

TEST(argument_parsing, no_compression_direction) {
  const char *argv[] = {"frle", "-r", "input", NULL};
  const char argc = sizeof(argv) / sizeof(*argv) - 1;
  args_options_t options;
  char *output;
  char *input;
  ASSERT_EXIT(parse_args(argc, (char **)argv, &options, &output, &input),
              ::testing::ExitedWithCode(EXIT_FAILURE),
              "Please specify whether to compress or decompress the "
              "file. Use the necessary options");
}
*/
