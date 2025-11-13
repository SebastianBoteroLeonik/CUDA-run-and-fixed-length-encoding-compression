#include "cli.h"
#include "file_io.h"
#include "load_images.h"
#include "rle.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  // show_help();
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
      printf("decompressing fle\n");
    } else {
      printf("compressing fle\n");
    }
  } else {
    if (options & DECOMPRESS) {
      printf("decompressing rle\n");
    } else {
      if (!output_file_name) {
        output_file_name = "output.rle";
      }
      printf("compressing rle\n");
      int data_length;
      unsigned char *binary_data =
          read_binary_file(input_file_name, &data_length);
      printf("data_length = %d\n", data_length);
      struct rle_data *rle = compress_rle(binary_data, data_length);
      write_rle_to_file(rle, output_file_name);
    }
  }
  // imgRawImage_t *img = loadJpegImageFile("sample-images/docs/image-1.jpg");
  // // imgRawImage_t *img = loadJpegImageFile("graphic.jpeg");
  // struct rle_data *compressed =
  //     compress_rle(img->lpData, img->height * img->width);
  // FILE *outfile = fopen("outfile.rle", "wb");
  // char channels = 3;
  // // channels
  // fwrite(&(channels), sizeof(channels), 1, outfile);
  //
  // // R G B sizes
  // for (int i = 0; i < 3; i++) {
  //   unsigned int sum = 0;
  //   for (int j = 0; j < compressed[i].number_of_chunks; j++) {
  //     sum += compressed[i].chunks[j].array_length;
  //   }
  //   fwrite(&(sum), sizeof(sum), 1, outfile);
  //   printf("sum = %d\n", sum);
  // }
  // for (int i = 0; i < 3; i++) {
  //   // R G B values
  //   for (int j = 0; j < compressed[i].number_of_chunks; j++) {
  //     fwrite(compressed[i].chunks[j].values, sizeof(unsigned char),
  //            compressed[i].chunks[j].array_length, outfile);
  //   }
  //   // R G B lengths
  //   printf("compressed[i].chunks[0].lengths[0] = %d\n",
  //          compressed[i].chunks[0].lengths[0]);
  //   for (int j = 0; j < compressed[i].number_of_chunks; j++) {
  //     fwrite(compressed[i].chunks[j].lengths, sizeof(unsigned char),
  //            compressed[i].chunks[j].array_length, outfile);
  //   }
  // }
  // fclose(outfile);

  // #define LEN 1024
  //   unsigned char data[LEN];
  //   for (int i = 0; i < LEN; i++) {
  //     data[i] = rand() % 10 == 0;
  //   }
  //   compress_rle(data, LEN, 1);
  //   printf("\ndata: [");
  //   for (int i = 0; i < LEN; i++) {
  //     printf("%d, ", data[i]);
  //   }
  //   printf("]\n");
  // storeJpegImageFile(img, "new_image.jpg");
}
