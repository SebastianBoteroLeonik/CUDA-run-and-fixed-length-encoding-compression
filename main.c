#include "load_images.h"
#include "rle.h"
// #include <stdio.h>
// #include <stdlib.h>

int main(int argc, char *argv[]) {

// imgRawImage_t *img = loadJpegImageFile("sample-images/docs/image-1.jpg");
// compress_rle(img->lpData, img->height * img->width, 1);
#define LEN 1024
  unsigned char data[LEN];
  for (int i = 0; i < LEN; i++) {
    data[i] = i / 3;
  }
  compress_rle(data, LEN, 1);
  // storeJpegImageFile(img, "new_image.jpg");
}
