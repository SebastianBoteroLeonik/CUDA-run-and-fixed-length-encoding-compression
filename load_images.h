#ifndef IMAGE_LOAD_H
#define IMAGE_LOAD_H

typedef struct imgRawImage {
  unsigned int numComponents;
  unsigned long int width, height;

  unsigned char *lpData;
} imgRawImage_t;

/**
 * copied from https://www.tspi.at/2020/03/20/libjpegexample.html#gsc.tab=0
 */

struct imgRawImage *loadJpegImageFile(char *lpFilename);

int storeJpegImageFile(struct imgRawImage *lpImage, char *lpFilename);

#endif
