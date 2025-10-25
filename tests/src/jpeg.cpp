#include <cstdlib>
#include <gtest/gtest.h>

extern "C" {
#include "load_images.h"
}

TEST(jpeg, load_non_existant_file) {
  char false_file_name[5] = "file";
  imgRawImage_t *img = loadJpegImageFile(false_file_name);
  EXPECT_FALSE(img);
}

TEST(jpeg, load_good_file) {
  char good_file_name[30] = "test_data/graphic.jpeg";
  imgRawImage_t *img = loadJpegImageFile(good_file_name);
  EXPECT_NE(img, nullptr);
  // std::cout << "height: " << img->height << " wifth: " << img->width
  //           << " channels: " << img->numComponents;
  EXPECT_EQ(img->height, 2627);
  EXPECT_EQ(img->width, 4348);
  EXPECT_EQ(img->numComponents, 3);
  EXPECT_NE(img->lpData, nullptr);
  free(img->lpData);
  free(img);
}

TEST(jpeg, load_and_store_good_file) {
  char good_file_name[30] = "test_data/graphic.jpeg";
  imgRawImage_t *img = loadJpegImageFile(good_file_name);
  EXPECT_NE(img, nullptr);
  // std::cout << "height: " << img->height << " wifth: " << img->width
  //           << " channels: " << img->numComponents;
  EXPECT_EQ(img->height, 2627);
  EXPECT_EQ(img->width, 4348);
  EXPECT_EQ(img->numComponents, 3);
  EXPECT_NE(img->lpData, nullptr);
  char store_name[30] = "test_outputs/stored.jpeg";
  storeJpegImageFile(img, store_name);
  imgRawImage_t *img2 = loadJpegImageFile(store_name);
  EXPECT_NE(img2, nullptr);
  EXPECT_EQ(img2->height, 2627);
  EXPECT_EQ(img2->width, 4348);
  EXPECT_EQ(img2->numComponents, 3);
  EXPECT_NE(img2->lpData, nullptr);
  free(img->lpData);
  free(img);
  free(img2->lpData);
  free(img2);
}
