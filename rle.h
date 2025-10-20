#ifndef RLE_H
#define RLE_H

struct rle_chunk {
  unsigned char *lengths;
  unsigned char *values;
  unsigned int array_lenght;
};

struct rle_data {
  struct rle_chunk *chunks;
  unsigned int number_of_chunks;
};
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
struct rle_data *compress_rle(unsigned char *data, unsigned int data_len,
                              unsigned char jump_len);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // !RLE_H
