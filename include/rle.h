#ifndef RLE_H
#define RLE_H

struct rle_chunk {
  unsigned int array_length;
  unsigned char lengths[1024];
  unsigned char values[1024];
};

struct rle_data {
  unsigned int number_of_chunks;
  struct rle_chunk *chunks;
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
