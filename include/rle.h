#ifndef RLE_H
#define RLE_H
#include <stdio.h>

// struct rle_chunks {
//   unsigned long *chunk_starts;
//   unsigned int *chunk_lengths;
// };

struct rle_data {
  unsigned int total_data_length;
  unsigned int compressed_array_length;
  // unsigned int number_of_chunks;
  unsigned char *repetitions;
  unsigned char *values;
  // struct rle_chunks *chunks;
};

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifndef CPYKIND
#define CPYKIND
enum cpyKind { HostToDevice, HostToHost, DeviceToHost };
#endif // !CPYKIND

// struct rle_chunks *make_device_rle_chunks(size_t number_of_chunks,
//                                           size_t chunk_capacity);
//
// struct rle_chunks *make_host_rle_chunks(size_t number_of_chunks,
//                                         size_t chunk_capacity);
//
// void copy_rle_chunks(struct rle_chunks *src, struct rle_chunks *dst,
//                      enum cpyKind kind, size_t number_of_chunks,
//                      ssize_t total_array_length);
struct rle_data *make_device_rle_data(unsigned int capacity);

struct rle_data *make_host_rle_data(unsigned int capacity);

void copy_rle_data(struct rle_data *src, struct rle_data *dst,
                   enum cpyKind kind, unsigned capacity);

struct rle_data *compress_rle(unsigned char *data, size_t data_len);

unsigned char *decompress_rle(struct rle_data *compressed_data);
#ifdef __cplusplus
}
#endif // __cplusplus

#endif // !RLE_H
