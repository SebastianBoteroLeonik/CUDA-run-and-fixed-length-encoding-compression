#ifndef FLE_H
#define FLE_H
#include "define.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifndef CPYKIND
#define CPYKIND
enum cpyKind { HostToDevice, HostToHost, DeviceToHost };
#endif // !CPYKIND

struct fle_data {
  unsigned long total_data_length;
  unsigned long number_of_chunks;
  unsigned char *chunk_element_size;
  unsigned char (*chunk_data)[1024];
};

struct fle_data *fle_compress(unsigned char *binary_data,
                              unsigned long data_length);

unsigned char *fle_decompress(struct fle_data *compressed);

struct fle_data *make_device_fle_data(size_t number_of_chunks);

struct fle_data *make_host_fle_data(size_t number_of_chunks);

void copy_fle_data(struct fle_data *src, struct fle_data *dst,
                   enum cpyKind kind);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // !FLE_H
