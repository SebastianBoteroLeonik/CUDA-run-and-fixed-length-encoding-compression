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

// Main data structure for fixed length encoding
struct fle_data {
  unsigned long
      total_data_length; /* the length of the uncompressed data in bytes */
  unsigned long number_of_chunks;    /* number of chunks used in compression */
  unsigned char *chunk_element_size; /* number of bits used by each byte after
                                        compression */
  unsigned char (*chunk_data)[1024]; /* the compression data */
};

// Main compression function
struct fle_data *fle_compress(unsigned char *binary_data,
                              unsigned long data_length);

// Main decompression function
unsigned char *fle_decompress(struct fle_data *compressed);

// Device allocator for struct fle_data
struct fle_data *make_device_fle_data(size_t number_of_chunks);

// Host allocator for struct fle_data
struct fle_data *make_host_fle_data(size_t number_of_chunks);

// Funtion for copying fle_data between device and host
void copy_fle_data(struct fle_data *src, struct fle_data *dst,
                   enum cpyKind kind);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // !FLE_H
