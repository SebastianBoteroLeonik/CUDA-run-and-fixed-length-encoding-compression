#ifndef RLE_H
#define RLE_H
#include "define.h"
#include <stdio.h>

/***
 Main data structure for run length encoding
 @field aslnlasdn
 */
struct rle_data {
  unsigned int
      total_data_length; /* the length of the uncompressed data in bytes */
  unsigned int compressed_array_length; /* the length of the arrays used in the
                                           compression in bytes */
  unsigned char *repetitions; /* the number of repetitions of a value -1 (in
                                 order to utilize the value 0)*/
  unsigned char *values;      /* the value to be repeated */
};

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 A function that allocates struct rle_data on the device
 */
struct rle_data *make_device_rle_data(unsigned int capacity);

/*
 A function that allocates struct rle_data on the host
 */
struct rle_data *make_host_rle_data(unsigned int capacity);

#ifndef CPYKIND
#define CPYKIND
// A helper enum to specify the copy direction outside of cuda files
enum cpyKind { HostToDevice, HostToHost, DeviceToHost };
#endif // !CPYKIND

/**
 * A function to copy struct rle_data between devices
 * @param src - ads
 */
void copy_rle_data(struct rle_data *src, struct rle_data *dst,
                   enum cpyKind kind, unsigned capacity);

/***
 The compression function for rle.
 @param data -
 */
struct rle_data *compress_rle(unsigned char *data, size_t data_len);

unsigned char *decompress_rle(struct rle_data *compressed_data);
#ifdef __cplusplus
}
#endif // __cplusplus

#endif // !RLE_H
