#include "cuda_utils.cuh"
#include "rle.h"
#include <cuda.h>
#include <device_launch_parameters.h>
#include <pthread.h>
#include <stdio.h>

__global__ void
find_differing_neighbours(unsigned char *data,
                          unsigned int *diff_from_prev_indicators, size_t len) {
  const int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_thread_id >= len) {
    return;
  }
  diff_from_prev_indicators[global_thread_id] =
      global_thread_id ? (data[global_thread_id] != data[global_thread_id - 1])
                       : 0;
}

__global__ void find_segment_end(unsigned int *scan_result,
                                 unsigned *segment_ends, size_t len) {

  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_thread_id >= len) {
    return;
  }
  if (global_thread_id == len - 1) {
    segment_ends[scan_result[global_thread_id]] = global_thread_id;
  } else if (scan_result[global_thread_id] !=
             scan_result[global_thread_id + 1]) {
    segment_ends[scan_result[global_thread_id]] = global_thread_id;
  }
}

__global__ void subtract_segment_begining(unsigned int *scan_result,
                                          unsigned *segment_lengths_out,
                                          unsigned int *overflows,
                                          unsigned char *data,
                                          unsigned char *compressed_data_vals,
                                          size_t len) {

  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_thread_id >= len) {
    return;
  }
  int this_val = scan_result[global_thread_id];
  int previous_val;
  if (global_thread_id) {
    previous_val = scan_result[global_thread_id - 1];
  } else {
    previous_val = -1;
  }
  if (previous_val != this_val) {
    segment_lengths_out[this_val] -= global_thread_id;
    overflows[this_val] = segment_lengths_out[this_val] / 256;
    compressed_data_vals[this_val] = data[global_thread_id];
  }
}

__global__ void write_rle(unsigned char *values, unsigned int *og_lengths,
                          unsigned int *overflows, struct rle_data *rle,
                          unsigned int len) {
  const long long global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_thread_id >= len) {
    return;
  }
  unsigned int offset;
  if (global_thread_id == 0) {
    offset = 0;
  } else {
    offset = overflows[global_thread_id - 1];
  }
  unsigned int og_length = og_lengths[global_thread_id];
  unsigned char value = values[global_thread_id];
  int i = 0;
  for (; i < og_length / 256; i++) {
    rle->values[global_thread_id + offset + i] = value;
    rle->repetitions[global_thread_id + offset + i] = 255;
  }
  rle->values[global_thread_id + offset + i] = value;
  rle->repetitions[global_thread_id + offset + i] = og_length % 256;
  rle->compressed_array_length = len + overflows[len - 1];
}

struct rle_data_list_node {
  struct rle_data *dev_rle;
  int compressed_len;
  struct rle_data_list_node *next;
};

struct pthread_shared_data {
  int copied_bytes;
  bool all_copied;
  pthread_mutex_t copy_mtx;
  pthread_cond_t copy_cond;

  struct rle_data_list_node *compressed_head;
  struct rle_data_list_node *compressed_tail;
  bool all_compressed;
  pthread_mutex_t compression_mtx;
  pthread_cond_t compression_cond;

  unsigned char *data;
  unsigned char *dev_data;
  size_t data_len;

  int default_step_size;

  struct rle_data *host_rle;
  int compressed_len_total;
};

void *binary_copier_thread(void *vptr) {
  struct pthread_shared_data *shared_data = (struct pthread_shared_data *)vptr;
  cudaStream_t stream;
  CUDA_ERROR_CHECK(cudaStreamCreate(&stream));

  CUDA_ERROR_CHECK(
      cudaMalloc(&shared_data->dev_data,
                 sizeof(*shared_data->data) * shared_data->data_len));

  int step_size = shared_data->default_step_size;
  for (int offset = 0; offset < shared_data->data_len; offset += step_size) {
    if (offset + step_size > shared_data->data_len) {
      step_size = shared_data->data_len % step_size;
    }
    CUDA_ERROR_CHECK(cudaMemcpyAsync(shared_data->dev_data + offset,
                                     shared_data->data + offset,
                                     sizeof(*shared_data->data) * step_size,
                                     cudaMemcpyHostToDevice, stream));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    pthread_mutex_lock(&shared_data->copy_mtx);
    shared_data->copied_bytes += step_size;
    if (offset + step_size >= shared_data->data_len) {
      shared_data->all_copied = true;
    }
    pthread_cond_broadcast(&shared_data->copy_cond);
    pthread_mutex_unlock(&shared_data->copy_mtx);
    // fprintf(stderr, "Copied onto gpu %d bytes out of %lu\n",
    //         shared_data->copied_bytes, shared_data->data_len);
  }

  CUDA_ERROR_CHECK(cudaStreamDestroy(stream));
  return NULL;
}

// __global__ void verify_dev_rle(struct rle_data *rle, int offset) {
//   int reps = threadIdx.x ? 14 : 0;
//   int vals = threadIdx.x ? 2 : 3;
//   if (rle->repetitions[threadIdx.x] != reps) {
//     printf("rle->reps[%d] = %d; should be %d; offset: %d\n", threadIdx.x,
//            rle->repetitions[threadIdx.x], reps, offset);
//   }
//   if (rle->values[threadIdx.x] != vals) {
//     printf("rle->vals[%d] = %d; should be %d; offset: %d\n", threadIdx.x,
//            rle->values[threadIdx.x], vals, offset);
//   }
//   // printf("val[0]: %d\n", rle->values[0]);
//   // printf("val[1]: %d\n", rle->values[1]);
//   // printf("reps[0]: %d\n", rle->repetitions[0]);
//   // printf("reps[1]: %d\n", rle->repetitions[1]);
// }

void *compressor_thread(void *vptr) {
  struct pthread_shared_data *shared_data = (struct pthread_shared_data *)vptr;
  cudaStream_t stream;
  CUDA_ERROR_CHECK(cudaStreamCreate(&stream));

  int processed_bytes = 0;
  int copied_bytes = 0;
  unsigned int *scan_array;
  CUDA_ERROR_CHECK(
      cudaMalloc(&scan_array, sizeof(*scan_array) * shared_data->data_len));

  bool all_copied = false;
  do {
    pthread_mutex_lock(&shared_data->copy_mtx);
    if (shared_data->copied_bytes <= processed_bytes) {
      pthread_cond_wait(&shared_data->copy_cond, &shared_data->copy_mtx);
    }
    all_copied = shared_data->all_copied;
    copied_bytes = shared_data->copied_bytes;
    pthread_mutex_unlock(&shared_data->copy_mtx);
    if (shared_data->data_len == processed_bytes) {
      break;
    }

    int step_size = copied_bytes - processed_bytes;

    int number_of_blocks = CEIL_DEV(step_size, BLOCK_SIZE);

    find_differing_neighbours<<<number_of_blocks, BLOCK_SIZE, 0, stream>>>(
        shared_data->dev_data + processed_bytes, scan_array, step_size);
    recursive_cumsum(scan_array, step_size, stream);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    unsigned int compressed_len;
    CUDA_ERROR_CHECK(cudaMemcpyAsync(
        &compressed_len, &(scan_array[step_size - 1]), sizeof(compressed_len),
        cudaMemcpyDeviceToHost, stream));
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    compressed_len++;
    unsigned int *og_lengths;
    CUDA_ERROR_CHECK(
        cudaMalloc(&og_lengths, sizeof(*og_lengths) * compressed_len));

    find_segment_end<<<number_of_blocks, BLOCK_SIZE, 0, stream>>>(
        scan_array, og_lengths, step_size);
    unsigned int *overflows;
    unsigned char *values;
    CUDA_ERROR_CHECK(
        cudaMalloc(&overflows, sizeof(*overflows) * compressed_len));
    CUDA_ERROR_CHECK(cudaMalloc(&values, sizeof(*values) * compressed_len));
    subtract_segment_begining<<<number_of_blocks, BLOCK_SIZE, 0, stream>>>(
        scan_array, og_lengths, overflows,
        shared_data->dev_data + processed_bytes, values, step_size);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    recursive_cumsum(overflows, compressed_len, stream);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    unsigned int last_overflow;
    CUDA_ERROR_CHECK(
        cudaMemcpyAsync(&last_overflow, &(overflows[compressed_len - 1]),
                        sizeof(last_overflow), cudaMemcpyDeviceToHost, stream));
    struct rle_data *dev_rle =
        make_device_rle_data(compressed_len + last_overflow);
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    write_rle<<<number_of_blocks, BLOCK_SIZE, 0, stream>>>(
        values, og_lengths, overflows, dev_rle, compressed_len);
    struct rle_data_list_node *new_node =
        (struct rle_data_list_node *)malloc(sizeof(*new_node));
    new_node->next = NULL;
    new_node->dev_rle = dev_rle;
    new_node->compressed_len = compressed_len + last_overflow;
    CUDA_ERROR_CHECK(cudaStreamSynchronize(stream));
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());
    // verify_dev_rle<<<1, 2>>>(dev_rle, 0);
    processed_bytes += step_size;
    pthread_mutex_lock(&shared_data->compression_mtx);
    if (!shared_data->compressed_head) {
      shared_data->compressed_head = new_node;
    }
    if (shared_data->compressed_tail) {
      shared_data->compressed_tail->next = new_node;
    }
    shared_data->compressed_tail = new_node;
    shared_data->all_compressed = processed_bytes == shared_data->data_len;
    shared_data->compressed_len_total += compressed_len + last_overflow;
    if (shared_data->all_compressed) {
      pthread_cond_broadcast(&shared_data->compression_cond);
    }
    pthread_mutex_unlock(&shared_data->compression_mtx);
    fprintf(stderr,
            "processed %d bytes out of %lu; this step was %d bytes "
            "long\n"
            "all_copied: %d\nall_compressed: %d\n",
            processed_bytes, shared_data->data_len, step_size, all_copied,
            shared_data->all_compressed);
    cudaFree(overflows);
    cudaFree(values);
    cudaFree(og_lengths);
  } while (!all_copied);

  CUDA_ERROR_CHECK(cudaStreamDestroy(stream));
  return NULL;
}

void *rle_copier_thread(void *vptr) {
  struct pthread_shared_data *shared_data = (struct pthread_shared_data *)vptr;
  cudaStream_t stream;
  CUDA_ERROR_CHECK(cudaStreamCreate(&stream));
  struct rle_data *dev_rle;
  int dev_compressed_len;
  int host_compressed_len = 0;
  bool all_compressed = false;
  pthread_mutex_lock(&shared_data->compression_mtx);
  if (!shared_data->all_compressed) {
    pthread_cond_wait(&shared_data->compression_cond,
                      &shared_data->compression_mtx);
  }
  shared_data->host_rle = make_host_rle_data(shared_data->compressed_len_total);
  pthread_mutex_unlock(&shared_data->compression_mtx);
  struct rle_data_list_node *list_node = shared_data->compressed_head;
  while (list_node) {
    dev_rle = list_node->dev_rle;
    dev_compressed_len = list_node->compressed_len;
    struct rle_data_list_node *next_list_node = list_node->next;
    free(list_node);
    list_node = next_list_node;
    if (list_node) {
      fprintf(stderr, "more\n");
    } else {
      fprintf(stderr, "end\n");
    }
    fprintf(stderr, "host_compressed_len: %d\n", host_compressed_len);
    fprintf(stderr, "comp_tot_len: %d\n", shared_data->compressed_len_total);
    fprintf(stderr, "chunk_len: %d\n", dev_compressed_len);
    // verify_dev_rle<<<1, 2>>>(dev_rle, 0);
    // if (shared_data->host_rle->repetitions == shared_data->host_rle->values)
    // {
    //   fprintf(stderr, "same vec\n");
    //   fprintf(stderr, "comp_len: %d\n", shared_data->compressed_len_total);
    //   exit(EXIT_FAILURE);
    // }
    struct rle_data dummy;
    CUDA_ERROR_CHECK(
        cudaMemcpy(&dummy, dev_rle, sizeof(dummy), cudaMemcpyDeviceToHost));

    CUDA_ERROR_CHECK(
        cudaMemcpy(shared_data->host_rle->values + host_compressed_len,
                   dummy.values, dev_compressed_len, cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaMemcpy(
        shared_data->host_rle->repetitions + host_compressed_len,
        dummy.repetitions, dev_compressed_len, cudaMemcpyDeviceToHost));
    // copy_rle_data(dev_rle, shared_data->host_rle, DeviceToHost,
    //               dev_compressed_len);
    // printf("val[0]: %d\n", shared_data->host_rle->values[0]);
    // printf("val[1]: %d\n", shared_data->host_rle->values[1]);
    // printf("reps[0]: %d\n", shared_data->host_rle->repetitions[0]);
    // printf("reps[1]: %d\n", shared_data->host_rle->repetitions[1]);
    host_compressed_len += dev_compressed_len;
    CUDA_ERROR_CHECK(cudaFree(dev_rle));
  }
  shared_data->host_rle->compressed_array_length = host_compressed_len;
  shared_data->host_rle->total_data_length = shared_data->data_len;

  CUDA_ERROR_CHECK(cudaStreamDestroy(stream));
  return NULL;
}

__host__ struct rle_data *compress_rle(unsigned char *data, size_t data_len) {
  struct pthread_shared_data shared_data;
  shared_data.copied_bytes = 0;
  shared_data.all_copied = false;
  shared_data.copy_mtx = PTHREAD_MUTEX_INITIALIZER;
  shared_data.copy_cond = PTHREAD_COND_INITIALIZER;
  shared_data.compressed_tail = NULL;
  shared_data.compressed_head = NULL;
  shared_data.all_compressed = false;
  shared_data.compression_mtx = PTHREAD_MUTEX_INITIALIZER;
  shared_data.compression_cond = PTHREAD_COND_INITIALIZER;
  shared_data.default_step_size = 1 << 14;
  shared_data.compressed_len_total = 0;
  shared_data.data_len = data_len;
  shared_data.data = data;
  pthread_t byte_copier, compressor, rle_copier;
  pthread_create(&byte_copier, NULL, binary_copier_thread, &shared_data);
  pthread_create(&compressor, NULL, compressor_thread, &shared_data);
  pthread_create(&rle_copier, NULL, rle_copier_thread, &shared_data);
  pthread_join(byte_copier, NULL);
  pthread_join(compressor, NULL);
  pthread_join(rle_copier, NULL);
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  return shared_data.host_rle;
}

__host__ struct rle_data *compress_rle2(unsigned char *data, size_t data_len) {
  INITIALIZE_CUDA_PERFORMANCE_CHECK(20)
  int number_of_blocks = CEIL_DEV(data_len, BLOCK_SIZE);

  unsigned char *dev_data;
  CUDA_PERFORMANCE_CHECKPOINT(binary_data_alloc)
  CUDA_ERROR_CHECK(cudaMalloc(&dev_data, sizeof(*data) * data_len));
  CUDA_PERFORMANCE_CHECKPOINT(binary_data_memcpy)
  CUDA_ERROR_CHECK(cudaMemcpy(dev_data, data, sizeof(*data) * data_len,
                              cudaMemcpyHostToDevice));

  CUDA_PERFORMANCE_CHECKPOINT(malloc_scan_array)
  unsigned int *scan_array;
  CUDA_ERROR_CHECK(cudaMalloc(&scan_array, sizeof(*scan_array) * data_len));
  CUDA_PERFORMANCE_CHECKPOINT(diff_kernel)
  find_differing_neighbours<<<number_of_blocks, BLOCK_SIZE>>>(
      dev_data, scan_array, data_len);

  CUDA_PERFORMANCE_CHECKPOINT(recursive_cumsum)
  // CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  recursive_cumsum(scan_array, data_len);
  unsigned int compressed_len;
  CUDA_PERFORMANCE_CHECKPOINT(memcpy_comp_len)
  CUDA_ERROR_CHECK(cudaMemcpy(&compressed_len, &(scan_array[data_len - 1]),
                              sizeof(compressed_len), cudaMemcpyDeviceToHost));
  compressed_len++;
  unsigned int *og_lengths;
  CUDA_PERFORMANCE_CHECKPOINT(og_len_malloc)
  CUDA_ERROR_CHECK(
      cudaMalloc(&og_lengths, sizeof(*og_lengths) * compressed_len));
  CUDA_PERFORMANCE_CHECKPOINT(find_end)
  find_segment_end<<<number_of_blocks, BLOCK_SIZE>>>(scan_array, og_lengths,
                                                     data_len);
  // CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  unsigned int *overflows;
  unsigned char *values;
  CUDA_PERFORMANCE_CHECKPOINT(malloc_overflows_and_vals)
  CUDA_ERROR_CHECK(cudaMalloc(&overflows, sizeof(*overflows) * compressed_len));
  CUDA_ERROR_CHECK(cudaMalloc(&values, sizeof(*values) * compressed_len));
  CUDA_PERFORMANCE_CHECKPOINT(sub_begining)
  subtract_segment_begining<<<number_of_blocks, BLOCK_SIZE>>>(
      scan_array, og_lengths, overflows, dev_data, values, data_len);
  // CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  CUDA_PERFORMANCE_CHECKPOINT(recursive_cumsum_overflows)
  recursive_cumsum(overflows, compressed_len);
  CUDA_PERFORMANCE_CHECKPOINT(after_recursive_cumsum_overflows)
  CUDA_ERROR_CHECK(cudaFree(dev_data));
  CUDA_PERFORMANCE_CHECKPOINT(rle_malloc)

  unsigned int last_overflow;
  CUDA_ERROR_CHECK(cudaMemcpy(&last_overflow, &(overflows[compressed_len - 1]),
                              sizeof(last_overflow), cudaMemcpyDeviceToHost));
  struct rle_data *dev_rle =
      make_device_rle_data(compressed_len + last_overflow);
  CUDA_PERFORMANCE_CHECKPOINT(write_rle)
  write_rle<<<number_of_blocks, BLOCK_SIZE>>>(values, og_lengths, overflows,
                                              dev_rle, compressed_len);
  CUDA_PERFORMANCE_CHECKPOINT(after_write_rle)
  CUDA_ERROR_CHECK(cudaDeviceSynchronize());
  struct rle_data *out_rle = make_host_rle_data(compressed_len + last_overflow);

  CUDA_PERFORMANCE_CHECKPOINT(before_rle_copy)
  copy_rle_data(dev_rle, out_rle, DeviceToHost, compressed_len + last_overflow);

  CUDA_PERFORMANCE_CHECKPOINT(after_rle_copy)
  CUDA_ERROR_CHECK(cudaFree(dev_rle));
  PRINT_AND_TERMINATE_CUDA_PERFORMANCE_CHECK()
  out_rle->total_data_length = data_len;
  out_rle->compressed_array_length = compressed_len + last_overflow;
  return out_rle;
}
