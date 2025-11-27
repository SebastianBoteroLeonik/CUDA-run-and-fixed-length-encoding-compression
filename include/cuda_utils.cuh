#ifndef RLE_UTILS
#define RLE_UTILS
#define BLOCK_SIZE 1024
#define WARP_SIZE 32

__device__ int warp_cumsum(int val, unsigned int mask);

__device__ int block_cumsum(int val);

__host__ void recursive_cumsum(unsigned int *array, unsigned int array_len);

#define CEIL_DEV(num, div) (((num) / (div)) + ((num) % (div) != 0))

#define CUDA_ERROR_CHECK(expr)                                                 \
  do {                                                                         \
    cudaError_t cudaStatus = expr;                                             \
    if (cudaStatus != cudaSuccess) {                                           \
      fprintf(stderr, "%s failed! At line %d, in %s\nError: %s\n\t %s\n",      \
              #expr, __LINE__, __FILE__, cudaGetErrorName(cudaStatus),         \
              cudaGetErrorString(cudaStatus));                                 \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define PERFORMANCE_TEST

#ifdef PERFORMANCE_TEST

#define INITIALIZE_CUDA_PERFORMANCE_CHECK(capacity)                            \
  cudaEvent_t __cuda_performance_test_events[capacity];                        \
  int __cuda_performance_test_counter = 0;                                     \
  int __cuda_performance_test_capacity = capacity;                             \
  char *__cuda_performance_test_names[capacity];                               \
  for (int i = 0; i < capacity; i++) {                                         \
    CUDA_ERROR_CHECK(cudaEventCreate(&(__cuda_performance_test_events[i])));   \
  }

#define CUDA_PERFORMANCE_CHECKPOINT(name)                                      \
  if (__cuda_performance_test_counter >= __cuda_performance_test_capacity) {   \
    fprintf(stderr,                                                            \
            "Performance test capacity overflow detected!\n"                   \
            "counter = %d, capacity = %d. Please adjust the capacity.\n"       \
            "Occured at line %d in %s",                                        \
            __cuda_performance_test_counter, __cuda_performance_test_capacity, \
            __LINE__, __FILE__);                                               \
  }                                                                            \
  __cuda_performance_test_names[__cuda_performance_test_counter] =             \
      (char *)#name;                                                           \
  CUDA_ERROR_CHECK(cudaEventRecord(                                            \
      __cuda_performance_test_events[__cuda_performance_test_counter++]));

#define PRINT_AND_TERMINATE_CUDA_PERFORMANCE_CHECK()                           \
  CUDA_ERROR_CHECK(cudaEventSynchronize(                                       \
      __cuda_performance_test_events[--__cuda_performance_test_counter]));     \
  for (int i = 0; i < __cuda_performance_test_counter; i++) {                  \
    float time;                                                                \
    CUDA_ERROR_CHECK(                                                          \
        cudaEventElapsedTime(&time, __cuda_performance_test_events[i],         \
                             __cuda_performance_test_events[i + 1]));          \
    printf("Time elapsed in %s --> %s: %f ms\n",                               \
           __cuda_performance_test_names[i],                                   \
           __cuda_performance_test_names[i + 1], time);                        \
  }                                                                            \
  for (int i = 0; i < __cuda_performance_test_capacity; i++) {                 \
    CUDA_ERROR_CHECK(cudaEventDestroy(__cuda_performance_test_events[i]));     \
  }

#else

#define INITIALIZE_CUDA_PERFORMANCE_CHECK(capacity)
#define CUDA_PERFORMANCE_CHECKPOINT(name)
#define PRINT_AND_TERMINATE_CUDA_PERFORMANCE_CHECK()

#endif /* ifdef PERFORMANCE_TEST */

// void f() {
//   // init
// #define capacity 10
//   cudaEvent_t __cuda_performance_test_events[capacity];
//   int __cuda_performance_test_counter = 0;
//   char *__cuda_performance_test_names[capacity];
//   for (int i = 0; i < capacity; i++) {
//     CUDA_ERROR_CHECK(cudaEventCreate(&(__cuda_performance_test_events[i])));
//   }
//   // checkpoint
// #define name "name"
//   __cuda_performance_test_names[__cuda_performance_test_counter]
//   = (char
//   *)name; CUDA_ERROR_CHECK(cudaEventRecord(
//       __cuda_performance_test_events[__cuda_performance_test_counter++]));
//   // terminate
//   CUDA_ERROR_CHECK(cudaEventSynchronize(
//       __cuda_performance_test_events[--__cuda_performance_test_counter]));
//   for (int i = 0; i < __cuda_performance_test_counter; i++) {
//     float time;
//     CUDA_ERROR_CHECK(
//         cudaEventElapsedTime(&time,
//         __cuda_performance_test_events[i],
//                              __cuda_performance_test_events[i +
//                              1]));
//     printf("Time elapsed in %s --> %s: %f ms\n",
//            __cuda_performance_test_names[i],
//            __cuda_performance_test_names[i + 1], time);
//   }
//   for (int i = 0; i < capacity; i++) {
//     CUDA_ERROR_CHECK(cudaEventDestroy(__cuda_performance_test_events[i]));
//   }
// }

#endif // !RLE_UTILS
