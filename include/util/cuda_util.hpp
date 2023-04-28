#ifndef PENTADSOLVER_CUDA_UTIL_HPP
#define PENTADSOLVER_CUDA_UTIL_HPP

#include <cstdio>

#define CHECK_CUDA(XXX)                                                        \
  do {                                                                         \
    hipError_t CHECK_CUDART_res = XXX;                                        \
    if (CHECK_CUDART_res != hipSuccess) {                                     \
      auto s = hipGetErrorString(CHECK_CUDART_res);                           \
      printf("cudart error in '" #XXX "' - %d: %s\n", CHECK_CUDART_res, s);    \
      exit(3);                                                                 \
    }                                                                          \
  } while (0)

#endif /* ifndef PENTADSOLVER_CUDA_UTIL_HPP */
