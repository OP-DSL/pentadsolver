option(BUILD_BENCHMARKS "Build library for CPU architecture." OFF)

if(BUILD_BENCHMARKS)
  find_package(benchmark CONFIG REQUIRED)
  if(BUILD_FOR_CPU)
    add_subdirectory(cpu)
  endif(BUILD_FOR_CPU)

  if(BUILD_FOR_CUDA)
    add_subdirectory(cuda)
  endif(BUILD_FOR_CUDA)
endif(BUILD_BENCHMARKS)
