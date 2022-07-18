option(BUILD_FOR_CPU "Build library for CPU architecture." ON)
option(BUILD_FOR_CUDA "Build library for CUDA architecture." OFF)
option(BUILD_FOR_MPI "Build distributer library." OFF)

if(BUILD_FOR_CPU)
  add_subdirectory(cpu)
endif(BUILD_FOR_CPU)

if(BUILD_FOR_CUDA)
  add_subdirectory(cuda)
endif(BUILD_FOR_CUDA)
