option(BUILD_BENCHMARKS "Build library for CPU architecture." OFF)

if(BUILD_BENCHMARKS)
  find_package(benchmark CONFIG REQUIRED)
  if(BUILD_FOR_CPU)
    add_subdirectory(cpu)
  endif(BUILD_FOR_CPU)

  if(BUILD_FOR_CUDA)
    add_subdirectory(cuda)
  endif(BUILD_FOR_CUDA)

  if(BUILD_FOR_MPI)
    if(BUILD_FOR_CPU)
      add_executable(cpu_perf_mpi performance_test.cpp)
      target_link_libraries(cpu_perf_mpi PRIVATE pentadsolver::project_warnings pentadsolver::project_options
                                                 pentadsolver::mpi_cpu)

      target_include_directories(cpu_perf_mpi PUBLIC ${PROJECT_SOURCE_DIR})
    endif(BUILD_FOR_CPU)

    if(BUILD_FOR_CUDA)
      add_executable(cuda_perf_mpi performance_test.cu)
      target_link_cuda(cuda_perf_mpi)
      target_link_libraries(
        cuda_perf_mpi
        PRIVATE OpenMP::OpenMP_CXX
                pentadsolver::project_warnings
                pentadsolver::project_options
                pentadsolver::mpi_cuda)
      target_compile_definitions(cuda_perf_mpi PRIVATE PENTAD_PERF_CUDA)
    endif(BUILD_FOR_CUDA)
  endif(BUILD_FOR_MPI)
endif(BUILD_BENCHMARKS)
