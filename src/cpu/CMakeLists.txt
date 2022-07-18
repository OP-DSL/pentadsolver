add_library(cpu ./singlenode/pentad_cpu.cpp)
add_library(${PROJECT_NAME}::cpu ALIAS cpu)

target_link_libraries(cpu PRIVATE OpenMP::OpenMP_CXX ${PROJECT_NAME}::project_options ${PROJECT_NAME}::project_warnings)

target_include_directories(cpu PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                      $<INSTALL_INTERFACE:${PROJECT_NAME}/include>)

if(BUILD_FOR_MPI)
  find_package(MPI REQUIRED)
  add_library(mpi_cpu ./mpi/pentad_cpu.cpp)
  add_library(${PROJECT_NAME}::mpi_cpu ALIAS mpi_cpu)

  target_link_libraries(
    mpi_cpu
    PRIVATE OpenMP::OpenMP_CXX ${PROJECT_NAME}::project_options ${PROJECT_NAME}::project_warnings
    PUBLIC MPI::MPI_CXX)

  target_include_directories(mpi_cpu PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                            $<INSTALL_INTERFACE:${PROJECT_NAME}/include>)
endif(BUILD_FOR_MPI)
