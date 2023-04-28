add_executable(my_bm_cu my_bm.cu)
#target_link_cuda(cuda)
target_link_libraries(
  my_bm_cu
  PRIVATE OpenMP::OpenMP_CXX
  	  hip::device
          pentadsolver::project_warnings
          pentadsolver::project_options
          pentadsolver::cuda
          benchmark::benchmark)
