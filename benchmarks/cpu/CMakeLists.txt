add_executable(my_bm my_bm.cpp)
target_link_libraries(
  my_bm
  PRIVATE pentadsolver::project_warnings
          pentadsolver::project_options
          pentadsolver::cpu
          benchmark::benchmark)
