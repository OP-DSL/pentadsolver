add_library(cpu ./pentad_cpu.cpp)
add_library(${PROJECT_NAME}::cpu ALIAS cpu)

target_link_libraries(cpu PRIVATE OpenMP::OpenMP_CXX ${PROJECT_NAME}::project_options ${PROJECT_NAME}::project_warnings)

target_include_directories(cpu PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                      $<INSTALL_INTERFACE:${PROJECT_NAME}/include>)
