add_library(cuda ./pentad_cuda.cu)
add_library(${PROJECT_NAME}::cuda ALIAS cuda)

target_link_libraries(cuda PRIVATE OpenMP::OpenMP_CXX ${PROJECT_NAME}::project_options ${PROJECT_NAME}::project_warnings)
target_link_cuda(cuda)

# TODO CUDA_ARCH??

target_include_directories(cuda PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                       $<INSTALL_INTERFACE:${PROJECT_NAME}/include>)
