add_executable(cuda_tests cuda_tests.cu)
target_link_libraries(
  cuda_tests
  PRIVATE pentadsolver::project_warnings
          pentadsolver::project_options
          pentadsolver::cuda
          catch_main)

target_include_directories(cuda_tests PUBLIC ${PROJECT_SOURCE_DIR})
target_link_cuda(cuda_tests)

# automatically discover tests that are defined in catch based test files you can modify the unittests. Set TEST_PREFIX
# to whatever you want, or use different for different binaries
catch_discover_tests(
  cuda_tests
  TEST_PREFIX
  "unittests."
  REPORTER
  xml
  WORKING_DIRECTORY
  ${PROJECT_SOURCE_DIR}
  OUTPUT_DIR
  .OUTPUT_PREFIX
  "unittests."
  OUTPUT_SUFFIX.xml)
